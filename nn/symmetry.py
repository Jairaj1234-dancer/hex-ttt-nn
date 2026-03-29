"""D6 symmetry augmentation for hex grid tensors in brick-wall layout.

The dihedral group D6 has 12 elements: 6 rotations (0, 60, 120, 180,
240, 300 degrees) and 6 rotation-then-reflect combinations.  Applying
all 12 symmetries to training data is a powerful regularizer that
teaches the network translational/rotational invariance cheaply.

Because the neural network operates on a *brick-wall* (even-r offset)
rectangular grid rather than native axial coordinates, rotation and
reflection require coordinate remapping:

    1. For each output cell ``(row, col)``, convert to axial ``(q, r)``.
    2. Apply the *inverse* symmetry transform in axial space to find the
       source coordinate.
    3. Convert the source back to brick-wall ``(row_src, col_src)``.
    4. Copy the value from the source cell (or 0 if out-of-bounds).

We pre-compute integer index-remapping tensors for each of the 12
symmetries at a given grid size, so runtime augmentation is a single
``torch.gather`` -- no per-cell branching.
"""

from __future__ import annotations

import functools
from typing import Callable, List, Tuple

import torch


# ======================================================================
# Coordinate conversions (mirrored from game.hex_grid to keep nn/ self-
# contained for deployment / export; these are trivial arithmetic).
# ======================================================================

def _brick_to_axial(row: int, col: int, half: int) -> Tuple[int, int]:
    """Brick-wall (row, col) -> axial (q, r) with window centre at grid midpoint.

    ``half = grid_size // 2``.  The centre cell ``(half, half)`` maps to
    axial ``(0, 0)``.  This is the inverse of :func:`_axial_to_brick`.

    Derivation (from ``axial_to_brick`` in ``game/hex_grid.py`` with
    ``center_q = center_r = 0``, then shifting so grid index ``half``
    corresponds to axial ``0``):

        row = r + half          =>  r = row - half
        col = q + r//2 + half   =>  q = (col - half) - (row - half) // 2
    """
    r_ax = row - half
    q_ax = (col - half) - r_ax // 2
    return (q_ax, r_ax)


def _axial_to_brick(q: int, r: int, half: int) -> Tuple[int, int]:
    """Axial (q, r) -> brick-wall (row, col) with window centre at grid midpoint."""
    row = r + half
    col = q + r // 2 + half
    return (row, col)


# ======================================================================
# Axial symmetry transforms (pure coordinate math)
# ======================================================================

def _rotate_60_axial(q: int, r: int) -> Tuple[int, int]:
    """60-degree clockwise rotation: (q, r) -> (-r, q+r)."""
    return (-r, q + r)


def _reflect_axial(q: int, r: int) -> Tuple[int, int]:
    """Reflect across q=r line: (q, r) -> (r, q)."""
    return (r, q)


def _compose_axial(q: int, r: int, n_rotations: int, do_reflect: bool) -> Tuple[int, int]:
    """Apply ``n_rotations`` 60-degree CW rotations then optionally reflect."""
    for _ in range(n_rotations % 6):
        q, r = _rotate_60_axial(q, r)
    if do_reflect:
        q, r = _reflect_axial(q, r)
    return (q, r)


def _inverse_compose(q: int, r: int, n_rotations: int, do_reflect: bool) -> Tuple[int, int]:
    """Apply the *inverse* of (rotate-n-then-reflect).

    If the forward transform is  T = Reflect . Rot(n),  then
        T^{-1} = Rot(-n) . Reflect     if do_reflect
        T^{-1} = Rot(-n)               otherwise

    (Reflect is its own inverse.)
    """
    if do_reflect:
        q, r = _reflect_axial(q, r)
    inv_rot = (6 - n_rotations) % 6
    for _ in range(inv_rot):
        q, r = _rotate_60_axial(q, r)
    return (q, r)


# ======================================================================
# Pre-computed remapping indices (cached per grid_size)
# ======================================================================

@functools.lru_cache(maxsize=8)
def _build_remap_indices(grid_size: int) -> List[torch.Tensor]:
    """Build flat gather-index tensors for all 12 D6 symmetries.

    Returns a list of 12 tensors, each of shape ``(grid_size * grid_size,)``
    containing the flat source index for every destination cell.  Source
    indices that fall out-of-bounds are set to ``grid_size * grid_size``
    (a sentinel that will be clamped and zeroed after gather).
    """
    half = grid_size // 2
    hw = grid_size * grid_size
    sentinel = hw  # one past last valid index

    transforms: List[Tuple[int, bool]] = []
    for n_rot in range(6):
        transforms.append((n_rot, False))
        transforms.append((n_rot, True))

    indices: List[torch.Tensor] = []
    for n_rot, do_ref in transforms:
        idx = torch.full((hw,), sentinel, dtype=torch.long)
        for dst_row in range(grid_size):
            for dst_col in range(grid_size):
                # Destination cell in axial coords
                q_dst, r_dst = _brick_to_axial(dst_row, dst_col, half)
                # Inverse transform to find source in axial
                q_src, r_src = _inverse_compose(q_dst, r_dst, n_rot, do_ref)
                # Convert source axial back to brick
                src_row, src_col = _axial_to_brick(q_src, r_src, half)
                if 0 <= src_row < grid_size and 0 <= src_col < grid_size:
                    idx[dst_row * grid_size + dst_col] = src_row * grid_size + src_col
        indices.append(idx)

    return indices


def _apply_remap(
    tensor: torch.Tensor,
    idx: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Remap a ``(B, C, H, W)`` tensor using pre-computed flat indices.

    Args:
        tensor: input feature map ``(B, C, H, W)``.
        idx: ``(H*W,)`` flat gather indices (sentinel = H*W for OOB).
        grid_size: spatial dimension (H = W = grid_size).

    Returns:
        Remapped ``(B, C, H, W)`` tensor.  OOB cells are filled with 0.
    """
    B, C, H, W = tensor.shape
    hw = H * W
    device = tensor.device

    idx_dev = idx.to(device)  # (hw,)

    # Clamp sentinel to last valid index (will be zeroed via mask).
    valid_mask = (idx_dev < hw)  # (hw,) bool
    idx_clamped = idx_dev.clamp(max=hw - 1)  # (hw,)

    # Flatten spatial dims, gather, then mask.
    flat = tensor.reshape(B, C, hw)  # (B, C, hw)
    idx_exp = idx_clamped.unsqueeze(0).unsqueeze(0).expand(B, C, hw)  # (B, C, hw)
    gathered = torch.gather(flat, 2, idx_exp)  # (B, C, hw)

    # Zero out OOB positions.
    mask = valid_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, hw)
    gathered = gathered * mask

    return gathered.reshape(B, C, H, W)


def _apply_remap_1d(
    policy: torch.Tensor,
    idx: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Remap a ``(B, H*W)`` policy vector using pre-computed flat indices.

    Args:
        policy: ``(B, H*W)`` policy distribution.
        idx: ``(H*W,)`` flat gather indices.
        grid_size: spatial dimension.

    Returns:
        Remapped ``(B, H*W)`` policy.
    """
    B, hw = policy.shape
    device = policy.device

    idx_dev = idx.to(device)
    valid_mask = (idx_dev < hw)
    idx_clamped = idx_dev.clamp(max=hw - 1)

    idx_exp = idx_clamped.unsqueeze(0).expand(B, hw)  # (B, hw)
    gathered = torch.gather(policy, 1, idx_exp)  # (B, hw)

    mask = valid_mask.unsqueeze(0)  # (1, hw)
    gathered = gathered * mask

    return gathered


# ======================================================================
# Public API
# ======================================================================

def rotate_tensor_60(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate a ``(B, C, H, W)`` brick-wall hex tensor by 60 degrees CW.

    Applies the axial transform ``(q, r) -> (-r, q+r)`` via pre-computed
    index remapping.  Cells that map outside the grid are filled with 0.
    """
    _, _, H, W = tensor.shape
    assert H == W, f"Expected square spatial dims, got {H}x{W}"
    grid_size = H
    all_idx = _build_remap_indices(grid_size)
    # Index 2 is (n_rot=1, reflect=False) -- single 60-degree rotation.
    return _apply_remap(tensor, all_idx[2], grid_size)


def reflect_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reflect a ``(B, C, H, W)`` brick-wall hex tensor.

    Applies the axial transform ``(q, r) -> (r, q)`` via pre-computed
    index remapping.
    """
    _, _, H, W = tensor.shape
    assert H == W, f"Expected square spatial dims, got {H}x{W}"
    grid_size = H
    all_idx = _build_remap_indices(grid_size)
    # Index 1 is (n_rot=0, reflect=True) -- pure reflection.
    return _apply_remap(tensor, all_idx[1], grid_size)


def augment_batch(
    states: torch.Tensor,
    policies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply all 12 D6 symmetries to a batch of states and policy targets.

    Args:
        states: ``(B, C, H, W)`` input feature tensors.
        policies: ``(B, H*W)`` MCTS visit-count distributions.

    Returns:
        ``(states_aug, policies_aug)`` where:
            - ``states_aug``:   ``(12*B, C, H, W)``
            - ``policies_aug``: ``(12*B, H*W)``
    """
    _, _, H, W = states.shape
    assert H == W, f"Expected square spatial dims, got {H}x{W}"
    grid_size = H
    all_idx = _build_remap_indices(grid_size)

    aug_states: List[torch.Tensor] = []
    aug_policies: List[torch.Tensor] = []

    for idx in all_idx:
        aug_states.append(_apply_remap(states, idx, grid_size))
        aug_policies.append(_apply_remap_1d(policies, idx, grid_size))

    return torch.cat(aug_states, dim=0), torch.cat(aug_policies, dim=0)


def get_symmetry_transforms() -> List[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
    """Return a list of 12 D6 transform functions.

    Each function takes ``(state_tensor, policy_tensor)`` where
    ``state_tensor`` is ``(B, C, H, W)`` and ``policy_tensor`` is ``(B, H*W)``,
    and returns the transformed ``(state, policy)`` pair.

    The grid size is inferred from the spatial dimensions of the input.
    """
    transforms: List[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = []

    for sym_idx in range(12):

        def _make_transform(si: int) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
            def transform(
                state: torch.Tensor,
                policy: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                _, _, H, W = state.shape
                assert H == W
                grid_size = H
                all_idx = _build_remap_indices(grid_size)
                idx = all_idx[si]
                return (
                    _apply_remap(state, idx, grid_size),
                    _apply_remap_1d(policy, idx, grid_size),
                )
            return transform

        transforms.append(_make_transform(sym_idx))

    return transforms
