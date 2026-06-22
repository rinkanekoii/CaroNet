"""Board symmetry augmentation shared by replay buffer and standalone tools."""

from __future__ import annotations

import numpy as np


_COORD_CACHE = {}


def _coord_planes(board_size: int):
    """Return cached canonical row/column coordinate planes for the board size."""
    cached = _COORD_CACHE.get(board_size)
    if cached is None:
        lin = np.linspace(-1.0, 1.0, board_size, dtype=np.float32)
        rows = np.broadcast_to(lin[:, None], (board_size, board_size)).copy()
        cols = np.broadcast_to(lin[None, :], (board_size, board_size)).copy()
        cached = (rows, cols)
        _COORD_CACHE[board_size] = cached
    return cached


def _with_canonical_coords(
    state_no_coords: np.ndarray, board_size: int, has_coords: bool
):
    """Append fresh absolute coordinate planes after rotating/flipping occupancy planes."""
    if not has_coords:
        return state_no_coords.astype(np.float32, copy=False)
    rows, cols = _coord_planes(board_size)
    return np.concatenate([state_no_coords, rows[None, :, :], cols[None, :, :]], axis=0).astype(
        np.float32, copy=False
    )


def augment_sample(state: np.ndarray, policy: np.ndarray, board_size: int):
    """Return 8 symmetry variants: 4 rotations times horizontal flip/no flip.

    Only the three board-occupancy planes are transformed. If coordinate planes are
    present, they are regenerated after the transform. Coordinate planes represent
    absolute board positions, so they must stay canonical after augmentation.
    """
    policy_2d = policy.reshape(board_size, board_size)
    original_channels = state.shape[0]
    has_coords = original_channels >= 5
    num_board_planes = original_channels - 2 if has_coords else original_channels
    board_planes = state[:num_board_planes]
    augmented = []
    for rotation in range(4):
        rotated_board = np.rot90(board_planes, rotation, axes=(1, 2)).copy()
        rotated_policy_2d = np.rot90(policy_2d, rotation)
        rotated_policy = rotated_policy_2d.copy().reshape(-1)
        augmented.append(
            (
                _with_canonical_coords(rotated_board, board_size, has_coords),
                rotated_policy,
            )
        )

        flipped_board = np.ascontiguousarray(np.flip(rotated_board, axis=2))
        flipped_policy = np.ascontiguousarray(np.flip(rotated_policy_2d, axis=1)).reshape(-1)
        augmented.append(
            (
                _with_canonical_coords(flipped_board, board_size, has_coords),
                flipped_policy,
            )
        )
    return augmented


def random_augment_single(state: np.ndarray, policy: np.ndarray, board_size: int):
    """Apply one random symmetry transformation. Much faster than augment_sample.
    
    Instead of computing all 8 variants and picking one, this directly
    applies a single random rotation + optional flip.
    """
    policy_2d = policy.reshape(board_size, board_size)
    original_channels = state.shape[0]
    has_coords = original_channels >= 5
    num_board_planes = original_channels - 2 if has_coords else original_channels
    board_planes = state[:num_board_planes]
    
    transform_id = np.random.randint(8)
    rotation = transform_id >> 1  # 0-3
    do_flip = transform_id & 1    # 0 or 1
    
    rotated_board = np.rot90(board_planes, rotation, axes=(1, 2))
    rotated_policy = np.rot90(policy_2d, rotation)
    
    if do_flip:
        rotated_board = np.flip(rotated_board, axis=2)
        rotated_policy = np.flip(rotated_policy, axis=1)
    
    aug_state = _with_canonical_coords(np.ascontiguousarray(rotated_board), board_size, has_coords)
    aug_policy = np.ascontiguousarray(rotated_policy).reshape(-1)
    return aug_state, aug_policy


def augment_batch(states, policies, values, board_size: int):
    """Expand a batch to 8x using all board symmetries."""
    aug_states, aug_policies, aug_values = [], [], []
    for state, policy, value in zip(states, policies, values):
        for aug_state, aug_policy in augment_sample(state, policy, board_size):
            aug_states.append(aug_state)
            aug_policies.append(aug_policy)
            aug_values.append(value)
    return (
        np.asarray(aug_states, dtype=np.float32),
        np.asarray(aug_policies, dtype=np.float32),
        np.asarray(aug_values, dtype=np.float32),
    )


def random_augment_batch(states, policies, values, board_size: int):
    """Apply one random symmetry per sample, keeping batch size unchanged.

    Important: do not call augment_sample() here. That builds all 8 variants and
    then throws 7 away, which is comedy, but not the useful kind.
    """
    n = len(states)
    aug_states = np.empty_like(states)
    aug_policies = np.empty_like(policies)
    for i in range(n):
        aug_states[i], aug_policies[i] = random_augment_single(states[i], policies[i], board_size)
    return aug_states, aug_policies, np.asarray(values, dtype=np.float32)
