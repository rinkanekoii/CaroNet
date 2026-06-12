import numpy as np
from numba import njit
from scipy.ndimage import binary_dilation

__all__ = ["check_win_adaptive", "state_to_tensor", "state_to_tensor_out", "filter_edge_moves"]

_COORD_CACHE = {}

# Pre-allocate once at module level — radius-2 neighborhood (5x5 structuring element)
_DILATION_STRUCT = np.ones((5, 5), dtype=bool)


@njit(cache=True, nogil=True)
def check_win_adaptive(board, r, c, player, win_length=5):
    N = board.shape[0]
    actual_win_length = win_length if win_length <= N else N  # avoid int() cast

    count = 1
    for i in range(1, actual_win_length):
        if c + i < N and board[r, c + i] == player:
            count += 1
        else:
            break
    for i in range(1, actual_win_length):
        if c - i >= 0 and board[r, c - i] == player:
            count += 1
        else:
            break
    if count >= actual_win_length:
        return True

    count = 1
    for i in range(1, actual_win_length):
        if r + i < N and board[r + i, c] == player:
            count += 1
        else:
            break
    for i in range(1, actual_win_length):
        if r - i >= 0 and board[r - i, c] == player:
            count += 1
        else:
            break
    if count >= actual_win_length:
        return True

    count = 1
    for i in range(1, actual_win_length):
        if r + i < N and c + i < N and board[r + i, c + i] == player:
            count += 1
        else:
            break
    for i in range(1, actual_win_length):
        if r - i >= 0 and c - i >= 0 and board[r - i, c - i] == player:
            count += 1
        else:
            break
    if count >= actual_win_length:
        return True

    count = 1
    for i in range(1, actual_win_length):
        if r + i < N and c - i >= 0 and board[r + i, c - i] == player:
            count += 1
        else:
            break
    for i in range(1, actual_win_length):
        if r - i >= 0 and c + i < N and board[r - i, c + i] == player:
            count += 1
        else:
            break
    if count >= actual_win_length:
        return True

    return False


@njit(cache=True)
def _fill_state_planes(board, player, out):
    """Single-pass, zero-allocation state plane fill.

    Replaces three separate numpy boolean comparisons (which each create a
    temporary bool array and then cast to float32) with one JIT-compiled loop
    that writes directly into `out`.
    """
    N = board.shape[0]
    opp = -player
    for r in range(N):
        for c in range(N):
            v = board[r, c]
            out[0, r, c] = 1.0 if v == player else 0.0
            out[1, r, c] = 1.0 if v == opp   else 0.0
            out[2, r, c] = 1.0 if v == 0     else 0.0


def _get_coord_planes(N: int):
    """Return cached row/column coordinate planes for an NxN board."""
    cached = _COORD_CACHE.get(N)
    if cached is None:
        # broadcast_to avoids the extra multiply-by-ones allocation
        lin = np.linspace(-1.0, 1.0, N, dtype=np.float32)
        rows = np.broadcast_to(lin[:, None], (N, N)).copy()
        cols = np.broadcast_to(lin[None, :], (N, N)).copy()
        cached = (rows, cols)
        _COORD_CACHE[N] = cached
    return cached


def state_to_tensor_out(board, player, out, use_coords=False):
    """Fill `out` with state planes without allocating a new stacked array."""
    _fill_state_planes(board, player, out)  # single pass, no temporaries
    if use_coords:
        rows, cols = _get_coord_planes(board.shape[0])
        out[3] = rows
        out[4] = cols
    return out


def state_to_tensor(board, player, use_coords=False):
    """Return tensor (C, N, N): current player, opponent, empty, optionally row/col coords."""
    N = board.shape[0]
    channels = 3 + (2 if use_coords else 0)
    out = np.empty((channels, N, N), dtype=np.float32)
    return state_to_tensor_out(board, player, out, use_coords=use_coords)


def filter_edge_moves(board, moves, margin=2):
    """Filter edge moves using one precomputed allowed-cell mask."""
    if not moves:
        return moves

    N = board.shape[0]
    allowed = np.zeros((N, N), dtype=bool)

    if margin <= 0:
        allowed[:, :] = True
    elif 2 * margin < N:
        allowed[margin:N - margin, margin:N - margin] = True

    occupied = board != 0
    if occupied.any():
        # Vectorized dilation replaces the O(occupied) Python loop.
        # _DILATION_STRUCT (5x5) covers the same radius-2 neighbourhood.
        allowed |= binary_dilation(occupied, structure=_DILATION_STRUCT)

    filtered = {mv: prob for mv, prob in moves.items() if allowed[mv]}
    if not filtered:
        return moves

    total = sum(filtered.values())
    if total <= 0:
        return moves
    inv_total = 1.0 / total
    return {k: v * inv_total for k, v in filtered.items()}
