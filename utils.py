import numpy as np
from numba import njit
from scipy.ndimage import binary_dilation

__all__ = ["check_win_adaptive", "state_to_tensor", "state_to_tensor_out", "filter_edge_moves"]

_COORD_CACHE = {}

# Pre-allocate once at module level — radius-6 neighborhood (13x13 structuring element)
# Expanded to 13x13 so AI can learn long-range strategic moves
_DILATION_STRUCT = np.ones((13, 13), dtype=bool)


@njit(cache=True, nogil=True)
def check_win_adaptive(board, r, c, player, win_length=5, rule_type=0):
    N = board.shape[0]
    actual_win_length = win_length if win_length <= N else N

    # Horizontal
    count = 1
    left_i = 1
    while c - left_i >= 0 and board[r, c - left_i] == player:
        count += 1
        left_i += 1
    right_i = 1
    while c + right_i < N and board[r, c + right_i] == player:
        count += 1
        right_i += 1
    if count >= actual_win_length:
        if rule_type == 0:
            return True
        else:
            # Edge of board counts as a block (consistent with Caro rules)
            left_blocked = (c - left_i < 0) or (board[r, c - left_i] == -player)
            right_blocked = (c + right_i >= N) or (board[r, c + right_i] == -player)
            if not (left_blocked and right_blocked):
                return True

    # Vertical
    count = 1
    up_i = 1
    while r - up_i >= 0 and board[r - up_i, c] == player:
        count += 1
        up_i += 1
    down_i = 1
    while r + down_i < N and board[r + down_i, c] == player:
        count += 1
        down_i += 1
    if count >= actual_win_length:
        if rule_type == 0:
            return True
        else:
            up_blocked = (r - up_i < 0) or (board[r - up_i, c] == -player)
            down_blocked = (r + down_i >= N) or (board[r + down_i, c] == -player)
            if not (up_blocked and down_blocked):
                return True

    # Diagonal 1 (\)
    count = 1
    up_l = 1
    while r - up_l >= 0 and c - up_l >= 0 and board[r - up_l, c - up_l] == player:
        count += 1
        up_l += 1
    down_r = 1
    while r + down_r < N and c + down_r < N and board[r + down_r, c + down_r] == player:
        count += 1
        down_r += 1
    if count >= actual_win_length:
        if rule_type == 0:
            return True
        else:
            up_l_blocked = (r - up_l < 0 or c - up_l < 0) or (board[r - up_l, c - up_l] == -player)
            down_r_blocked = (r + down_r >= N or c + down_r >= N) or (board[r + down_r, c + down_r] == -player)
            if not (up_l_blocked and down_r_blocked):
                return True

    # Diagonal 2 (/)
    count = 1
    down_l = 1
    while r + down_l < N and c - down_l >= 0 and board[r + down_l, c - down_l] == player:
        count += 1
        down_l += 1
    up_r = 1
    while r - up_r >= 0 and c + up_r < N and board[r - up_r, c + up_r] == player:
        count += 1
        up_r += 1
    if count >= actual_win_length:
        if rule_type == 0:
            return True
        else:
            down_l_blocked = (r + down_l >= N or c - down_l < 0) or (board[r + down_l, c - down_l] == -player)
            up_r_blocked = (r - up_r < 0 or c + up_r >= N) or (board[r - up_r, c + up_r] == -player)
            if not (down_l_blocked and up_r_blocked):
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


def state_to_tensor_out(board, player, out, use_coords=False, rule_type=0):
    """Fill `out` with state planes without allocating a new stacked array."""
    _fill_state_planes(board, player, out)  # single pass, no temporaries
    out[3].fill(float(rule_type))
    if use_coords:
        rows, cols = _get_coord_planes(board.shape[0])
        out[4] = rows
        out[5] = cols
    return out


def state_to_tensor(board, player, use_coords=False, rule_type=0):
    """Return tensor (C, N, N): current player, opponent, empty, rule_type, optionally row/col coords."""
    N = board.shape[0]
    channels = 4 + (2 if use_coords else 0)
    out = np.empty((channels, N, N), dtype=np.float32)
    return state_to_tensor_out(board, player, out, use_coords=use_coords, rule_type=rule_type)


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
        # _DILATION_STRUCT (13x13) covers the radius-6 neighbourhood.
        allowed |= binary_dilation(occupied, structure=_DILATION_STRUCT)

    filtered = {mv: prob for mv, prob in moves.items() if allowed[mv]}
    if not filtered:
        return moves

    total = sum(filtered.values())
    if total <= 0:
        return moves
    inv_total = 1.0 / total
    return {k: v * inv_total for k, v in filtered.items()}
