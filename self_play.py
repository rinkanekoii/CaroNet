import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

from mcts import ProgressiveMCTS, find_tactical_moves
from utils import check_win_adaptive, state_to_tensor_out

@dataclass
class GameResult:
    samples: List[Tuple[np.ndarray, np.ndarray, float]]
    winner: int
    num_moves: int
    bypassed_count: int
    adjudicated: bool
    resigned: bool

__all__ = ["play_game", "GameResult"]


def _get_temperature(move_count: int, temp_threshold: int) -> float:
    """AlphaZero-style temperature decay."""
    if move_count < temp_threshold:
        return 1.0
    if move_count < temp_threshold * 2:
        return 0.5
    return 0.25


@torch.no_grad()
def play_game(
    net,
    device,
    net2=None,
    board_size: int = 20,
    sims: int = 400,
    c_puct: float = 2.5,
    temp_threshold: int = 15,
    batch_size: int = 64,
    win_length: int = 5,
    progressive_widening: bool = True,
    use_coords: bool = False,
    first_noise_moves: int = 4,
    start_player: int | None = None,
    center_bias_strength: float = 0.0,
    center_bias_moves: int = 0,
    draw_penalty: float = 0.0,
    max_moves: int | None = None,
    noise_eps: float = 0.35,
    dirichlet_alpha: float | None = None,
    pw_alpha: float = 3.0,
    pw_min: int = 5,
    adjudication_threshold: float = 0.3,
    resign_threshold: Optional[float] = -1.1,
    rule_type: int = 0,
    mixed_rules: bool = False,
) -> GameResult:
    if mixed_rules:
        rule_type = int(np.random.choice([0, 1]))

    board = np.zeros((board_size, board_size), dtype=np.int8)
    history = []
    if start_player is None or start_player == 0:
        player = int(np.random.choice([1, -1]))
    else:
        player = int(np.sign(start_player))
        
    move_count = 0
    bypassed_count = 0
    adjudicated = False
    resigned = False

    mcts_p1 = ProgressiveMCTS(
        net,
        board_size=board_size,
        device=device,
        c_puct=c_puct,
        batch_size=batch_size,
        win_length=win_length,
        progressive_widening=progressive_widening,
        use_coords=use_coords,
        noise_eps=noise_eps,
        dirichlet_alpha=dirichlet_alpha,
        pw_alpha=pw_alpha,
        pw_min=pw_min,
        rule_type=rule_type,
    )
    mcts_p1.reset_root()

    mcts_p2 = None
    if net2 is not None:
        mcts_p2 = ProgressiveMCTS(
            net2,
            board_size=board_size,
            device=device,
            c_puct=c_puct,
            batch_size=batch_size,
            win_length=win_length,
            progressive_widening=progressive_widening,
            use_coords=use_coords,
            noise_eps=noise_eps,
            dirichlet_alpha=dirichlet_alpha,
            pw_alpha=pw_alpha,
            pw_min=pw_min,
            rule_type=rule_type,
        )
        mcts_p2.reset_root()

    # ── Pre-allocate buffers used every move ──
    board_cells = board_size * board_size
    # Max moves cap: default to 100% of board cells to let games draw naturally
    if max_moves is None:
        max_moves = board_cells
    max_moves = min(max_moves, board_cells)  # never exceed board capacity

    pi_buffer = np.zeros(board_cells, dtype=np.float32)
    state_channels = 4 + (2 if use_coords else 0)
    state_buffer = np.empty((state_channels, board_size, board_size), dtype=np.float32)

    use_center_bias = center_bias_strength > 0.0 and center_bias_moves > 0

    while True:
        temp = _get_temperature(move_count, temp_threshold)
        add_noise = move_count < first_noise_moves

        # ── Fast Win/Block Check (Bypass MCTS) ──
        winning_moves, blocking_moves = find_tactical_moves(board, player, win_length, rule_type)
        pi_buffer[:] = 0.0

        bypassed_mcts = False
        if winning_moves:
            selected_move = next(iter(winning_moves))
            pi_buffer[selected_move[0] * board_size + selected_move[1]] = 1.0
            bypassed_mcts = True
        elif len(blocking_moves) == 1:
            selected_move = next(iter(blocking_moves))
            pi_buffer[selected_move[0] * board_size + selected_move[1]] = 1.0
            bypassed_mcts = True
        else:
            active_mcts = mcts_p1 if player == 1 or mcts_p2 is None else mcts_p2
            cb_strength = center_bias_strength if (use_center_bias and move_count < center_bias_moves) else 0.0
            counts = active_mcts.run_simulations(
                board, player, num_sims=sims, add_noise=add_noise, center_bias_strength=cb_strength
            )
            if not counts:
                winner = 0
                break

            moves = list(counts.keys())
            num_moves = len(moves)
            visits = np.array([counts[m] for m in moves], dtype=np.float32)

            # ── Build moves_arr once, reuse below ──
            moves_arr = np.array(moves, dtype=np.int32)


            # ── Guard against zero visits ──
            visits_sum = float(visits.sum())
            if visits_sum <= 0:
                visits = np.ones(num_moves, dtype=np.float32)

            # ── Select move ──
            if temp < 0.3:
                idx = int(np.argmax(visits))
            else:
                inv_temp = 1.0 / temp  # temp >= 0.25, always safe
                probs = visits ** inv_temp
                prob_sum = float(probs.sum())
                if prob_sum > 0:
                    probs *= (1.0 / prob_sum)
                else:
                    probs = np.full(num_moves, 1.0 / num_moves, dtype=np.float32)
                idx = np.random.choice(num_moves, p=probs)
            selected_move = moves[idx]

            # ── Build policy target (reuse pi_buffer) ──
            indices = moves_arr[:, 0] * board_size + moves_arr[:, 1]
            
            # AlphaZero standard: always use tau=1.0 for the target policy distribution.
            pi_buffer[indices] = visits
            v_sum = float(visits.sum())
            if v_sum > 0:
                pi_buffer[indices] *= (1.0 / v_sum)
            else:
                pi_buffer[indices] = 1.0 / num_moves

        # ── Record state (reuse state_buffer to avoid allocation) ──
        state_to_tensor_out(board, player, state_buffer, use_coords=use_coords, rule_type=rule_type)
        history.append((state_buffer.copy(), pi_buffer.copy(), player))

        r, c = selected_move
        if board[r, c] != 0:
            raise RuntimeError(
                f"Illegal move selected at ({r}, {c}); cell already contains {board[r, c]}"
            )
        board[r, c] = player
        move_count += 1

        if bypassed_mcts:
            bypassed_count += 1
            
        mcts_p1.move_root(selected_move)
        if mcts_p2 is not None:
            mcts_p2.move_root(selected_move)

        if check_win_adaptive(board, r, c, player, win_length, rule_type):
            winner = player
            break
        if move_count >= board_cells:
            winner = 0
            break

        next_player = -player

        # ── Resign mechanism ──
        if resign_threshold is not None:
            # Evaluate from the perspective of the player who is about to move.
            state_to_tensor_out(board, next_player, state_buffer, use_coords=use_coords, rule_type=rule_type)
            inp = torch.from_numpy(state_buffer).unsqueeze(0).to(device)
            active_net = net if (next_player == 1 or net2 is None) else net2
            with torch.inference_mode():
                val, _ = active_net(inp)
            v = float(val.item())
            
            if v < resign_threshold:
                winner = -next_player  # Side to move resigns; opponent wins.
                resigned = True
                break

        # ── Max moves cap: use value network to adjudicate ──
        if move_count >= max_moves:
            # Evaluate from the perspective of the player who is about to move.
            state_to_tensor_out(board, next_player, state_buffer, use_coords=use_coords, rule_type=rule_type)
            inp = torch.from_numpy(state_buffer).unsqueeze(0).to(device)
            active_net = net if (next_player == 1 or net2 is None) else net2
            with torch.inference_mode():
                val, _ = active_net(inp)
            v = float(val.item())
            
            if v > adjudication_threshold:
                winner = next_player
            elif v < -adjudication_threshold:
                winner = -next_player
            else:
                winner = 0
            adjudicated = True
            break
        player = next_player

    # ── Build results ──
    if winner == 0:
        # Draw penalty applies a small negative reward to BOTH players when configured.
        dp = -abs(draw_penalty)
        results = [(state, pi, dp) for state, pi, _ in history]
    else:
        results = [
            (state, pi, 1.0 if winner == p else -1.0)
            for state, pi, p in history
        ]
        
    return GameResult(
        samples=results,
        winner=winner,
        num_moves=move_count,
        bypassed_count=bypassed_count,
        adjudicated=adjudicated,
        resigned=resigned,
    )
