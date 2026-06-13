import os
import torch
import numpy as np
from pathlib import Path

from self_play import play_game

def evaluate_models(net_new, net_old, device, args, iteration):
    """
    Pit net_new against net_old.
    Returns the win rate of net_new.
    net_new plays as Player 1, net_old plays as Player -1.
    """
    num_games = getattr(args, "eval_games", 20)
    if num_games <= 0:
        print("\n--- ARENA DISABLED: eval_games <= 0; no win-rate will be reported. ---")
        return None

    print(f"\n--- ARENA: Evaluating latest model vs best model ({num_games} games) ---")
    
    net_new.eval()
    net_old.eval()

    win_count_new = 0
    draw_count = 0

    for i in range(num_games):
        start_player = 1 if i % 2 == 0 else -1
        
        game_res = play_game(
            net=net_new,
            device=device,
            net2=net_old,
            board_size=args.size,
            sims=args.sims,
            c_puct=args.c_puct,
            temp_threshold=0,  # Greedy play
            batch_size=args.batch,
            win_length=args.win_length,
            progressive_widening=args.progressive_widening,
            use_coords=args.use_coords,
            first_noise_moves=0,  # No initial noise
            start_player=start_player,
            center_bias_strength=0.0,
            center_bias_moves=0,
            adjudication_threshold=getattr(args, "adjudication_threshold", 0.3),
            resign_threshold=None,
        )
        
        # In sequential play_game, play_game returns GameResult
        winner = game_res.winner

        if winner == 1:
            win_count_new += 1
            res_str = "NEW won"
        elif winner == -1:
            res_str = "OLD won"
        else:
            draw_count += 1
            res_str = "DRAW"
            
        print(f"  Arena Game {i+1}/{num_games}: {res_str} (start: {'NEW' if start_player == 1 else 'OLD'})")

    score = win_count_new + (draw_count * 0.5)
    win_rate = score / num_games
    print(f"Arena Result: NEW won {win_count_new}, OLD won {num_games - win_count_new - draw_count}, Draws {draw_count}")
    print(f"Win Rate: {win_rate * 100:.1f}%")
    
    return win_rate
