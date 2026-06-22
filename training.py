#!/usr/bin/env python3
"""AlphaZero Gomoku training CLI."""

import argparse
import torch

from config import add_training_args, normalize_args, print_training_config
from training_loop import create_model, training_loop, _can_use_compile
from self_play import play_game


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Gomoku Training")
    add_training_args(parser)
    return normalize_args(parser.parse_args())


def run_selfplay_once(args):
    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    net = create_model(args, device)
    net.eval()

    if args.use_compile and hasattr(torch, "compile") and _can_use_compile():
        try:
            net = torch.compile(net)
        except Exception as exc:
            print(f"torch.compile skipped: {exc}")

    game_res = play_game(
        net,
        device,
        board_size=args.size,
        sims=args.sims,
        c_puct=args.c_puct,
        temp_threshold=args.temp_threshold,
        batch_size=args.batch,
        win_length=args.win_length,
        progressive_widening=args.progressive_widening,
        use_coords=args.use_coords,
        first_noise_moves=args.first_noise_moves,
        start_player=1,
        center_bias_strength=args.center_bias_strength,
        center_bias_moves=args.center_bias_moves,
        adjudication_threshold=getattr(args, "adjudication_threshold", 0.3),
        resign_threshold=getattr(args, "resign_threshold", -1.1),
        rule_type=getattr(args, "rule_type", 0),
        mixed_rules=getattr(args, "mixed_rules", False),
    )
    print(f"Game finished. Winner: {game_res.winner}, Moves: {game_res.num_moves}")


def main():
    args = parse_args()
    print_training_config(args)

    if args.mode == "train":
        training_loop(args)
    elif args.mode == "selfplay":
        run_selfplay_once(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")


if __name__ == "__main__":
    main()
