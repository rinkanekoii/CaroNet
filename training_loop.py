"""
Training Loop for AlphaZero Gomoku.

Key features:
- Learning rate scheduling (warmup + cosine annealing)
- Gradient accumulation for memory efficiency
- Training quality monitoring
- Early warning for training issues
"""

import random
import time
import copy
from evaluate import evaluate_models
import math
from pathlib import Path
import importlib
import platform
import shutil
import os
import csv
import json
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from loss import train_network, train_network_dataloader
from checkpoint_utils import (
    load_checkpoint,
    save_checkpoint,
    unwrap_model,
    export_to_onnx,
)
from models import ModelV8
from replay_buffer import PrioritizedReplayBuffer
from self_play import play_game

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "training_loop",
    "parallel_self_play",
    "create_model",
]


def create_model(args, device):
    """Create model and move to device."""
    from models import get_model_class
    model_class = getattr(args, "model_class", "v8")
    ModelClass = get_model_class(model_class)
    return ModelClass(
        board_size=args.size,
        channels=args.channels,
        num_res_blocks=args.res_blocks,
        dropout=args.dropout,
        use_coords=args.use_coords,
        use_checkpoint=getattr(args, "use_checkpoint", False),
    ).to(device)


def _clone_state_dict(model):
    """Clone an unwrapped, CPU state_dict for compiled/DataParallel-safe reloads."""
    return {
        k: v.detach().cpu().clone()
        for k, v in unwrap_model(model).state_dict().items()
    }


def _clone_state_mapping(state_dict):
    """Clone a plain state mapping while preserving CPU tensors."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _load_state_into_model(model, state_dict, strict: bool = True):
    """Load plain, unwrapped state_dict into compiled/DataParallel-safe model."""
    return unwrap_model(model).load_state_dict(state_dict, strict=strict)


def _push_game_to_buffer(buffer, results, priority):
    if results:
        states = np.stack([s for (s, _, _) in results], axis=0)
        pis = np.stack([p for (_, p, _) in results], axis=0)
        zs = np.asarray([z for (_, _, z) in results], dtype=np.float32)
        prios = np.full(len(results), priority, dtype=np.float32)
        buffer.push_batch(states, pis, zs, priorities=prios)


def _can_use_compile():
    import sys
    if sys.platform == "win32":
        return False
    try:
        import torch._dynamo  # noqa: F401
        return hasattr(torch, "compile")
    except ImportError:
        return False


def _make_grad_scaler(use_amp, device_type):
    """Create gradient scaler for mixed precision training."""
    if not use_amp:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type, enabled=use_amp)
        except TypeError:
            pass
    if device_type == "cuda":
        from torch.cuda.amp import GradScaler as CudaGradScaler

        return CudaGradScaler()
    return None


def create_lr_scheduler(optimizer, warmup_iters, total_iters, min_lr_ratio=0.1):
    """Create learning rate scheduler with non-zero warmup + cosine annealing."""
    warmup_iters = max(0, int(warmup_iters))
    total_iters = max(1, int(total_iters))

    def lr_lambda(current_iter):
        if warmup_iters > 0 and current_iter < warmup_iters:
            # LambdaLR applies this before the first optimizer step, so use current_iter + 1.
            return float(current_iter + 1) / float(max(1, warmup_iters))
        progress = float(current_iter - warmup_iters) / float(
            max(1, total_iters - warmup_iters)
        )
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda)


def _checkpoint_metadata(args):
    return {
        "config": {
            "board_size": int(args.size),
            "win_length": int(args.win_length),
            "channels": int(args.channels),
            "num_res_blocks": int(args.res_blocks),
            "dropout": float(args.dropout),
            "use_coords": bool(args.use_coords),
            "model_class": getattr(args, "model_class", "v8"),
        }
    }


def _append_training_log(log_path: Path, row: dict):
    fieldnames = [
        "iteration",
        "lr",
        "selfplay_time_sec",
        "train_time_sec",
        "buffer_size",
        "p1_wins",
        "p2_wins",
        "draws",
        "avg_game_length",
        "total_loss",
        "policy_loss",
        "value_loss",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


class TrainingMonitor:
    """
    Monitor training quality and detect issues early.
    """

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.policy_losses = deque(maxlen=window_size)
        self.value_losses = deque(maxlen=window_size)
        self.game_lengths = deque(maxlen=window_size)
        self.win_rates = deque(maxlen=window_size)  # P1 win rate

    def update(
        self,
        loss=None,
        policy_loss=None,
        value_loss=None,
        game_length=None,
        p1_wins=None,
        total_games=None,
    ):
        if loss is not None:
            self.losses.append(loss)
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if value_loss is not None:
            self.value_losses.append(value_loss)
        if game_length is not None:
            self.game_lengths.append(game_length)
        if p1_wins is not None and total_games is not None and total_games > 0:
            self.win_rates.append(p1_wins / total_games)

    def check_issues(self):
        """Check for training issues. Returns list of warnings."""
        warnings = []
        losses = list(self.losses)
        game_lengths = list(self.game_lengths)
        win_rates = list(self.win_rates)

        # Check loss explosion
        if len(losses) >= 3:
            recent = losses[-3:]
            if any(v > 10 for v in recent):
                warnings.append("CRITICAL: Loss explosion detected (>10)")
            elif any(v > 5 for v in recent):
                warnings.append("WARNING: High loss values (>5)")

        # Check loss stagnation
        if len(losses) >= self.window_size:
            recent = losses[-self.window_size :]
            if np.std(recent) < 0.01 and np.mean(recent) > 2.0:
                warnings.append("WARNING: Loss appears stagnant")

        # Check game length collapse
        if len(game_lengths) >= 5:
            recent = game_lengths[-5:]
            avg = np.mean(recent)
            if avg < 15:
                warnings.append(
                    f"CRITICAL: Very short games ({avg:.0f} moves) - possible collapse!"
                )
            elif avg < 25:
                warnings.append(f"WARNING: Short games ({avg:.0f} moves)")

        # Check win rate imbalance
        if len(win_rates) >= 5:
            recent = win_rates[-5:]
            avg = np.mean(recent)
            if avg > 0.75 or avg < 0.25:
                warnings.append(f"WARNING: Win rate imbalance (P1: {avg*100:.0f}%)")

        # Check for loss increasing trend
        if len(losses) >= self.window_size:
            first_half = np.mean(
                losses[-self.window_size : -self.window_size // 2]
            )
            second_half = np.mean(losses[-self.window_size // 2 :])
            if second_half > first_half * 1.2:
                warnings.append("WARNING: Loss increasing trend detected")

        return warnings

    def get_summary(self):
        """Get training summary."""
        summary = {}
        losses = list(self.losses)
        game_lengths = list(self.game_lengths)
        win_rates = list(self.win_rates)
        
        if losses:
            summary["avg_loss"] = np.mean(losses[-self.window_size :])
            summary["loss_trend"] = (
                "decreasing"
                if len(losses) >= 2 and losses[-1] < losses[-2]
                else "stable/increasing"
            )
        if game_lengths:
            summary["avg_game_length"] = np.mean(game_lengths[-self.window_size :])
        if win_rates:
            summary["p1_winrate"] = np.mean(win_rates[-self.window_size :])
        return summary



_SELFPLAY_WORKER_NET = None
_SELFPLAY_WORKER_DEVICE = None


def _seed_selfplay_game(seed: int, iteration: int, game_idx: int):
    """Deterministic per-game seed for reproducible parallel/sequential self-play."""
    game_seed = int(seed) + int(iteration) * 10000 + int(game_idx)
    np.random.seed(game_seed)
    random.seed(game_seed)
    torch.manual_seed(game_seed)
    return game_seed


def _start_player_for_game(args, global_game_idx: int) -> int:
    if getattr(args, "alternate_start_player", False):
        return 1 if (int(global_game_idx) % 2 == 0) else -1
    return 1


def _init_selfplay_worker(
    net_state_dict,
    device_str,
    board_size,
    channels,
    res_blocks,
    dropout,
    use_coords,
    use_checkpoint,
    model_class="v8",
):
    """
    ProcessPool initializer for CPU self-play.

    Loading the PyTorch model once per worker is much cheaper than rebuilding it for
    every single game. Humanity survived long enough to invent multiprocessing, so
    we may as well not sabotage it with per-task model construction.
    """
    global _SELFPLAY_WORKER_NET, _SELFPLAY_WORKER_DEVICE
    _SELFPLAY_WORKER_DEVICE = torch.device(device_str)
    try:
        # Avoid every worker spawning a full BLAS party and oversubscribing the CPU.
        torch.set_num_threads(1)
    except Exception:
        pass
    from models import get_model_class
    ModelClass = get_model_class(model_class)
    _SELFPLAY_WORKER_NET = ModelClass(
        board_size=board_size,
        channels=channels,
        num_res_blocks=res_blocks,
        dropout=dropout,
        use_coords=use_coords,
        use_checkpoint=use_checkpoint,
    ).to(_SELFPLAY_WORKER_DEVICE)
    _SELFPLAY_WORKER_NET.load_state_dict(net_state_dict)
    _SELFPLAY_WORKER_NET.eval()


def _run_selfplay_game_in_process(
    net,
    device,
    args,
    game_idx: int,
    game_index_offset: int,
):
    """Run one game directly with an already-created model."""
    global_game_idx = game_index_offset + game_idx
    iteration = game_index_offset // max(1, getattr(args, "games_per_iter", 1))
    _seed_selfplay_game(args.seed, iteration, game_idx)
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
        start_player=_start_player_for_game(args, global_game_idx),
        center_bias_strength=getattr(args, "center_bias_strength", 0.0),
        center_bias_moves=getattr(args, "center_bias_moves", 0),
        adjudication_threshold=getattr(args, "adjudication_threshold", 0.3),
        resign_threshold=getattr(args, "resign_threshold", -1.1),
    )
    return game_res.samples, game_res.winner

@dataclass
class SelfPlayTask:
    game_idx: int
    net_state_dict: dict | None
    device_str: str
    board_size: int
    sims: int
    c_puct: float
    temp_threshold: int
    batch_size: int
    win_length: int
    progressive_widening: bool
    use_coords: bool
    first_noise_moves: int
    channels: int
    res_blocks: int
    dropout: float
    use_checkpoint: bool
    seed: int
    start_player: int
    center_bias_strength: float
    center_bias_moves: int
    onnx_path: str | None
    model_class: str
    shared_onnx_net: Optional[Any]
    shared_pytorch_net: Optional[Any]
    iteration: int
    adjudication_threshold: float
    resign_threshold: float
    use_compile: bool

def _play_single_game(task: SelfPlayTask):
    """Worker function for parallel self-play."""


    game_idx = task.game_idx
    net_state_dict = task.net_state_dict
    device_str = task.device_str
    board_size = task.board_size
    sims = task.sims
    c_puct = task.c_puct
    temp_threshold = task.temp_threshold
    batch_size = task.batch_size
    win_length = task.win_length
    progressive_widening = task.progressive_widening
    use_coords = task.use_coords
    first_noise_moves = task.first_noise_moves
    channels = task.channels
    res_blocks = task.res_blocks
    dropout = task.dropout
    use_checkpoint = task.use_checkpoint
    seed = task.seed
    start_player = task.start_player
    center_bias_strength = task.center_bias_strength
    center_bias_moves = task.center_bias_moves
    onnx_path = task.onnx_path
    model_class = task.model_class
    shared_onnx_net = task.shared_onnx_net
    shared_pytorch_net = getattr(task, 'shared_pytorch_net', None)
    iteration = task.iteration
    adjudication_threshold = task.adjudication_threshold
    resign_threshold = task.resign_threshold
    use_compile = task.use_compile

    # Set different seed per game per iteration
    _seed_selfplay_game(seed, iteration, game_idx)

    device = torch.device(device_str)

    if shared_onnx_net is not None:
        net = shared_onnx_net
    elif shared_pytorch_net is not None:
        net = shared_pytorch_net
    elif onnx_path:
        from mcts import ONNXNetWrapper

        net = ONNXNetWrapper(onnx_path)
    elif _SELFPLAY_WORKER_NET is not None and net_state_dict is None:
        # CPU ProcessPool fast path: model was loaded once in the initializer.
        net = _SELFPLAY_WORKER_NET
        device = _SELFPLAY_WORKER_DEVICE
    else:
        # Fallback path for non-initialized workers.
        from models import get_model_class
        ModelClass = get_model_class(model_class)
        net = ModelClass(
            board_size=board_size,
            channels=channels,
            num_res_blocks=res_blocks,
            dropout=dropout,
            use_coords=use_coords,
            use_checkpoint=use_checkpoint,
        ).to(device)

        net.load_state_dict(net_state_dict)
        if device.type == "cuda":
            net = net.to(memory_format=torch.channels_last)
            # Compile if it's PyTorch 2.0+, we're not using ONNX, and OS is not Windows
            if use_compile and _can_use_compile():
                try:
                    net = torch.compile(net, mode="reduce-overhead")
                except Exception:
                    pass
        net.eval()
    game_res = play_game(
        net,
        device,
        board_size=board_size,
        sims=sims,
        c_puct=c_puct,
        temp_threshold=temp_threshold,
        batch_size=batch_size,
        win_length=win_length,
        progressive_widening=progressive_widening,
        use_coords=use_coords,
        first_noise_moves=first_noise_moves,
        start_player=start_player,
        center_bias_strength=center_bias_strength,
        center_bias_moves=center_bias_moves,
        adjudication_threshold=adjudication_threshold,
        resign_threshold=resign_threshold,
    )
    return game_res.samples, game_res.winner, game_idx


def parallel_self_play(
    net,
    device,
    args,
    num_games,
    num_workers=None,
    use_multiprocessing=False,
    game_index_offset=0,
):
    """
    Run self-play games in parallel using ThreadPoolExecutor or ProcessPoolExecutor.
    For GPU: use threads (share GPU)
    For CPU: use processes (utilize multiple cores)
    """
    if num_workers is None:
        num_workers = min(4, num_games)

    device_str = str(device)
    center_bias_strength = getattr(args, "center_bias_strength", 0.0)
    center_bias_moves = getattr(args, "center_bias_moves", 0)
    iteration = game_index_offset // max(1, getattr(args, "games_per_iter", 1))

    onnx_path = None
    if getattr(args, "use_onnx", False) and not getattr(args, "_disable_onnx_selfplay", False):
        try:
            import onnxruntime  # noqa: F401

            test_path = str(
                Path(getattr(args, "checkpoint_dir", "checkpoints"))
                / "current_model.onnx"
            )
            export_to_onnx(
                net, test_path, board_size=args.size, use_coords=args.use_coords
            )
            onnx_path = test_path
        except ImportError as e:
            print(
                f"ONNX Export/Import failed ({e}), falling back to PyTorch for self-play. Disabling ONNX retries."
            )
            setattr(args, "_disable_onnx_selfplay", True)
            onnx_path = None
        except Exception as e:
            print(
                f"ONNX Export failed ({e}), falling back to PyTorch for self-play. Disabling ONNX retries."
            )
            setattr(args, "_disable_onnx_selfplay", True)
            onnx_path = None

    all_results = []
    win_stats = {1: 0, -1: 0, 0: 0}
    game_lengths = []

    # Use ProcessPoolExecutor for CPU, ThreadPoolExecutor for GPU/ONNX.
    use_process_pool = bool(use_multiprocessing and device.type == "cpu")
    executor_class = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor

    shared_onnx_net = None
    if onnx_path and executor_class == ThreadPoolExecutor:
        from mcts import ONNXNetWrapper

        shared_onnx_net = ONNXNetWrapper(onnx_path)

    worker_initializer = None
    worker_initargs = None
    net_state_dict = None
    if onnx_path is None:
        net_state_dict = _clone_state_dict(net)
        if use_process_pool:
            # Load once per worker process, not once per game.
            worker_initializer = _init_selfplay_worker
            worker_initargs = (
                net_state_dict,
                device_str,
                args.size,
                args.channels,
                args.res_blocks,
                args.dropout,
                args.use_coords,
                getattr(args, "use_checkpoint", False),
                getattr(args, "model_class", "v8"),
            )
            task_state_dict = None
        else:
            task_state_dict = net_state_dict
    else:
        task_state_dict = None

    game_args = [
        SelfPlayTask(
            game_idx=i,
            net_state_dict=task_state_dict,
            device_str=device_str,
            board_size=args.size,
            sims=args.sims,
            c_puct=args.c_puct,
            temp_threshold=args.temp_threshold,
            batch_size=args.batch,
            win_length=args.win_length,
            progressive_widening=args.progressive_widening,
            use_coords=args.use_coords,
            first_noise_moves=args.first_noise_moves,
            channels=args.channels,
            res_blocks=args.res_blocks,
            dropout=args.dropout,
            use_checkpoint=getattr(args, "use_checkpoint", False),
            seed=args.seed,
            start_player=_start_player_for_game(args, game_index_offset + i),
            center_bias_strength=center_bias_strength,
            center_bias_moves=center_bias_moves,
            onnx_path=onnx_path,
            model_class=getattr(args, "model_class", "v8"),
            shared_onnx_net=shared_onnx_net,
            shared_pytorch_net=net if (onnx_path is None and not use_process_pool) else None,
            iteration=iteration,
            adjudication_threshold=getattr(args, "adjudication_threshold", 0.3),
            resign_threshold=getattr(args, "resign_threshold", -1.1),
            use_compile=getattr(args, "use_compile", False),
        )
        for i in range(num_games)
    ]

    executor_kwargs = {"max_workers": num_workers}
    if worker_initializer is not None:
        executor_kwargs["initializer"] = worker_initializer
        executor_kwargs["initargs"] = worker_initargs

    with executor_class(**executor_kwargs) as executor:
        futures = [executor.submit(_play_single_game, arg) for arg in game_args]
        completed_games = 0
        try:
            for future in as_completed(futures):
                idx = completed_games
                try:
                    results, winner, game_idx = future.result()
                    all_results.append((results, game_idx))
                    win_stats[winner] += 1
                    game_lengths.append(len(results))

                    completed_games += 1
                    avg_len = np.mean(game_lengths[-10:]) if game_lengths else 0.0
                    print(
                        f"  Games: {completed_games}/{num_games} | Avg Len(last10): {avg_len:.1f}"
                    )
                except Exception as e:
                    print(f"Game failed in worker: {e}")
        except KeyboardInterrupt:
            print("\n[!] Self-play interrupted. Cancelling pending games...")
            for future in futures:
                future.cancel()
            raise

    completed_count = len(all_results)
    if completed_count == 0:
        raise RuntimeError(
            "All self-play workers failed; refusing to train on stale or empty replay data."
        )
    if completed_count < num_games:
        print(f"[!] Only {completed_count}/{num_games} self-play games completed successfully.")

    return all_results, win_stats, game_lengths


def training_loop(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    if device.type == "cpu":
        # Avoid CPU thread oversubscription during small CPU runs.
        cpu_threads = min(8, max(1, os.cpu_count() or 1))
        try:
            torch.set_num_threads(cpu_threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
        except Exception:
            pass
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Using device: {device}")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    net = create_model(args, device)

    if device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)

    if args.use_compile and hasattr(torch, "compile") and _can_use_compile():
        try:
            net = torch.compile(net)
            print("torch.compile applied to model")
        except Exception as e:
            print(f"torch.compile failed, continuing without compile: {e}")

    # PyTorch bug: fused AdamW fails with GradScaler if we call scaler.unscale_() for grad clipping.
    safe_fused = args.fused_adamw and not (args.use_amp and getattr(args, "grad_clip", 0.0) > 0.0)

    if safe_fused and device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(
                net.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True
            )
            print("Using fused AdamW optimizer")
        except TypeError:
            optimizer = optim.AdamW(
                net.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            print("Fused AdamW not available, falling back to standard AdamW")
    else:
        optimizer = optim.AdamW(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    effective_amp = bool(args.use_amp and device.type == "cuda")
    if args.use_amp and not effective_amp:
        print("AMP disabled on CPU; use CUDA for mixed precision")
    scaler = _make_grad_scaler(effective_amp, device.type)

    # Learning rate scheduler with warmup
    warmup_iters = getattr(args, "warmup_iters", None)
    if warmup_iters is None:
        warmup_iters = max(1, args.iters // 20)
    warmup_iters = min(max(0, int(warmup_iters)), max(1, args.iters))
    scheduler = create_lr_scheduler(
        optimizer, warmup_iters, args.iters, min_lr_ratio=0.1
    )
    print(f"LR Schedule: warmup={warmup_iters} iters, then cosine annealing")

    start_iteration = 0
    if args.init_model:
        try:
            it = load_checkpoint(
                args.init_model, net, optimizer=optimizer, scaler=scaler, device=device,
                strict=(args.model_class == "v8")
            )
            if it is not None:
                start_iteration = int(it)
                # Properly advance scheduler to the correct state
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for _ in range(start_iteration):
                        scheduler.step()
            print(f"Loaded initial model: {args.init_model}")
        except Exception as e:
            raise RuntimeError(
                f"failed to load init_model {args.init_model!r}: {e}"
            ) from e

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Network parameters: {total_params:,}")

    buffer = PrioritizedReplayBuffer(capacity=args.replay_capacity)
    monitor = TrainingMonitor(window_size=10)
    log_path = checkpoint_dir / "training_log.csv"

    # Initialize state dicts
    best_state_dict = _clone_state_dict(net)
    latest_state_dict = _clone_state_dict(net)
    best_net = create_model(args, device)
    _load_state_into_model(best_net, best_state_dict)
    best_net.eval()


    # Determine if we should use parallel self-play
    use_parallel = (
        bool(getattr(args, "parallel_selfplay", False)) and args.games_per_iter > 1
    )
    num_selfplay_workers = getattr(args, "selfplay_workers", None)
    use_dataloader = getattr(args, "use_dataloader", True)

    for iteration in range(start_iteration, args.iters):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'='*60}\nIteration {iteration+1}/{args.iters}\n{'='*60}")
        print(f"Self-play: {args.games_per_iter} games with {args.sims} simulations")
        start_time = time.time()

        # Load best model weights for self-play data generation
        _load_state_into_model(net, best_state_dict)
        net.eval()

        if use_parallel:
            # Parallel self-play
            all_results, win_stats, game_lengths = parallel_self_play(
                net,
                device,
                args,
                num_games=args.games_per_iter,
                num_workers=num_selfplay_workers,
                use_multiprocessing=(device.type == "cpu"),
                game_index_offset=iteration * args.games_per_iter,
            )
            for results, game_idx in all_results:
                priority = 1.0 + (game_idx / max(1, args.games_per_iter))
                _push_game_to_buffer(buffer, results, priority)
        else:
            # Sequential self-play (original behavior)
            win_stats = {1: 0, -1: 0, 0: 0}
            game_lengths = []
            for game_idx in range(args.games_per_iter):
                start_player = _start_player_for_game(args, iteration * args.games_per_iter + game_idx)
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
                    start_player=start_player,
                    center_bias_strength=getattr(args, "center_bias_strength", 0.0),
                    center_bias_moves=getattr(args, "center_bias_moves", 0),
                    adjudication_threshold=getattr(args, "adjudication_threshold", 0.3),
                    resign_threshold=getattr(args, "resign_threshold", -1.1),
                )
                priority = 1.0 + (game_idx / max(1, args.games_per_iter))
                _push_game_to_buffer(buffer, game_res.samples, priority)
                win_stats[game_res.winner] += 1
                game_lengths.append(game_res.num_moves)
                if (game_idx + 1) % 10 == 0 or (game_idx + 1) == args.games_per_iter:
                    print(
                        f"  Games: {game_idx+1}/{args.games_per_iter} | Buffer: {len(buffer)} | Avg Len(last10): {np.mean(game_lengths[-10:]):.1f}"
                    )

        selfplay_time = time.time() - start_time
        total_games = sum(win_stats.values())
        avg_game_len = np.mean(game_lengths) if game_lengths else 0

        print(f"\nSelf-play completed in {selfplay_time:.1f}s")
        print(
            f"  Player 1 wins: {win_stats[1]} ({100*win_stats[1]/max(1,total_games):.1f}%)"
        )
        print(
            f"  Player -1 wins: {win_stats[-1]} ({100*win_stats[-1]/max(1,total_games):.1f}%)"
        )
        print(f"  Draws: {win_stats[0]} ({100*win_stats[0]/max(1,total_games):.1f}%)")
        print(f"  Avg game length: {avg_game_len:.1f} moves")
        print(f"  Buffer size: {len(buffer)}")
        print(f"  Learning rate: {current_lr:.2e}")

        # Update training monitor
        monitor.update(
            game_length=avg_game_len, p1_wins=win_stats[1], total_games=total_games
        )

        did_train = False
        train_time = 0.0
        avg_loss = avg_policy = avg_value = None
        if len(buffer) >= args.batch_train:
            accum = getattr(args, "accumulation_steps", 1)
            micro_batch = (
                max(1, args.batch_train // max(1, accum))
                if accum > 1
                else max(1, args.batch_train)
            )
            print(
                f"\nTraining for {args.train_epochs} epochs (micro_batch={micro_batch}, accum={accum}, effective={args.batch_train})..."
            )
            train_start = time.time()

            # Load latest model weights to resume training
            _load_state_into_model(net, latest_state_dict)

            # Use DataLoader-based training for better GPU utilization
            if use_dataloader and device.type == "cuda":
                avg_loss, avg_policy, avg_value = train_network_dataloader(
                    net,
                    buffer,
                    device,
                    optimizer,
                    scaler=scaler,
                    batch_size=args.batch_train,
                    epochs=args.train_epochs,
                    use_amp=effective_amp,
                    label_smoothing=args.label_smoothing,
                    grad_clip=args.grad_clip,
                    policy_weight=args.policy_weight,
                    value_weight=args.value_weight,
                    entropy_weight=args.entropy_weight,
                    num_workers=getattr(args, "dataloader_workers", 2),
                    accumulation_steps=accum,
                )
            else:
                avg_loss, avg_policy, avg_value = train_network(
                    net,
                    buffer,
                    device,
                    optimizer,
                    scaler=scaler,
                    batch_size=args.batch_train,
                    epochs=args.train_epochs,
                    use_amp=effective_amp,
                    label_smoothing=args.label_smoothing,
                    grad_clip=args.grad_clip,
                    policy_weight=args.policy_weight,
                    value_weight=args.value_weight,
                    entropy_weight=args.entropy_weight,
                    accumulation_steps=getattr(args, "accumulation_steps", 1),
                )

            train_time = time.time() - train_start
            print(f"Training completed in {train_time:.1f}s")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  Policy Loss: {avg_policy:.4f}")
            print(f"  Value Loss: {avg_value:.4f}")

            # Update monitor with loss values
            monitor.update(loss=avg_loss, policy_loss=avg_policy, value_loss=avg_value)
            did_train = True
            
            # Save latest weights
            latest_state_dict = _clone_state_dict(net)

            # Evaluate against best model periodically
            eval_freq = getattr(args, "eval_freq", 10)
            if (iteration + 1) % eval_freq == 0 or (iteration + 1) == getattr(args, "iters", 200):
                _load_state_into_model(best_net, best_state_dict)
                best_net.eval()
                win_rate = evaluate_models(net, best_net, device, args, iteration)
                
                if win_rate >= getattr(args, "eval_win_threshold", 0.55):
                    print(f"[NEW BEST MODEL] Win rate {win_rate*100:.1f}% >= threshold. Promoting!")
                    best_state_dict = _clone_state_dict(net)
                    _load_state_into_model(best_net, best_state_dict)
                    best_net.eval()
                    latest_state_dict = _clone_state_mapping(best_state_dict)
                else:
                    print(f"[REJECTED] Win rate {win_rate*100:.1f}% < threshold. Discarding latest weights.")
                    # Revert latest weights to best
                    latest_state_dict = _clone_state_mapping(best_state_dict)
                    _load_state_into_model(net, best_state_dict)
                    optimizer.state.clear()  # Drop Adam moments from the rejected candidate.
            else:
                print(f"[SKIP ARENA] Evaluation scheduled every {eval_freq} iters.")


        _append_training_log(
            log_path,
            {
                "iteration": iteration + 1,
                "lr": current_lr,
                "selfplay_time_sec": round(selfplay_time, 3),
                "train_time_sec": round(train_time, 3),
                "buffer_size": len(buffer),
                "p1_wins": win_stats[1],
                "p2_wins": win_stats[-1],
                "draws": win_stats[0],
                "avg_game_length": round(float(avg_game_len), 3),
                "total_loss": "" if avg_loss is None else round(float(avg_loss), 6),
                "policy_loss": (
                    "" if avg_policy is None else round(float(avg_policy), 6)
                ),
                "value_loss": "" if avg_value is None else round(float(avg_value), 6),
            },
        )

        # Step learning rate scheduler (only after optimizer updates)
        if did_train:
            scheduler.step()

        # Check for training issues
        training_warnings = monitor.check_issues()
        if training_warnings:
            print("\n[!] Training Warnings:")
            for w in training_warnings:
                print(f"    {w}")

        if (iteration + 1) % args.save_freq == 0:
            checkpoint_path = checkpoint_dir / f"model_iter{iteration+1}.pth"
            save_checkpoint(
                checkpoint_path,
                net,
                optimizer=optimizer,
                scaler=scaler,
                iteration=iteration + 1,
                **_checkpoint_metadata(args),
            )
            print(f"\nCheckpoint saved: {checkpoint_path}")
        
        _load_state_into_model(best_net, best_state_dict)
        best_net.eval()
        best_path = checkpoint_dir / "model_best.pth"
        # Best checkpoint is for evaluation/deployment. Do not attach optimizer
        # state from a different latest/candidate model.
        save_checkpoint(
            best_path,
            best_net,
            iteration=iteration + 1,
            **_checkpoint_metadata(args),
        )

        latest_path = checkpoint_dir / "model_latest.pth"
        _load_state_into_model(net, latest_state_dict)
        net.eval()
        save_checkpoint(
            latest_path,
            net,
            optimizer=optimizer,
            scaler=scaler,
            iteration=iteration + 1,
            **_checkpoint_metadata(args),
        )

    # Print final training summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    summary = monitor.get_summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    final_path = checkpoint_dir / f"model_final_size{args.size}.pth"
    save_checkpoint(
        final_path,
        net,
        optimizer=optimizer,
        scaler=scaler,
        iteration=args.iters,
        **_checkpoint_metadata(args),
    )
    print(f"\nFinal model saved: {final_path}")
    print(f"{'='*60}")
