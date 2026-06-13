"""Central CLI defaults for AlphaZero Gomoku training.

Keep commonly changed hyperparameters in one place so the main training code
stays easier to read and maintain.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoardConfig:
    size: int = 15
    win_length: int = 5


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 320
    res_blocks: int = 20
    dropout: float = 0.1
    use_coords: bool = True
    use_checkpoint: bool = True


@dataclass(frozen=True)
class MCTSConfig:
    sims: int = 512
    c_puct: float = 3.5
    batch: int = 256
    temp_threshold: int = 30
    progressive_widening: bool = True
    first_noise_moves: int = 15
    alternate_start_player: bool = False
    center_bias_strength: float = 0.0
    center_bias_moves: int = 0


@dataclass(frozen=True)
class TrainConfig:
    iters: int = 200
    games_per_iter: int = 24
    batch_train: int = 256
    accumulation_steps: int = 4
    train_epochs: int = 2
    lr: float = 7.5e-4
    warmup_iters: int | None = None
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    label_smoothing: float = 0.02
    policy_weight: float = 1.0
    value_weight: float = 2.0
    entropy_weight: float = 0.01
    replay_capacity: int = 250_000


@dataclass(frozen=True)
class RuntimeConfig:
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 8
    init_model: str = ""
    use_cuda: bool = True
    use_amp: bool = True
    use_compile: bool = True
    use_onnx: bool = True
    fused_adamw: bool = True
    parallel_selfplay: bool = True
    selfplay_workers: int | None = 4
    dataloader_workers: int = 2
    use_dataloader: bool = True
    seed: int = 42
    resume_replay_buffer: bool = True
    save_replay_buffer: bool = False
    mode: str = "train"


BOARD = BoardConfig()
MODEL = ModelConfig()
MCTS = MCTSConfig()
TRAIN = TrainConfig()
RUNTIME = RuntimeConfig()


def add_bool_arg(parser, name: str, default: bool, help_text: str = ""):
    """Add --foo / --no_foo flags with a sane default."""
    parser.add_argument(
        f"--{name}", dest=name, action="store_true", default=default, help=help_text
    )
    parser.add_argument(
        f"--no_{name}",
        dest=name,
        action="store_false",
        help=f"Disable {name.replace('_', ' ')}",
    )


def add_training_args(parser):
    """Register all training CLI arguments in one place."""
    parser.add_argument("--size", type=int, default=BOARD.size, help="Board size")
    parser.add_argument(
        "--win_length", type=int, default=BOARD.win_length, help="Win condition length"
    )

    parser.add_argument(
        "--channels", type=int, default=MODEL.channels, help="Model channels"
    )
    parser.add_argument(
        "--res_blocks", type=int, default=MODEL.res_blocks, help="Residual blocks"
    )
    parser.add_argument("--dropout", type=float, default=MODEL.dropout)
    add_bool_arg(parser, "use_coords", MODEL.use_coords, "Use coordinate input planes")

    add_bool_arg(
        parser, "use_checkpoint", MODEL.use_checkpoint, "Use gradient checkpointing"
    )

    parser.add_argument(
        "--sims", type=int, default=MCTS.sims, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--c_puct", type=float, default=MCTS.c_puct, help="PUCT exploration constant"
    )
    parser.add_argument("--batch", type=int, default=MCTS.batch, help="MCTS batch size")
    parser.add_argument(
        "--temp_threshold",
        type=int,
        default=MCTS.temp_threshold,
        help="Temperature threshold move",
    )
    add_bool_arg(parser, "progressive_widening", MCTS.progressive_widening)
    parser.add_argument(
        "--first_noise_moves",
        type=int,
        default=MCTS.first_noise_moves,
        help="Dirichlet noise moves",
    )
    parser.add_argument(
        "--alternate_start_player",
        action="store_true",
        default=MCTS.alternate_start_player,
        help="Alternate the starting player across self-play games",
    )
    parser.add_argument(
        "--center_bias_strength",
        type=float,
        default=MCTS.center_bias_strength,
        help="Boost central moves during early self-play moves",
    )
    parser.add_argument(
        "--center_bias_moves",
        type=int,
        default=MCTS.center_bias_moves,
        help="Number of opening moves affected by center bias",
    )
    parser.add_argument(
        "--adjudication_threshold",
        type=float,
        default=0.3,
        help="Value threshold to adjudicate game at max moves",
    )
    parser.add_argument(
        "--resign_threshold",
        type=float,
        default=-1.1,
        help="Value threshold to auto-resign; <= -1 disables resign for tanh value heads",
    )

    parser.add_argument(
        "--iters", type=int, default=TRAIN.iters, help="Training iterations"
    )
    parser.add_argument(
        "--games_per_iter",
        type=int,
        default=TRAIN.games_per_iter,
        help="Self-play games per iteration",
    )
    parser.add_argument(
        "--batch_train", type=int, default=TRAIN.batch_train, help="Training batch size"
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=TRAIN.accumulation_steps,
        help="Gradient accumulation",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=TRAIN.train_epochs,
        help="Epochs per iteration",
    )
    parser.add_argument("--lr", type=float, default=TRAIN.lr, help="Learning rate")
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=TRAIN.warmup_iters,
        help="LR warmup iterations",
    )
    parser.add_argument("--weight_decay", type=float, default=TRAIN.weight_decay)
    parser.add_argument("--grad_clip", type=float, default=TRAIN.grad_clip)
    parser.add_argument("--label_smoothing", type=float, default=TRAIN.label_smoothing)
    parser.add_argument("--policy_weight", type=float, default=TRAIN.policy_weight)
    parser.add_argument(
        "--value_weight",
        type=float,
        default=TRAIN.value_weight,
        help="Value loss weight",
    )
    parser.add_argument("--entropy_weight", type=float, default=TRAIN.entropy_weight)
    parser.add_argument("--replay_capacity", type=int, default=TRAIN.replay_capacity)

    parser.add_argument("--checkpoint_dir", type=str, default=RUNTIME.checkpoint_dir)
    parser.add_argument(
        "--save_freq",
        type=int,
        default=RUNTIME.save_freq,
        help="Save every N iterations",
    )
    parser.add_argument(
        "--init_model",
        type=str,
        default=RUNTIME.init_model,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="v8",
        choices=["v5", "v6", "v7", "v8", "v8_legacy"],
        help="Model architecture version to use",
    )
    
    # --- Arena Evaluation ---
    parser.add_argument(
        "--eval_games",
        type=int,
        default=8,
        help="Number of games to pit latest model vs best model",
    )
    parser.add_argument(
        "--eval_win_threshold",
        type=float,
        default=0.55,
        help="Win rate threshold for latest model to become new best model",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Run arena evaluation every N iterations",
    )

    add_bool_arg(parser, "use_cuda", RUNTIME.use_cuda, "Use GPU")
    add_bool_arg(parser, "use_amp", RUNTIME.use_amp, "Use mixed precision")

    add_bool_arg(parser, "use_compile", RUNTIME.use_compile, "Use torch.compile()")
    add_bool_arg(parser, "fused_adamw", RUNTIME.fused_adamw, "Fused AdamW optimizer")

    add_bool_arg(parser, "use_onnx", RUNTIME.use_onnx, "Use ONNX Runtime for self-play")

    add_bool_arg(parser, "parallel_selfplay", RUNTIME.parallel_selfplay)
    parser.add_argument(
        "--selfplay_workers", type=int, default=RUNTIME.selfplay_workers
    )
    parser.add_argument(
        "--dataloader_workers", type=int, default=RUNTIME.dataloader_workers
    )
    add_bool_arg(parser, "resume_replay_buffer", RUNTIME.resume_replay_buffer)
    add_bool_arg(parser, "save_replay_buffer", RUNTIME.save_replay_buffer)
    add_bool_arg(parser, "use_dataloader", RUNTIME.use_dataloader)

    parser.add_argument("--seed", type=int, default=RUNTIME.seed)
    parser.add_argument("--mode", choices=["train", "selfplay"], default=RUNTIME.mode)
    parser.add_argument(
        "--model_version", default="v8", help="argparse compatibility only"
    )
    return parser


def normalize_args(args):
    """Clamp invalid CLI combinations before they become runtime nonsense."""
    # Backward compatibility: older commands may pass --model_version instead of --model_class.
    if getattr(args, "model_version", "v8") != "v8" and getattr(args, "model_class", "v8") == "v8":
        args.model_class = args.model_version

    args.size = max(3, int(args.size))
    args.win_length = max(3, min(int(args.win_length), args.size))
    args.channels = max(3, int(args.channels))
    args.res_blocks = max(1, int(args.res_blocks))
    args.dropout = max(0.0, min(0.9, float(args.dropout)))

    args.accumulation_steps = max(1, int(args.accumulation_steps))
    args.batch_train = max(1, int(args.batch_train))
    args.batch = max(1, int(args.batch))
    args.games_per_iter = max(1, int(args.games_per_iter))
    args.iters = max(1, int(args.iters))
    args.train_epochs = max(1, int(args.train_epochs))
    args.center_bias_moves = max(0, int(args.center_bias_moves))
    args.center_bias_strength = max(0.0, float(args.center_bias_strength))

    args.sims = max(1, int(args.sims))
    args.lr = max(1e-8, float(args.lr))
    args.weight_decay = max(0.0, float(args.weight_decay))
    args.grad_clip = max(0.0, float(args.grad_clip))
    args.label_smoothing = max(0.0, min(0.2, float(args.label_smoothing)))
    args.policy_weight = max(0.0, float(args.policy_weight))
    args.value_weight = max(0.0, float(args.value_weight))
    args.entropy_weight = max(0.0, float(args.entropy_weight))
    args.replay_capacity = max(args.batch_train, int(args.replay_capacity))
    args.eval_games = max(0, int(getattr(args, "eval_games", 8)))
    args.eval_freq = max(1, int(getattr(args, "eval_freq", 10)))
    args.eval_win_threshold = max(0.0, min(1.0, float(getattr(args, "eval_win_threshold", 0.55))))

    if args.warmup_iters is not None:
        args.warmup_iters = max(0, int(args.warmup_iters))
    if args.accumulation_steps > args.batch_train:
        args.accumulation_steps = args.batch_train
    return args


def print_training_config(args):
    print(f"""
AlphaZero Gomoku Training
{'='*40}
Board: {args.size}x{args.size} (win={args.win_length})
Model: {getattr(args, 'model_class', 'v8')} | {args.channels}ch x {args.res_blocks} blocks | coords={args.use_coords}
MCTS: {args.sims} sims, c_puct={args.c_puct}
Training: {args.games_per_iter} games/iter, {args.iters} iters
Batch: {args.batch_train} (accum={args.accumulation_steps})
Device: {'CUDA' if args.use_cuda else 'CPU'}
{'='*40}
""")
