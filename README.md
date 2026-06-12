# CaroNet

**CaroNet** is an AlphaZero-style training system for Caro/Gomoku.  
It combines **self-play reinforcement learning**, **Monte Carlo Tree Search (MCTS)**, **deep Policy-Value neural networks**, tactical move detection, replay buffer training, checkpoint recovery, and GPU-oriented training optimizations.

The project is designed for research, experimentation, and long-running training of Caro/Gomoku AI models on local machines, Kaggle, Google Colab, or cloud GPU servers.

---

## 1. Overview

CaroNet trains a Caro/Gomoku agent without relying on human game records.  
The model improves by repeatedly playing against itself.

At each move:

1. The neural network evaluates the current board.
2. MCTS searches possible continuations.
3. The MCTS visit distribution becomes the policy target.
4. The final game result becomes the value target.
5. The model is trained from generated self-play data.

The general pipeline is:

```text
Current Model
     |
     v
MCTS-guided Self-Play
     |
     v
Replay Buffer
     |
     v
Policy-Value Training
     |
     v
Updated Checkpoint
     |
     v
Repeat
```

---

## 2. Main Features

- AlphaZero-style self-play training.
- Progressive Monte Carlo Tree Search.
- Deep Policy-Value neural network.
- Support for legacy model checkpoints.
- Tactical move detection for immediate wins, blocks, forks, and double threats.
- Prioritized replay buffer.
- Board symmetry augmentation.
- Checkpoint save/load and resume support.
- Mixed precision training with AMP.
- Optional ONNX Runtime inference (experimental; disabled by default in recommended commands).
- Optional PyTorch compile support.
- CLI-based configuration.
- Training log export.
- Support for local, Kaggle, Colab, and cloud GPU environments.

---

## 3. Core Algorithms

### 3.1 Self-Play Reinforcement Learning

CaroNet uses self-play to generate its own training data.  
Each game produces samples in the form:

```text
(state, policy_target, value_target)
```

| Component | Meaning |
|---|---|
| `state` | Encoded board state from the current player's perspective |
| `policy_target` | MCTS-improved move distribution |
| `value_target` | Final result of the game |

This allows the model to improve through repeated competition against itself.

---

### 3.2 Policy-Value Network

The neural network has two output heads:

| Head | Output | Purpose |
|---|---|---|
| Policy head | Logits over board positions | Predicts promising moves |
| Value head | Scalar in `[-1, 1]` | Predicts expected game outcome |

For a 15x15 board:

```text
15 × 15 = 225 possible moves
```

The policy head outputs 225 logits.  
The value head outputs a scalar evaluation of the current position.

---

### 3.3 Board Representation

A board state is encoded into multiple tensor planes:

```text
[current_player_stones, opponent_stones, empty_cells, row_coordinates, column_coordinates]
```

The coordinate planes help the model understand absolute board location.  
This is useful because center control is important in Caro/Gomoku openings.

---

### 3.4 Monte Carlo Tree Search

MCTS improves raw neural network policy by performing simulations before selecting a move.

Each simulation includes:

1. **Selection**: traverse the tree using a PUCT-style score.
2. **Expansion**: add promising child nodes.
3. **Evaluation**: evaluate a leaf state using the Policy-Value network.
4. **Backup**: propagate the value estimate back through the tree.
5. **Move selection**: choose a move from visit counts.

The main parameter is:

```bash
--sims
```

Higher simulation counts produce stronger self-play targets but require more compute.

---

### 3.5 PUCT Selection

CaroNet uses a PUCT-style formula:

```text
score(s, a) = Q(s, a) + U(s, a)
```

Where:

| Term | Meaning |
|---|---|
| `Q(s, a)` | Estimated value of action `a` |
| `U(s, a)` | Exploration bonus based on policy prior |
| `c_puct` | Exploration strength |

Controlled by:

```bash
--c_puct
```

---

### 3.6 Progressive Widening

Progressive widening avoids expanding all legal moves at once.  
Instead, the number of expanded children grows with node visits.

This reduces search cost in early-game positions where the board has many legal moves.

Controlled by:

```bash
--progressive_widening
--no_progressive_widening
```

---

### 3.7 Tactical Move Detection

Before running full MCTS, CaroNet checks for tactical situations such as:

- immediate winning moves;
- immediate blocking moves;
- fork creation;
- double-threat opportunities.

This improves tactical reliability and prevents wasting simulations on obvious forced moves.

---

### 3.8 Fork and Double-Threat Search

Fork detection checks whether a move creates multiple winning threats.  
The implementation uses localized scanning around candidate moves instead of scanning the whole board, reducing computation during self-play.

This is important because double threats are often decisive in Caro/Gomoku.

---

### 3.9 Opening Exploration

Early-game exploration is controlled by:

```bash
--first_noise_moves
```

Example:

```bash
--first_noise_moves 18
```

This prevents self-play from repeatedly generating the same openings.

---

### 3.10 Center Bias

Caro/Gomoku openings are usually stronger near the center.  
CaroNet supports optional center bias during early moves:

```bash
--center_bias_strength 0.7
--center_bias_moves 10
```

This encourages stronger opening positions during self-play.

---

### 3.11 Alternating Start Player

The flag:

```bash
--alternate_start_player
```

alternates the first player across self-play games.  
This reduces first-player bias in the dataset.

---

## 4. Engineering Optimizations

CaroNet includes multiple optimizations to improve training speed and stability.

### 4.1 Numba-Accelerated Board Logic

Critical board operations are accelerated using Numba JIT compilation:

- win checking;
- neighbor detection;
- tactical mask generation;
- fork detection;
- PUCT child selection.

This reduces Python overhead during MCTS.

---

### 4.2 Zobrist Hashing

MCTS uses Zobrist hashing for efficient board state identification.  
This helps track and reuse search states.

---

### 4.3 Batched Neural Network Evaluation

MCTS supports batched inference through:

```bash
--batch
```

Example:

```bash
--batch 256
```

Batching improves GPU utilization when evaluating multiple leaf nodes.

---

### 4.4 Root Reuse

After a move is selected, the corresponding child node can be reused as the new root.  
This avoids discarding useful search information after each move.

---

### 4.5 Prioritized Replay Buffer

The replay buffer supports weighted priority sampling.  
A Fenwick tree is used for efficient priority updates and sampling.

Benefits:

- faster weighted sampling;
- better use of important positions;
- scalable replay storage.

---

### 4.6 Preallocated Replay Storage

Replay data is stored in preallocated NumPy arrays.  
This avoids repeated memory allocation during long training runs.

---

### 4.7 Board Symmetry Augmentation

CaroNet supports board symmetry augmentation using rotations and flips.

This increases data diversity without requiring more self-play games.

When coordinate planes are used, they are regenerated after augmentation so that absolute board coordinates remain correct.

---

### 4.8 Mixed Precision Training

Automatic mixed precision can be enabled with:

```bash
--use_amp
```

AMP reduces VRAM usage and can speed up training on CUDA GPUs.

---

### 4.9 Gradient Accumulation

Gradient accumulation is controlled by:

```bash
--accumulation_steps
```

In this codebase, accumulation splits the training batch into smaller micro-batches to reduce VRAM usage while preserving the intended training batch behavior.

---

### 4.10 Gradient Clipping

Gradient clipping helps prevent unstable updates:

```bash
--grad_clip 1.0
```

---

### 4.11 Learning Rate Warmup and Cosine Annealing

The training loop supports:

- warmup phase;
- cosine annealing;
- non-zero minimum learning rate ratio.

This improves stability during long-running training.

---

### 4.12 Optional PyTorch Compile

PyTorch compilation can be enabled when supported:

```bash
--use_compile
```

It can also be disabled for compatibility:

```bash
--no_use_compile
```

---

### 4.13 Optional ONNX Runtime

CaroNet includes optional ONNX Runtime inference support, but this path should be treated as experimental.

```bash
--use_onnx
--no_use_onnx
```

The recommended default is:

```bash
--no_use_onnx
```

PyTorch inference is the primary and most compatible backend. ONNX/TensorRT can be useful for speed experiments, but it may fail depending on CUDA, ONNX Runtime, TensorRT, driver versions, model export compatibility, and runtime providers.

Use ONNX only when the environment is known to support it reliably.

---

### 4.14 Parallel Self-Play

Parallel self-play workers are controlled by:

```bash
--selfplay_workers
```

The best value depends on:

- CPU speed;
- GPU model;
- board size;
- number of MCTS simulations;
- memory bandwidth;
- environment limitations.

---

### 4.15 Atomic Checkpoint Saving

Checkpoints are saved through a temporary file and then atomically replaced.  
This reduces the chance of corrupting checkpoint files if training is interrupted.

---

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Neural network training and inference |
| CUDA | GPU acceleration |
| AMP | Mixed precision training |
| NumPy | Board data and replay buffer storage |
| Numba | JIT acceleration for board logic |
| SciPy | Board utility operations |
| ONNX Runtime | Optional experimental inference backend |
| TensorRT | Optional experimental ONNX acceleration backend |
| CSV logging | Training metrics |

---

## 6. Project Structure

```text
CaroNet/
├── training.py              # CLI entry point
├── training_loop.py         # Main training loop
├── self_play.py             # Self-play game generation
├── mcts.py                  # MCTS and tactical search
├── models.py                # Main model architectures
├── legacy_models.py         # Legacy model architectures
├── loss.py                  # Policy-value loss and optimizer logic
├── replay_buffer.py         # Prioritized replay buffer
├── augmentation.py          # Board symmetry augmentation
├── checkpoint_utils.py      # Checkpoint save/load helpers
├── evaluate.py              # Arena evaluation
├── config.py                # CLI argument definitions
├── utils.py                 # Board utilities and state encoding
├── requirements.txt         # Core dependencies
├── requirements-optional.txt# Optional dependencies
├── LICENSE                  # License information
└── README.md                # Project documentation
```

---

## 7. Installation

### 7.1 Core Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
numpy>=1.24
scipy>=1.10
numba>=0.58
torch>=2.0
```

---

### 7.2 Optional Dependencies

These dependencies are only needed for experimental ONNX inference. They are not required for standard PyTorch training.

For experimental ONNX support:

```bash
pip install -r requirements-optional.txt
```

Example `requirements-optional.txt`:

```txt
onnx
onnxruntime
```

ONNX is not required for normal training. The recommended training commands use `--no_use_onnx`.

For GPU ONNX Runtime, install the correct package for the target CUDA environment. Version compatibility can be fragile, so PyTorch should be considered the default backend.

---

---

## 8. ONNX Compatibility Note

ONNX support is included for experimentation, but it is not the default training path.

Recommended default:

```bash
--no_use_onnx
```

Use ONNX only when the environment has compatible versions of ONNX Runtime, CUDA, drivers, and optional TensorRT providers. If ONNX export or runtime inference fails, fall back to PyTorch inference.

## 10. Running CaroNet

CaroNet is executed through the same `training.py` entry point on Linux, macOS, Windows, Kaggle, Colab, or cloud GPU servers.

The only environment-specific part is the path to the checkpoint. Replace:

```text
<CHECKPOINT_DIR>
```

with the actual checkpoint directory.

---

### 8.1 Training Command

The default command disables ONNX because the PyTorch backend is more reliable across local machines, Kaggle, Colab, and cloud GPU environments.

```bash
python training.py --mode train --init_model "<CHECKPOINT_DIR>/model_latest.pth" --model_class v8_legacy --size 15 --channels 320 --res_blocks 20 --iters 200 --games_per_iter 64 --sims 800 --batch 256 --batch_train 128 --accumulation_steps 8 --train_epochs 1 --lr 1e-4 --grad_clip 1.0 --value_weight 1.0 --alternate_start_player --center_bias_strength 0.7 --center_bias_moves 10 --first_noise_moves 18 --resign_threshold -1.1 --use_amp --no_use_compile --no_use_onnx --selfplay_workers 2 --eval_games 0 --save_freq 5
```

For notebook environments such as Kaggle, Colab, or Jupyter, prefix the command with `!`:

```python
!python training.py --mode train --init_model "<CHECKPOINT_DIR>/model_latest.pth" --model_class v8_legacy --size 15 --channels 320 --res_blocks 20 --iters 200 --games_per_iter 64 --sims 800 --batch 256 --batch_train 128 --accumulation_steps 8 --train_epochs 1 --lr 1e-4 --grad_clip 1.0 --value_weight 1.0 --alternate_start_player --center_bias_strength 0.7 --center_bias_moves 10 --first_noise_moves 18 --resign_threshold -1.1 --use_amp --no_use_compile --no_use_onnx --selfplay_workers 2 --eval_games 0 --save_freq 5
```

---

### 8.2 Example Checkpoint Paths

| Environment | Example checkpoint path |
|---|---|
| Local Linux/macOS | `checkpoints/model_latest.pth` |
| Windows | `checkpoints/model_latest.pth` |
| Kaggle | `/kaggle/working/CaroNet/checkpoints/model_latest.pth` |
| Google Colab | `/content/drive/MyDrive/CaroNet/checkpoints/model_latest.pth` |
| Cloud server | `/path/to/CaroNet/checkpoints/model_latest.pth` |

Example:

```bash
python training.py --mode train --init_model "checkpoints/model_latest.pth" --model_class v8_legacy --size 15 --channels 320 --res_blocks 20 --iters 200 --games_per_iter 64 --sims 800 --batch 256 --batch_train 128 --accumulation_steps 8 --train_epochs 1 --lr 1e-4 --grad_clip 1.0 --value_weight 1.0 --alternate_start_player --center_bias_strength 0.7 --center_bias_moves 10 --first_noise_moves 18 --resign_threshold -1.1 --use_amp --no_use_compile --no_use_onnx --selfplay_workers 2 --eval_games 0 --save_freq 5
```


## 10. Running One Self-Play Game

Use self-play mode to test whether the model and MCTS are working:

```bash
python training.py \
    --mode selfplay \
    --init_model checkpoints/model_latest.pth \
    --model_class v8_legacy \
    --size 15 \
    --channels 320 \
    --res_blocks 20 \
    --sims 800
```

---

## 11. Important Parameters

| Parameter | Description |
|---|---|
| `--mode` | `train` or `selfplay` |
| `--init_model` | Path to a checkpoint |
| `--model_class` | Model architecture version |
| `--size` | Board size |
| `--win_length` | Number of stones required to win |
| `--channels` | Model width |
| `--res_blocks` | Number of residual blocks |
| `--sims` | MCTS simulations per move |
| `--c_puct` | PUCT exploration constant |
| `--batch` | MCTS inference batch size |
| `--games_per_iter` | Self-play games per iteration |
| `--batch_train` | Training batch size |
| `--accumulation_steps` | Micro-batch accumulation steps |
| `--train_epochs` | Training passes per iteration |
| `--lr` | Learning rate |
| `--grad_clip` | Gradient clipping threshold |
| `--value_weight` | Value loss weight |
| `--entropy_weight` | Entropy bonus weight |
| `--first_noise_moves` | Number of opening moves with exploration noise |
| `--center_bias_strength` | Strength of center opening bias |
| `--center_bias_moves` | Duration of center bias |
| `--alternate_start_player` | Alternate starting player |
| `--resign_threshold` | Auto-resignation threshold |
| `--use_amp` | Enable mixed precision |
| `--use_compile` | Enable PyTorch compile |
| `--use_onnx` | Enable ONNX inference |
| `--selfplay_workers` | Number of self-play workers |
| `--eval_games` | Number of arena games; use `0` to disable |
| `--save_freq` | Periodic checkpoint save frequency |

---

## 12. Checkpointing

CaroNet saves several checkpoint types:

```text
checkpoints/model_latest.pth
checkpoints/model_best.pth
checkpoints/model_iterX.pth
checkpoints/model_final_size15.pth
```

| File | Meaning |
|---|---|
| `model_latest.pth` | Most recent checkpoint |
| `model_best.pth` | Best checkpoint according to evaluation |
| `model_iterX.pth` | Periodic checkpoint |
| `model_final_size15.pth` | Final checkpoint |

Checkpoint files may store:

- model weights;
- optimizer state;
- AMP scaler state;
- model configuration;
- iteration number.

---

## 13. Logging

Training metrics are written to:

```text
checkpoints/training_log.csv
```

Typical fields:

| Field | Meaning |
|---|---|
| `iteration` | Current iteration |
| `lr` | Learning rate |
| `selfplay_time_sec` | Self-play duration |
| `train_time_sec` | Training duration |
| `buffer_size` | Replay buffer size |
| `p1_wins` | Player 1 wins |
| `p2_wins` | Player 2 wins |
| `draws` | Draw count |
| `avg_game_length` | Average game length |
| `total_loss` | Combined loss |
| `policy_loss` | Policy head loss |
| `value_loss` | Value head loss |

---

## 14. Stability Checks

A healthy training run should show:

```text
No NaN or Inf loss
No repeated CUDA out-of-memory errors
No repeated game-length collapse
Policy loss remains bounded
Value loss remains bounded
First-player win rate is not extremely imbalanced
The model keeps known tactical ability
```

Warning signs:

```text
Average game length repeatedly below 15 moves
Loss explosion
Value loss dominating total loss
Policy collapse
One side winning almost all games
New checkpoint playing worse than old checkpoint
```

---

## 15. Tactical Evaluation

Loss alone is not enough.  
A Caro/Gomoku model should be tested on fixed tactical positions.

Suggested tests:

1. Win in one move.
2. Block opponent's immediate win.
3. Create open three.
4. Create open four.
5. Create double threat.
6. Block double threat.
7. Convert forced win sequence.
8. Avoid obvious trap.
9. Choose strong center opening.
10. Defend against fork setup.

---

## 16. Suggested Experiment Presets

| Preset | Board | Model | Sims | Games/iter | LR | Workers | Purpose |
|---|---:|---|---:|---:|---:|---:|---|
| Debug | 15x15 | v8_legacy 320x20 | 128 | 8 | 1e-4 | 1 | Quick pipeline check |
| Fast test | 15x15 | v8_legacy 320x20 | 256 | 16 | 1e-4 | 1-2 | Short sanity run |
| Stable recovery | 15x15 | v8_legacy 320x20 | 800 | 64 | 1e-4 | 2 | Continue strong checkpoint |
| Heavy training | 15x15 | v8_legacy 320x20 | 800 | 128 | 1e-4 | 2-4 | Long high-quality run |

---

## 17. Environment Notes

### Local GPU

Best for long training if the GPU has enough VRAM.  
Use AMP and tune self-play workers.

### Kaggle

Avoid hardcoded personal paths.  
Use `/kaggle/working` for generated outputs if persistence is needed.

### Google Colab

Mount Drive only if checkpoints must persist.  
Disable `torch.compile` if compatibility issues occur.

### CPU-only

CPU training is possible but slow. Reduce:

```text
sims
games_per_iter
channels
res_blocks
```

---

## 18. Files Not Included

Large generated files should not be committed directly to the repository.

Do not commit:

```text
*.pth
*.pt
*.onnx
*.engine
*.npz
large logs
training outputs
TensorRT cache
```

Use external storage or release assets for large checkpoints.

---

## 19. Future Improvements

Possible next steps:

1. Add a fixed tactical benchmark suite.
2. Add model-vs-model arena outside the training loop.
3. Add policy entropy logging.
4. Add game length histogram logging.
5. Add first-player win-rate tracking.
6. Improve batched MCTS GPU inference.
7. Benchmark ONNX/TensorRT inference as an optional experimental backend.
8. Add a playable UI.
9. Add model export scripts.
10. Compare model sizes and simulation budgets.

---

## 20. License

Copyright (c) 2026 Rin. All rights reserved.

This project is publicly available for viewing and reference purposes only.  
Copying, modifying, redistributing, sublicensing, publishing, selling, or using this code in other projects is not permitted without explicit written permission from the author.

See the `LICENSE` file for details.

---

## 21. Conclusion

CaroNet is a complete AlphaZero-style training system for Caro/Gomoku.  
It combines deep Policy-Value networks, MCTS, self-play, tactical move detection, prioritized replay, board augmentation, checkpoint recovery, and GPU-focused optimization.

The system is suitable for experiments on both model strength and training efficiency.  
A strong checkpoint should be evaluated not only by loss values, but also by practical tactical behavior such as blocking, fork creation, double-threat handling, and forced-win conversion.
