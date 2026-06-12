# Gomoku AlphaZero Training System

A GPU-accelerated AlphaZero-style training system for Gomoku/Caro. The project combines **self-play reinforcement learning**, **Monte Carlo Tree Search (MCTS)**, **deep Policy-Value neural networks**, tactical search shortcuts, prioritized replay, checkpoint recovery, and multiple performance optimizations for long-running training.

The codebase is designed to run in different environments, including local machines, Google Colab, Kaggle notebooks, and cloud GPU instances. Paths and runtime flags should be adjusted according to the target environment.

---

## 1. What This Project Does

This project trains a Gomoku agent without human game records. The model improves by repeatedly playing against itself.

At each move:

1. The neural network evaluates the board.
2. MCTS searches candidate continuations.
3. The search visit distribution becomes the training target.
4. The final game outcome becomes the value target.

The training loop is:

```text
Current checkpoint
      |
      v
Self-play with MCTS
      |
      v
Replay buffer
      |
      v
Policy-Value training
      |
      v
New checkpoint
      |
      v
Repeat
```

---

## 2. Core Algorithms

### 2.1 AlphaZero-style Self-Play

The system follows the AlphaZero idea: instead of learning from expert games, the model creates its own dataset through self-play.

Each generated sample contains:

```text
(state, policy_target, value_target)
```

Where:

| Component | Meaning |
|---|---|
| `state` | Encoded board state from the current player's perspective |
| `policy_target` | Improved move distribution produced by MCTS |
| `value_target` | Final game result from the current player's perspective |

The network learns both move selection and position evaluation.

---

### 2.2 Policy-Value Network

The neural network has two heads:

| Head | Output | Purpose |
|---|---|---|
| Policy head | Move logits over all board cells | Predicts promising moves |
| Value head | Scalar in `[-1, 1]` | Predicts win/loss expectation |

For a 15x15 board, the policy head outputs:

```text
15 × 15 = 225 logits
```

The value output represents the expected result:

| Value | Interpretation |
|---:|---|
| `1.0` | Current player is winning |
| `0.0` | Balanced or uncertain position |
| `-1.0` | Current player is losing |

---

### 2.3 Board State Encoding

The board is encoded using multiple planes:

```text
[current_player_stones, opponent_stones, empty_cells, row_coordinates, column_coordinates]
```

The optional coordinate planes help the network distinguish center, edge, and corner positions. This matters in Gomoku because opening strength is not position-invariant in practical play.

---

### 2.4 Monte Carlo Tree Search

MCTS improves the raw policy from the neural network by running simulations before selecting a move.

Each MCTS simulation performs:

1. **Selection**: choose a child node using PUCT.
2. **Expansion**: add candidate child moves.
3. **Evaluation**: use the Policy-Value network to evaluate a leaf position.
4. **Backup**: propagate the value estimate through the path.
5. **Move selection**: use visit counts to form the final policy target.

The key MCTS parameter is:

```bash
--sims
```

Example:

```bash
--sims 800
```

Higher simulation counts produce stronger targets but increase self-play time significantly.

---

### 2.5 PUCT Selection

The MCTS implementation uses a PUCT-style selection score:

```text
score(s, a) = Q(s, a) + U(s, a)
```

Where:

| Term | Meaning |
|---|---|
| `Q(s, a)` | Estimated value of action `a` |
| `U(s, a)` | Exploration bonus based on prior probability |
| `c_puct` | Controls exploration strength |

The parameter is controlled by:

```bash
--c_puct
```

---

### 2.6 Progressive Widening

The MCTS implementation supports progressive widening. Instead of expanding every legal move at once, it gradually expands more moves as the node receives more visits.

This reduces wasted computation in early-game positions where the branching factor is large.

Controlled by:

```bash
--progressive_widening
--no_progressive_widening
```

---

### 2.7 Tactical Move Bypass

Before running full MCTS, the self-play logic checks tactical conditions such as:

- immediate winning move;
- immediate blocking move;
- fork or double-threat creation.

If a forced tactical move exists, the system can bypass full MCTS and directly assign the policy target to that move.

This improves tactical reliability and avoids wasting simulations on obvious forced positions.

---

### 2.8 Fork and Double-Threat Detection

The code includes logic for detecting moves that create multiple winning threats. Instead of scanning the whole board for every candidate, the optimized fork search checks only a local window around the candidate move.

This is important for Gomoku because double threats are often decisive.

---

### 2.9 Zobrist Hashing

MCTS uses Zobrist hashing to represent board states efficiently. Hashing allows fast state identification and supports search reuse across moves.

This helps reduce repeated computation during tree search.

---

### 2.10 Opening Exploration

Early self-play moves can receive exploration noise:

```bash
--first_noise_moves 18
```

This prevents the model from always generating the same openings and improves dataset diversity.

---

### 2.11 Center Bias

The training pipeline supports optional opening center bias:

```bash
--center_bias_strength 0.7
--center_bias_moves 10
```

This encourages early self-play moves near the board center. It is useful because center control is usually stronger in Gomoku openings.

---

### 2.12 Alternating Start Player

The flag:

```bash
--alternate_start_player
```

alternates the first player across self-play games. This reduces first-player bias in the generated dataset.

---

## 3. Engineering and Performance Optimizations

The project includes several algorithmic and systems-level optimizations.

### 3.1 Numba-Accelerated Board Logic

Critical board operations are accelerated with Numba JIT compilation, including:

- win checking;
- neighbor detection;
- tactical winning/blocking masks;
- fork detection;
- PUCT child selection.

This reduces Python overhead inside MCTS.

---

### 3.2 Localized Tactical Scanning

Fork detection only scans within the win-length radius of a candidate move instead of scanning the whole board. This reduces unnecessary board checks during tactical analysis.

---

### 3.3 Progressive MCTS Expansion

Progressive widening limits how many child nodes are expanded early, reducing search cost in high-branching positions.

---

### 3.4 Batched Neural Network Evaluation

MCTS supports batched neural network evaluation through the `--batch` parameter:

```bash
--batch 256
```

Batching improves GPU utilization when many leaf nodes need network inference.

---

### 3.5 Root Reuse

After a move is selected, the MCTS tree can reuse the corresponding child node as the new root. This avoids discarding all search information after every move.

---

### 3.6 Prioritized Replay Buffer

The replay buffer uses a Fenwick tree for efficient weighted priority sampling. This allows sampling important positions more efficiently than uniform-only replay.

---

### 3.7 Preallocated Replay Storage

The replay buffer uses preallocated NumPy arrays for states, policies, values, and priorities. This avoids repeated memory allocation during long training runs.

---

### 3.8 Board Symmetry Augmentation

The augmentation module supports rotations and flips of board states and policy targets. This increases data diversity without requiring extra self-play games.

When coordinate planes are used, they are regenerated after augmentation to preserve correct absolute coordinates.

---

### 3.9 Mixed Precision Training

The training loop supports automatic mixed precision:

```bash
--use_amp
```

AMP can reduce VRAM usage and speed up training on CUDA GPUs.

---

### 3.10 Channels-Last Memory Format

When CUDA is used, the model can be moved to channels-last memory format, which may improve convolution performance on supported GPUs.

---

### 3.11 TF32 Acceleration

On compatible NVIDIA GPUs, TF32 matrix multiplication can be enabled for faster training and inference.

---

### 3.12 Gradient Accumulation

The training code supports gradient accumulation:

```bash
--accumulation_steps 8
```

In this codebase, accumulation splits the training batch into smaller micro-batches to reduce VRAM usage while preserving the intended effective batch behavior.

---

### 3.13 Gradient Clipping

Gradient clipping improves stability:

```bash
--grad_clip 1.0
```

This prevents very large gradient updates from destabilizing training.

---

### 3.14 Learning Rate Warmup and Cosine Annealing

The optimizer uses a learning rate schedule with:

1. warmup phase;
2. cosine annealing phase;
3. non-zero minimum learning rate ratio.

This helps avoid unstable early updates while still reducing learning rate over time.

---

### 3.15 Fused AdamW Fallback

The training loop tries to use fused AdamW when safe and available. If it is not supported, it falls back to standard AdamW.

---

### 3.16 Optional `torch.compile`

The code supports optional PyTorch compilation:

```bash
--use_compile
--no_use_compile
```

This can improve performance in some Linux/CUDA environments, but it may be disabled for compatibility.

---

### 3.17 Optional ONNX Runtime Inference

The project includes an ONNX wrapper for inference acceleration. It can use available providers such as CUDA or TensorRT when supported.

Controlled by:

```bash
--use_onnx
--no_use_onnx
```

---

### 3.18 Parallel Self-Play

The training loop supports parallel self-play workers:

```bash
--selfplay_workers 2
```

On CPU, multiprocessing can be used. On CUDA, thread-based parallelism can be used to share the GPU model.

The optimal number of workers depends on:

- CPU speed;
- GPU type;
- board size;
- MCTS simulations;
- batch size;
- memory bandwidth.

---

### 3.19 Atomic Checkpoint Saving

Checkpoint saving uses a temporary file followed by atomic replacement. This reduces the chance of corrupting a checkpoint if the process is interrupted while saving.

---

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main language |
| PyTorch | Neural network training and inference |
| CUDA | GPU acceleration |
| AMP | Mixed precision training |
| NumPy | Board and replay data handling |
| Numba | JIT acceleration for board logic |
| SciPy | Board utility operations |
| ONNX Runtime | Optional inference backend |
| TensorRT | Optional accelerated ONNX provider |
| CSV logging | Training metric tracking |

---

## 5. Project Structure

```text
Gomoku-Training/
├── training.py              # CLI entry point
├── training_loop.py         # Main training loop, checkpointing, scheduling
├── self_play.py             # Self-play game generation
├── mcts.py                  # Progressive MCTS and tactical search
├── models.py                # Main model architectures
├── legacy_models.py         # Legacy-compatible model architectures
├── loss.py                  # Policy-value loss and optimizer steps
├── replay_buffer.py         # Prioritized replay buffer
├── augmentation.py          # Board symmetry augmentation
├── checkpoint_utils.py      # Checkpoint save/load helpers
├── evaluate.py              # Arena evaluation
├── config.py                # CLI arguments and default configs
├── utils.py                 # Board encoding and win checking
└── checkpoints/             # Saved checkpoints
```

---

## 6. Installation

### 6.1 Basic Requirements

Install the main dependencies:

```bash
pip install torch numpy scipy numba
```

For ONNX inference:

```bash
pip install onnx onnxruntime
```

For GPU ONNX Runtime, install the correct package for the target environment.

---

## 7. Running the Project

The repository is environment-agnostic. Use the same `training.py` entry point on local machines, Kaggle, Colab, or cloud GPU servers.

---

### 7.1 Generic Training Command

Use this form for any environment:

```bash
python training.py \
    --mode train \
    --init_model "<PATH_TO_CHECKPOINT>/model_latest.pth" \
    --model_class v8_legacy \
    --size 15 \
    --channels 320 \
    --res_blocks 20 \
    --iters 200 \
    --games_per_iter 64 \
    --sims 800 \
    --batch 256 \
    --batch_train 128 \
    --accumulation_steps 8 \
    --train_epochs 1 \
    --lr 1e-4 \
    --grad_clip 1.0 \
    --value_weight 1.0 \
    --alternate_start_player \
    --center_bias_strength 0.7 \
    --center_bias_moves 10 \
    --first_noise_moves 18 \
    --resign_threshold -1.1 \
    --use_amp \
    --no_use_compile \
    --no_use_onnx \
    --selfplay_workers 2 \
    --eval_games 0 \
    --save_freq 5
```

Replace:

```text
<PATH_TO_CHECKPOINT>
```

with the actual checkpoint directory.

---

### 7.2 Local Linux / macOS

```bash
cd /path/to/Gomoku-Training

python training.py \
    --mode train \
    --init_model checkpoints/model_latest.pth \
    --model_class v8_legacy \
    --size 15 \
    --channels 320 \
    --res_blocks 20 \
    --iters 200 \
    --games_per_iter 64 \
    --sims 800 \
    --batch 256 \
    --batch_train 128 \
    --accumulation_steps 8 \
    --train_epochs 1 \
    --lr 1e-4 \
    --grad_clip 1.0 \
    --value_weight 1.0 \
    --alternate_start_player \
    --center_bias_strength 0.7 \
    --center_bias_moves 10 \
    --first_noise_moves 18 \
    --resign_threshold -1.1 \
    --use_amp \
    --no_use_compile \
    --no_use_onnx \
    --selfplay_workers 2 \
    --eval_games 0 \
    --save_freq 5
```

---

### 7.3 Windows PowerShell

```powershell
cd "C:\path\to\Gomoku-Training"

python training.py `
    --mode train `
    --init_model "checkpoints\model_latest.pth" `
    --model_class v8_legacy `
    --size 15 `
    --channels 320 `
    --res_blocks 20 `
    --iters 200 `
    --games_per_iter 64 `
    --sims 800 `
    --batch 256 `
    --batch_train 128 `
    --accumulation_steps 8 `
    --train_epochs 1 `
    --lr 1e-4 `
    --grad_clip 1.0 `
    --value_weight 1.0 `
    --alternate_start_player `
    --center_bias_strength 0.7 `
    --center_bias_moves 10 `
    --first_noise_moves 18 `
    --resign_threshold -1.1 `
    --use_amp `
    --no_use_compile `
    --no_use_onnx `
    --selfplay_workers 2 `
    --eval_games 0 `
    --save_freq 5
```

---

### 7.4 Notebook Environments

For notebook environments such as Kaggle or Colab, set the project directory first:

```python
PROJECT_DIR = "/path/to/Gomoku-Training"
%cd $PROJECT_DIR
```

Then run:

```python
!python training.py \
    --mode train \
    --init_model "checkpoints/model_latest.pth" \
    --model_class v8_legacy \
    --size 15 \
    --channels 320 \
    --res_blocks 20 \
    --iters 200 \
    --games_per_iter 64 \
    --sims 800 \
    --batch 256 \
    --batch_train 128 \
    --accumulation_steps 8 \
    --train_epochs 1 \
    --lr 1e-4 \
    --grad_clip 1.0 \
    --value_weight 1.0 \
    --alternate_start_player \
    --center_bias_strength 0.7 \
    --center_bias_moves 10 \
    --first_noise_moves 18 \
    --resign_threshold -1.1 \
    --use_amp \
    --no_use_compile \
    --no_use_onnx \
    --selfplay_workers 2 \
    --eval_games 0 \
    --save_freq 5
```

---

## 8. Running One Self-Play Game

To test whether the model and MCTS run correctly:

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

## 9. Important Parameters

| Parameter | Description |
|---|---|
| `--mode` | `train` or `selfplay` |
| `--init_model` | Path to checkpoint for resume/fine-tuning |
| `--model_class` | Model architecture version |
| `--size` | Board size |
| `--win_length` | Number of stones needed to win |
| `--channels` | Model width |
| `--res_blocks` | Number of residual blocks |
| `--sims` | MCTS simulations per move |
| `--c_puct` | PUCT exploration constant |
| `--batch` | MCTS inference batch size |
| `--games_per_iter` | Self-play games per training iteration |
| `--batch_train` | Training batch size |
| `--accumulation_steps` | Micro-batch gradient accumulation |
| `--train_epochs` | Training passes per iteration |
| `--lr` | Learning rate |
| `--grad_clip` | Gradient clipping threshold |
| `--value_weight` | Weight of value loss |
| `--entropy_weight` | Entropy bonus weight |
| `--first_noise_moves` | Number of opening moves with exploration noise |
| `--center_bias_strength` | Strength of center opening bias |
| `--center_bias_moves` | Duration of center bias |
| `--alternate_start_player` | Alternate first player across self-play games |
| `--resign_threshold` | Auto-resign threshold |
| `--use_amp` | Enable mixed precision on CUDA |
| `--use_compile` | Enable PyTorch compile when supported |
| `--use_onnx` | Enable ONNX inference backend |
| `--selfplay_workers` | Parallel self-play workers |
| `--eval_games` | Number of arena games; set `0` to disable |
| `--save_freq` | Periodic checkpoint frequency |

---

## 10. Checkpointing

The system saves several checkpoint types:

```text
checkpoints/model_latest.pth
checkpoints/model_best.pth
checkpoints/model_iterX.pth
checkpoints/model_final_size15.pth
```

| File | Meaning |
|---|---|
| `model_latest.pth` | Most recent checkpoint |
| `model_best.pth` | Best checkpoint according to arena evaluation |
| `model_iterX.pth` | Periodic checkpoint |
| `model_final_size15.pth` | Final checkpoint after training ends |

Checkpoint files may include:

- model weights;
- optimizer state;
- AMP scaler state;
- model configuration;
- iteration number.

---

## 11. Logging

Training metrics are written to:

```text
checkpoints/training_log.csv
```

Typical logged fields include:

| Field | Meaning |
|---|---|
| `iteration` | Current training iteration |
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

## 12. Stability Checks

A healthy training run should show:

```text
No NaN or Inf loss
No repeated CUDA out-of-memory errors
No repeated game-length collapse
Policy loss remains bounded
Value loss remains bounded
First-player win rate is not extremely imbalanced
The model keeps tactical ability after further training
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

## 13. Tactical Evaluation

Loss alone is not enough. A Gomoku model should also be tested on fixed tactical positions.

Suggested tactical tests:

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

## 14. Recommended Experiment Table

| Run | Board | Model | Sims | Games/iter | LR | Workers | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| Debug | 15x15 | v8_legacy 320x20 | 128 | 8 | 1e-4 | 1 | Quick pipeline check |
| Fast test | 15x15 | v8_legacy 320x20 | 256 | 16 | 1e-4 | 1-2 | Short sanity run |
| Stable recovery | 15x15 | v8_legacy 320x20 | 800 | 64 | 1e-4 | 2 | Continue strong checkpoint |
| Heavy training | 15x15 | v8_legacy 320x20 | 800 | 128 | 1e-4 | 2-4 | Long high-quality run |

---

## 15. Environment Notes

### Local GPU

Best for long training if the GPU has enough VRAM. Use `--use_amp` and tune `--selfplay_workers`.

### Kaggle

Use notebook commands with the correct project path. Avoid hardcoding Colab paths. Store outputs under `/kaggle/working` if persistence is needed.

### Google Colab

Mount Drive if checkpoints must persist. Disable `torch.compile` if compatibility issues occur.

### CPU-only

CPU training is possible but slow. Reduce:

```text
sims
games_per_iter
channels
res_blocks
```

---

## 16. Future Improvements

Possible next steps:

1. Add a fixed tactical benchmark suite.
2. Add model-vs-model arena outside the training loop.
3. Add policy entropy logging.
4. Add game length histogram logging.
5. Add first-player win-rate tracking.
6. Improve batched MCTS GPU inference.
7. Benchmark ONNX/TensorRT inference.
8. Add a playable UI.
9. Add model export scripts.
10. Compare model sizes and simulation budgets.

---

## 17. Conclusion

This project implements a complete AlphaZero-style Gomoku training system. It combines deep Policy-Value networks, MCTS, self-play, tactical move detection, prioritized replay, board augmentation, checkpoint recovery, and GPU-focused training optimizations.

The system is suitable for experiments on both model strength and training efficiency. A strong checkpoint should be evaluated not only by loss values, but also by practical tactical behavior such as blocking, double-threat creation, fork handling, and forced-win conversion.
