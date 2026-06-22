# CaroNet

**CaroNet** is an AlphaZero-style training system for Caro/Gomoku.  
It trains a neural agent through **self-play**, **Monte Carlo Tree Search (MCTS)**, and a deep **Policy-Value network**.

The project focuses on practical Caro/Gomoku training with tactical move detection, replay buffer learning, checkpoint recovery, and GPU-oriented optimization.

---

## Features

- AlphaZero-style self-play training.
- Monte Carlo Tree Search with PUCT.
- Deep Policy-Value neural network.
- Tactical detection for immediate wins, blocks, forks, and double threats.
- Progressive widening for efficient MCTS expansion.
- Prioritized replay buffer.
- Board symmetry augmentation.
- Checkpoint save/load and resume support.
- Mixed precision training with AMP.
- Optional experimental ONNX inference path.
- CLI-based training configuration.
- Training log export.

---

## Game Rules Configuration

CaroNet supports two different rule sets during training. By default, it trains on **Gomoku rules**.

```bash
--rule_type 0       # Default: Gomoku rules (5-in-a-row is a win regardless of blocks)
--rule_type 1       # Caro rules (5-in-a-row blocked at both ends is NOT a win)
```

If you want the AI to learn both rule sets simultaneously, you can enable mixed rules:

```bash
--mixed_rules       # Default: False. If True, randomly switches between Gomoku and Caro each game.
```

When training with `--rule_type 1` (Caro), the AI will:
- Correctly evaluate board edges/walls as blocks.
- Learn to avoid wasting moves on blocked lines.
- Explore distant blocks and strategic traps thanks to its expanded 13x13 (Radius 6) tactical vision.

---

## Training Pipeline

CaroNet trains without human game records.  
The model improves by repeatedly playing against itself.

```text
Current Model
     |
     v
MCTS Self-Play
     |
     v
Replay Buffer
     |
     v
Policy-Value Training
     |
     v
Updated Checkpoint
```

Each self-play sample has the form:

```text
(state, policy_target, value_target)
```

| Component | Meaning |
|---|---|
| `state` | Encoded board position |
| `policy_target` | MCTS-improved move distribution |
| `value_target` | Final game result from the current player's perspective |

---

## Board Representation

For each position, the board is encoded as tensor planes:

```text
[current_player_stones, opponent_stones, empty_cells, row_coordinates, column_coordinates]
```

The coordinate planes help the model recognize absolute board positions such as center, edge, and corner regions.

---

## Policy-Value Network

The neural network predicts two outputs:

```text
(value, policy_logits) = fθ(state)
```

| Output | Meaning |
|---|---|
| `policy_logits` | Raw scores for all board positions |
| `value` | Expected outcome in `[-1, 1]` |

For a 15x15 board:

```text
policy_size = 15 × 15 = 225
```

The policy head learns move selection.  
The value head learns position evaluation.

---

## Monte Carlo Tree Search

MCTS improves the raw neural network policy by running simulations before selecting a move.

Each simulation includes:

1. **Selection**: choose child nodes using PUCT.
2. **Expansion**: add promising legal moves.
3. **Evaluation**: evaluate the leaf state with the neural network.
4. **Backup**: propagate the value estimate through the search path.
5. **Move selection**: build a policy target from visit counts.

---

## PUCT Formula

CaroNet uses a PUCT-style score:

```text
score(s, a) = Q(s, a) + U(s, a)
```

Where:

```text
Q(s, a) = W(s, a) / N(s, a)
```

```text
U(s, a) = c_puct × P(s, a) × sqrt(Σb N(s, b)) / (1 + N(s, a))
```

| Symbol | Meaning |
|---|---|
| `s` | Current state |
| `a` | Candidate move |
| `Q(s, a)` | Mean value of action `a` |
| `W(s, a)` | Accumulated value |
| `N(s, a)` | Visit count |
| `P(s, a)` | Policy prior from the neural network |
| `c_puct` | Exploration constant |

The key CLI parameter is:

```bash
--c_puct
```

---

## MCTS Policy Target

After MCTS finishes, the policy target is derived from visit counts:

```text
π(a | s) = N(s, a)^(1/τ) / Σb N(s, b)^(1/τ)
```

Where:

| Symbol | Meaning |
|---|---|
| `π(a | s)` | Training target for move `a` |
| `N(s, a)` | Visit count of move `a` |
| `τ` | Temperature parameter |

Higher temperature increases exploration.  
Lower temperature makes the policy more deterministic.

---

## Temperature Schedule

CaroNet uses a move-based temperature schedule:

```text
τ = 1.0   if move_count < temp_threshold
τ = 0.5   if move_count < 2 × temp_threshold
τ = 0.25  otherwise
```

Controlled by:

```bash
--temp_threshold
```

This encourages exploration in the opening and more decisive play later.

---

## Progressive Widening

Progressive widening limits how many children are expanded in MCTS.

Conceptually:

```text
expanded_children = max(pw_min, floor(pw_alpha × sqrt(total_visits + 1)))
```

The number of expanded children is capped by the number of legal moves.

This reduces early-game branching cost, especially on large boards.

---

## Tactical Move Detection

Before running full MCTS, CaroNet checks for tactical moves:

1. Immediate winning move.
2. Immediate blocking move.
3. Fork or double-threat move.

A move is treated as an immediate win if placing a stone there satisfies:

```text
count_in_any_direction >= win_length
```

For normal Gomoku/Caro:

```text
win_length = 5
```

Fork detection checks whether a candidate move creates at least two future winning threats:

```text
threat_count(move) >= 2
```

This is important because double threats often decide the game.

---

## Opening Exploration

Early-game noise is controlled by:

```bash
--first_noise_moves
```

Example:

```bash
--first_noise_moves 18
```

This adds exploration during early moves and helps prevent repetitive openings.

---

## Center Bias

CaroNet can bias early self-play moves toward the center:

```bash
--center_bias_strength 0.7
--center_bias_moves 10
```

A conceptual Gaussian-like center prior is used:

```text
center_score(r, c) ∝ exp(-distance_to_center² / (2σ²))
```

This improves opening quality because central positions are generally stronger in Caro/Gomoku.

---

## Alternating Start Player

The flag:

```bash
--alternate_start_player
```

alternates the first player across self-play games.

This reduces first-player bias in the generated dataset.

---

## Loss Function

The model is trained with a combined Policy-Value objective:

```text
L = policy_weight × L_policy + value_weight × L_value - entropy_weight × H(policy)
```

### Policy Loss

The policy loss uses cross-entropy between the MCTS target and the model policy:

```text
L_policy = - Σa π(a | s) log pθ(a | s)
```

Where:

| Symbol | Meaning |
|---|---|
| `π(a | s)` | MCTS policy target |
| `pθ(a | s)` | Model policy prediction |

---

### Value Loss

The value loss uses mean squared error:

```text
L_value = (z - vθ(s))²
```

Where:

| Symbol | Meaning |
|---|---|
| `z` | Final game result |
| `vθ(s)` | Predicted value |

---

### Entropy Bonus

Entropy can be used to prevent early policy collapse:

```text
H(policy) = - Σa pθ(a | s) log pθ(a | s)
```

The entropy term is subtracted from the loss:

```text
L_total = ... - entropy_weight × H(policy)
```

This encourages exploration among legal moves.

---

## Replay Buffer

Self-play samples are stored in a replay buffer.

Stored data:

```text
states
policy targets
value targets
priorities
```

CaroNet uses prioritized sampling with a Fenwick tree for efficient prefix-sum queries.

Conceptually, sampling probability can be written as:

```text
P(i) = priority_i / Σj priority_j
```

This allows important samples to be selected more often.

---

## Data Augmentation

CaroNet applies board symmetry augmentation:

- rotations;
- horizontal flips;
- equivalent board transformations.

For each transformed board, the policy target is transformed in the same way.

If coordinate planes are used, they are regenerated after augmentation to keep absolute coordinates correct.

---

## Engineering Optimizations

CaroNet includes several performance-oriented optimizations:

- Numba-accelerated win checking and tactical scans.
- Zobrist hashing for board state identification.
- Progressive widening in MCTS.
- Batched neural network inference.
- Prioritized replay sampling with Fenwick tree.
- Preallocated replay buffer arrays.
- AMP mixed precision training.
- Gradient accumulation for lower VRAM usage.
- Gradient clipping for stability.
- Learning rate warmup and cosine annealing.
- Atomic checkpoint saving.

---

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main language |
| PyTorch | Neural network training and inference |
| CUDA | GPU acceleration |
| AMP | Mixed precision training |
| NumPy | Board and replay data |
| Numba | JIT acceleration for board logic |
| SciPy | Board utility operations |
| ONNX Runtime | Optional experimental inference backend |
| CSV logging | Training metrics |

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Current `requirements.txt`:

```txt
torch>=2.0.0
numpy>=1.20.0
numba>=0.56.0
scipy>=1.9.0
```

ONNX code exists in the project, but ONNX is experimental and not required for normal training.  
The recommended training path uses:

```bash
--no_use_onnx
```

---

## Training Command

Generic command:

```bash
python training.py --mode train --init_model "<CHECKPOINT_DIR>/model_latest.pth" --model_class v8_legacy --size 15 --channels 320 --res_blocks 20 --iters 200 --games_per_iter 64 --sims 800 --batch 256 --batch_train 128 --accumulation_steps 8 --train_epochs 1 --lr 1e-4 --grad_clip 1.0 --value_weight 1.0 --alternate_start_player --center_bias_strength 0.7 --center_bias_moves 10 --first_noise_moves 18 --resign_threshold -1.1 --use_amp --no_use_compile --no_use_onnx --selfplay_workers 2 --eval_games 0 --save_freq 5
```

Replace:

```text
<CHECKPOINT_DIR>
```

with the path to the checkpoint directory.

Example:

```bash
python training.py --mode train --init_model "checkpoints/model_latest.pth" --model_class v8_legacy --size 15 --channels 320 --res_blocks 20 --iters 200 --games_per_iter 64 --sims 800 --batch 256 --batch_train 128 --accumulation_steps 8 --train_epochs 1 --lr 1e-4 --grad_clip 1.0 --value_weight 1.0 --alternate_start_player --center_bias_strength 0.7 --center_bias_moves 10 --first_noise_moves 18 --resign_threshold -1.1 --use_amp --no_use_compile --no_use_onnx --selfplay_workers 2 --eval_games 0 --save_freq 5
```

For notebook environments such as Kaggle, Colab, or Jupyter, prefix the command with `!`.

---

## Self-Play Test

Run one self-play game:

```bash
python training.py --mode selfplay --init_model "checkpoints/model_latest.pth" --model_class v8_legacy --size 15 --channels 320 --res_blocks 20 --sims 800 --no_use_onnx
```

---

## Important Parameters

| Parameter | Description |
|---|---|
| `--mode` | `train` or `selfplay` |
| `--rule_type` | `1` for Caro (block at both ends doesn't win), `0` for Gomoku |
| `--mixed_rules` | Randomly swap between Gomoku and Caro per game |
| `--init_model` | Checkpoint path |
| `--model_class` | Model architecture |
| `--size` | Board size |
| `--channels` | Model width |
| `--res_blocks` | Number of residual blocks |
| `--sims` | MCTS simulations per move |
| `--games_per_iter` | Self-play games per iteration |
| `--batch_train` | Training batch size |
| `--accumulation_steps` | Micro-batch accumulation steps |
| `--train_epochs` | Training passes per iteration |
| `--lr` | Learning rate |
| `--grad_clip` | Gradient clipping threshold |
| `--use_amp` | Enable mixed precision |
| `--no_use_onnx` | Use PyTorch inference instead of ONNX |
| `--selfplay_workers` | Number of self-play workers |
| `--eval_games 0` | Disable arena evaluation |
| `--save_freq` | Periodic checkpoint interval |

---

## Checkpoints

Common checkpoint files:

```text
model_latest.pth
model_best.pth
model_iterX.pth
model_final_size15.pth
```

| File | Purpose |
|---|---|
| `model_latest.pth` | Most recent checkpoint |
| `model_best.pth` | Best checkpoint if evaluation is enabled |
| `model_iterX.pth` | Periodic checkpoint |
| `model_final_size15.pth` | Final checkpoint |

Large model files should not be committed directly to Git.

---

## Logs

Training metrics are written to:

```text
checkpoints/training_log.csv
```

Typical fields:

- iteration;
- learning rate;
- self-play time;
- training time;
- buffer size;
- player 1 wins;
- player 2 wins;
- draws;
- average game length;
- total loss;
- policy loss;
- value loss.

---

## Stability Checks

A healthy run should show:

```text
No NaN or Inf loss
No repeated CUDA out-of-memory errors
No repeated game-length collapse
Policy loss remains bounded
Value loss remains bounded
First-player win rate is not extremely imbalanced
```

Warning signs:

```text
Average game length repeatedly below 15 moves
Loss explosion
Policy collapse
One side winning almost all games
New checkpoint playing worse than older checkpoints
```

---

## Tactical Evaluation

Loss is not enough to judge model strength.  
A Caro model should also be tested on tactical positions:

- win in one move;
- block immediate win;
- create open three;
- create open four;
- create double threat;
- block double threat;
- convert forced win;
- avoid obvious trap.

---

## Suggested Presets

| Preset | Sims | Games/iter | Purpose |
|---|---:|---:|---|
| Debug | 128 | 8 | Quick pipeline test |
| Fast test | 256 | 16 | Short sanity run |
| Stable training | 800 | 64 | Continue strong checkpoint |
| Heavy training | 800 | 128 | Long high-quality run |

---

## Project Structure

```text
CaroNet/
├── training.py
├── training_loop.py
├── self_play.py
├── mcts.py
├── models.py
├── legacy_models.py
├── loss.py
├── replay_buffer.py
├── augmentation.py
├── checkpoint_utils.py
├── evaluate.py
├── config.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Files Not Included

Do not commit large generated files:

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
