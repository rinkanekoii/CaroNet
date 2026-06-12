# Gomoku AlphaZero Training System

A deep reinforcement learning system for training a Gomoku/Caro AI using **self-play**, **Monte Carlo Tree Search (MCTS)**, and a deep **Policy-Value neural network**.

This repository is designed for training a strong 15x15 Gomoku model, with support for checkpoint recovery, tactical move detection, replay buffer training, mixed precision, and long-running experiments on GPU environments such as Google Colab.

---

## 1. Overview

The project follows the AlphaZero-style training approach:

1. A neural network predicts:
   - a policy distribution over legal moves;
   - a value estimate for the current board state.

2. MCTS uses the network to search possible continuations and produce a stronger move distribution.

3. The model plays games against itself.

4. The generated self-play data is stored in a replay buffer.

5. The neural network is trained from the self-play data.

6. The updated checkpoint is saved and used for the next training iteration.

In simple terms:

```text
Neural Network → MCTS → Self-play Games → Replay Buffer → Training → New Checkpoint
```

---

## 2. Main Algorithms

### 2.1 AlphaZero-style Self-Play

The system trains without human game records. Instead, the model improves by repeatedly playing against itself.

Each self-play game produces training samples:

```text
(state, improved_policy, final_result)
```

Where:

- `state` is the current board position.
- `improved_policy` is the move distribution produced by MCTS.
- `final_result` is the final game outcome from the current player's perspective.

The model is then trained to imitate the MCTS policy and predict the final result.

---

### 2.2 Monte Carlo Tree Search

MCTS is used to improve move selection. Instead of directly choosing the move predicted by the neural network, the system performs multiple simulations from the current position.

The MCTS process consists of:

1. **Selection**  
   Traverse the search tree using a PUCT-style score.

2. **Expansion**  
   Expand promising child nodes.

3. **Evaluation**  
   Use the Policy-Value network to evaluate leaf positions.

4. **Backup**  
   Backpropagate the value estimate through the visited nodes.

5. **Action Selection**  
   Choose a move based on visit counts.

The number of simulations is controlled by:

```bash
--sims
```

For the current strong model setup, the system commonly uses:

```text
sims = 800
```

This produces higher-quality self-play targets, but it is computationally expensive.

---

### 2.3 PUCT Search Formula

MCTS uses a PUCT-style balance between exploitation and exploration.

Conceptually:

```text
score = Q(s, a) + U(s, a)
```

Where:

- `Q(s, a)` estimates the value of choosing action `a`.
- `U(s, a)` encourages exploration based on the prior probability from the neural network.
- `c_puct` controls the exploration strength.

The parameter is controlled by:

```bash
--c_puct
```

Default / common value:

```text
c_puct = 3.5
```

---

### 2.4 Progressive Widening

The MCTS implementation supports progressive widening. Instead of expanding all legal moves immediately, the search gradually expands more moves as visit count increases.

This is useful for Gomoku because the branching factor is large, especially in early game positions.

Benefits:

- reduces search overhead;
- focuses simulations on promising moves;
- improves efficiency on large boards.

Controlled by:

```bash
--progressive_widening
--no_progressive_widening
```

---

### 2.5 Tactical Move Detection

The MCTS system includes tactical shortcuts for obvious forced moves.

Before running full MCTS, the engine checks for:

- immediate winning moves;
- immediate blocking moves;
- fork-like tactical threats.

This is important in Gomoku because missing a direct win or a direct block is usually catastrophic.

The tactical logic helps the model handle patterns such as:

- five-in-a-row completion;
- blocking opponent's four;
- forced winning moves;
- double-threat situations.

---

### 2.6 Opening Exploration

The self-play system supports early-game exploration through:

```bash
--first_noise_moves
```

This adds noise to early moves so that self-play games do not always follow the same opening.

Example:

```text
first_noise_moves = 18
```

This means the first 18 moves can receive additional exploration noise.

---

### 2.7 Center Bias

Gomoku openings are usually stronger near the center of the board. The project supports optional center bias during early moves.

Controlled by:

```bash
--center_bias_strength
--center_bias_moves
```

Example:

```text
center_bias_strength = 0.7
center_bias_moves = 10
```

This encourages early self-play moves to stay near the center, reducing low-quality corner openings during training.

---

### 2.8 Alternating Start Player

To reduce first-player bias, the training system can alternate which side starts each self-play game.

Controlled by:

```bash
--alternate_start_player
```

This helps prevent the dataset from becoming too biased toward one side.

---

## 3. Neural Network

The project uses a deep convolutional Policy-Value network.

The current strong checkpoint setup uses:

```text
model_class = v8_legacy
board_size = 15
channels = 320
res_blocks = 20
use_coords = true
```

---

### 3.1 Input Representation

Each board state is converted into tensor planes.

Typical input planes:

```text
[current_player_stones, opponent_stones, empty_cells, row_coordinates, col_coordinates]
```

The coordinate planes help the model understand absolute board position, which is useful because center control matters in Gomoku.

---

### 3.2 Policy Head

The policy head outputs logits for all board positions.

For a 15x15 board:

```text
15 × 15 = 225 possible moves
```

So the policy output has 225 logits.

The model learns to match the MCTS visit distribution.

---

### 3.3 Value Head

The value head outputs one scalar:

```text
value ∈ [-1, 1]
```

Interpretation:

| Value | Meaning |
|---:|---|
| `1` | Current player is likely winning |
| `0` | Balanced / uncertain |
| `-1` | Current player is likely losing |

The value head allows MCTS to evaluate positions without rolling out games to the end.

---

## 4. Training Objective

The model is trained using a combined Policy-Value loss.

```text
loss = policy_loss + value_weight × value_loss - entropy_weight × entropy
```

### Policy Loss

The policy loss trains the network to imitate the MCTS-improved move distribution.

### Value Loss

The value loss trains the network to predict the final game result.

### Entropy Bonus

The entropy term prevents the policy from becoming too confident too early.

### Gradient Clipping

Gradient clipping is used to prevent unstable updates:

```bash
--grad_clip 1.0
```

---

## 5. Replay Buffer

Self-play data is stored in a replay buffer.

The replay buffer stores:

```text
state tensors
policy targets
value targets
sample priorities
```

The system supports prioritized sampling, which can help the model learn more from important or difficult positions.

---

## 6. Data Augmentation

The project supports board symmetry augmentation.

Gomoku positions can be transformed by:

- rotation;
- horizontal flip;
- equivalent board symmetries.

This increases the effective dataset size and helps the model generalize.

If coordinate planes are used, they are regenerated after augmentation so that they remain correct absolute coordinate maps.

---

## 7. Checkpoint System

The training system supports checkpoint saving and recovery.

Common checkpoint files:

```text
checkpoints/model_latest.pth
checkpoints/model_best.pth
checkpoints/model_iterX.pth
checkpoints/model_final_size15.pth
```

### `model_latest.pth`

The newest checkpoint. Used for continuing training.

### `model_best.pth`

The best checkpoint according to evaluation, if arena evaluation is enabled.

### `model_iterX.pth`

Periodic checkpoint saved every `save_freq` iterations.

### `model_final_size15.pth`

Final model after the training run completes.

Checkpoints store:

- model weights;
- optimizer state;
- AMP scaler state;
- model configuration;
- iteration number.

This allows training to resume from the next iteration.

---

## 8. Technologies Used

### Core Technologies

| Technology | Usage |
|---|---|
| Python | Main programming language |
| PyTorch | Neural network training and inference |
| NumPy | Board representation and numerical operations |
| Numba | JIT acceleration for tactical board checks |
| SciPy | Board utility operations |
| CUDA | GPU acceleration |
| AMP | Mixed precision training |
| Google Colab | Cloud GPU training environment |

---

### PyTorch

PyTorch is used for:

- defining neural network models;
- GPU training;
- mixed precision;
- optimizer and scheduler;
- checkpoint saving/loading.

---

### CUDA

CUDA is used when available:

```text
Device: CUDA
```

Self-play and training can both use GPU acceleration, although MCTS also relies heavily on CPU-side tree search logic.

---

### Automatic Mixed Precision

AMP is enabled with:

```bash
--use_amp
```

Benefits:

- lower VRAM usage;
- faster neural network inference/training;
- better suitability for Colab GPUs.

---

### Numba

Numba is used to accelerate low-level board logic, especially:

- win checking;
- tactical move detection;
- local board scanning;
- fork detection.

This matters because MCTS calls board evaluation logic many times.

---

## 9. Main Training Command

Recommended command for continuing training from a strong checkpoint on Google Colab:

```python
%cd /content/drive/MyDrive/Gomoku-Training

!python training.py \
    --mode train \
    --init_model "/content/drive/MyDrive/Gomoku-Training/checkpoints/model_latest.pth" \
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
    --eval_freq 1000 \
    --eval_games 0 \
    --save_freq 5
```

---

## 10. Parameter Explanation

| Parameter | Meaning |
|---|---|
| `--mode train` | Run full training loop |
| `--init_model` | Load an existing checkpoint |
| `--model_class v8_legacy` | Use architecture compatible with old checkpoint |
| `--size 15` | Use 15x15 board |
| `--channels 320` | Network width |
| `--res_blocks 20` | Number of residual blocks |
| `--iters 200` | Final training iteration |
| `--games_per_iter 64` | Number of self-play games per iteration |
| `--sims 800` | MCTS simulations per move |
| `--batch 256` | MCTS inference batch size |
| `--batch_train 128` | Training batch size |
| `--accumulation_steps 8` | Split training batch into smaller micro-batches |
| `--train_epochs 1` | One training pass per iteration |
| `--lr 1e-4` | Conservative learning rate for checkpoint continuation |
| `--grad_clip 1.0` | Prevent gradient explosion |
| `--value_weight 1.0` | Weight of value loss |
| `--alternate_start_player` | Alternate first player |
| `--center_bias_strength 0.7` | Opening center preference |
| `--center_bias_moves 10` | Center bias duration |
| `--first_noise_moves 18` | Exploration noise duration |
| `--resign_threshold -1.1` | Disable resignation |
| `--use_amp` | Enable mixed precision |
| `--no_use_compile` | Disable torch.compile |
| `--no_use_onnx` | Disable ONNX inference |
| `--selfplay_workers 2` | Parallel self-play workers |
| `--eval_games 0` | Disable arena evaluation |
| `--save_freq 5` | Save periodic checkpoint every 5 iterations |

---

## 11. Monitoring Training

Important training indicators:

| Metric | Meaning |
|---|---|
| `Avg Len(last10)` | Average length of recent games |
| `policy_loss` | How well the model learns MCTS policy |
| `value_loss` | How well the model predicts result |
| `total_loss` | Combined loss |
| `p1_wins` / `p2_wins` | First-player balance |
| `draws` | Draw count |
| `selfplay_time_sec` | Time spent generating games |
| `train_time_sec` | Time spent training network |
| `buffer_size` | Replay buffer size |

---

## 12. Signs of a Stable Run

A training run is considered stable when:

```text
No NaN or Inf loss
No CUDA out-of-memory error
Game length does not collapse repeatedly
Policy loss does not explode
Value loss remains bounded
First-player win rate is not extremely imbalanced
The model still handles known tactical positions
```

---

## 13. Tactical Skills to Test

A strong Gomoku model should be tested on positions involving:

1. Immediate win.
2. Immediate block.
3. Open three.
4. Open four.
5. Double threat.
6. Fork creation.
7. Forced defense.
8. Central opening control.
9. Avoiding obvious traps.
10. Converting winning sequences.

These tests are important because loss values alone do not prove that the model is stronger.

---

## 14. Why 800 MCTS Simulations?

Lower simulation counts such as 128 or 256 are useful for debugging, but they produce weaker search targets.

Using 800 simulations provides:

- stronger move targets;
- better tactical awareness;
- better double-threat detection;
- more reliable self-play games.

The trade-off is much slower training.

---

## 15. Why Use a Lower Learning Rate?

When continuing from a strong checkpoint, the goal is not to relearn everything quickly. The goal is to improve without damaging previously learned knowledge.

A learning rate of:

```text
1e-4
```

is safer than:

```text
3e-4
```

for long-term fine-tuning of a large model.

---

## 16. Worker Count

`--selfplay_workers` controls how many self-play games are generated in parallel.

For Colab, a safe default is:

```text
selfplay_workers = 2
```

Four workers may or may not be faster. More workers can cause:

- CPU bottleneck;
- GPU inference contention;
- RAM pressure;
- slower CPU-GPU data transfer.

The best approach is to benchmark 2 vs 4 workers for one iteration.

---

## 17. Suggested Experiment Log

| Iteration | Sims | Games/iter | LR | Avg length | Policy loss | Value loss | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| 106 | 800 | 64 | 3e-4 | TBD | TBD | TBD | Before LR change |
| 110 | 800 | 64 | 1e-4 | TBD | TBD | TBD | After LR change |
| 120 | 800 | 64 | 1e-4 | TBD | TBD | TBD | Stability check |
| 150 | 800 | 64 | 1e-4 | TBD | TBD | TBD | Long run |
| 200 | 800 | 64 | 1e-4 | TBD | TBD | TBD | Final |

---

## 18. Future Improvements

Possible improvements:

1. Add a fixed tactical benchmark suite.
2. Add model-vs-model arena testing outside the training loop.
3. Log policy entropy.
4. Log game length distribution.
5. Log first-player win rate separately.
6. Benchmark ONNX Runtime or TensorRT.
7. Improve batched MCTS inference.
8. Add a simple playable UI.
9. Export trained models for deployment.
10. Compare different model sizes and MCTS simulation counts.

---

## 19. Conclusion

This project implements a complete AlphaZero-style Gomoku training system. It combines a deep Policy-Value network, MCTS, self-play, tactical move detection, replay buffer training, data augmentation, checkpoint recovery, and GPU acceleration.

The current setup is designed for continuing training from a strong 15x15 `v8_legacy` checkpoint. The emphasis is on stable improvement rather than aggressive retraining.

A good checkpoint should not only have lower loss, but should also demonstrate practical tactical strength: blocking threats, creating forks, recognizing double threats, and converting forced wins.
