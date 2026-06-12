# AlphaZero Gomoku AI: Self-Play Training with MCTS and Policy-Value Networks

## Abstract

This project implements an AlphaZero-style training pipeline for Gomoku/Caro. The system trains a neural agent through self-play, using Monte Carlo Tree Search (MCTS) to generate improved move targets and a deep Policy-Value network to estimate both move probabilities and game outcome.

The current recovery/training setup focuses on continuing training from a strong checkpoint using a `v8_legacy` model with 320 channels and 20 residual blocks on a 15x15 Gomoku board. The training strategy prioritizes stability: high-quality self-play data with 800 MCTS simulations per move, conservative learning rate, checkpoint recovery, mixed precision training, and periodic checkpoint saving.

This README is written in a research-oriented format so the repository can be used as part of an academic or experimental AI training project.

---

## 1. Research Objective

The main objective of this project is to study whether an AlphaZero-style pipeline can train a strong Gomoku agent under limited computational resources.

The specific objectives are:

1. Build a full self-play reinforcement learning pipeline for Gomoku.
2. Combine MCTS with a Policy-Value neural network.
3. Train a deep convolutional model capable of learning tactical patterns such as forced wins, blocking moves, forks, and double threats.
4. Support checkpoint recovery and long-running training.
5. Monitor training stability using game length, policy loss, value loss, total loss, and win/draw statistics.
6. Evaluate practical trade-offs between MCTS simulations, games per iteration, learning rate, training epochs, and self-play workers.

---

## 2. Problem Definition

Gomoku is a two-player, zero-sum, perfect-information board game. Players alternate placing stones on an empty board. The first player to form a continuous line of five stones horizontally, vertically, or diagonally wins.

The main experimental setting is:

| Component | Value |
|---|---|
| Board size | 15x15 |
| Win condition | 5 stones in a row |
| Game type | Two-player zero-sum game |
| Learning method | Self-play reinforcement learning |
| Search algorithm | Monte Carlo Tree Search |
| Neural network type | Policy-Value network |

---

## 3. Methodology

The system follows the AlphaZero training paradigm:

1. The current model plays games against itself.
2. MCTS is used at each move to produce a stronger policy target.
3. Each self-play game produces training samples:
   - board state;
   - MCTS policy distribution;
   - final game outcome.
4. The neural network is trained to predict both:
   - the MCTS-improved policy;
   - the final game result.
5. The updated model is saved and used in the next iteration.

The high-level training loop is:

```text
Load checkpoint or initialize model
        |
        v
Generate self-play games using MCTS
        |
        v
Store samples in replay buffer
        |
        v
Train Policy-Value network
        |
        v
Log metrics and save checkpoints
        |
        v
Repeat for multiple iterations
```

---

## 4. System Architecture

The project is organized into modular components:

| File | Role |
|---|---|
| `training.py` | Main command-line entry point |
| `training_loop.py` | Main training loop, checkpointing, scheduling, logging |
| `self_play.py` | Self-play game generation |
| `mcts.py` | Progressive MCTS implementation and tactical move detection |
| `models.py` | Current neural network architectures |
| `legacy_models.py` | Legacy model architectures |
| `loss.py` | Policy-value loss and training steps |
| `replay_buffer.py` | Prioritized replay buffer |
| `augmentation.py` | Board symmetry augmentation |
| `checkpoint_utils.py` | Save/load checkpoint utilities |
| `evaluate.py` | Arena evaluation between models |
| `config.py` | CLI argument definitions and default hyperparameters |
| `utils.py` | Board utilities and state encoding |

---

## 5. Neural Network Model

The current recovery configuration uses:

| Parameter | Value |
|---|---|
| Model class | `v8_legacy` |
| Board size | 15 |
| Channels | 320 |
| Residual blocks | 20 |
| Coordinate input planes | Enabled |
| Policy output size | 225 logits |
| Value output size | 1 scalar |

The network receives state planes representing:

```text
[current player stones, opponent stones, empty cells, row coordinates, column coordinates]
```

The model produces two outputs:

```text
policy: logits over all board positions
value: scalar evaluation in the range [-1, 1]
```

The policy head learns to imitate MCTS-improved move distributions.  
The value head learns to predict the final outcome of the game from the current player's perspective.

---

## 6. Monte Carlo Tree Search

MCTS is used to improve move selection during self-play. Instead of directly sampling from the neural network policy, MCTS performs multiple simulations per move and produces a stronger policy target.

Important MCTS-related parameters:

| Parameter | Meaning |
|---|---|
| `--sims` | Number of MCTS simulations per move |
| `--c_puct` | Exploration-exploitation coefficient |
| `--batch` | MCTS inference batch size |
| `--progressive_widening` | Restricts expansion for efficiency |
| `--first_noise_moves` | Adds exploration noise in early moves |
| `--center_bias_strength` | Biases early moves toward the center |
| `--center_bias_moves` | Number of opening moves affected by center bias |
| `--alternate_start_player` | Alternates the first player across games |
| `--resign_threshold` | Auto-resignation threshold |

Current recovery setting:

```text
sims = 800
games_per_iter = 64
first_noise_moves = 18
center_bias_strength = 0.7
center_bias_moves = 10
resign_threshold = -1.1
```

A `resign_threshold` of `-1.1` effectively disables resignation because value predictions usually fall within `[-1, 1]`. This is useful during recovery because it prevents premature resignation caused by unstable value estimates.

---

## 7. Replay Buffer and Data Augmentation

Self-play samples are stored in a replay buffer. Each sample contains:

```text
(state, policy_target, value_target)
```

The project supports board symmetry augmentation through rotations and flips. These transformations help the model generalize because many Gomoku positions are strategically equivalent under board symmetries.

When coordinate planes are used, they are regenerated after augmentation so that they continue to represent absolute board positions correctly.

---

## 8. Loss Function

The model is trained with a combined Policy-Value objective:

```text
loss = policy_weight * policy_loss
     + value_weight * value_loss
     - entropy_weight * policy_entropy
```

Where:

| Component | Meaning |
|---|---|
| `policy_loss` | Cross-entropy between MCTS policy and model policy |
| `value_loss` | Mean squared error between predicted value and game outcome |
| `policy_entropy` | Optional entropy bonus to discourage early policy collapse |
| `grad_clip` | Gradient clipping threshold |

Current stable recovery configuration:

```text
policy_weight = 1.0
value_weight = 1.0
grad_clip = 1.0
learning_rate = 1e-4
```

---

## 9. Recommended Training Command

The following command continues training from the latest checkpoint while disabling arena evaluation:

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

## 10. Hyperparameter Explanation

| Parameter | Explanation |
|---|---|
| `--init_model` | Loads an existing checkpoint for continued training |
| `--model_class v8_legacy` | Uses the architecture compatible with the old checkpoint |
| `--size 15` | Uses a 15x15 Gomoku board |
| `--channels 320` | Width of the neural network |
| `--res_blocks 20` | Number of residual blocks |
| `--games_per_iter 64` | Number of self-play games per training iteration |
| `--sims 800` | Number of MCTS simulations per move |
| `--batch 256` | MCTS inference batch size |
| `--batch_train 128` | Training batch size |
| `--accumulation_steps 8` | Splits the training batch into smaller micro-batches to reduce VRAM usage |
| `--train_epochs 1` | Number of training passes per iteration |
| `--lr 1e-4` | Conservative learning rate for stable fine-tuning |
| `--grad_clip 1.0` | Clips gradients to prevent instability |
| `--value_weight 1.0` | Weight of the value loss |
| `--use_amp` | Enables mixed precision training |
| `--no_use_compile` | Disables `torch.compile` for better Colab compatibility |
| `--no_use_onnx` | Uses PyTorch inference instead of ONNX Runtime |
| `--selfplay_workers 2` | Number of parallel self-play workers |
| `--eval_games 0` | Disables arena evaluation |
| `--save_freq 5` | Saves periodic checkpoints every 5 iterations |

---

## 11. Checkpointing and Resume Behavior

The training system saves several checkpoint types:

```text
checkpoints/model_latest.pth
checkpoints/model_best.pth
checkpoints/model_iterX.pth
checkpoints/model_final_size15.pth
```

Meaning:

| Checkpoint | Purpose |
|---|---|
| `model_latest.pth` | Most recent model checkpoint |
| `model_best.pth` | Best model according to arena evaluation if enabled |
| `model_iterX.pth` | Periodic checkpoint saved every `save_freq` iterations |
| `model_final_size15.pth` | Final model after training completes |

The checkpoint contains model state, optimizer state, AMP scaler state, model configuration, and iteration metadata. This allows training to resume from the next iteration after loading `model_latest.pth`.

---

## 12. Monitoring Training Stability

During training, the following metrics should be monitored:

| Metric | Purpose |
|---|---|
| `Avg Len(last10)` | Detects game length collapse or abnormal games |
| `policy_loss` | Indicates how well the model learns MCTS targets |
| `value_loss` | Indicates how well the model predicts game outcomes |
| `total_loss` | Overall training objective |
| `p1_wins`, `p2_wins`, `draws` | Detects first-player imbalance |
| `selfplay_time_sec` | Measures self-play cost |
| `train_time_sec` | Measures neural network training cost |
| `buffer_size` | Tracks replay buffer growth |

A training run is considered stable if:

```text
No NaN or Inf loss
No repeated game-length collapse
Policy loss does not explode
Value loss remains bounded
First-player win rate is not extremely imbalanced
The new checkpoint does not lose known tactical ability
```

---

## 13. Tactical Evaluation Criteria

Loss alone is not sufficient to evaluate a Gomoku model. A strong model should be tested on tactical positions such as:

1. Winning in one move.
2. Blocking an opponent's immediate win.
3. Creating an open three.
4. Creating a double threat.
5. Blocking a double threat.
6. Converting a forced line.
7. Avoiding obvious tactical traps.
8. Maintaining strong central control in the opening.

A checkpoint is stronger only if it preserves or improves practical gameplay strength, not merely if it reports lower loss.

---

## 14. Experimental Log Template

The following table should be filled after training:

| Iteration | Games/iter | Sims | Avg game length | Policy loss | Value loss | Total loss | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| 106 | 64 | 800 | TBD | TBD | TBD | TBD | Recovery checkpoint |
| 110 | 64 | 800 | TBD | TBD | TBD | TBD | After LR reduction |
| 120 | 64 | 800 | TBD | TBD | TBD | TBD | Stability check |
| 150 | 64 | 800 | TBD | TBD | TBD | TBD | Long training |
| 200 | 64 | 800 | TBD | TBD | TBD | TBD | Final checkpoint |

---

## 15. Practical Notes

### Number of self-play workers

Using more workers is not always faster. On Colab, `--selfplay_workers 2` is usually safer than `4` for a large 39M-parameter model because multiple workers can compete for CPU-GPU transfer and inference resources.

Recommended procedure:

```text
Start with 2 workers.
Benchmark one iteration.
Try 4 workers only if CPU, RAM, and GPU utilization suggest underuse.
Keep 4 workers only if self-play time improves clearly.
```

### MCTS simulations

`800` simulations provide stronger self-play targets but significantly increase training time. Lower values such as `128`, `256`, or `512` can be used for debugging or warm-up, but long-term training should use higher simulations if model strength is the priority.

### Learning rate

For continued training from a strong checkpoint, `1e-4` is safer than `3e-4`. A lower learning rate reduces the risk of damaging already learned tactical knowledge.

---

## 16. Limitations

Current limitations include:

1. Self-play is computationally expensive with 800 MCTS simulations.
2. A 320-channel, 20-block model requires substantial GPU memory and compute.
3. Disabling arena evaluation makes progress harder to quantify automatically.
4. Loss metrics do not fully capture practical gameplay strength.
5. Opening bias and noise must be tuned carefully to avoid repetitive self-play.
6. Worker scaling may be limited by CPU-GPU synchronization overhead.
7. A dedicated tactical test suite is still needed for more rigorous evaluation.

---

## 17. Future Work

Possible future improvements:

1. Build a tactical benchmark suite for Gomoku:
   - immediate win;
   - immediate block;
   - open three;
   - double threat;
   - forced win sequence.

2. Add stronger arena evaluation:
   - new model vs old model;
   - model vs heuristic bot;
   - model vs fixed tactical benchmark;
   - model vs human player.

3. Improve self-play speed:
   - optimized batched inference;
   - ONNX Runtime;
   - TensorRT;
   - better worker scheduling.

4. Improve logging:
   - first-player win rate;
   - game length distribution;
   - number of tactical MCTS bypasses;
   - per-iteration learning rate;
   - policy entropy;
   - saved training curves.

5. Compare hyperparameter settings:
   - `games_per_iter = 64` vs `128`;
   - `sims = 400`, `512`, `800`;
   - `lr = 1e-4`, `2e-4`, `3e-4`;
   - `train_epochs = 1` vs `2`;
   - `selfplay_workers = 1`, `2`, `4`.

---

## 18. Reproducibility Checklist

To reproduce an experiment, record:

```text
Python version
PyTorch version
GPU type
CUDA version
Initial checkpoint path
Starting iteration
Ending iteration
Board size
Model class
Channels
Residual blocks
MCTS simulations
Games per iteration
Learning rate
Batch size
Accumulation steps
Training epochs
Self-play workers
AMP enabled/disabled
ONNX enabled/disabled
Arena evaluation enabled/disabled
```

Example configuration:

```text
Board: 15x15
Model: v8_legacy
Channels: 320
Residual blocks: 20
MCTS simulations: 800
Games per iteration: 64
Training batch: 128
Accumulation steps: 8
Training epochs: 1
Learning rate: 1e-4
AMP: enabled
Compile: disabled
ONNX: disabled
Arena evaluation: disabled
```

---

## 19. Conclusion

This project implements a research-oriented AlphaZero-style Gomoku training system. It combines self-play, MCTS, replay buffer training, deep Policy-Value networks, checkpoint recovery, and training stability monitoring.

The current training strategy is designed for continuing from a strong `v8_legacy` checkpoint rather than training from scratch. It uses high-quality MCTS targets with 800 simulations while applying a conservative learning rate to preserve previously learned tactical knowledge.

Future evaluation should focus not only on loss values, but also on tactical behavior such as forced wins, blocking moves, fork creation, and double-threat handling.
