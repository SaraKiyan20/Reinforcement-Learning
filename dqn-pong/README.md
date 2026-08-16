# DQN on Atari Pong — from Scratch

A from-scratch implementation of Deep Q-Networks ([Mnih et al., 2015](https://www.nature.com/articles/nature14236)),
trained to play Atari Pong directly from raw pixels. Built up in stages: environment exploration,
frame preprocessing (grayscale, resize, frame-skip, frame-stack), a replay buffer, the convolutional
Q-network, epsilon-greedy action selection, and a full agent with a target network and checkpointing.

## Results

The agent was trained for ~2,000,000 environment steps. Average evaluation reward (10 episodes,
greedy policy) at each included checkpoint:

| Checkpoint | Avg. reward |
|---|---|
| step 0 (untrained) | −21.0 |
| step 500,000 | −17.2 |
| step 1,000,000 | −5.1 |
| step 1,950,000 | +4.0 |

(Pong's reward range is −21 to +21; +4.0 means the agent is reliably winning.)

## Contents

- **Steps 1–2** — raw environment exploration and frame preprocessing (grayscale, resize to 84×84,
  frame-skip with max-pooling, 4-frame stacking)
- **Step 3** — a fixed-capacity replay buffer
- **Step 4** — the Q-network (the standard "Nature DQN" convolutional architecture)
- **Step 5** — epsilon-greedy action selection with linear epsilon decay
- **Step 6** — the full `DQNAgent` (policy net, target net, optimizer, checkpointing)
- **Training** — the full training loop (not re-executed in this notebook — see Notes below)
- **Evaluation** — loads each included checkpoint and re-evaluates it live, to verify the checkpoints
  reproduce the improvement shown in the training plot

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook dqn_pong.ipynb
```

The four checkpoint files are included in this folder, so the Evaluation section runs immediately
without needing to retrain. A CUDA GPU is used automatically if available; otherwise everything runs
on CPU (training is far slower on CPU — a GPU is recommended if you want to retrain from scratch).

## Notes

- Training (all 2,000,000 steps, producing the checkpoints and `training_progress.png` above) was
  run on a GPU. The Evaluation section's checkpoint comparison table was separately verified on CPU
  (loading the saved checkpoints and re-running inference only — no retraining) to confirm the
  checkpoints reproduce the improvement shown in the training plot.
- The training loop cell is included as written but **was not re-executed to produce this notebook**
  — training runs for several hours on a GPU. The actual result (`training_progress.png`) and the
  four checkpoints it produced are included and verified in the Evaluation section instead.
- The sanity-check cell after `DQNAgent` originally checked whether `target_net`'s weights changed
  after calling `update()` — but `target_net` is only synced every `target_sync_every` steps (10,000
  by default) and is never touched by backprop directly, so it correctly never changes on this test,
  which made the check look like a false negative. Fixed to check `policy_net` instead, which is the
  network `update()` actually trains — it now correctly shows `changed = True` on every call.
- Uses `render_mode="rgb_array"` throughout so the notebook runs headlessly (e.g. on GitHub, Colab,
  or CI); use `render_mode="human"` instead if you want a live pop-up window when running locally.
