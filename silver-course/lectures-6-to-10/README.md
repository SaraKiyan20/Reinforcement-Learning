# Reinforcement Learning — David Silver's UCL Course, Lectures 6–10

The second half of my implementations from David Silver's
[Reinforcement Learning course](https://www.davidsilver.uk/teaching/) (UCL), continuing on from
[lectures 1–5](../rl-lectures-1-to-5). Everything here — environments, agents, and algorithms — is
built from scratch with NumPy, not called from an existing RL library.

## Contents

- **Lecture 6 — Value Function Approximation:** Linear SARSA with two feature encodings (coarse
  coding and radial basis functions) on Mountain Car; Least-Squares Policy Iteration (LSPI-TD) on a
  Chain Walk MDP
- **Lecture 7 — Policy Gradient Methods:** REINFORCE (Monte Carlo policy gradient) on a custom
  PuckWorld environment; Q Actor-Critic (QAC)
- **Lecture 8 — Integrating Learning and Planning:** Dyna-Q, Dyna-Q+ (with an exploration bonus for
  stale state-action pairs), and Dyna actor-critic (Dyna-AC), compared on a static maze and a maze
  whose layout changes mid-training
- **Lecture 9 — Exploration and Exploitation:** UCB (Upper Confidence Bound) for multi-armed bandits,
  with a visualization of how the belief distribution over each arm's mean sharpens over time

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook rl_lectures_6_to_10.ipynb
```

No external data — every environment is implemented in the notebook itself, so it runs offline.
Some training loops (REINFORCE at 7000 episodes, the Dyna maze comparison across 3 runs) take a few
minutes to execute in full.

## Notes

- The QAC section originally had a leftover duplicate `run` method (the first definition was dead
  code, silently overwritten by a second version with logging) — removed here.
