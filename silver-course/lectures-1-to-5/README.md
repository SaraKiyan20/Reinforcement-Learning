# Reinforcement Learning — David Silver's UCL Course, Lectures 1–5

My own implementations of the core algorithms from the first five lectures of David Silver's
[Reinforcement Learning course](https://www.davidsilver.uk/teaching/) (UCL), alongside my notes on
the theory. Everything here — environments, agents, and algorithms — is built from scratch with
NumPy (using a `gymnasium.Env`-style interface for the grid-world environments), not called from an
existing RL library.

## Contents

- **Lectures 1–2:** MDPs, policies, value functions, Bellman equations (theory notes)
- **Lecture 3 — Planning by Dynamic Programming:** Value Iteration and Policy Iteration on a custom
  Gridworld with walls and two terminal states
- **Lecture 4 — Model-Free Prediction:** Monte Carlo prediction (first-visit / every-visit) and
  TD(0) / TD(λ) (forward & backward view, online & offline) on a Random Walk chain
- **Lecture 5 — Model-Free Control:** On-policy Monte Carlo control with GLIE exploration, SARSA and
  SARSA(λ) on a Windy Gridworld, and off-policy Q-Learning — including the classic SARSA-vs-Q-Learning
  comparison on Cliff Walking

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook rl_lectures_1_to_5.ipynb
```

No external data — every environment is implemented in the notebook itself, so it runs offline.

## Notes

- The TD(λ) section originally had an environment-interface bug: the agent code expected a
  `gymnasium`-style `(state, info)` reset / 5-tuple step, but the Random Walk environment it was
  actually run against returns plain values. This version fixes the agent code to match the Random
  Walk environment's real interface, so the TD estimates now correctly converge toward the known
  analytical solution for a 6-state random walk (≈ 1/7, 2/7, ..., 6/7).
