# Easy21 — Monte Carlo Control, TD(λ), and Linear Function Approximation

An implementation of the "Easy21" assignment from David Silver's
[Reinforcement Learning course](https://www.davidsilver.uk/teaching/) (UCL) — a simplified Blackjack
variant used to compare Monte Carlo control, tabular Sarsa(λ), and Sarsa(λ) with linear function
approximation on the same environment, all implemented from scratch with NumPy.

## Contents

- **Environment** — a from-scratch Easy21 implementation (black/red card draws, dealer/player rules)
- **Monte Carlo control** — every-visit MC with a GLIE policy, visualized as a 3D value-function surface
- **Sarsa(λ)** — tabular TD control with eligibility traces, evaluated against MC-estimated Q* across
  λ ∈ [0, 1]
- **Linear Sarsa(λ)** — the same algorithm with a 36-feature coarse-coded linear approximator instead
  of a table

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook easy21.ipynb
```

No external data — the environment is simulated. Full execution (~1.5M Monte Carlo episodes total,
plus the λ sweeps) takes roughly a minute.

## Notes

- `LinearSarsaLambda.Q_value` originally summed `theta[i]` for list-position `i` rather than
  `theta[idx]` for each active feature index `idx` — since `encode()` returns a short list of feature
  *indices* (not a 36-length one-hot vector), this meant the linear approximator almost always returned
  0 and never learned properly. This is fixed here (`total += self.theta[idx]` for `idx in phi`). This
  changes the linear Sarsa(λ) results substantially — MSE now stays in a stable, low range across all
  λ, with no collapse at λ=1, which is closer to the theoretically expected behavior than the original
  run's "always stick" degeneracy at λ=1.
- `LinearSarsaLambda.epsilon_greedy` originally referenced an undefined `state` — it happened to run
  under the original notebook's kernel session (likely a leftover global from earlier experimentation),
  but would fail on a clean run. Fixed to take `state` as a parameter.
- The two separate 1,000,000-episode Monte Carlo runs used as ground truth (`Q*`) for evaluating Sarsa(λ)
  and linear Sarsa(λ) are consolidated into one shared run, so both methods are compared against the
  same ground truth instead of two independently-sampled estimates, and the notebook runs faster.
