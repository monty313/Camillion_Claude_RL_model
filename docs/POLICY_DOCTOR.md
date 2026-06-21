# POLICY DOCTOR — interpretability & alpha-vs-policy diagnostics

All metrics here are **read-only diagnostics**. None is an observation feature
and **none is ever a reward term** (reward = the real objective only: risk-adjusted
PnL, FTMO progress, drawdown control, trade quality).

## Policy 1/3/10-bar directional accuracy (definition)
Each step's action becomes a directional call `d_t`:
`BUY -> +1`, `SELL -> -1`, `HOLD/CLOSE -> 0` (0 = no directional bet, not graded).
For horizon `h`, `d_t` is "correct" if `sign(d_t) == sign(close[t+h] - close[t])`.
Policy accuracy at bar `t` = fraction of correct directional calls `X` in the
rolling window with `X <= t-h`. Computed identically to alpha accuracy so the
two are directly comparable.

## No-leakage guarantee (alpha AND policy)
Both use one primitive, `signal_accuracy.rolling_accuracy_counts`:
a signal at bar `X` is graded against `close[X+h]`, and at decision bar `t` only
signals with `X <= t-h` are counted. Therefore `out[t]` depends on `close` only up
to index `t` — **modifying any bar > t cannot change it**, and a bar never sees its
own future. Proven in `tests/test_alpha_accuracy_no_leakage.py` and
`tests/test_signal_accuracy_no_leakage.py`.

## Alpha reliability aggregates (exact)
Per alpha: rolling accuracy + valid-sample count at horizons 1/3/10. Aggregated
**only over ASSIGNED alphas with >= min_samples graded outcomes** (default 5):
- `mean` — mean accuracy across valid alphas
- `best` — max accuracy across valid alphas
- `dispersion` — std across valid alphas (alpha disagreement / regime signal)
- `n_valid` — how many alphas qualified
These live in diagnostics; a compact aggregate may enter the observation later
(v1.2.0) only if ablation shows it helps.

## Leader-chasing test (explicit, numeric)
- `leader(t)` = the assigned, valid alpha with the highest rolling 3-bar accuracy.
- `leader_follow_rate` = fraction of *decision bars* (policy directional != 0 and
  leader active) where `policy_dir == sign(leader_signal)`.
- `policy_outperforms_best` = policy mean 3-bar accuracy > the best single alpha's
  mean 3-bar accuracy.
- **flag = (leader_follow_rate > 0.75) AND (NOT policy_outperforms_best).**
A copycat policy lights this up; a contextual policy clears it.

## Best-single-alpha comparison
Reports the best assigned alpha's mean 3-bar accuracy vs the policy's, the margin,
and whether the policy beats the best single alpha (the bar it must clear to be
worth more than the alpha layer).

## Block saliency (why)
Model-agnostic **ablation**: zero each observation block, measure the L1 shift in
the action distribution `pi(a|s)`, normalize to fractions, and group into
`raw_indicators / alpha / account / portfolio / time`. Answers "which part of the
state drove this decision." A gradient-saliency path is added when torch is present.
