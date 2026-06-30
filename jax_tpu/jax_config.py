# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  All JAX/TPU knobs in ONE place: (a) PPO hyperparams that MIRROR the CPU
#      trainer EXACTLY (so a JAX-trained policy is comparable to a CPU one), and
#      (b) the SCALE knobs that push a Colab TPU v2-8 to 70-80% utilization, and
#      (c) the STOP condition (40 consecutive held-out challenge passes) + where
#      progress/policies are saved on Drive.
# WHERE jax_tpu/jax_config.py
# HOW   Plain module constants (no logic). Read by jax_ppo / jax_trainer / jax_eval.
#       The PPO block is copied 1:1 from src/training/trainer.py PPO_HPARAMS +
#       SB3 MlpPolicy defaults; DO NOT drift it without bumping the CPU side too.
# DEPENDS_ON: (nothing — pure constants)
# USED_BY: jax_tpu/jax_ppo.py, jax_tpu/jax_trainer.py, jax_tpu/jax_eval.py, notebook
# CHANGE_NOTES(IRAC): I: a JAX run must be comparable to a CPU run AND saturate the
#   TPU AND stop on a real consistency bar. R: operator 2026-06-28 — 70-80% TPU,
#   train to 40 straight passes, save progress+policies to Colab. A: mirror CPU
#   PPO, expose pmap/vmap scale, set TARGET_CONSECUTIVE_PASSES=40, Drive paths.
#   C: same-fingerprint policies, a saturated TPU, and a documented march to 40-in-a-row.
# =====================================================================
"""JAX/TPU configuration: PPO hyperparams (mirror CPU) + TPU scale + stop/save."""
from __future__ import annotations

# ---------------------------------------------------------------------
# OBS / ACTION (the locked contract — imported from config.constants at use site;
# duplicated here only as a readable reference, asserted equal in jax_env).
# ---------------------------------------------------------------------
OBS_SIZE: int = 526                 # == config.constants.OBS_TOTAL_SIZE (v1.9.0: +9 momentum-perception block)
N_ACTIONS: int = 4                  # HOLD, BUY, SELL, CLOSE
N_STATIC_OBS: int = 468             # precomputed per-bar blocks (see jax_static_features); +9 momentum in v1.9.0
N_DYNAMIC_OBS: int = 58             # account/sizing/recent (40) + trade_risk (14) + consistency (4, v1.8.0)

# ---------------------------------------------------------------------
# PPO HYPERPARAMETERS — MIRROR src/training/trainer.py EXACTLY.
# (trainer.py PPO_HPARAMS + SB3 PPO/MlpPolicy defaults for the unset ones.)
# Changing any of these makes JAX policies NOT comparable to CPU policies.
# ---------------------------------------------------------------------
GAMMA: float = 0.9999              # operator 2026-06-29: STRETCHED horizon (~1/(1-g)=10000 steps ~1.7 days). With
                                   # the shared-pot env cycling N symbols (~5760 steps/day at 4 symbols), 0.9995 was
                                   # only ~1/3 day -> the multi-day WON-DAY STREAK reward was discounted to ~0 a day
                                   # out (the bot couldn't VALUE/PROTECT the streak). 0.9999 lets a breach TODAY also
                                   # forfeit tomorrow's bigger streak reward. MUST match src/training/trainer.py.
GAE_LAMBDA: float = 0.97            # trainer.py
CLIP_RANGE: float = 0.2            # SB3 default
ENT_COEF_START: float = 0.01       # trainer.py (start of the linear anneal)
ENT_COEF_END: float = 0.0          # trainer.py _make_entropy_anneal -> ~0
VF_COEF: float = 0.5               # SB3 default
LEARNING_RATE: float = 3e-4        # trainer.py
MAX_GRAD_NORM: float = 0.5         # SB3 default
N_EPOCHS: int = 10                 # SB3 default (PPO update epochs per rollout)
NET_ARCH: tuple[int, ...] = (256, 256, 256)   # trainer.py policy_kwargs.net_arch
ACTIVATION: str = "tanh"           # SB3 MlpPolicy default activation

# VecNormalize (trainer.py VECNORM_KW) — the JAX PPO replicates this with an online
# running-mean/std normalizer over observations, frozen (training=False) at eval.
NORM_OBS: bool = True
NORM_REWARD: bool = False
CLIP_OBS: float = 10.0

# ---------------------------------------------------------------------
# TPU SCALE — push a Colab TPU v2-8 to ~70-80% utilization (operator 2026-06-28).
# Effective batch per update = N_ENVS_PER_CORE * N_DEVICES * N_STEPS.
# Default ~ 2048 * 8 * 128 = 2.1M states/update (saturates the matrix engine on
# the tiny 3x256 MLP). RAISE N_ENVS_PER_CORE until HBM ~80% or step time plateaus
# (the notebook prints a utilization probe + a suggested value).
# ---------------------------------------------------------------------
N_ENVS_PER_CORE: int = 2048        # parallel envs per TPU core (vmap)
N_STEPS: int = 128                 # rollout length per update (lax.scan)
MINIBATCH_SIZE: int = 8192         # PPO minibatch (large -> TPU-friendly)
MAX_BARS: int = 7200               # FIXED episode length (Rule 2: static shapes, ~5 trading days M1)
USE_PMAP: bool = True              # shard across all TPU cores (pmean grads)
COMPUTE_DTYPE: str = "bfloat16"    # matmuls in bf16 (TPU-native); env money math stays fp32
MONEY_DTYPE: str = "float32"       # equity/reward/FTMO — NEVER bf16 (C3: tiny moves must survive)

# Domain-randomized risk across the env army (so ONE policy handles the whole range,
# and you can dial target/risk at inference with no retrain — blueprint §5).
DOMAIN_RANDOMIZE_RISK: bool = True
DAILY_TARGET_MIN: float = 0.020    # 2.0%
DAILY_TARGET_MAX: float = 0.035    # 3.5%
TRAILING_DD_MIN: float = 0.030     # 3.0%
TRAILING_DD_MAX: float = 0.050     # 5.0%

# ---------------------------------------------------------------------
# STOP CONDITION + EVAL — train until 40 consecutive held-out challenge PASSES
# (operator 2026-06-28). One failed held-out window resets the streak to 0.
# ---------------------------------------------------------------------
# STOP CONDITION = 40 WINNING DAYS IN A ROW on held-out data (operator 2026-06-28). A "winning day" ends
# >= +2.5% of initial; a BREACH (or a failed day) RESETS the streak (start over) but training KEEPS GOING.
# We also still report the challenge pass-rate as a HEALTH metric, but the STOP is the won-day streak.
TARGET_WON_DAY_STREAK: int = 40       # <-- the real stop: 40 winning days back-to-back (held-out)
TARGET_CONSECUTIVE_PASSES: int = 40   # (health) longest run of consecutive CHALLENGE passes in the windowed eval
EVAL_EVERY: int = 50                  # run the held-out pass-rate eval every N updates
EVAL_N_WINDOWS: int = 256             # held-out challenge windows per eval (also the P(pass) sample)
WON_DAY_N_WALKS: int = 8              # parallel continuous held-out walks for the won-day-streak STOP metric
EVAL_DAILY_TARGET: float = 0.025      # the ACTUAL FTMO target used for the streak (2.5%)
EVAL_TRAILING_DD: float = 0.04        # the ACTUAL FTMO trailing wall used for the streak (4%)
PROFIT_TARGET_PCT: float = 0.10       # +10% challenge pass
TOTAL_UPDATES_CAP: int = 1_000_000    # OPT-IN safety cap; default training is UNBOUNDED (stop only at 40-in-a-row)
ANNEAL_UPDATES: int = 100_000         # entropy anneals 0.01->0 over this many updates (decoupled from the open-ended stop)

# ---------------------------------------------------------------------
# SAVE / PROGRESS — everything persisted to Colab Drive so a disconnect never
# loses progress; policies + their details are documented as we go.
# ---------------------------------------------------------------------
DRIVE_ROOT: str = "/content/drive/MyDrive/Camillion"
SAVE_DIR: str = DRIVE_ROOT + "/jax_models"           # checkpoints live here
PROGRESS_JSONL: str = "jax_progress.jsonl"           # one line per eval (the run ledger)
BEST_DIR: str = "best_policy"                         # rolling best (longest streak so far)
PASSED_DIR: str = "passed_40_in_a_row"               # final policy once the gate is hit
CHECKPOINT_EVERY: int = 25                            # lightweight crash-safe save every N updates (between evals);
                                                      # progress also saved at EVERY eval and on interrupt (Ctrl-C / disconnect)
SEED: int = 42
EVAL_SEED: int = 999                                  # fixed seed -> reproducible eval windows
