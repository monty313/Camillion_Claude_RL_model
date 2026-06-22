# Audit S1.1: SB3 check_env passes. Skips cleanly if SB3 absent (run in Colab).
try:
    import gymnasium  # noqa
    import stable_baselines3  # noqa
    _HAVE = True
except Exception:
    _HAVE = False
from tests._audit_helpers import cache
from src.strategies.registry import AlphaRegistry


def test_env_checker():
    if not _HAVE:
        print("SKIP test_env_checker: needs gymnasium + stable-baselines3 (run in Colab)")
        return
    from stable_baselines3.common.env_checker import check_env
    from src.training.gym_adapter import make_gym_env
    ind, cl, t = cache(n=800)
    env = make_gym_env(ind, cl, t, AlphaRegistry(), warmup=210)
    check_env(env)   # raises on any hard Gym/SB3 violation
