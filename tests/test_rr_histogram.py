# v1.12.0 Stage 4: the R:R-choice histogram (post-hoc analysis of the per-trade bracket log) + the freeze/
# unlock curriculum mask (config-driven).
import numpy as np
from src.analysis.rr_histogram import rr_histogram


def _close(rr, won, align=0.0, pnl=1.0):
    return {"event": "close", "rr": rr, "trade_won": won, "alignment_score_at_entry": align, "pnl": pnl,
            "clamped": False}


def test_rr_histogram_buckets_and_winrate():
    log = [{"event": "open"}] + [
        _close(0.3, False), _close(0.6, True), _close(1.2, True), _close(2.5, True, pnl=3.0), _close(4.0, False)]
    h = rr_histogram(log)
    assert h["n_trades"] == 5
    # buckets: edges (0,0.5,0.75,1,1.5,2,3,inf) -> 0.3->bin0, 0.6->bin1, 1.2->bin3, 2.5->bin5, 4.0->bin6
    assert h["overall_counts"] == [1, 1, 0, 1, 0, 1, 1]
    assert abs(h["win_rate"] - 0.6) < 1e-9
    assert "R:R HISTOGRAM" in h["text"]


def test_rr_histogram_segments_by_alignment():
    log = [_close(1.0, True, align=-0.8), _close(3.0, True, align=-0.7),
           _close(0.4, False, align=0.7), _close(0.5, False, align=0.8)]
    h = rr_histogram(log, n_align_bins=2)
    assert len(h["by_alignment"]) == 2          # low-alignment vs high-alignment buckets
    means = [v["mean_rr"] for v in h["by_alignment"].values()]
    assert max(means) >= 2.0 and min(means) <= 0.5   # the two conviction groups chose different ratios


def test_rr_histogram_empty():
    assert rr_histogram([{"event": "open"}])["n_trades"] == 0


def test_curriculum_mask_flips_with_stage():
    from jax_tpu import jax_config as JC
    assert JC.curriculum_head_mask(1) == (0.0, 0.0, 0.0)    # freeze all
    assert JC.curriculum_head_mask(2) == (0.0, 0.0, 1.0)    # unlock lot
    assert JC.curriculum_head_mask(3) == (1.0, 1.0, 1.0)    # unlock all
    assert len(JC.FROZEN_CONT) == 3
