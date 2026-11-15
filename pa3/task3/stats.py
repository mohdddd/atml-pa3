"""Paired question-level bootstrap (after averaging over rollout orders)."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(a, b, n_boot: int = 1000, seed: int = 6304) -> dict:
    """a, b: per-question values (already averaged over orders), aligned by question.
    Returns mean(a), mean(b), mean(a - b) and its 95% percentile interval over question resamples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi), "n_questions": int(len(d)), "n_boot": n_boot}


def bootstrap_mean(a, n_boot: int = 1000, seed: int = 6304) -> tuple[float, float]:
    a = np.asarray(a, float)
    rng = np.random.default_rng(seed)
    boots = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)
