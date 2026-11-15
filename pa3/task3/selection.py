"""Selection rules over one candidate set (a prefix of a rollout order).

All functions take arrays in the CURRENT ORDER (position 0 = first revealed rollout). Ties:
* highest-score selection and (weighted) answer votes: the earliest supporting rollout in the current order wins;
* RWC: tied scores receive average ranks; vote sums within 1e-9 are treated as ties.
At N = 1 every rule returns the only candidate.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

TIE_TOL = 1e-9


def _clean(scores) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    return np.where(np.isnan(s), -np.inf, s)       # undefined scores (zero-length rollouts) rank last


def highest(scores) -> int:
    """Position of the highest score; np.argmax returns the first maximum = earliest in the order."""
    return int(np.argmax(_clean(scores)))


def vote_totals(groups, weights) -> tuple[dict, dict]:
    W: dict = {}
    first: dict = {}
    for pos, (g, w) in enumerate(zip(groups, weights)):
        W[g] = W.get(g, 0.0) + float(w)
        first.setdefault(g, pos)
    return W, first


def weighted_vote(groups, weights) -> tuple[object, int]:
    """(winning group, position of its earliest supporting rollout)."""
    W, first = vote_totals(groups, weights)
    top = max(W.values())
    best = min((g for g in W if W[g] >= top - TIE_TOL), key=lambda g: first[g])
    return best, first[best]


def self_consistency(groups) -> tuple[object, int]:
    return weighted_vote(groups, np.ones(len(groups)))


def rwc_weights(scores) -> np.ndarray:
    """w_i = (N - r_i) / (N - 1) with descending average ranks r_i (rank 1 = best)."""
    s = _clean(scores)
    n = len(s)
    if n == 1:
        return np.ones(1)
    r = rankdata(-s, method="average")
    return (n - r) / (n - 1)


def rwc(groups, scores) -> tuple[object, int]:
    if len(groups) == 1:
        return groups[0], 0
    return weighted_vote(groups, rwc_weights(scores))


def rwc_margin(groups, scores) -> float:
    """(largest - second largest answer vote) / total vote weight; runner-up 0 when one group exists."""
    W, _ = vote_totals(groups, rwc_weights(scores))
    v = sorted(W.values(), reverse=True)
    tot = sum(v)
    if tot <= 0:
        return 0.0
    return (v[0] - (v[1] if len(v) > 1 else 0.0)) / tot


def sc_leader(groups) -> tuple[object, int, bool]:
    """(self-consistency winner, its vote count, whether the top count is held by a single group)."""
    W, first = vote_totals(groups, np.ones(len(groups)))
    top = max(W.values())
    leaders = [g for g in W if W[g] == top]
    best = min(leaders, key=lambda g: first[g])
    return best, int(top), len(leaders) == 1


def visconf_scores(kl_list, v_list, q=(5.0, 95.0), eps=1e-8):
    """VisConf over one candidate set G (list of per-rollout token arrays, current set only).
    Returns (C_i, V_i, Z_i = V_i C_i, Z_token_i = mean_t c_it v_it, (Q05, Q95))."""
    lens = np.array([len(k) for k in kl_list])
    kl = np.concatenate(kl_list).astype(np.float64)
    v = np.concatenate(v_list).astype(np.float64)
    lo, hi = np.percentile(kl, q, method="linear")
    c = np.clip((kl - lo) / max(hi - lo, eps), 0.0, 1.0)
    C, V, Zt = _seg_mean(c, lens), _seg_mean(v, lens), _seg_mean(c * v, lens)
    return C, V, V * C, Zt, (float(lo), float(hi))


def _seg_mean(x: np.ndarray, lens: np.ndarray) -> np.ndarray:
    out = np.full(len(lens), np.nan)
    ends = np.cumsum(lens)
    starts = ends - lens
    cs = np.concatenate([[0.0], np.cumsum(x)])
    nz = lens > 0
    out[nz] = (cs[ends[nz]] - cs[starts[nz]]) / lens[nz]
    return out
