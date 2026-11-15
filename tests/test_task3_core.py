"""Unit tests for the tricky parts of Task 3 (run: python -m pytest tests/test_task3_core.py -q)."""
import numpy as np
import pytest

from pa3.task3 import selection as S
from pa3.task3.answers import NO_ANSWER, clean_list, clean_number, group_answers, mathvista_normalize


# ------------------------------------------------------------------------------------------ selection
def test_highest_tie_earliest():
    assert S.highest([0.1, 0.5, 0.5, 0.2]) == 1
    assert S.highest([np.nan, -3.0]) == 1          # undefined score ranks last


def test_self_consistency_ties_and_no_answer():
    # B and A both have 2 votes; A's earliest supporter comes first
    assert S.self_consistency(["A", "B", "B", "A"]) == ("A", 0)
    assert S.self_consistency(["B", "A", "A", "B"]) == ("B", 0)
    # NO_ANSWER is a normal group in voting
    assert S.self_consistency([NO_ANSWER, NO_ANSWER, "A"])[0] == NO_ANSWER


def test_rwc_weights_average_ranks():
    w = S.rwc_weights([3.0, 1.0, 3.0, 0.0])        # ranks 1.5, 3, 1.5, 4
    assert np.allclose(w, [(4 - 1.5) / 3, (4 - 3) / 3, (4 - 1.5) / 3, 0.0])
    assert np.allclose(S.rwc_weights([1, 2, 3]), [0.0, 0.5, 1.0])
    assert S.rwc_weights([5.0]).tolist() == [1.0]


def test_rwc_winner_and_tie():
    # scores 4>3>2>1 -> weights 1, 2/3, 1/3, 0. A = 1 + 0 = 1, B = 2/3 + 1/3 = 1 -> tie -> earliest (A at 0)
    assert S.rwc(["A", "B", "B", "A"], [4, 3, 2, 1]) == ("A", 0)
    assert S.rwc(["B", "A", "A", "B"], [4, 3, 2, 1]) == ("B", 0)
    # quality beats count: A has 2 low-ranked votes, B one top vote
    g, _ = S.rwc(["A", "A", "B", "C"], [0.1, 0.2, 0.9, 0.3])  # w: 0, 1/3, 1, 2/3 -> A 1/3, B 1, C 2/3
    assert g == "B"
    assert S.rwc(["Z"], [1.0]) == ("Z", 0)


def test_rwc_quality_beats_count_example():
    # weights 0, 0.5, 1: A = 0.5, B = 1 -> the single top-ranked vote beats two low-ranked ones
    assert S.rwc(["A", "A", "B"], [0.1, 0.2, 0.9])[0] == "B"
    assert S.self_consistency(["A", "A", "B"])[0] == "A"


def test_rwc_margin():
    # weights 1, 2/3, 1/3, 0 (total 2): A = 1, B = 1 -> margin 0
    assert S.rwc_margin(["A", "B", "B", "A"], [4, 3, 2, 1]) == pytest.approx(0.0)
    # one group -> runner-up 0 -> margin 1
    assert S.rwc_margin(["A"] * 4, [4, 3, 2, 1]) == pytest.approx(1.0)
    # A = 1 + 2/3, B = 1/3 + 0 -> (5/3 - 1/3) / 2 = 2/3
    assert S.rwc_margin(["A", "A", "B", "B"], [4, 3, 2, 1]) == pytest.approx(2 / 3)


def test_sc_leader_unique():
    assert S.sc_leader(["A", "A", "B"]) == ("A", 2, True)
    assert S.sc_leader(["A", "B"]) == ("A", 1, False)


def test_visconf_prefix_only_and_identity():
    rng = np.random.default_rng(0)
    kl = [rng.normal(10, 2, size=n).astype(np.float32) for n in (5, 9, 3, 12)]
    v = [rng.uniform(0, 0.4, size=len(k)).astype(np.float32) for k in kl]
    C2, V2, Z2, Zt2, (lo2, hi2) = S.visconf_scores(kl[:2], v[:2])
    pooled = np.concatenate(kl[:2]).astype(np.float64)
    assert lo2 == pytest.approx(np.percentile(pooled, 5)) and hi2 == pytest.approx(np.percentile(pooled, 95))
    # prefix scores must not depend on unseen rollouts
    C4, *_ = S.visconf_scores(kl, v)
    C2b, *_ = S.visconf_scores(kl[:2], v[:2])
    assert np.allclose(C2, C2b) and not np.allclose(C2, C4[:2])
    c0 = np.clip((kl[0].astype(np.float64) - lo2) / (hi2 - lo2), 0, 1)
    assert C2[0] == pytest.approx(c0.mean()) and V2[0] == pytest.approx(v[0].mean())
    # tokenwise fusion identity: Z_token - Z = population covariance of (c, v)
    v0 = v[0].astype(np.float64)
    cov = np.mean(c0 * v0) - c0.mean() * v0.mean()
    assert Zt2[0] - Z2[0] == pytest.approx(cov, abs=1e-9)
    # degenerate spread -> eps denominator, values clipped
    C, *_ = S.visconf_scores([np.ones(3)], [np.ones(3)])
    assert np.allclose(C, 0.0)


# -------------------------------------------------------------------------------------------- answers
def test_grouping_without_gold():
    eq = lambda a, b: float(a) == float(b)
    gids, reps = group_answers(["2", None, "2.0", "3", None], eq)
    assert gids == [0, 1, 0, 2, 1] and reps == ["2", NO_ANSWER, "3"]


def test_math_scorer_equivalences():
    pytest.importorskip("math_verify")
    from pa3.task3.answers import MathScorer
    m = MathScorer()
    assert m.equiv("\\frac{1}{2}", "0.5") and m.equiv("\\$40", "40") and m.equiv("\\dfrac{\\sqrt7}{14}", "\\frac{\\sqrt{7}}{14}")
    assert not m.equiv("2", "3")


def test_clean_number():
    assert clean_number("51.045\\%") == pytest.approx(51.045)
    assert clean_number("\\frac{3}{14}") == pytest.approx(3 / 14)
    assert clean_number("9 \\, \\text{cm}") == 9
    assert clean_number("2,300") == 2300
    assert clean_number("7.119 \\times 10^3 \\, \\text{s}") == pytest.approx(7119)
    assert clean_number("45^\\circ") == 45
    assert clean_number("10^8") == 1e8 and clean_number("600 mL") == 600 and clean_number("13 years") == 13
    assert clean_number("5.4974 trillion") is None
    assert clean_number("10-15 years") is None and clean_number("junk") is None


def test_clean_list():
    assert clean_list("2014 \\text{ and } 2016") == [2014, 2016]
    assert clean_list("2014-2015") == [2014, 2015]
    assert clean_list("[0, 2, 0, 2]") == [0, 2, 0, 2]
    assert clean_list("several") is None


def test_mathvista_normalize():
    mc = {"question_type": "multi_choice", "answer_type": "text", "choices": ["135°", "140°", "145°"], "precision": None}
    assert mathvista_normalize("B", mc) == ("B", True)
    assert mathvista_normalize("(C) 145°", mc) == ("C", True)
    assert mathvista_normalize("140°", mc) == ("B", True)
    assert mathvista_normalize("145", mc) == ("C", True)        # most similar option (Levenshtein)
    assert mathvista_normalize(None, mc) == (NO_ANSWER, False)
    i = {"question_type": "free_form", "answer_type": "integer", "choices": None, "precision": None}
    assert mathvista_normalize("2.7", i) == ("2", True)        # official int(float(x))
    assert mathvista_normalize("10-15 years", i) == ("raw:10-15 years", False)
    f = {"question_type": "free_form", "answer_type": "float", "choices": None, "precision": 1.0}
    assert mathvista_normalize("1.15", f)[0] in ("1.1", "1.2") and mathvista_normalize("1.24", f)[0] == "1.2"
    lst = {"question_type": "free_form", "answer_type": "list", "choices": None, "precision": None}
    assert mathvista_normalize("2014 \\text{ and } 2016", lst) == ("[2014, 2016]", True)
