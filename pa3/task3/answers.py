"""Answer normalisation, answer groups and correctness labels (the Task 3 scorer).

* Missing extractions (null) form one NO_ANSWER group per question, always graded incorrect, and stay in
  voting, cost and accuracy.
* MATH: answers are parsed with math-verify (`parse("\\boxed{<answer>}")`). Groups are formed per question
  WITHOUT gold answers: distinct answer strings are visited in order of their first rollout ID; a string joins
  the first existing group whose representative it equals (whitespace-insensitive) or is equivalent to by
  `verify(parse(rep), parse(s))`; otherwise it starts a new group. Strings that math-verify cannot parse only
  match by exact (whitespace-insensitive) string. A group is graded with the same rule against the gold answer
  (`verify(parse(gold), parse(rep))`), and every rollout gets the label of its group.
* MathVista: the official MathVista normalisation (`normalize_extracted_answer` in the released evaluation
  code), using each question's released metadata:
    multi_choice: "(X)" -> letter X; an option letter is kept; an exact option text maps to its letter;
                  otherwise the most similar option by Levenshtein distance (first on ties). Answers are
                  represented as option letters.
    integer:      str(int(float(x)));  float: str(round(float(x), precision));  list: str(list).
  Because the supplied extractor keeps LaTeX (e.g. `\\frac{3}{14}`, `51.045\\%`, `9 \\, \\text{cm}`), free-form
  answers are first passed through a conservative LaTeX-to-number cleaner (`clean_number`). Extractions that are
  still not numbers of the required type are kept as their own (always incorrect) groups keyed by the raw
  string, not merged into NO_ANSWER. Grading is exact match of the normalised answer with the gold answer.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pa3.task3.config import NO_ANSWER

SCORER_VERSION = "pa3-task3-scorer-1"


def _ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


# --------------------------------------------------------------------------------------------------- MATH
class MathScorer:
    def __init__(self):
        from math_verify import parse, verify

        self._parse, self._verify = parse, verify
        self._cache: dict[str, list] = {}
        self.timeout_s = 5                       # math-verify default; a timed-out comparison = not equivalent
        self.timeouts: list[tuple[str, str]] = []
        import logging

        logging.getLogger("math_verify").setLevel(logging.ERROR)

    def parse(self, s: str):
        if s not in self._cache:
            try:
                self._cache[s] = self._parse("\\boxed{" + s + "}")
            except Exception:
                self._cache[s] = []
        return self._cache[s]

    def equiv(self, ref: str, s: str) -> bool:
        """`s` equivalent to reference `ref` (reference first, as in grading)."""
        if _ws(ref) == _ws(s):
            return True
        a, b = self.parse(ref), self.parse(s)
        if not a or not b:
            return False
        import time

        t0 = time.perf_counter()
        try:
            res = bool(self._verify(a, b, timeout_seconds=self.timeout_s))
        except Exception:
            res = False
        if time.perf_counter() - t0 >= 0.95 * self.timeout_s:
            self.timeouts.append((ref, s))
        return res

    def key(self, s: str) -> str:
        return _ws(s)


def group_answers(answers: list, equiv) -> tuple[list[int], list[str]]:
    """Group one question's answers (rollout order 0..31) without gold answers.
    None -> NO_ANSWER group. Returns (group id per rollout, representative per group).
    Group ids are numbered by first appearance."""
    reps: list[str] = []
    gid_of_string: dict[str, int] = {}
    out = []
    for a in answers:
        if a is None:
            a = NO_ANSWER
        if a in gid_of_string:
            out.append(gid_of_string[a])
            continue
        g = None
        if a != NO_ANSWER:
            for j, r in enumerate(reps):
                if r != NO_ANSWER and equiv(r, a):
                    g = j
                    break
        if g is None:
            reps.append(a)
            g = len(reps) - 1
        gid_of_string[a] = g
        out.append(g)
    return out, reps


# ---------------------------------------------------------------------------------------------- MathVista
_TEXT_CMD = re.compile(r"\\(?:text|mathrm|textbf|mbox|operatorname|textrm|mathit)\s*\{[^{}]*\}")
_FRAC = re.compile(r"^(-?)\\frac\{(-?[0-9.]+)\}\{(-?[0-9.]+)\}$")
_SLASH = re.compile(r"^(-?[0-9.]+)/(-?[0-9.]+)$")
_SCI = re.compile(r"^(-?[0-9.]+)(?:\\times|\\cdot|x|\*)10\^\{?(-?[0-9]+)\}?$")
_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_POW = re.compile(r"^(-?[0-9.]+)\^\{?(-?[0-9]+)\}?$")
_UNIT_WORD = re.compile(r"\s+(?:mm|cm|m|km|g|kg|mg|ml|mL|l|L|s|sec|seconds|min|minutes|h|hours|years?|days?|"
                        r"ft|feet|inches|in|yards?|miles?|degrees?|units?|dollars?|cents?|people|percent)\.?$")


def clean_number(s: str):
    """Conservative LaTeX/plain-text -> float. Returns None when the string is not a single number."""
    if s is None:
        return None
    x = s.strip()
    x = x.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for t in ["$", "\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "~"]:
        x = x.replace(t, " ")
    x = _TEXT_CMD.sub(" ", x)                                     # unit / word annotations
    x = _UNIT_WORD.sub("", x.strip())                             # one trailing plain unit word
    for t in ["^{\\circ}", "^\\circ", "\\circ", "°", "\\%", "%", "\\$"]:
        x = x.replace(t, " ")
    x = re.sub(r"\s+", "", x)
    x = x.replace("−", "-")
    if _THOUSANDS.match(x):
        x = x.replace(",", "")
    m = _FRAC.match(x)
    if m:
        sign, a, b = m.groups()
        try:
            v = float(a) / float(b)
            return -v if sign else v
        except (ValueError, ZeroDivisionError):
            return None
    m = _SLASH.match(x)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    m = _POW.match(x)
    if m:
        try:
            v = float(m.group(1)) ** int(m.group(2))
            return v if math.isfinite(v) else None
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
    m = _SCI.match(x)
    if m:
        try:
            return float(m.group(1)) * 10.0 ** int(m.group(2))
        except ValueError:
            return None
    if not re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", x):
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def clean_list(s: str):
    x = _TEXT_CMD.sub(lambda m: "," if re.search(r"\b(and|to)\b", m.group(0)) else " ", s)
    x = re.sub(r"\b(and|to)\b", ",", x)
    num = r"(?<!\d)-?\d+(?:\.\d+)?"
    nums = re.findall(num, x)
    rest = re.sub(num, "", x)
    if not nums or re.sub(r"[\s\[\](),;\-$\\]", "", rest):
        return None
    return [int(float(n)) if float(n).is_integer() else float(n) for n in nums]


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mathvista_normalize(extraction, meta: dict) -> tuple[str, bool]:
    """Normalised answer key and whether it is a valid answer of the required type.
    `meta`: question_type, answer_type, choices, precision. None extraction -> (NO_ANSWER, False)."""
    if extraction is None:
        return NO_ANSWER, False
    if meta["question_type"] == "multi_choice":
        choices = list(meta["choices"])
        x = str(extraction).strip()
        letter = re.findall(r"\(([a-zA-Z])\)", x)
        if letter:
            x = letter[0].upper()
        options = [chr(ord("A") + i) for i in range(len(choices))]
        if x in options:
            return x, True
        if x in choices:
            return options[choices.index(x)], True
        d = [levenshtein(x, c) for c in choices]
        return options[d.index(min(d))], True
    at = meta["answer_type"]
    raw = "raw:" + re.sub(r"\s+", " ", str(extraction).strip())
    if at == "integer":
        v = clean_number(extraction)
        return (str(int(v)), True) if v is not None else (raw, False)
    if at == "float":
        v = clean_number(extraction)
        return (str(round(v, int(meta["precision"]))), True) if v is not None else (raw, False)
    if at == "list":
        v = clean_list(str(extraction))
        return (str(v), True) if v is not None else (raw, False)
    return raw, False


# ---------------------------------------------------------------------------------------------- building
def build_math(gen: pd.DataFrame, qids: list, scorer: MathScorer):
    rows, groups = [], []
    for qi, qid in enumerate(qids):
        sub = gen.iloc[qi * 32:(qi + 1) * 32]
        ans = [None if a is None else str(a).strip() or None for a in sub.extracted_answer]
        gids, reps = group_answers(ans, scorer.equiv)
        gold = str(sub.gold_answer.iloc[0])
        gcorr = [r != NO_ANSWER and scorer.equiv(gold, r) for r in reps]
        own = [a is not None and scorer.equiv(gold, a) for a in ans]
        for rid in range(32):
            rows.append({"sample_id": qid, "rollout_id": rid, "answer_raw": sub.extracted_answer.iloc[rid],
                         "answer_key": reps[gids[rid]], "group": gids[rid], "no_answer": ans[rid] is None,
                         "valid": ans[rid] is not None and bool(scorer.parse(ans[rid])),
                         "correct": bool(gcorr[gids[rid]]), "correct_own_string": bool(own[rid])})
        for g, r in enumerate(reps):
            groups.append({"sample_id": qid, "group": g, "representative": r, "correct": bool(gcorr[g]),
                           "size": gids.count(g),
                           "members": sorted({a for a, x in zip(ans, gids) if x == g and a is not None})})
    return pd.DataFrame(rows), pd.DataFrame(groups)


def build_mathvista(gen: pd.DataFrame, qids: list, meta: pd.DataFrame):
    rows, groups = [], []
    for qi, qid in enumerate(qids):
        sub = gen.iloc[qi * 32:(qi + 1) * 32]
        m = meta.loc[qid].to_dict()
        gold = str(sub.gold_answer.iloc[0])
        norm = [mathvista_normalize(None if a is None else str(a), m) for a in sub.extracted_answer]
        keys = [k for k, _ in norm]
        gids, reps = group_answers([None if k == NO_ANSWER else k for k in keys], lambda a, b: a == b)
        gcorr = [r != NO_ANSWER and not r.startswith("raw:") and r == gold for r in reps]
        for rid in range(32):
            rows.append({"sample_id": qid, "rollout_id": rid, "answer_raw": sub.extracted_answer.iloc[rid],
                         "answer_key": keys[rid], "group": gids[rid], "no_answer": keys[rid] == NO_ANSWER,
                         "valid": norm[rid][1], "correct": bool(gcorr[gids[rid]]),
                         "correct_own_string": bool(gcorr[gids[rid]])})
        for g, r in enumerate(reps):
            groups.append({"sample_id": qid, "group": g, "representative": r, "correct": bool(gcorr[g]),
                           "size": gids.count(g),
                           "members": sorted({str(a) for a, x in zip(sub.extracted_answer, gids)
                                              if x == g and a is not None})})
    return pd.DataFrame(rows), pd.DataFrame(groups)


def scorer_info() -> dict:
    def v(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None

    src = Path(__file__).read_bytes()
    return {"scorer_version": SCORER_VERSION, "answers_py_sha256": hashlib.sha256(src).hexdigest(),
            "math_verify": v("math-verify"), "latex2sympy2_extended": v("latex2sympy2_extended"),
            "sympy": v("sympy"), "mathvista_rules": "official normalize_extracted_answer + clean_number/clean_list",
            "numpy": np.__version__}
