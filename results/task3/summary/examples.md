# Task 3 - qualitative examples

## MATH (Qwen2.5-3B): selection disagreement
Selection rule: held-out questions in split order, rollout order 0, N = 32: the first question where Self-certainty + RWC is right and self-consistency wrong, and the first where the reverse holds.

**Question test:2460** - self-consistency -> `58`, Self-certainty + RWC -> `154`

| answer | correct | votes | rwc_weight |
|---|---|---|---|
| 154 | True | 11 | 6.81 |
| 58 | False | 11 | 5.81 |
| NO_ANSWER | False | 3 | 0.65 |
| 13 | False | 1 | 0.39 |
| 45 | False | 1 | 0.68 |
| 346 | False | 1 | 0.45 |

**Question test:2198** - self-consistency -> `11`, Self-certainty + RWC -> `14`

| answer | correct | votes | rwc_weight |
|---|---|---|---|
| 11 | True | 5 | 0.81 |
| 14 | False | 4 | 3.16 |
| 10^{200} - 7 \cdot 10^{100} + 12 | False | 3 | 2.26 |
| 10^{200} - 7 \cdot 10^{100} + 7 | False | 2 | 1.58 |
| 30 | False | 2 | 0.81 |
| 6 | False | 2 | 0.87 |

## MATH (Qwen2.5-3B): premature stopping
Selection rule: dev-selected quality-aware controller on held-out runs; questions in split order, first order where it stopped early with a wrong answer while fixed-32 self-consistency (same order) was right; first 2.

**Question test:3504, order 7** - stopped at n = 4 with votes {'120': 2, '72': 1, '216': 1}; the full pool of 32: {'216 (correct)': 13, '120': 4, '36': 2, 'NO_ANSWER': 2, '1440': 1, '1152': 1, '144': 1, '1728': 1, '168': 1, '15120': 1, '1944': 1, '24': 1, '360': 1, '576': 1, '72': 1}

**Question test:2460, order 7** - stopped at n = 4 with votes {'58': 3, '154': 1}; the full pool of 32: {'154 (correct)': 11, '58': 11, 'NO_ANSWER': 3, '13': 1, '45': 1, '346': 1, '50': 1, '80': 1, '90': 1, '98': 1}

## MathVista (Qwen2.5-VL-3B): selection disagreement
Selection rule: held-out questions in split order, rollout order 0, N = 32: the first question where VisConf + RWC is right and self-consistency wrong, and the first where the reverse holds.

**Question 643** - self-consistency -> `C`, VisConf + RWC -> `A`

| answer | correct | votes | rwc_weight |
|---|---|---|---|
| C | False | 11 | 4.13 |
| A | True | 10 | 5.45 |
| NO_ANSWER | False | 8 | 5.16 |
| D | False | 3 | 1.26 |

**Question 552** - self-consistency -> `4`, VisConf + RWC -> `2`

| answer | correct | votes | rwc_weight |
|---|---|---|---|
| 4 | True | 14 | 4.77 |
| 2 | False | 9 | 7.42 |
| 3 | False | 9 | 3.81 |

## MathVista (Qwen2.5-VL-3B): premature stopping
Selection rule: dev-selected quality-aware controller on held-out runs; questions in split order, first order where it stopped early with a wrong answer while fixed-32 self-consistency (same order) was right; first 2.

**Question 601, order 2** - stopped at n = 4 with votes {'0.12': 3, '0.33': 1}; the full pool of 32: {'0.33 (correct)': 12, '0.12': 11, '0.47': 6, '0.11': 2, '0.71': 1}

**Question 821, order 8** - stopped at n = 4 with votes {'B': 3, 'NO_ANSWER': 1}; the full pool of 32: {'A (correct)': 21, 'B': 8, 'NO_ANSWER': 3}

## Live MATH: sequential refinement
Selection rule: live questions in their fixed order: the first 2 wrong-to-correct and the first 2 correct-to-wrong revisions (relative to the shared initial answer).

**wrong -> correct: test:3504** - gold `216`, initial `NO_ANSWER (initial response truncated at 1,024)`, revised `216`

> :    \[    2^1 \times 3^1 \times 5^1 \times 7^1 = 2 \times 3 \times 5 \times 7 = 210    \]    Check if 210 is a multiple of 72:    \[    210 \div 72 = 2.9167 \quad (\text{not an integer})    \]    This case is not valid.  From the valid cases, the smallest number that meets the criteria is 216. Therefore, the least positive multiple of 72 that has exactly 16 positive factors is:  \[ \boxed{216} \]

**wrong -> correct: test:2732** - gold `\frac{266664}{5}`, initial `NO_ANSWER (initial response truncated at 1,024)`, revised `53332.8`

> s: \[ 576 + 5760 + 57600 + 576000 + 5760000 \]  Adding these contributions: \[ 576 + 5760 = 6336 \] \[ 6336 + 57600 = 63936 \] \[ 63936 + 576000 = 639936 \] \[ 639936 + 5760000 = 6399936 \]  ### Final Step: Calculate the average  The average is the total sum divided by the number of permutations: \[ \text{Average} = \frac{6399936}{120} = 53332.8 \]  Thus, the final answer is: \[ \boxed{53332.8} \]

**correct -> wrong**: no such case among the 50 questions.
