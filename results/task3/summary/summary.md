# Task 3 - summary (all numbers from results/task3/; held-out = 800 evaluation questions per dataset)

Scorer: `pa3-task3-scorer-1`, math-verify 0.9.0, latex2sympy2_extended 1.11.0 (fixed 2026-11-15 14:07:09).

## MATH (Qwen2.5-3B): accuracy (%) by candidate budget N

| method | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| FirstAns | 64.3 | 64.3 | 64.3 | 64.3 | 64.3 | 64.3 |
| Mean log-prob (max) | 64.3 | 65.8 | 67.2 | 67.5 | 66.4 | 64.0 |
| Mean log-prob + RWC | 64.3 | 65.8 | 68.5 | 72.2 | 74.0 | 74.9 |
| Self-consistency | 64.3 | 64.3 | 69.2 | 71.8 | 73.1 | 73.8 |
| Self-certainty (max) | 64.3 | 66.8 | 68.8 | 70.1 | 70.7 | 70.6 |
| Self-certainty + RWC | 64.3 | 66.8 | 69.7 | 72.9 | 74.7 | 75.5 |
| Shortest (length control) | 64.3 | 65.3 | 65.2 | 64.6 | 63.2 | 62.4 |
| ORACLE coverage | 64.3 | 72.5 | 79.1 | 84.3 | 88.5 | 91.4 |
| distinct groups | 1.0 | 1.4 | 1.9 | 2.7 | 4.0 | 5.9 |
| mean tokens (all N) | 669.8 | 1339.9 | 2679.7 | 5344.1 | 10677.7 | 21347.0 |

### MATH (Qwen2.5-3B): paired bootstrap at N = 32 (pp, 95% CI, 1,000 question resamples)

| comparison | mean_a | mean_b | diff | ci_lo | ci_hi |
|---|---|---|---|---|---|
| FirstAns - Self-consistency | 64.35 | 73.79 | -9.44 | -11.01 | -7.97 |
| Mean log-prob (max) - Self-consistency | 64.00 | 73.79 | -9.79 | -12.39 | -7.40 |
| Mean log-prob + RWC - Self-consistency | 74.88 | 73.79 | 1.09 | 0.06 | 2.14 |
| Self-certainty (max) - Self-consistency | 70.62 | 73.79 | -3.16 | -5.51 | -0.82 |
| Self-certainty + RWC - Self-consistency | 75.55 | 73.79 | 1.76 | 0.69 | 2.91 |
| Shortest (length control) - Self-consistency | 62.38 | 73.79 | -11.41 | -14.13 | -8.99 |
| Mean log-prob + RWC - Mean log-prob (max) | 74.88 | 64.00 | 10.88 | 8.50 | 13.38 |
| Self-certainty + RWC - Self-certainty (max) | 75.55 | 70.62 | 4.92 | 2.80 | 7.17 |

### MATH (Qwen2.5-3B): failure decomposition at N = 32 (%)

| method | accuracy | missed_given_covered | no_correct_candidate |
|---|---|---|---|
| FirstAns | 64.3 | 27.0 | 8.6 |
| Mean log-prob (max) | 64.0 | 27.4 | 8.6 |
| Mean log-prob + RWC | 74.9 | 16.5 | 8.6 |
| Self-consistency | 73.8 | 17.6 | 8.6 |
| Self-certainty (max) | 70.6 | 20.8 | 8.6 |
| Self-certainty + RWC | 75.5 | 15.8 | 8.6 |
| Shortest (length control) | 62.4 | 29.0 | 8.6 |

## MathVista (Qwen2.5-VL-3B): accuracy (%) by candidate budget N

| method | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| FirstAns | 54.2 | 54.2 | 54.2 | 54.2 | 54.2 | 54.2 |
| Mean log-prob (max) | 54.2 | 54.6 | 54.6 | 53.5 | 52.1 | 49.5 |
| Mean log-prob + RWC | 54.2 | 54.6 | 56.9 | 60.2 | 62.4 | 62.5 |
| Self-consistency | 54.2 | 54.2 | 59.1 | 61.5 | 63.4 | 64.0 |
| Self-certainty (max) | 54.2 | 55.4 | 56.4 | 56.3 | 56.0 | 56.0 |
| Self-certainty + RWC | 54.2 | 55.4 | 58.2 | 61.0 | 63.5 | 64.7 |
| Shortest (length control) | 54.2 | 55.1 | 55.2 | 55.2 | 55.2 | 54.4 |
| Tokenwise fusion (max) | 54.2 | 55.8 | 56.2 | 57.0 | 56.9 | 57.1 |
| Tokenwise fusion + RWC | 54.2 | 55.8 | 58.6 | 61.4 | 63.7 | 64.5 |
| VisConf (max) | 54.2 | 55.8 | 56.7 | 57.1 | 57.8 | 58.4 |
| VisConf + RWC | 54.2 | 55.8 | 58.8 | 61.7 | 64.0 | 65.1 |
| Vision-only (max) | 54.2 | 55.2 | 54.6 | 54.8 | 54.6 | 54.5 |
| Vision-only + RWC | 54.2 | 55.2 | 57.7 | 60.7 | 63.0 | 64.6 |
| ORACLE coverage | 54.2 | 66.9 | 76.6 | 83.6 | 87.7 | 90.2 |
| distinct groups | 1.0 | 1.5 | 2.0 | 2.7 | 3.6 | 4.8 |
| mean tokens (all N) | 281.8 | 560.3 | 1113.9 | 2225.9 | 4442.6 | 8897.8 |

### MathVista (Qwen2.5-VL-3B): paired bootstrap at N = 32 (pp, 95% CI, 1,000 question resamples)

| comparison | mean_a | mean_b | diff | ci_lo | ci_hi |
|---|---|---|---|---|---|
| FirstAns - Self-consistency | 54.21 | 64.00 | -9.79 | -11.54 | -7.92 |
| Mean log-prob (max) - Self-consistency | 49.50 | 64.00 | -14.50 | -17.44 | -11.41 |
| Mean log-prob + RWC - Self-consistency | 62.50 | 64.00 | -1.50 | -2.88 | -0.09 |
| Self-certainty (max) - Self-consistency | 56.00 | 64.00 | -8.00 | -10.94 | -4.99 |
| Self-certainty + RWC - Self-consistency | 64.70 | 64.00 | 0.70 | -0.68 | 2.26 |
| Shortest (length control) - Self-consistency | 54.42 | 64.00 | -9.57 | -12.55 | -6.61 |
| Tokenwise fusion (max) - Self-consistency | 57.12 | 64.00 | -6.88 | -9.64 | -3.91 |
| Tokenwise fusion + RWC - Self-consistency | 64.50 | 64.00 | 0.50 | -1.00 | 1.94 |
| VisConf (max) - Self-consistency | 58.38 | 64.00 | -5.62 | -8.43 | -2.79 |
| VisConf + RWC - Self-consistency | 65.12 | 64.00 | 1.12 | -0.30 | 2.60 |
| Vision-only (max) - Self-consistency | 54.50 | 64.00 | -9.50 | -12.71 | -6.31 |
| Vision-only + RWC - Self-consistency | 64.62 | 64.00 | 0.62 | -1.00 | 2.19 |
| Mean log-prob + RWC - Mean log-prob (max) | 62.50 | 49.50 | 13.00 | 10.00 | 16.00 |
| Self-certainty + RWC - Self-certainty (max) | 64.70 | 56.00 | 8.70 | 5.77 | 11.57 |
| Vision-only + RWC - Vision-only (max) | 64.62 | 54.50 | 10.12 | 7.00 | 13.00 |
| VisConf + RWC - VisConf (max) | 65.12 | 58.38 | 6.75 | 4.00 | 9.62 |
| VisConf + RWC - Self-certainty + RWC | 65.12 | 64.70 | 0.42 | -0.85 | 1.88 |
| VisConf (max) - Self-certainty (max) | 58.38 | 56.00 | 2.38 | -0.50 | 5.62 |
| VisConf + RWC - Vision-only + RWC | 65.12 | 64.62 | 0.50 | -0.75 | 1.75 |
| Tokenwise fusion (max) - VisConf (max) | 57.12 | 58.38 | -1.25 | -3.25 | 0.62 |
| Tokenwise fusion + RWC - VisConf + RWC | 64.50 | 65.12 | -0.62 | -1.38 | 0.00 |

### MathVista (Qwen2.5-VL-3B): failure decomposition at N = 32 (%)

| method | accuracy | missed_given_covered | no_correct_candidate |
|---|---|---|---|
| FirstAns | 54.2 | 36.0 | 9.8 |
| Mean log-prob (max) | 49.5 | 40.8 | 9.8 |
| Mean log-prob + RWC | 62.5 | 27.8 | 9.8 |
| Self-consistency | 64.0 | 26.2 | 9.8 |
| Self-certainty (max) | 56.0 | 34.2 | 9.8 |
| Self-certainty + RWC | 64.7 | 25.5 | 9.8 |
| Shortest (length control) | 54.4 | 35.8 | 9.8 |
| Tokenwise fusion (max) | 57.1 | 33.1 | 9.8 |
| Tokenwise fusion + RWC | 64.5 | 25.8 | 9.8 |
| VisConf (max) | 58.4 | 31.9 | 9.8 |
| VisConf + RWC | 65.1 | 25.1 | 9.8 |
| Vision-only (max) | 54.5 | 35.7 | 9.8 |
| Vision-only + RWC | 64.6 | 25.6 | 9.8 |

## Answer novelty of added candidates (held-out)

| dataset | added_candidates | frac_new_answer_group | frac_repeating_existing | frac_first_correct | frac_no_answer |
|---|---|---|---|---|---|
| math | 2-2 | 0.371 | 0.629 | 0.081 | 0.018 |
| math | 3-4 | 0.278 | 0.722 | 0.033 | 0.018 |
| math | 5-8 | 0.203 | 0.797 | 0.013 | 0.017 |
| math | 9-16 | 0.158 | 0.842 | 0.005 | 0.017 |
| math | 17-32 | 0.119 | 0.881 | 0.002 | 0.017 |
| mathvista | 2-2 | 0.454 | 0.546 | 0.127 | 0.097 |
| mathvista | 3-4 | 0.283 | 0.717 | 0.048 | 0.094 |
| mathvista | 5-8 | 0.179 | 0.821 | 0.018 | 0.099 |
| mathvista | 9-16 | 0.114 | 0.886 | 0.005 | 0.097 |
| mathvista | 17-32 | 0.070 | 0.930 | 0.002 | 0.098 |

## MATH (Qwen2.5-3B): signal associations at N = 32 (held-out)

| score | mean_spearman_with_length | mean_pearson_with_correct_mixed | top_minus_bottom_pp_all | top_minus_bottom_pp_mixed | ci_lo_mixed | ci_hi_mixed |
|---|---|---|---|---|---|---|
| K | -0.161 | 0.220 | 11.984 | 20.442 | 17.804 | 23.241 |
| L | 0.225 | 0.144 | 7.828 | 13.353 | 10.844 | 15.886 |
| n_tokens | nan | -0.100 | -4.594 | -7.836 | -10.634 | -4.823 |
| negT | nan | 0.100 | 4.562 | 7.783 | 4.769 | 10.608 |

## MathVista (Qwen2.5-VL-3B): signal associations at N = 32 (held-out)

| score | mean_spearman_with_length | mean_pearson_with_correct_mixed | top_minus_bottom_pp_all | top_minus_bottom_pp_mixed | ci_lo_mixed | ci_hi_mixed |
|---|---|---|---|---|---|---|
| C32 | 0.342 | 0.057 | 3.984 | 4.996 | 2.919 | 6.975 |
| K | 0.349 | 0.056 | 3.672 | 4.604 | 2.468 | 6.622 |
| L | 0.329 | 0.034 | 2.016 | 2.527 | 0.372 | 4.585 |
| V | -0.686 | 0.044 | 3.469 | 4.350 | 2.095 | 6.721 |
| Z32 | -0.279 | 0.096 | 7.234 | 9.071 | 6.719 | 11.462 |
| Ztok32 | -0.325 | 0.089 | 7.062 | 8.856 | 6.563 | 11.207 |
| n_tokens | nan | -0.078 | -3.578 | -4.487 | -6.995 | -2.155 |
| negT | nan | 0.078 | 3.688 | 4.624 | 2.311 | 7.093 |

## MathVista: within-rollout corr(c_t, v_t)

| split | rollouts | n_rollouts | n_questions | mean_of_question_means | median_of_question_means | q25 | q75 | frac_rollouts_positive | mean_corr_raw_kl_v |
|---|---|---|---|---|---|---|---|---|---|
| eval | False | 11570 | 716 | -0.199 | -0.216 | -0.306 | -0.094 | 0.114 | -0.199 |
| eval | True | 14002 | 722 | -0.209 | -0.204 | -0.308 | -0.090 | 0.175 | -0.207 |
| eval | all | 25572 | 800 | -0.211 | -0.212 | -0.306 | -0.094 | 0.147 | -0.210 |

Fusion identity check: {'identity': "Z_token - Z = Cov_t(c, v) (population covariance over a rollout's tokens)", 'max_abs_error': 1.2607970223399434e-14, 'n_rollouts': 32000}

## Adaptive sampling (thresholds selected on dev, evaluated on held-out)

| dataset | family | policy | tau | gamma | dev_accuracy | dev_cost | dev_fixed32_sc | eval_accuracy | eval_mean_cost | eval_mean_rollouts | eval_fixed32_sc | eval_fixed32_cost | diff_vs_fixed32_pp | ci_lo | ci_hi | stop@4 | stop@8 | stop@16 | stop@32 | incorrect_early_stops_frac | incorrect_early_recoverable_by_fixed32 | correct_early_where_fixed32_wrong |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| math | agreement | agreement_tau0.625 | 0.625 | nan | 82.600 | 7088.955 | 82.500 | 73.638 | 10139.981 | 11.824 | 73.787 | 21347.018 | -0.150 | -0.500 | 0.163 | 0.663 | 0.059 | 0.013 | 0.265 | 0.083 | 51 | 39 |
| math | quality | quality_tau0.5_gamma0.0 | 0.500 | 0.000 | 82.150 | 5479.528 | 82.500 | 73.175 | 7663.055 | 9.008 | 73.787 | 21347.018 | -0.613 | -1.200 | -0.062 | 0.764 | 0.058 | 0.013 | 0.165 | 0.144 | 131 | 82 |
| mathvista | agreement | agreement_tau0.625 | 0.625 | nan | 63.700 | 5209.820 | 64.250 | 63.962 | 4541.589 | 13.380 | 64.000 | 8897.757 | -0.037 | -0.675 | 0.613 | 0.585 | 0.081 | 0.019 | 0.315 | 0.140 | 142 | 139 |
| mathvista | quality | quality_tau0.625_gamma0.0 | 0.625 | 0.000 | 63.700 | 5283.434 | 64.250 | 64.088 | 4604.004 | 13.613 | 64.000 | 8897.757 | 0.087 | -0.500 | 0.700 | 0.570 | 0.086 | 0.022 | 0.321 | 0.136 | 127 | 134 |

## Live MATH

_Pending: run kaggle/31_task3_full.ipynb, then `python -m pa3.task3.aggregate`._
