# Task 2 summary

## Development metrics at stage endpoints (t196)

| update | cap loss | cap loss (no img) | G NLL corr | G NLL comp | G NLL no-img | G dNLL | V NLL corr | V NLL comp | V NLL no-img | V dNLL | G dev acc % | V dev acc % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.435 | 4.559 | 10.158 | 10.390 | 5.114 | 0.232 | 11.269 | 11.199 | 4.631 | -0.069 | 4.2 | 1.6 |
| 2250 | 2.305 | 4.559 | 3.194 | 4.193 | 5.114 | 0.999 | 3.136 | 3.925 | 4.631 | 0.789 | 20.8 | 21.3 |
| 3750 | 2.295 | 4.503 | 0.649 | 4.915 | 1.430 | 4.265 | 1.035 | 3.507 | 2.249 | 2.472 | 66.7 | 71.1 |
| 4500 | 2.314 | 4.297 | 0.684 | 3.527 | 1.253 | 2.842 | 0.626 | 2.768 | 1.595 | 2.142 | 66.7 | 74.2 |

## Development metrics at stage endpoints (t49)

| update | cap loss | cap loss (no img) | G NLL corr | G NLL comp | G NLL no-img | G dNLL | V NLL corr | V NLL comp | V NLL no-img | V dNLL | G dev acc % | V dev acc % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.646 | 4.559 | 4.920 | 5.579 | 5.114 | 0.659 | 4.961 | 4.857 | 4.631 | -0.104 | 4.2 | 8.3 |
| 2250 | 2.397 | 4.559 | 3.231 | 3.876 | 5.114 | 0.644 | 3.291 | 3.546 | 4.631 | 0.254 | 16.7 | 21.2 |
| 3750 | 2.383 | 4.492 | 0.663 | 4.427 | 1.854 | 3.764 | 1.781 | 3.093 | 2.377 | 1.313 | 75.0 | 49.7 |
| 4500 | 2.400 | 4.314 | 0.691 | 3.166 | 1.254 | 2.475 | 1.014 | 2.193 | 1.622 | 1.179 | 75.0 | 64.4 |

## Final test

### t196 - Annotation-derived grounding (64 pairs / 128 questions)

| condition | accuracy % [95% CI] | counting acc % | presence acc % | pair acc % [CI] | identical % | empty % | ref NLL [CI] | ms / answer |
|---|---|---|---|---|---|---|---|---|
| correct image | 70.3 [60.9, 78.9] | 50.0 | 90.6 | 54.7 [42.2, 67.2] | 15.6 | 0.0 | 0.717 [0.522, 0.934] | 44.2 |
| complementary swap | 12.5 [7.0, 18.8] | 15.6 | 9.4 | 3.1 [0.0, 7.8] | 15.6 | 0.0 | 4.081 [3.612, 4.539] | 41.6 |
| mismatched image | 37.5 [31.2, 43.8] | 25.0 | 50.0 | 3.1 [0.0, 7.8] | 78.1 | 0.0 | 2.594 [2.337, 2.860] | 43.3 |
| no image | 40.6 [35.9, 45.3] | 31.2 | 50.0 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 1.203 [1.083, 1.326] | 19.4 |

Correct-image advantages (t196, grounding; NLL advantage = NLL(other) - NLL(correct)):

| comparison | subset | accuracy adv. (pp) [95% CI] | pair-acc adv. (pp) [CI] | NLL adv. (nats) [CI] |
|---|---|---|---|---|
| correct vs complementary | all | 57.8 [43.8, 70.3] | 51.6 [37.5, 64.1] | 3.364 [2.748, 3.948] |
| correct vs complementary | counting | 34.4 [14.1, 54.7] | 21.9 [3.1, 40.6] | 2.351 [1.485, 3.239] |
| correct vs complementary | presence | 81.2 [68.8, 93.8] | 81.2 [68.8, 93.8] | 4.377 [3.689, 5.012] |
| correct vs mismatched | all | 32.8 [23.4, 42.2] | 51.6 [39.1, 64.1] | 1.877 [1.551, 2.207] |
| correct vs mismatched | counting | 25.0 [7.8, 42.2] | 25.0 [9.4, 43.8] | 1.383 [0.911, 1.867] |
| correct vs mismatched | presence | 40.6 [32.8, 48.4] | 78.1 [62.5, 90.6] | 2.371 [1.966, 2.742] |
| correct vs no_image | all | 29.7 [20.3, 38.3] | 54.7 [42.2, 67.2] | 0.486 [0.302, 0.652] |
| correct vs no_image | counting | 18.8 [4.7, 32.8] | 28.1 [12.5, 43.8] | 0.482 [0.196, 0.747] |
| correct vs no_image | presence | 40.6 [34.4, 46.9] | 81.2 [68.8, 93.8] | 0.490 [0.262, 0.650] |

Complementary swap outcomes (t196, grounding): follows partner answer 70.3%, keeps original answer 12.5%, other answer 17.2%; unchanged from the correct-image prediction 15.6%; equal to the partner's correct-image prediction (same input, sanity check) 100.0%.

### t196 - Natural VQAv2 (96 pairs / 192 questions)

| condition | accuracy % [95% CI] | number acc % | other acc % | yes/no acc % | pair acc % [CI] | identical % | empty % | ref NLL [CI] | ms / answer |
|---|---|---|---|---|---|---|---|---|---|
| correct image | 67.8 [61.1, 74.4] | 57.3 | 67.4 | 75.1 | 41.7 [31.2, 52.1] | 37.5 | 0.0 | 0.978 [0.768, 1.209] | 43.7 |
| complementary swap | 22.5 [16.7, 28.7] | 12.3 | 11.5 | 40.3 | 2.1 [0.0, 5.2] | 37.5 | 0.0 | 2.976 [2.620, 3.360] | 42.2 |
| mismatched image | 33.1 [27.4, 38.9] | 24.2 | 14.7 | 57.4 | 3.1 [0.0, 7.3] | 71.9 | 0.0 | 2.605 [2.247, 3.000] | 40.9 |
| no image | 39.1 [34.0, 44.1] | 40.0 | 21.7 | 55.8 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 1.828 [1.570, 2.119] | 20.8 |

Correct-image advantages (t196, vqa; NLL advantage = NLL(other) - NLL(correct)):

| comparison | subset | accuracy adv. (pp) [95% CI] | pair-acc adv. (pp) [CI] | NLL adv. (nats) [CI] |
|---|---|---|---|---|
| correct vs complementary | all | 45.3 [35.3, 55.3] | 39.6 [29.2, 50.0] | 1.998 [1.607, 2.410] |
| correct vs complementary | number | 45.0 [29.0, 61.7] | 29.2 [12.5, 45.8] | 2.395 [1.785, 3.027] |
| correct vs complementary | other | 55.8 [40.1, 71.7] | 50.0 [33.3, 66.7] | 2.702 [1.882, 3.585] |
| correct vs complementary | yes/no | 34.9 [17.1, 52.6] | 36.1 [16.7, 55.6] | 1.029 [0.702, 1.374] |
| correct vs mismatched | all | 34.7 [26.6, 43.0] | 38.5 [28.1, 49.0] | 1.627 [1.298, 1.984] |
| correct vs mismatched | number | 33.1 [16.9, 49.8] | 29.2 [12.5, 45.8] | 1.903 [1.276, 2.591] |
| correct vs mismatched | other | 52.6 [37.9, 66.9] | 50.0 [33.3, 66.7] | 2.169 [1.507, 2.893] |
| correct vs mismatched | yes/no | 17.8 [7.9, 27.5] | 33.3 [13.9, 52.8] | 0.901 [0.631, 1.182] |
| correct vs no_image | all | 28.7 [20.7, 36.9] | 41.7 [31.2, 52.1] | 0.850 [0.596, 1.115] |
| correct vs no_image | number | 17.3 [3.5, 29.8] | 29.2 [12.5, 45.8] | 0.561 [0.096, 0.937] |
| correct vs no_image | other | 45.7 [30.0, 61.7] | 50.0 [33.3, 66.7] | 1.626 [1.101, 2.173] |
| correct vs no_image | yes/no | 19.3 [10.8, 27.8] | 41.7 [25.0, 58.3] | 0.266 [0.111, 0.405] |

Complementary swap outcomes (t196, vqa): follows partner answer 63.5%, keeps original answer 18.8%, other answer 17.7%; unchanged from the correct-image prediction 37.5%; equal to the partner's correct-image prediction (same input, sanity check) 100.0%.

### t49 - Annotation-derived grounding (64 pairs / 128 questions)

| condition | accuracy % [95% CI] | counting acc % | presence acc % | pair acc % [CI] | identical % | empty % | ref NLL [CI] | ms / answer |
|---|---|---|---|---|---|---|---|---|
| correct image | 71.1 [62.5, 79.7] | 51.6 | 90.6 | 56.2 [43.8, 68.8] | 20.3 | 0.0 | 0.690 [0.510, 0.889] | 23.7 |
| complementary swap | 10.2 [5.5, 15.6] | 10.9 | 9.4 | 0.0 [0.0, 0.0] | 20.3 | 0.0 | 3.665 [3.256, 4.061] | 19.8 |
| mismatched image | 35.9 [29.7, 42.2] | 21.9 | 50.0 | 4.7 [0.0, 10.9] | 62.5 | 0.0 | 2.129 [1.922, 2.338] | 19.5 |
| no image | 39.1 [34.4, 43.8] | 28.1 | 50.0 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 1.212 [1.092, 1.337] | 19.6 |

Correct-image advantages (t49, grounding; NLL advantage = NLL(other) - NLL(correct)):

| comparison | subset | accuracy adv. (pp) [95% CI] | pair-acc adv. (pp) [CI] | NLL adv. (nats) [CI] |
|---|---|---|---|---|
| correct vs complementary | all | 60.9 [48.4, 72.7] | 56.2 [43.8, 68.8] | 2.975 [2.413, 3.499] |
| correct vs complementary | counting | 40.6 [23.4, 57.8] | 31.2 [15.6, 46.9] | 2.191 [1.444, 2.976] |
| correct vs complementary | presence | 81.2 [65.6, 93.8] | 81.2 [65.6, 93.8] | 3.760 [3.100, 4.381] |
| correct vs mismatched | all | 35.2 [26.6, 43.0] | 51.6 [37.5, 64.1] | 1.440 [1.154, 1.703] |
| correct vs mismatched | counting | 29.7 [14.1, 45.3] | 28.1 [12.5, 46.9] | 0.977 [0.653, 1.301] |
| correct vs mismatched | presence | 40.6 [32.8, 46.9] | 75.0 [59.4, 90.6] | 1.902 [1.508, 2.288] |
| correct vs no_image | all | 32.0 [24.2, 39.1] | 56.2 [43.8, 68.8] | 0.522 [0.345, 0.679] |
| correct vs no_image | counting | 23.4 [10.9, 35.9] | 31.2 [15.6, 46.9] | 0.541 [0.276, 0.799] |
| correct vs no_image | presence | 40.6 [32.8, 46.9] | 81.2 [65.6, 93.8] | 0.503 [0.290, 0.678] |

Complementary swap outcomes (t49, grounding): follows partner answer 71.1%, keeps original answer 10.2%, other answer 18.8%; unchanged from the correct-image prediction 20.3%; equal to the partner's correct-image prediction (same input, sanity check) 100.0%.

### t49 - Natural VQAv2 (96 pairs / 192 questions)

| condition | accuracy % [95% CI] | number acc % | other acc % | yes/no acc % | pair acc % [CI] | identical % | empty % | ref NLL [CI] | ms / answer |
|---|---|---|---|---|---|---|---|---|---|
| correct image | 59.6 [53.1, 66.4] | 49.0 | 63.2 | 63.2 | 29.2 [20.8, 38.5] | 47.9 | 0.0 | 1.085 [0.889, 1.307] | 23.8 |
| complementary swap | 28.6 [22.6, 34.4] | 14.4 | 15.7 | 51.0 | 2.1 [0.0, 5.2] | 47.9 | 0.0 | 2.242 [1.977, 2.520] | 23.9 |
| mismatched image | 33.0 [27.2, 39.1] | 29.0 | 14.3 | 54.4 | 4.2 [1.0, 8.3] | 65.6 | 0.0 | 2.287 [1.958, 2.634] | 22.1 |
| no image | 39.7 [34.8, 44.8] | 44.8 | 20.3 | 55.8 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 1.909 [1.642, 2.206] | 21.6 |

Correct-image advantages (t49, vqa; NLL advantage = NLL(other) - NLL(correct)):

| comparison | subset | accuracy adv. (pp) [95% CI] | pair-acc adv. (pp) [CI] | NLL adv. (nats) [CI] |
|---|---|---|---|---|
| correct vs complementary | all | 31.0 [21.2, 41.5] | 27.1 [17.7, 37.5] | 1.157 [0.875, 1.471] |
| correct vs complementary | number | 34.6 [12.9, 56.9] | 33.3 [16.7, 54.2] | 1.473 [0.855, 2.118] |
| correct vs complementary | other | 47.5 [30.6, 63.6] | 36.1 [19.4, 52.8] | 1.615 [1.075, 2.205] |
| correct vs complementary | yes/no | 12.2 [0.0, 25.3] | 13.9 [0.0, 27.8] | 0.488 [0.281, 0.707] |
| correct vs mismatched | all | 26.6 [18.6, 35.1] | 25.0 [15.6, 34.4] | 1.202 [0.888, 1.534] |
| correct vs mismatched | number | 20.0 [4.6, 36.0] | 29.2 [12.5, 45.8] | 1.250 [0.558, 2.000] |
| correct vs mismatched | other | 48.9 [36.0, 62.6] | 36.1 [22.2, 52.8] | 1.765 [1.187, 2.398] |
| correct vs mismatched | yes/no | 8.8 [-1.0, 18.9] | 11.1 [-2.8, 25.0] | 0.609 [0.365, 0.865] |
| correct vs no_image | all | 19.9 [12.1, 28.2] | 29.2 [20.8, 38.5] | 0.825 [0.574, 1.090] |
| correct vs no_image | number | 4.2 [-11.9, 20.2] | 33.3 [16.7, 54.2] | 0.675 [0.213, 1.163] |
| correct vs no_image | other | 42.9 [28.5, 57.5] | 38.9 [22.2, 55.6] | 1.549 [1.087, 2.071] |
| correct vs no_image | yes/no | 7.4 [1.4, 13.5] | 16.7 [5.6, 30.6] | 0.200 [0.081, 0.319] |

Complementary swap outcomes (t49, vqa): follows partner answer 55.7%, keeps original answer 25.0%, other answer 19.3%; unchanged from the correct-image prediction 47.9%; equal to the partner's correct-image prediction (same input, sanity check) 100.0%.

## Resources

| run | stage | trainable params | work | time | rate | peak GPU alloc (GB) | note |
|---|---|---|---|---|---|---|---|
| SigLIP feature caching (once, shared) | - | frozen encoder | 17050 images | 1.2 min | 4.3 ms/img | 0.69 | cache 4.78 GB |
| t196 | caption | 1,837,056 (conn 1,837,056 + LoRA 0) | 2250 upd | 0.77 h | 1.24 s/upd | 1.64 |  |
| t196 | instruct | 6,424,576 (conn 1,837,056 + LoRA 4,587,520) | 1500 upd | 0.61 h | 1.46 s/upd | 1.69 |  |
| t196 | vqa | 6,424,576 (conn 1,837,056 + LoRA 4,587,520) | 750 upd | 0.30 h | 1.45 s/upd | 1.69 |  |
| t196 | final-test generation | - | 1280 answers | 1.3 min (incl. NLL) | 37.0 ms/answer | 1.48 | batch 8, greedy, 16 tokens |
| t196 | dev evaluations | - | - | 3.0 min | - | - |  |
| t49 | caption | 1,837,056 (conn 1,837,056 + LoRA 0) | 2250 upd | 0.31 h | 0.50 s/upd | 1.46 |  |
| t49 | instruct | 6,424,576 (conn 1,837,056 + LoRA 4,587,520) | 1500 upd | 0.30 h | 0.73 s/upd | 1.49 |  |
| t49 | vqa | 6,424,576 (conn 1,837,056 + LoRA 4,587,520) | 750 upd | 0.15 h | 0.72 s/upd | 1.49 |  |
| t49 | final-test generation | - | 1280 answers | 0.7 min (incl. NLL) | 22.0 ms/answer | 1.26 | batch 8, greedy, 16 tokens |
| t49 | dev evaluations | - | - | 1.4 min | - | - |  |

## Optional visual-token compression (196 vs 49)

| metric | 196 tokens (t196) | 49 tokens (t49) |
|---|---|---|
| grounding accuracy % (correct image) | 70.3 [60.9, 78.9] | 71.1 [62.5, 79.7] |
| grounding accuracy % (complementary swap) | 12.5 [7.0, 18.8] | 10.2 [5.5, 15.6] |
| grounding accuracy % (no image) | 40.6 [35.9, 45.3] | 39.1 [34.4, 43.8] |
| grounding pair accuracy % (correct) | 54.7 [42.2, 67.2] | 56.2 [43.8, 68.8] |
| grounding accuracy adv. vs complementary (pp) | 57.8 [43.8, 70.3] | 60.9 [48.4, 72.7] |
| grounding NLL adv. vs complementary (nats) | 3.364 [2.748, 3.948] | 2.975 [2.413, 3.499] |
| vqa accuracy % (correct image) | 67.8 [61.1, 74.4] | 59.6 [53.1, 66.4] |
| vqa accuracy % (complementary swap) | 22.5 [16.7, 28.7] | 28.6 [22.6, 34.4] |
| vqa accuracy % (no image) | 39.1 [34.0, 44.1] | 39.7 [34.8, 44.8] |
| vqa pair accuracy % (correct) | 41.7 [31.2, 52.1] | 29.2 [20.8, 38.5] |
| vqa accuracy adv. vs complementary (pp) | 45.3 [35.3, 55.3] | 31.0 [21.2, 41.5] |
| vqa NLL adv. vs complementary (nats) | 1.998 [1.607, 2.410] | 1.157 [0.875, 1.471] |
| training time (h) | 1.69 | 0.76 |
| peak GPU alloc, stage 2 (GB) | 1.69 | 1.49 |
| generation ms / answer | 37.0 | 22.0 |

## Qualitative examples

For each suite, final-test pairs are sorted by pair ID and classified from the correct-image predictions of the main (196-token) model: (A) grounded - both questions exactly correct, so each complementary-swap prediction equals the partner's answer; (B) image-insensitive - identical predictions for both images; (C) image-sensitive but wrong - different predictions, at least one wrong. The first two pairs of each class are shown (fewer if a class is smaller).

#### Annotation-derived grounding

| class | question | image | reference (canonical) | correct | swap | mismatched | no image |
|---|---|---|---|---|---|---|---|
| grounded | How many bears can you see? | COCO_val2014_000000531334.jpg | 2 (2) | '2' | '1' | '0' | '1' |
| grounded | How many bears can you see? | COCO_val2014_000000573869.jpg | 1 (1) | '1' | '2' | '0' | '1' |
| grounded | How many giraffes can you see? | COCO_val2014_000000475150.jpg | 1 (1) | '1' | '4' | '0' | '1' |
| grounded | How many giraffes can you see? | COCO_val2014_000000543492.jpg | 4 (4) | '4' | '1' | '0' | '1' |
| image-insensitive | How many keyboards can you see? | COCO_val2014_000000385719.jpg | 4 (4) | '2' | '2' | '1' | '1' |
| image-insensitive | How many keyboards can you see? | COCO_val2014_000000570782.jpg | 3 (3) | '2' | '2' | '4' | '1' |
| image-insensitive | How many boats can you see? | COCO_val2014_000000175469.jpg | 3 (3) | '3' | '3' | '3' | '1' |
| image-insensitive | How many boats can you see? | COCO_val2014_000000455040.jpg | 4 (4) | '3' | '3' | '1' | '1' |
| image-sensitive, wrong | How many zebras can you see? | COCO_val2014_000000186991.jpg | 2 (2) | '2' | '3' | '0' | '1' |
| image-sensitive, wrong | How many zebras can you see? | COCO_val2014_000000218424.jpg | 4 (4) | '3' | '2' | '0' | '1' |
| image-sensitive, wrong | How many wine glasses can you see? | COCO_val2014_000000369986.jpg | 1 (1) | '2' | '3' | '1' | '1' |
| image-sensitive, wrong | How many wine glasses can you see? | COCO_val2014_000000401833.jpg | 4 (4) | '3' | '2' | '1' | '1' |

#### Natural VQAv2

| class | question | image | reference (canonical) | correct | swap | mismatched | no image |
|---|---|---|---|---|---|---|---|
| grounded | Are these giraffes in the wild? | COCO_val2014_000000101780.jpg | no (no) | 'no' | 'yes' | 'no' | 'yes' |
| grounded | Are these giraffes in the wild? | COCO_val2014_000000526645.jpg | yes (yes) | 'yes' | 'no' | 'no' | 'yes' |
| grounded | What sport are they playing? | COCO_val2014_000000126895.jpg | baseball (baseball) | 'baseball' | 'soccer' | 'soccer' | 'soccer' |
| grounded | What sport are they playing? | COCO_val2014_000000146723.jpg | soccer (soccer) | 'soccer' | 'baseball' | 'soccer' | 'soccer' |
| image-insensitive | What color is the sky? | COCO_val2014_000000001029.jpg | gray (gray) | 'blue' | 'blue' | 'blue' | 'blue' |
| image-insensitive | What color is the sky? | COCO_val2014_000000202573.jpg | blue (blue) | 'blue' | 'blue' | 'blue' | 'blue' |
| image-insensitive | Is this real or a toy? | COCO_val2014_000000014473.jpg | toy (toy) | 'toy' | 'toy' | 'toy' | 'toy' |
| image-insensitive | Is this real or a toy? | COCO_val2014_000000282940.jpg | real (real) | 'toy' | 'toy' | 'toy' | 'toy' |
| image-sensitive, wrong | What is he doing? | COCO_val2014_000000010023.jpg | skateboarding (skateboarding) | 'skating' | 'playing tennis' | 'showing horses' | 'playing' |
| image-sensitive, wrong | What is he doing? | COCO_val2014_000000540193.jpg | tennis (tennis) | 'playing tennis' | 'skating' | 'riding' | 'playing' |
| image-sensitive, wrong | How many cups are there? | COCO_val2014_000000172079.jpg | 2 (2) | '4' | '1' | '1' | '1' |
| image-sensitive, wrong | How many cups are there? | COCO_val2014_000000536990.jpg | 0 (0) | '1' | '4' | '1' | '1' |

## Connector calibration

- t196: connector output multiplied by a fixed scalar s = mean_token ||text embedding||_2 / mean_token ||initial connector output||_2 (training prompts+targets / training images); text 0.8329, visual before 15.9054 -> scale 0.0524 (after 0.8329)
- t49: connector output multiplied by a fixed scalar s = mean_token ||text embedding||_2 / mean_token ||initial connector output||_2 (training prompts+targets / training images); text 0.8329, visual before 11.4870 -> scale 0.0725 (after 0.8329)
