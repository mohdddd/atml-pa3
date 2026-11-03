# Task 2 summary

## Development metrics at stage endpoints (t196)

| update | cap loss | cap loss (no img) | G NLL corr | G NLL comp | G NLL no-img | G dNLL | V NLL corr | V NLL comp | V NLL no-img | V dNLL | G dev acc % | V dev acc % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.434 | 4.559 | 10.153 | 10.384 | 5.114 | 0.231 | 11.255 | 11.188 | 4.631 | -0.067 | 4.2 | 1.6 |
| 6 | 5.159 | 4.559 | 7.759 | 8.301 | 5.114 | 0.542 | 9.117 | 8.934 | 4.631 | -0.182 | 2.1 | 3.1 |
| 10 | 5.085 | 4.548 | 5.865 | 6.390 | 4.788 | 0.524 | 6.817 | 6.767 | 4.428 | -0.050 | 10.4 | 3.6 |
| 14 | 4.927 | 4.535 | 4.144 | 4.530 | 4.015 | 0.385 | 4.966 | 4.852 | 4.036 | -0.114 | 10.4 | 13.0 |

## Development metrics at stage endpoints (t49)

| update | cap loss | cap loss (no img) | G NLL corr | G NLL comp | G NLL no-img | G dNLL | V NLL corr | V NLL comp | V NLL no-img | V dNLL | G dev acc % | V dev acc % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.645 | 4.559 | 4.916 | 5.572 | 5.114 | 0.656 | 4.959 | 4.856 | 4.631 | -0.103 | 4.2 | 8.3 |
| 6 | 4.500 | 4.559 | 5.505 | 5.978 | 5.114 | 0.473 | 5.720 | 5.586 | 4.631 | -0.135 | 6.2 | 9.2 |
| 10 | 4.423 | 4.526 | 3.715 | 4.146 | 3.616 | 0.432 | 4.127 | 3.857 | 3.724 | -0.270 | 16.7 | 14.4 |
| 14 | 4.442 | 4.519 | 2.959 | 3.418 | 2.870 | 0.459 | 3.386 | 3.421 | 3.300 | 0.035 | 29.2 | 23.1 |

## Final test

### t196 - Annotation-derived grounding (64 pairs / 128 questions)

| condition | accuracy % [95% CI] | counting acc % | presence acc % | pair acc % [CI] | identical % | empty % | ref NLL [CI] | ms / answer |
|---|---|---|---|---|---|---|---|---|
| correct image | 15.6 [10.2, 21.1] | 21.9 | 9.4 | 0.0 [0.0, 0.0] | 60.9 | 1.6 | 4.576 [4.071, 5.087] | 56.2 |
| complementary swap | 11.7 [7.0, 17.2] | 17.2 | 6.2 | 0.0 [0.0, 0.0] | 60.9 | 1.6 | 4.604 [4.138, 5.089] | 54.3 |
| mismatched image | 17.2 [11.7, 23.4] | 21.9 | 12.5 | 0.0 [0.0, 0.0] | 68.8 | 0.8 | 4.377 [3.860, 4.926] | 53.3 |
| no image | 5.5 [2.3, 9.4] | 4.7 | 6.2 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 3.888 [3.657, 4.119] | 26.1 |

Correct-image advantages (t196, grounding; NLL advantage = NLL(other) - NLL(correct)):

| comparison | subset | accuracy adv. (pp) [95% CI] | pair-acc adv. (pp) [CI] | NLL adv. (nats) [CI] |
|---|---|---|---|---|
| correct vs complementary | all | 3.9 [-2.3, 9.4] | 0.0 [0.0, 0.0] | 0.029 [-0.345, 0.371] |
| correct vs complementary | counting | 4.7 [-4.7, 14.1] | 0.0 [0.0, 0.0] | 0.177 [-0.065, 0.410] |
| correct vs complementary | presence | 3.1 [-4.7, 10.9] | 0.0 [0.0, 0.0] | -0.120 [-0.823, 0.571] |
| correct vs mismatched | all | -1.6 [-7.8, 4.7] | 0.0 [0.0, 0.0] | -0.199 [-0.810, 0.407] |
| correct vs mismatched | counting | 0.0 [-9.4, 9.4] | 0.0 [0.0, 0.0] | -0.170 [-0.802, 0.437] |
| correct vs mismatched | presence | -3.1 [-12.5, 6.2] | 0.0 [0.0, 0.0] | -0.228 [-1.264, 0.791] |
| correct vs no_image | all | 10.2 [3.9, 16.4] | 0.0 [0.0, 0.0] | -0.687 [-1.266, -0.158] |
| correct vs no_image | counting | 17.2 [7.8, 26.6] | 0.0 [0.0, 0.0] | 0.479 [-0.033, 0.957] |
| correct vs no_image | presence | 3.1 [-4.7, 12.5] | 0.0 [0.0, 0.0] | -1.853 [-2.702, -1.079] |

Complementary swap outcomes (t196, grounding): follows partner answer 15.6%, keeps original answer 11.7%, other answer 72.7%; unchanged from the correct-image prediction 60.9%; equal to the partner's correct-image prediction (same input, sanity check) 100.0%.

### t196 - Natural VQAv2 (96 pairs / 192 questions)

| condition | accuracy % [95% CI] | number acc % | other acc % | yes/no acc % | pair acc % [CI] | identical % | empty % | ref NLL [CI] | ms / answer |
|---|---|---|---|---|---|---|---|---|---|
| correct image | 18.1 [13.2, 23.4] | 35.2 | 11.0 | 13.8 | 1.0 [0.0, 3.1] | 50.0 | 0.5 | 4.392 [3.954, 4.846] | 65.9 |
| complementary swap | 18.2 [13.1, 23.6] | 29.6 | 8.2 | 20.7 | 2.1 [0.0, 5.2] | 50.0 | 0.5 | 4.491 [4.077, 4.930] | 63.7 |
| mismatched image | 16.7 [12.2, 21.4] | 28.3 | 4.6 | 21.1 | 0.0 [0.0, 0.0] | 46.9 | 0.0 | 4.392 [3.968, 4.832] | 59.9 |
| no image | 12.4 [8.3, 16.8] | 19.4 | 16.1 | 4.2 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 4.097 [3.734, 4.460] | 29.2 |

Correct-image advantages (t196, vqa; NLL advantage = NLL(other) - NLL(correct)):

| comparison | subset | accuracy adv. (pp) [95% CI] | pair-acc adv. (pp) [CI] | NLL adv. (nats) [CI] |
|---|---|---|---|---|
| correct vs complementary | all | -0.2 [-5.4, 5.2] | -1.0 [-4.2, 2.1] | 0.099 [-0.102, 0.280] |
| correct vs complementary | number | 5.6 [-6.9, 16.0] | -4.2 [-12.5, 0.0] | 0.096 [-0.277, 0.460] |
| correct vs complementary | other | 2.8 [-5.6, 11.1] | 0.0 [-8.3, 8.3] | 0.158 [-0.116, 0.415] |
| correct vs complementary | yes/no | -6.9 [-15.3, 1.0] | 0.0 [0.0, 0.0] | 0.041 [-0.337, 0.405] |
| correct vs mismatched | all | 1.4 [-4.5, 7.3] | 1.0 [0.0, 3.1] | -0.000 [-0.360, 0.360] |
| correct vs mismatched | number | 6.9 [-3.5, 17.9] | 0.0 [0.0, 0.0] | 0.399 [-0.201, 1.233] |
| correct vs mismatched | other | 6.4 [-0.6, 14.6] | 2.8 [0.0, 8.3] | -0.183 [-0.748, 0.320] |
| correct vs mismatched | yes/no | -7.4 [-18.3, 4.2] | 0.0 [0.0, 0.0] | -0.084 [-0.661, 0.509] |
| correct vs no_image | all | 5.6 [0.1, 11.3] | 1.0 [0.0, 3.1] | -0.295 [-0.682, 0.103] |
| correct vs no_image | number | 15.8 [4.8, 26.9] | 0.0 [0.0, 0.0] | 0.542 [0.032, 1.162] |
| correct vs no_image | other | -5.1 [-12.9, 2.5] | 2.8 [0.0, 8.3] | -1.669 [-2.198, -1.194] |
| correct vs no_image | yes/no | 9.6 [0.4, 18.8] | 0.0 [0.0, 0.0] | 0.520 [-0.096, 1.136] |

Complementary swap outcomes (t196, vqa): follows partner answer 16.7%, keeps original answer 16.7%, other answer 66.7%; unchanged from the correct-image prediction 50.0%; equal to the partner's correct-image prediction (same input, sanity check) 100.0%.

### t49 - Annotation-derived grounding (64 pairs / 128 questions)

| condition | accuracy % [95% CI] | counting acc % | presence acc % | pair acc % [CI] | identical % | empty % | ref NLL [CI] | ms / answer |
|---|---|---|---|---|---|---|---|---|
| correct image | 26.6 [20.3, 32.8] | 20.3 | 32.8 | 1.6 [0.0, 4.7] | 64.1 | 0.0 | 2.856 [2.581, 3.141] | 30.5 |
| complementary swap | 26.6 [20.3, 32.8] | 23.4 | 29.7 | 1.6 [0.0, 4.7] | 64.1 | 0.0 | 2.940 [2.694, 3.220] | 26.6 |
| mismatched image | 25.0 [18.8, 31.2] | 18.8 | 31.2 | 0.0 [0.0, 0.0] | 62.5 | 0.0 | 2.878 [2.630, 3.148] | 24.9 |
| no image | 21.1 [14.8, 27.3] | 20.3 | 21.9 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 2.720 [2.545, 2.901] | 25.1 |

Correct-image advantages (t49, grounding; NLL advantage = NLL(other) - NLL(correct)):

| comparison | subset | accuracy adv. (pp) [95% CI] | pair-acc adv. (pp) [CI] | NLL adv. (nats) [CI] |
|---|---|---|---|---|
| correct vs complementary | all | 0.0 [-7.0, 7.0] | 0.0 [-4.7, 4.7] | 0.084 [-0.136, 0.332] |
| correct vs complementary | counting | -3.1 [-12.5, 4.7] | -3.1 [-9.4, 0.0] | -0.061 [-0.216, 0.095] |
| correct vs complementary | presence | 3.1 [-7.8, 14.1] | 3.1 [0.0, 9.4] | 0.230 [-0.209, 0.685] |
| correct vs mismatched | all | 1.6 [-4.7, 8.6] | 1.6 [0.0, 4.7] | 0.022 [-0.286, 0.347] |
| correct vs mismatched | counting | 1.6 [-7.8, 10.9] | 0.0 [0.0, 0.0] | -0.304 [-0.756, 0.061] |
| correct vs mismatched | presence | 1.6 [-7.8, 10.9] | 3.1 [0.0, 9.4] | 0.348 [-0.106, 0.821] |
| correct vs no_image | all | 5.5 [-1.6, 12.5] | 1.6 [0.0, 4.7] | -0.135 [-0.436, 0.161] |
| correct vs no_image | counting | 0.0 [-9.4, 9.4] | 0.0 [0.0, 0.0] | 0.393 [-0.100, 0.787] |
| correct vs no_image | presence | 10.9 [0.0, 21.9] | 3.1 [0.0, 9.4] | -0.664 [-1.007, -0.339] |

Complementary swap outcomes (t49, grounding): follows partner answer 26.6%, keeps original answer 26.6%, other answer 46.9%; unchanged from the correct-image prediction 64.1%; equal to the partner's correct-image prediction (same input, sanity check) 100.0%.

### t49 - Natural VQAv2 (96 pairs / 192 questions)

| condition | accuracy % [95% CI] | number acc % | other acc % | yes/no acc % | pair acc % [CI] | identical % | empty % | ref NLL [CI] | ms / answer |
|---|---|---|---|---|---|---|---|---|---|
| correct image | 22.3 [16.5, 28.8] | 30.4 | 16.1 | 23.1 | 5.2 [1.0, 10.4] | 60.4 | 1.0 | 3.124 [2.801, 3.488] | 36.9 |
| complementary swap | 21.6 [16.3, 27.1] | 35.2 | 14.7 | 19.3 | 3.1 [0.0, 7.3] | 60.4 | 1.0 | 3.139 [2.790, 3.508] | 36.1 |
| mismatched image | 24.3 [18.8, 30.1] | 40.0 | 18.5 | 19.7 | 3.1 [0.0, 7.3] | 62.5 | 0.5 | 3.192 [2.851, 3.548] | 33.4 |
| no image | 20.9 [15.7, 26.2] | 31.0 | 17.5 | 17.5 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 3.464 [3.159, 3.785] | 33.2 |

Correct-image advantages (t49, vqa; NLL advantage = NLL(other) - NLL(correct)):

| comparison | subset | accuracy adv. (pp) [95% CI] | pair-acc adv. (pp) [CI] | NLL adv. (nats) [CI] |
|---|---|---|---|---|
| correct vs complementary | all | 0.7 [-5.8, 7.8] | 2.1 [-4.2, 8.3] | 0.015 [-0.126, 0.155] |
| correct vs complementary | number | -4.8 [-20.8, 11.9] | 0.0 [-16.7, 16.7] | -0.094 [-0.378, 0.190] |
| correct vs complementary | other | 1.4 [-5.6, 8.3] | 2.8 [0.0, 8.3] | 0.016 [-0.163, 0.193] |
| correct vs complementary | yes/no | 3.8 [-8.2, 16.2] | 2.8 [-5.6, 11.1] | 0.088 [-0.188, 0.347] |
| correct vs mismatched | all | -2.0 [-7.2, 3.5] | 2.1 [-3.1, 7.3] | 0.068 [-0.105, 0.256] |
| correct vs mismatched | number | -9.6 [-25.6, 5.8] | 0.0 [-16.7, 16.7] | 0.237 [-0.153, 0.694] |
| correct vs mismatched | other | -2.4 [-6.9, 0.8] | 0.0 [0.0, 0.0] | -0.106 [-0.358, 0.152] |
| correct vs mismatched | yes/no | 3.3 [-5.0, 12.2] | 5.6 [0.0, 13.9] | 0.129 [-0.132, 0.416] |
| correct vs no_image | all | 1.4 [-4.4, 7.2] | 5.2 [1.0, 10.4] | 0.340 [0.087, 0.594] |
| correct vs no_image | number | -0.6 [-15.4, 13.3] | 8.3 [0.0, 20.8] | 0.685 [0.218, 1.206] |
| correct vs no_image | other | -1.4 [-6.9, 4.2] | 2.8 [0.0, 8.3] | -0.501 [-0.815, -0.207] |
| correct vs no_image | yes/no | 5.6 [-4.2, 15.8] | 5.6 [0.0, 13.9] | 0.953 [0.609, 1.282] |

Complementary swap outcomes (t49, vqa): follows partner answer 21.4%, keeps original answer 20.3%, other answer 58.3%; unchanged from the correct-image prediction 60.4%; equal to the partner's correct-image prediction (same input, sanity check) 100.0%.

## Resources

| run | stage | trainable params | work | time | rate | peak GPU alloc (GB) | note |
|---|---|---|---|---|---|---|---|
| SigLIP feature caching (once, shared) | - | frozen encoder | 720 images | 0.2 min | 12.9 ms/img | 0.69 | cache 0.20 GB |
| t196 | caption | 1,837,056 (conn 1,837,056 + LoRA 0) | 6 upd | 0.00 h | 1.16 s/upd | 1.63 |  |
| t196 | instruct | 6,424,576 (conn 1,837,056 + LoRA 4,587,520) | 4 upd | 0.00 h | 1.39 s/upd | 1.65 |  |
| t196 | vqa | 6,424,576 (conn 1,837,056 + LoRA 4,587,520) | 4 upd | 0.00 h | 1.42 s/upd | 1.69 |  |
| t196 | final-test generation | - | 1280 answers | 1.6 min (incl. NLL) | 51.8 ms/answer | 1.48 | batch 8, greedy, 16 tokens |
| t196 | dev evaluations | - | - | 2.1 min | - | - |  |
| t49 | caption | 1,837,056 (conn 1,837,056 + LoRA 0) | 6 upd | 0.00 h | 0.54 s/upd | 1.44 |  |
| t49 | instruct | 6,424,576 (conn 1,837,056 + LoRA 4,587,520) | 4 upd | 0.00 h | 0.73 s/upd | 1.49 |  |
| t49 | vqa | 6,424,576 (conn 1,837,056 + LoRA 4,587,520) | 4 upd | 0.00 h | 0.72 s/upd | 1.49 |  |
| t49 | final-test generation | - | 1280 answers | 0.9 min (incl. NLL) | 31.6 ms/answer | 1.26 | batch 8, greedy, 16 tokens |
| t49 | dev evaluations | - | - | 0.9 min | - | - |  |

## Optional visual-token compression (196 vs 49)

| metric | 196 tokens (t196) | 49 tokens (t49) |
|---|---|---|
| grounding accuracy % (correct image) | 15.6 [10.2, 21.1] | 26.6 [20.3, 32.8] |
| grounding accuracy % (complementary swap) | 11.7 [7.0, 17.2] | 26.6 [20.3, 32.8] |
| grounding accuracy % (no image) | 5.5 [2.3, 9.4] | 21.1 [14.8, 27.3] |
| grounding pair accuracy % (correct) | 0.0 [0.0, 0.0] | 1.6 [0.0, 4.7] |
| grounding accuracy adv. vs complementary (pp) | 3.9 [-2.3, 9.4] | 0.0 [-7.0, 7.0] |
| grounding NLL adv. vs complementary (nats) | 0.029 [-0.345, 0.371] | 0.084 [-0.136, 0.332] |
| vqa accuracy % (correct image) | 18.1 [13.2, 23.4] | 22.3 [16.5, 28.8] |
| vqa accuracy % (complementary swap) | 18.2 [13.1, 23.6] | 21.6 [16.3, 27.1] |
| vqa accuracy % (no image) | 12.4 [8.3, 16.8] | 20.9 [15.7, 26.2] |
| vqa pair accuracy % (correct) | 1.0 [0.0, 3.1] | 5.2 [1.0, 10.4] |
| vqa accuracy adv. vs complementary (pp) | -0.2 [-5.4, 5.2] | 0.7 [-5.8, 7.8] |
| vqa NLL adv. vs complementary (nats) | 0.099 [-0.102, 0.280] | 0.015 [-0.126, 0.155] |
| training time (h) | 0.01 | 0.00 |
| peak GPU alloc, stage 2 (GB) | 1.65 | 1.49 |
| generation ms / answer | 51.8 | 31.6 |

## Qualitative examples

For each suite, final-test pairs are sorted by pair ID and classified from the correct-image predictions of the main (196-token) model: (A) grounded - both questions exactly correct, so each complementary-swap prediction equals the partner's answer; (B) image-insensitive - identical predictions for both images; (C) image-sensitive but wrong - different predictions, at least one wrong. The first two pairs of each class are shown (fewer if a class is smaller).

#### Annotation-derived grounding

| class | question | image | reference (canonical) | correct | swap | mismatched | no image |
|---|---|---|---|---|---|---|---|
| image-insensitive | How many bears can you see? | COCO_val2014_000000531334.jpg | 2 (2) | '1' | '1' | '1' | 'bear' |
| image-insensitive | How many bears can you see? | COCO_val2014_000000573869.jpg | 1 (1) | '1' | '1' | '1' | 'bear' |
| image-insensitive | How many chairs can you see? | COCO_val2014_000000074209.jpg | 4 (4) | 'chairs' | 'chairs' | '1' | 'chair' |
| image-insensitive | How many chairs can you see? | COCO_val2014_000000180487.jpg | 1 (1) | 'chairs' | 'chairs' | 'chair' | 'chair' |
| image-sensitive, wrong | How many keyboards can you see? | COCO_val2014_000000385719.jpg | 4 (4) | 'yes' | 'keyboards' | 'None' | 'keyboard' |
| image-sensitive, wrong | How many keyboards can you see? | COCO_val2014_000000570782.jpg | 3 (3) | 'keyboards' | 'yes' | '10' | 'keyboard' |
| image-sensitive, wrong | How many zebras can you see? | COCO_val2014_000000186991.jpg | 2 (2) | 'zebra' | '1' | '1' | 'zebras' |
| image-sensitive, wrong | How many zebras can you see? | COCO_val2014_000000218424.jpg | 4 (4) | '1' | 'zebra' | '3' | 'zebras' |

#### Natural VQAv2

| class | question | image | reference (canonical) | correct | swap | mismatched | no image |
|---|---|---|---|---|---|---|---|
| grounded | Does this likely belong to a male or female? | COCO_val2014_000000286285.jpg | female (female) | 'female' | 'male' | 'male' | 'female' |
| grounded | Does this likely belong to a male or female? | COCO_val2014_000000364205.jpg | male (male) | 'male' | 'female' | 'female' | 'female' |
| image-insensitive | How many children are here? | COCO_val2014_000000142564.jpg | 2 (2) | '1' | '1' | '1' | '1' |
| image-insensitive | How many children are here? | COCO_val2014_000000121849.jpg | 1 (1) | '1' | '1' | 'here' | '1' |
| image-insensitive | Was the picture taken outdoors? | COCO_val2014_000000148282.jpg | yes (yes) | 'outdoors' | 'outdoors' | 'outdoors' | 'outdoors' |
| image-insensitive | Was the picture taken outdoors? | COCO_val2014_000000299288.jpg | no (no) | 'outdoors' | 'outdoors' | 'outdoors' | 'outdoors' |
| image-sensitive, wrong | What is he doing? | COCO_val2014_000000010023.jpg | skateboarding (skateboarding) | 'I' | 'The question is asking what the user is doing. The answer...' | 'Aurora' | 'doing' |
| image-sensitive, wrong | What is he doing? | COCO_val2014_000000540193.jpg | tennis (tennis) | 'The question is asking what the user is doing. The answer...' | 'I' | 'The answer is: **"He is playing a game."**' | 'doing' |
| image-sensitive, wrong | Are these giraffes in the wild? | COCO_val2014_000000101780.jpg | no (no) | 'giraffe' | 'None' | 'giraffes' | 'wild' |
| image-sensitive, wrong | Are these giraffes in the wild? | COCO_val2014_000000526645.jpg | yes (yes) | 'None' | 'giraffe' | 'no' | 'wild' |

## Connector calibration

- t196: connector output multiplied by a fixed scalar s = mean_token ||text embedding||_2 / mean_token ||initial connector output||_2 (training prompts+targets / training images); text 0.8401, visual before 16.0096 -> scale 0.0525 (after 0.8401)
- t49: connector output multiplied by a fixed scalar s = mean_token ||text embedding||_2 / mean_token ||initial connector output||_2 (training prompts+targets / training images); text 0.8401, visual before 11.6364 -> scale 0.0722 (after 0.8401)
