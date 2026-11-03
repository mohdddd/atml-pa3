
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
