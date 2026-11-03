
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
