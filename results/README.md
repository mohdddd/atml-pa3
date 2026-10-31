# results/

Machine-readable outputs behind every number in the report. One folder per run so concurrent
Kaggle pushes never conflict.

Conventions
* `config.yaml` — fully resolved config actually used (copied at run start).
* `train_log.jsonl` — one JSON object per optimizer update (loss, lr, grad norm, scaler, skipped flag, ...).
* `resources.json` — wall-clock times, peak GPU memory, trainable params, checkpoint sizes, overheads.
* `eval/<stage>/...` — aggregate metrics (`*.json`) and per-example outputs (`*.jsonl.gz`).
* Files must stay < 95 MB (GitHub limit); large per-example dumps are gzipped.
