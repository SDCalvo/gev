# systemone — working notes

- Package `systemone` (a Gemma 4 port of Kev; NOTICE has the attribution). Env: `uv sync --extra serve`. Entry points
  are modules: `uv run python -m systemone.train|benchmark|serve|calibrate|evaluate`.
- Backbone: `google/gemma-4-E4B` text model, loaded through `systemone.model.load_backbone`; delimiters are Gemma's
  `<unused0..4>` (`systemone.model.delimiters`). Gemma 4 has sliding-window layers, so the model always runs the row form
  (one causal row per question, `rows_only`), never the packed 4D mask.
- Mac training: `--weights_dtype bf16` (frozen backbone bf16, LoRA/head fp32). `--dtype bf16` autocast is CUDA-only.
  One training job at a time on MPS. Don't enable `output_hidden_states` or peft `trainable_token_indices` on MPS
  (memory problems inherited from Kev's notes).
- Suites under `evals/` are frozen and sha256-checked byte for byte: never reformat their JSON/JSONL. `train.jsonl` of
  decision-v7 is fetched from `jaredpalmer/kev-suites` on first use. Manifests pin `google/gemma-4-E4B` to commit
  `411aa17b`.
- Locked test partitions are read once per final candidate (`--allow-test`); select models on `development`.
- Fast tests: `uv run python -m pytest tests/test_unit.py tests/test_generators.py -q    # tests/test_api.py needs a running server`.
  `tests/test_model.py` needs a smoke checkpoint at `runs/smoke-hl/00-trial-0/checkpoint` (or edit SMOKE).
