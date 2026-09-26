# Gev — working notes

- Package `systemone` (a Gemma 4 port of Kev; NOTICE has the attribution). Env: `uv sync --extra serve`. Entry points
  are modules: `uv run python -m gev.train|benchmark|serve|calibrate|evaluate|publish`; the released recipe is `scripts/recipe.sh`.
- Backbone: `google/gemma-4-E4B` text model, loaded through `gev.model.load_backbone`; delimiters are `<bos>` for the state
  and Gemma's `<unused1..4>` for q/opt/close/decide (`gev.model.delimiters`); never drop the `<bos>`, Gemma reads
  documents badly without it. Gemma 4 has sliding-window layers, so the model always runs the row form
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
- Released model: `runs/e4b-v9` (Gev-E4B v1, google/gemma-4-E4B-it base, readout layers 12,24, Brier term, 400-step head
  warm-up, 1 epoch on 36k records). `docs/results.md` has every run; `docs/gemma-port.md` the diagnosis story. Experiment
  drivers live under `runs/logs/` (gitignored), the repo keeps only `scripts/recipe.sh` and the data builders.
- Repo identity is pinned to the SDCalvo GitHub account (repo-local user.email and credential.username); public at
  github.com/SDCalvo/gev. Weights ship as a GitHub release tarball (gev-e4b.tar.gz); the Hub upload (`gev.publish`)
  needs `hf auth login`.
