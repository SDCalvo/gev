# systemone

A **System One decision model** on Google's **Gemma 4 E4B**, trained on a Mac.

One document (the *state*) and a set of typed questions go in; a calibrated probability distribution per question
comes out, in a single forward pass. No text is generated, so the answer always matches the schema. The three question
types follow TypeSafe's public `/v1/systemone` contract:

| type   | in                                   | out                                   |
|--------|--------------------------------------|---------------------------------------|
| choice | named options, optionally described  | the option, its probability, all probabilities |
| score  | ordered levels                       | expected level and per-level probabilities |
| noul   | a yes/no question                    | p(yes)                                |

## How it works

The recipe is [Kev's](https://github.com/jaredpalmer/kev) (Apache-2.0; see NOTICE), with the Qwen backbone swapped
for Gemma 4 (also Apache-2.0):

- **Encoder.** `[<state> text] [<q> instruction <opt> option </opt> ... <decide>]` per question, using five reserved
  `<unusedN>` tokens of the Gemma tokenizer as delimiters. Caller text can never produce a delimiter.
- **Backbone.** The text model of `google/gemma-4-E4B` (4.5B effective / 8B total parameters, 42 layers, 35 of them
  sliding-window), frozen, with rank-16 LoRA adapters on the attention and MLP projections. Because of the sliding
  window, every question runs as its own causal row continuing from the state, which is exact isolation by
  construction (`systemone/model.py`).
- **Pointer head.** Each option's `</opt>` hidden state is scored against the question's `<decide>` hidden state and
  softmaxed. Trained from scratch with cross-entropy, then temperature-scaled on held-out data.
- **Data.** The frozen `decision-v7` suite: 10,000 records from ten public datasets (Banking77, BoolQ, AG News, MNLI,
  SST-5, Yelp, TREC, DBpedia, Amazon, IMDB) plus 2,576 generated policy and rule records. `transfer-v4` holds six
  never-trained sources (MMLU, Emotion, TweetEval, QNLI, PAWS, SciQ) for out-of-domain evaluation. Training partitions
  are fetched from the Hub mirror on first use and sha256-verified against the manifests.

## Setup

```bash
uv sync --extra serve          # Python 3.12 venv with torch, transformers 5.17, peft; FastAPI + TypeSafe SDK for serving
uv run python -m pytest tests/test_unit.py tests/test_generators.py -q    # tests/test_api.py needs a running server
```

The first training run downloads `google/gemma-4-E4B` (16 GB, bf16) into the Hugging Face cache.

## Train

```bash
# smoke test: 40 records per source, a few minutes, proves the pipeline end to end
uv run python -m systemone.train --n_per_source 40 --accum 4 --weights_dtype bf16 --out runs/smoke

# the Kev-4B recipe on the frozen suite: two epochs, LoRA r=16, lr 5e-5
uv run python -m systemone.train --suite evals/v7/decision-v7 --weights_dtype bf16 --lr 5e-5 --epochs 2 --accum 8 --out runs/e4b-v7

# delta fine-tune on your own labelled JSONL (see systemone.data.load_records for the format)
uv run python -m systemone.train --data mine.jsonl --init_from runs/e4b-v7 --lr 2e-5 --epochs 1 --out runs/mine
```

`--weights_dtype bf16` keeps the frozen backbone at 16 GB on a Mac; LoRA and the head train in fp32. Run one training
job at a time on Apple Silicon: two on the same GPU slow each other down by an order of magnitude.

## Evaluate, calibrate, serve

```bash
uv run python -m systemone.benchmark --run runs/e4b-v7 --suite evals/v7/decision-v7 --split development --out runs/e4b-v7/dev
uv run python -m systemone.benchmark --run runs/e4b-v7 --suite evals/v4/transfer-v4 --split development --out runs/e4b-v7/transfer
uv run python scripts/calibrate_checkpoint.py --run runs/e4b-v7 --suite evals/v7/decision-v7     # writes the temperature into head.pt
uv run --extra serve python -m systemone.serve --run runs/e4b-v7 --port 8008
```

The server speaks TypeSafe's API, so any System One client works:

```python
from typesafe import TypeSafeClient
client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008")
```

`SYSTEMONE_DTYPE=fp32` serves the exact path; `SYSTEMONE_TEMPERATURE=1.0` returns raw logits; `SYSTEMONE_DATE_FACTS=1`
adds deterministic date arithmetic to the state.

## Layout

```
systemone/   api (request/response mapping), model (encoder, backbone, pointer head), data (public sources -> records),
             suite (frozen suites), train, checkpoint, benchmark + predictors + metrics, calibrate, evaluate, serve
evals/       frozen suites: v7/decision-v7 (train), v4/transfer-v4 (out of domain), smoke-v1
tests/       unit tests (no weights), API tests, generator tests, model parity tests (need a smoke checkpoint)
scripts/     calibrate_checkpoint.py
runs/        training runs (gitignored except configs and reports)
```

## License

Apache-2.0. Derived from Kev by Jared Palmer (Apache-2.0); Gemma 4 is released by Google under Apache-2.0. Datasets in
the frozen suites carry their own licenses.
