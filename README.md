# Gev

A **decision model** on Google's Gemma 4 E4B, trained on a Mac. Gev answers typed questions about a document in one
forward pass and returns a calibrated probability distribution per question. No text is generated, so the answer
always matches the schema. It speaks TypeSafe's `/v1/systemone` API, like [Jev](https://typesafe.ai) and
[Kev](https://github.com/jaredpalmer/kev), whose recipe it ports to Gemma.

| question type | in | out |
|---|---|---|
| `choice` | named options, optionally described | the option, its probability, all probabilities |
| `score`  | ordered levels | expected level and per-level probabilities |
| `noul`   | a yes/no question | p(yes) |

## Results

Development = the same ten public datasets and generated policy families as training, different records. Transfer =
six datasets and rule structures never trained on. Gev's numbers are calibrated (one temperature fitted on development
rows; the answers are unchanged); Kev-4B's are from its model card, Jev's from Kev's evaluation of it.

| | Gev-E4B | Kev-4B (Qwen3.5) | Jev |
|---|---|---|---|
| development accuracy | 0.863 | 0.872 | 0.845 |
| development Brier | 0.191 | | |
| transfer accuracy | 0.800 | 0.797 | 0.857 |
| transfer Brier | 0.271 | 0.264 | 0.211 |
| transfer ECE | 0.030 | 0.040 | 0.049 |
| transfer decisions automatable at 5% error | 0.50 | 0.57 | 0.70 |

Every run, per source, with paired bootstraps: [`docs/results.md`](docs/results.md). Model card:
[`docs/model-cards/gev-e4b.md`](docs/model-cards/gev-e4b.md).

## How it works

Gev is Kev's recipe with the backbone swapped for Gemma 4, which needed five changes to work at all. Each one was
measured, not guessed; [`docs/gemma-port.md`](docs/gemma-port.md) has the evidence.

- **Encoder.** `[<bos> state] [<q> instruction <opt> option </opt> ... <decide>]` per question. Gemma's reserved
  `<unusedN>` tokens are the delimiters, except the state marker, which is `<bos>`: Gemma forms its attention sink on
  it and reads a document badly without it (a head on frozen features: 0.39 without, 0.61 with).
- **Backbone.** The text model of `google/gemma-4-E4B-it` (4.5B effective / 8B total parameters), frozen, with rank-16
  LoRA on the attention and MLP projections. The instruction-tuned checkpoint is worth 15 points out of domain over the
  base one with the same recipe. Gemma 4 has sliding-window layers, so every question runs as its own causal row
  continuing from the state, which is exact isolation by construction.
- **Pointer head.** Each option's `</opt>` state is scored against the question's `<decide>` state and softmaxed. The
  head reads a learned mix of the final layer and layers 12 and 24 (`--readout_layers 12,24`), and layer-normalizes its
  inputs: Gemma's hidden norms are in the hundreds, and without the norm the adapter collapsed every Choice question to
  a uniform answer.
- **Training.** The head trains alone for the first 400 optimizer steps, then jointly with the LoRA (linear probe, then
  fine-tune): with a random head the adapter's fastest loss reduction was to homogenize the option states. Head lr 2e-4,
  LoRA lr 1e-4, cross-entropy plus a Brier term, one epoch, resumable from checkpoints every 100 steps. Optional
  **delta passes** at lr 2e-5 on targeted generated data (policy rules over hundreds of random structures, date
  arithmetic, unknowable cases with uniform targets, a multiple-choice mix, paraphrase pairs) are worth +5 to +9 points
  on the base checkpoint and tie on the instruction-tuned one.
- **Data.** Kev's frozen `decision-v7` suite (10 public datasets, 1,000 records each, plus generated policy records),
  a 2,000-per-source public pool, and the generated delta sets. Eval-only datasets (MMLU, Emotion, TweetEval, QNLI,
  PAWS, SciQ) are never trained on. Training partitions download from Kev's Hub mirror on first use and are
  sha256-checked.

## Setup

```bash
git clone https://github.com/SDCalvo/gev.git && cd gev
uv sync --extra serve                                  # torch, transformers 5.17, peft; FastAPI + TypeSafe SDK for serving
uv run python -m pytest tests/test_unit.py tests/test_generators.py -q
```

The first run downloads `google/gemma-4-E4B-it` (16 GB) into the Hugging Face cache. Everything here ran on an
M3 Max with 48 GB; the backbone trains and serves in bf16 at about 16 GB.

## Use

The weights (LoRA adapter, pointer head, tokenizer, eval reports) are in the
[`gev-e4b-v1` release](https://github.com/SDCalvo/gev/releases/tag/gev-e4b-v1) with SHA-256 checksums:

```bash
curl -LO https://github.com/SDCalvo/gev/releases/download/gev-e4b-v1/gev-e4b.tar.gz && tar -xzf gev-e4b.tar.gz -C runs
uv run --extra serve python -m gev.serve --run runs/gev-e4b --port 8008
```

```python
from typesafe import TypeSafeClient
client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008")
```

`GEV_TEMPERATURE=1.0` returns raw logits; `GEV_DTYPE=fp32` serves the exact path the evaluations use. A Hugging Face
Hub upload (`python -m gev.publish`) follows; `--run` then also accepts the Hub id.

## Fine-tune on your own decisions

Write your labelled requests as JSONL, one per line, in the API's shape plus a `label` per question:

```json
{"state": {"subject": "Charged twice", "body": "Two charges for order 4411."},
 "questions": {"team":     {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": null}, "label": "billing"},
               "angry":    {"type": "noul",   "instructions": "Is the customer angry?", "label": false},
               "priority": {"type": "score",  "instructions": "How urgent?", "criteria": ["low", "normal", "high"], "label": 1}}}
```

Then a delta pass from the released checkpoint, replaying some of the original training data so it keeps what it
knows (the flags must match the checkpoint's: instruction-tuned base, readout layers, Brier term):

```bash
uv run python -m gev.train --data mine.jsonl --suite evals/v7/decision-v7 --replay 2000 --init_from runs/gev-e4b \
  --base google/gemma-4-E4B-it --base_revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --lr 2e-5 --head_lr 2e-5 --epochs 1 --weights_dtype bf16 --readout_layers 12,24 --brier_w 0.5 --out runs/mine
uv run python -m gev.benchmark --run runs/mine --data mine-heldout.jsonl --out runs/mine/eval
uv run python scripts/calibrate_checkpoint.py --run runs/mine --rows runs/mine/eval/rows.json
```

A few hundred labelled records are enough for a new routing taxonomy. `--rotations 3` at benchmark time averages over
option orders if order sensitivity matters, and `scripts/reliability_head.py` re-ranks confidences for a higher share of
automatable decisions at a fixed error budget.

## Reproduce the released model

`scripts/recipe.sh` runs the whole thing: the base run on the suite plus the public pool, evaluation and calibration,
and with `DELTA=1` the two delta passes. Data builders are in `scripts/build_*.py`. About 9 hours on an M3 Max,
resumable from checkpoints every 100 steps.

## Layout

```
gev/       api (request/response mapping), model (encoder, backbone, pointer head), data (public sources -> records),
           suite (frozen suites), train (resumable), checkpoint, benchmark + predictors + metrics, calibrate, serve,
           compare, publish
evals/     frozen suites (v7/decision-v7, v4/transfer-v4, smoke-v1) and the generated delta sets (manifests; the data
           files are rebuilt by scripts/build_*.py)
scripts/   recipe.sh, calibrate_checkpoint.py, reliability_head.py, data builders, backbone and ceiling probes
docs/      results.md (every run, per source), gemma-port.md (what broke and how it was fixed), model-cards/
tests/     unit, generator, API and model-parity tests
```

## License

Apache-2.0. Derived from [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer (Apache-2.0); Gemma 4 is released
by Google under Apache-2.0. Datasets in the frozen suites carry their own licenses.
