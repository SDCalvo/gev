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

### JevBench public benchmark (2026-10-02)

The current release, **Gev-E4B v1**, scored **72.3% (167/231)** on all public
[JevBench](https://github.com/fstandhartinger/jevbench) decisions. Gev was evaluated through its native
`/v1/systemone` API; the other models' scores come from JevBench's published outcomes on the **same 231 items**.

| Model | Easy | Standard | Hard | Public accuracy |
|---|---|---|---|---|
| Jev 1.13.0 (TypeSafe AI) | 48/48 | 71/72 | 81/111 | 86.6% (200/231) |
| SemIf (Qwen3.5-4B) | 48/48 | 71/72 | 68/111 | 81.0% (187/231) |
| system-one-open (Gemma 4 E2B) | 48/48 | 67/72 | 54/111 | 73.2% (169/231) |
| **Gev-E4B v1 (current release)** | **48/48** | **67/72** | **52/111** | **72.3% (167/231)** |
| Kev-4B (research preview) | 48/48 | 64/72 | 41/111 | 66.2% (153/231) |

Gev beats Kev-4B by **6.1 percentage points** and places **24th among 49 compared rows** on public accuracy,
including Gev and 48 published models. All **231/231** responses passed strict schema/distribution validation.
Temporal/numeric reasoning (**0/15**) and long policies (**5/19**) are the current release's weakest hard families.

This is a **public accuracy comparison**. The official JevBench composite also requires private items and measures
calibration, speed and cost. The comparison covers models with complete published item-level outcomes in the
pinned source artifact; newer entrants may be absent. Ties share a rank.
[Method, calibration, latency, limitations and supporting data](docs/jevbench.md).

<details>
<summary>Full comparison: all 49 models on the same public JevBench items</summary>

| Public accuracy rank | Model | Public accuracy | Easy | Standard | Hard |
|---|---|---|---|---|---|
| 1 | DeepSeek V4.1 Flash (thinking default) | 97.8% (226/231) | 48/48 | 71/72 | 107/111 |
| 2 | GPT-5.6 Luna (low reasoning effort) | 97.4% (225/231) | 48/48 | 70/72 | 107/111 |
| 3 | OpenJev (thinking, BF16) | 88.7% (205/231) | 48/48 | 72/72 | 85/111 |
| 4 | djev (thinking) | 87.4% (202/231) | 46/48 | 71/72 | 85/111 |
| 5 | Gemini 3.1 Flash-Lite | 87.0% (201/231) | 48/48 | 71/72 | 82/111 |
| 5 | reflex-27b (Qwen3.8-27B) | 87.0% (201/231) | 48/48 | 69/72 | 84/111 |
| 7 | Jev 1.13.0 (TypeSafe AI) | 86.6% (200/231) | 48/48 | 71/72 | 81/111 |
| 7 | SimpleJev Qwen3.8-27B | 86.6% (200/231) | 48/48 | 70/72 | 82/111 |
| 9 | LitJev (Qwen3.8-27B) | 86.1% (199/231) | 48/48 | 71/72 | 80/111 |
| 10 | Winnow-12B Q8 | 85.7% (198/231) | 48/48 | 69/72 | 81/111 |
| 11 | openjev-sglang (Qwen3.6-35B-A3B on SGLang) | 85.3% (197/231) | 48/48 | 68/72 | 81/111 |
| 12 | djev (Maisa, diffusion-gemma) | 84.0% (194/231) | 48/48 | 71/72 | 75/111 |
| 13 | decider-35b-a3b (Mapika) | 83.1% (192/231) | 48/48 | 70/72 | 74/111 |
| 14 | OpenJev (DiffusionGemma 26B-A4B NVFP4, razorback16) | 81.8% (189/231) | 48/48 | 70/72 | 71/111 |
| 15 | SimpleJev Qwen3.6-35B-A3B | 81.4% (188/231) | 48/48 | 67/72 | 73/111 |
| 16 | SemIf, formerly OpenJev (Qwen3.5-4B, TheoLeeCJ) | 81.0% (187/231) | 48/48 | 71/72 | 68/111 |
| 17 | jqv (Qwen3-32B zero-shot) | 80.1% (185/231) | 48/48 | 69/72 | 68/111 |
| 18 | Bespoke Nimble 9B (Bespoke Labs) | 79.7% (184/231) | 48/48 | 67/72 | 69/111 |
| 19 | reflex 4B (kshetrajna12) | 79.2% (183/231) | 48/48 | 68/72 | 67/111 |
| 20 | Open-Jev 9B (Zefan Cai) | 77.5% (179/231) | 48/48 | 65/72 | 66/111 |
| 21 | jev-local (Qwen3.5-9B) | 74.9% (173/231) | 48/48 | 60/72 | 65/111 |
| 22 | open-alternative-jev (Qwen3.5-4B, IkerMoel) | 74.0% (171/231) | 48/48 | 60/72 | 63/111 |
| 23 | system-one-open (Gemma 4 E2B LoRA on an L4) | 73.2% (169/231) | 48/48 | 67/72 | 54/111 |
| 24 | **Gev-E4B v1 (current release)** | 72.3% (167/231) | 48/48 | 67/72 | 52/111 |
| 25 | system-one (Qwen3-8B, Sean Goedecke) | 71.9% (166/231) | 48/48 | 64/72 | 54/111 |
| 26 | kev 8B (research preview) | 71.4% (165/231) | 48/48 | 67/72 | 50/111 |
| 27 | decider-2b (Mapika) | 71.0% (164/231) | 48/48 | 61/72 | 55/111 |
| 28 | ZeroEntropy zerank-2 | 70.1% (162/231) | 48/48 | 57/72 | 57/111 |
| 29 | Qwen3-Reranker-4B | 68.0% (157/231) | 48/48 | 54/72 | 55/111 |
| 30 | decision-machine-1 (milliseconds.ai) | 67.5% (156/231) | 48/48 | 54/72 | 54/111 |
| 31 | kev 0.6B (research preview) | 66.7% (154/231) | 48/48 | 58/72 | 48/111 |
| 32 | kev 4B (research preview) | 66.2% (153/231) | 48/48 | 64/72 | 41/111 |
| 33 | Open-Jev 2B (Zefan Cai) | 64.5% (149/231) | 48/48 | 55/72 | 46/111 |
| 34 | jeff (Logan Markewich, GLiFormer 400M) | 62.8% (145/231) | 48/48 | 54/72 | 43/111 |
| 35 | smalljev semantic-v9 | 60.6% (140/231) | 47/48 | 49/72 | 44/111 |
| 36 | Laya (Convai Innovations, ModernBERT-large 421M) | 58.4% (135/231) | 46/48 | 50/72 | 39/111 |
| 37 | GLiNER2 (Fastino, gliner2.5-base) | 58.0% (134/231) | 47/48 | 46/72 | 41/111 |
| 38 | openJev Verdict 1.4 | 57.6% (133/231) | 42/48 | 50/72 | 41/111 |
| 39 | GLiNER2 large (Fastino) | 56.7% (131/231) | 48/48 | 42/72 | 41/111 |
| 40 | openJev Verdict (heman10x, ModernBERT-base 151M) | 55.4% (128/231) | 41/48 | 45/72 | 42/111 |
| 41 | OpenDecision (ModernBERT-large zero-shot) | 53.2% (123/231) | 42/48 | 43/72 | 38/111 |
| 42 | open-jev-deberta-v3-large (local CPU) | 52.4% (121/231) | 48/48 | 31/72 | 42/111 |
| 43 | kev 0.5B | 49.4% (114/231) | 46/48 | 35/72 | 33/111 |
| 44 | GLiNER2.5 multi (Fastino, 287M) | 48.9% (113/231) | 44/48 | 32/72 | 37/111 |
| 45 | GLiNER2.5 small (Fastino, 74M) | 45.9% (106/231) | 41/48 | 30/72 | 35/111 |
| 46 | BAAI bge-reranker-v2-m3 | 39.4% (91/231) | 23/48 | 26/72 | 42/111 |
| 47 | Mixedbread mxbai-rerank-base-v2 | 37.2% (86/231) | 22/48 | 24/72 | 40/111 |
| 48 | Alibaba GTE Reranker ModernBERT-base | 33.8% (78/231) | 17/48 | 26/72 | 35/111 |
| 49 | Certo v1 (AltSlate Labs) | 31.6% (73/231) | 12/48 | 24/72 | 37/111 |

</details>

### Development and transfer evaluation

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
