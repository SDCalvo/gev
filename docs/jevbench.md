# Gev-E4B v1 on JevBench public items

Measured on **2026-10-02**. The current released checkpoint, `e4b-v9` ([gev-e4b-v1 release](https://github.com/SDCalvo/gev/releases/tag/gev-e4b-v1)),
scored **72.3% (167/231)** on all public JevBench decisions. It beats Kev-4B (research preview) by 6.1 percentage
points on these items and trails SemIf 4B by 8.7 points and Jev 1.13.0 by 14.3 points.

## Comparison on the same public decisions

Only Gev was run for this evaluation. Other models' results are taken from
[JevBench's published item outcomes](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/results/v1.2/jevbench-v1.2-per-task.json)
and recomputed over the same 48 easy, 72 standard and 111 hard public items. These are accuracy scores,
not JevBench composite scores.

| Model | Easy | Standard | Hard | Public accuracy |
|---|---|---|---|---|
| Jev 1.13.0 (TypeSafe AI) | 48/48 | 71/72 | 81/111 | 86.6% (200/231) |
| SemIf (Qwen3.5-4B, TheoLeeCJ) | 48/48 | 71/72 | 68/111 | 81.0% (187/231) |
| system-one-open (Gemma 4 E2B LoRA on an L4) | 48/48 | 67/72 | 54/111 | 73.2% (169/231) |
| **Gev-E4B v1 (this run)** | **48/48** | **67/72** | **52/111** | **72.3% (167/231)** |
| Kev-4B (research preview) | 48/48 | 64/72 | 41/111 | 66.2% (153/231) |

Gev is **24th among 49 compared rows**, including Gev and 48 published models. Ties share a rank.
The [full comparison JSON](benchmarks/jevbench-2026-10-02/comparison.json) includes every compared model,
exclusions, and descriptive paired bootstrap intervals. The interval for Gev minus Kev-4B is +1.7 to +10.8
percentage points (10,000 paired decision resamples, fixed seed; no multiple-comparison adjustment).

This subset omits newer entrants without public item-level outcomes in the source artifact. It also excludes
partial/unmeasured rows and classifier.dev, which is not independently ranked by JevBench. It is not a rank
against the whole current JevBench board. JevBench's official composite uses private original, imported judge,
held-out hard and fresh sealed items, and scores calibration, speed and cost as well as intelligence.
Those private items were unavailable, so we do not report an official Gev composite score.

## Reliability and latency

All **231/231** requests succeeded and passed JevBench's strict probability validation; no distributions needed
renormalization. Brier score was **0.361** and top-label ECE **0.063** overall. On hard items, Brier was **0.670**
and ECE **0.202**, indicating that the development-fitted temperature does not fully calibrate this harder suite.
Brier here is JevBench's sum of squared errors over the label probabilities, averaged over decisions;
ECE uses its ten confidence bins. These are label-based metrics, separate from the official Calibration axis.

Median local request latency was **408 ms**, p95 **2.981 s**. The run used an **Apple M3 Max, 48 GB**, PyTorch/MPS,
BF16, and a loopback HTTP connection. Requests were serial, and model loading was outside the timed run.
No artificial production-load adjustment was applied. These timings are deployment-specific and cannot establish
a speed advantage over published results measured on other hardware and network paths.
The route fee was zero; local compute and electricity were not priced, so no hosted cost score is inferred.

## Where the current release struggles

| Public hard family | Correct | Accuracy |
|---|---|---|
| temporal/numeric | 0/15 | 0.0% |
| long policy | 5/19 | 26.3% |
| ambiguous | 2/7 | 28.6% |
| tradeoff | 2/6 | 33.3% |
| probability | 4/10 | 40.0% |
| hard answer judging | 9/17 | 52.9% |
| multi-hop | 11/18 | 61.1% |
| adversarial | 6/6 | 100.0% |
| hard routing | 5/5 | 100.0% |
| trap | 8/8 | 100.0% |

The strongest results are clear routing, extraction and simple decisions. Temporal/numeric reasoning and long
policies remain substantial weaknesses. These families are small, so their percentages describe this suite rather
than a general capability estimate.

## Method and reproduction

JevBench was pinned to commit `bb05a335bc809e61b20c0f745d25499a82b326fc`. Its unmodified `typesafe` adapter sent
each canonical state, instruction, rubric and label set to Gev's native `/v1/systemone` endpoint in the canonical
option order. Ground-truth labels and provenance were used only by the scorer. The existing checkpoint and its
temperature **1.4142135623730947**, fitted on Gev's development data, were fixed for the run. No benchmark-specific
fine-tuning, calibration, threshold search or rotation averaging was applied for this measurement.

The verification checked all 231 saved requests against the adapter's request builder, independently rescored
all saved distributions, and checked token counts. No states were truncated; the longest encoded request was
3,868 tokens. This is an evaluation of the released checkpoint, not a training-data contamination audit.

From a Gev checkout with the released weights unpacked into `runs/gev-e4b`:

```bash
git clone https://github.com/fstandhartinger/jevbench.git evals/jevbench-upstream
git -C evals/jevbench-upstream checkout bb05a335bc809e61b20c0f745d25499a82b326fc
uv sync --extra serve
uv run --extra serve python -m gev.serve --run runs/gev-e4b --port 8008
```

In a second terminal, use a fresh output directory for each run:

```bash
PYTHONPATH=evals/jevbench-upstream uv run python -m jevbench.cli run \
  --tasks evals/jevbench-upstream/datasets/public/easy.jsonl,evals/jevbench-upstream/datasets/public/original.jsonl,evals/jevbench-upstream/datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8008 --model gev-latest --key-env '' \
  --results runs/gev-jevbench-repro/results.jsonl --raw-dir runs/gev-jevbench-repro/raw \
  --ledger runs/gev-jevbench-repro/ledger.jsonl --manifest runs/gev-jevbench-repro/manifest.json \
  --cap-usd 0 --reserve-usd 0 --price-in-per-m 0 --price-out-per-m 0 \
  --cost-basis zero_route_fee_local_compute_excluded

PYTHONPATH=evals/jevbench-upstream uv run python -m jevbench.cli summarize \
  --tasks evals/jevbench-upstream/datasets/public/easy.jsonl,evals/jevbench-upstream/datasets/public/original.jsonl,evals/jevbench-upstream/datasets/public/hard.jsonl \
  --results runs/gev-jevbench-repro/results.jsonl \
  --public-export runs/gev-jevbench-repro/summary.json
```

## Published evidence

- [Summary and per-tier/per-family metrics](benchmarks/jevbench-2026-10-02/summary.json)
- [Full public model comparison](benchmarks/jevbench-2026-10-02/comparison.json)
- [Gev predictions, probabilities and per-item timings](benchmarks/jevbench-2026-10-02/predictions.jsonl)
- [Run provenance, checkpoint/dataset hashes and verification counts](benchmarks/jevbench-2026-10-02/provenance.json)

The upstream item outcomes artifact labels its revision `v1.3.0` and lives under `results/v1.2/`.
The official-rank context fields in the comparison come from the upstream `v1.4.2.2` artifact and are separate
from the public accuracy ranks calculated here. Public benchmark text and outcome data are credited to
[JevBench / Benchmark Heaven](https://github.com/fstandhartinger/jevbench); JevBench's harness and original public
decisions are MIT-licensed. JevBench is independent of this Gev evaluation.
