---
language: en
license: apache-2.0
library_name: peft
base_model: google/gemma-4-E4B-it
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - gemma-4
datasets:
  - legacy-datasets/banking77
  - google/boolq
  - fancyzhx/ag_news
  - nyu-mll/multi_nli
  - SetFit/sst5
  - Yelp/yelp_review_full
  - CogComp/trec
  - fancyzhx/dbpedia_14
  - SetFit/amazon_reviews_multi_en
  - stanfordnlp/imdb
metrics:
  - accuracy
  - brier_score
  - expected_calibration_error
model-index:
  - name: Gev-E4B
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v7 development (1,264 questions; ten trained public sources + generated policy data)" }
        metrics:
          - { type: accuracy, value: 0.863 }
          - { type: brier_score, value: 0.191, name: "Brier, calibrated" }
          - { type: expected_calibration_error, value: 0.016, name: "ECE, calibrated" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (656 questions; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.800 }
          - { type: brier_score, value: 0.271, name: "Brier, calibrated" }
          - { type: expected_calibration_error, value: 0.030, name: "ECE, calibrated" }
---

# Gev-E4B

Gev-E4B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution
per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16, 36M trainable parameters) plus a
pointer head on `google/gemma-4-E4B-it` (commit `ee0ef602`), serving TypeSafe's public `/v1/systemone` contract, and a
port of [Kev](https://github.com/jaredpalmer/kev) to Gemma 4 ([what that took](../gemma-port.md)).

## Results (calibrated; temperature 1.41 fitted on development rows, argmax unchanged)

| | Gev-E4B | Kev-4B | Jev |
|---|---|---|---|
| in-distribution accuracy (decision-v7 dev, 1,264 questions) | 0.863 | 0.872 | 0.845 |
| in-distribution Brier | 0.191 | | |
| out-of-domain accuracy (transfer-v4 dev, 656 questions) | 0.800 | 0.797 | 0.857 |
| out-of-domain Brier | 0.271 | 0.264 | 0.211 |
| out-of-domain ECE | 0.030 | 0.040 | 0.049 |
| decisions automatable at ≤ 5% error, out of domain | 0.50 | 0.57 | 0.70 |

Per source out of domain: QNLI 0.91, SciQ 0.975, TweetEval-offensive 0.75, PAWS 0.69, MMLU 0.64, Emotion 0.55,
`deadline` (date arithmetic) 0.95, held-out rule structures 0.81–0.97. In distribution: Banking77 0.91, DBpedia 0.95,
TREC 0.94, IMDB 0.94, AG News 0.86, BoolQ 0.86, MNLI 0.85, Yelp 0.71, SST-5 0.58.

## How it was built

- **Base**: `google/gemma-4-E4B-it`, the instruction-tuned checkpoint (4.5B effective / 8B total parameters, 42 layers,
  35 of them sliding-window). The same recipe on the base checkpoint scores 0.652 out of domain; the instruction tuning
  carries the knowledge (MMLU 0.45 → 0.64) and the reasoning (held-out conditionals 0.25 → 0.81).
- **Encoder**: `[<bos> state] [<q> instruction <opt> option </opt> … <decide>]`, Gemma's reserved `<unusedN>` tokens
  as delimiters, `<bos>` as the state marker; every question runs as its own causal row continuing from the state.
- **Head**: each option's `</opt>` state against the `<decide>` state, both a learned mix of layers 12, 24 and the final
  layer, layer-normalized; softmax over the options.
- **Training**: LoRA r=16 on attention and MLP projections (lr 1e-4), head lr 2e-4 with a 400-step head-only warm-up,
  cross-entropy + 0.5 × Brier, one epoch over 36,286 records: Kev's frozen `decision-v7` training partition, a
  2,000-per-source public pool, and generated policy rules, date cases and unknowable cases; bf16 backbone on an
  M3 Max, 8.7 hours. Two further delta passes were evaluated and tie with this checkpoint (`docs/results.md`).
- **Calibration**: temperature 1.41, fitted on the development partition; `GEV_TEMPERATURE=1.0` restores raw logits.

## Known limits

- Emotion (noisy labels) 0.55, SST-5 0.58 and Yelp 0.71: fine-grained sentiment is the weakest area.
- Options past ~500 tokens cannot see the state in the 35 sliding-window layers; very long option lists lose accuracy.
- Coverage at a 5% error budget (0.50) trails Kev-4B as served (0.57) and Jev (0.70).
- Needs `transformers >= 5.17` (the `gemma4` architecture) and `peft >= 0.21`; serving in bf16 needs ~16 GB.

## Use

```bash
uv run --extra serve python -m gev.serve --run runs/gev-e4b --port 8008     # GEV_DTYPE=fp32 for the exact path
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008")`.

## License

Apache-2.0 for the adapter and head; Gemma 4 is Apache-2.0; datasets carry their own licenses.
