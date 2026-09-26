# Porting Kev to Gemma 4: what broke and how it was fixed

Kev's recipe (a pointer head over a LoRA-adapted decoder, trained on typed decisions) was built on Qwen bases. Applied
verbatim to `google/gemma-4-E4B` it trained without errors, learned every yes/no question, and left every Choice
question at exactly chance. This is the record of the diagnosis, in the order it happened, with the measurements
that decided each step. Everything ran on one M3 Max with 48 GB.

## Symptom

First pilot (1 epoch on 4,576 records, Kev-4B settings): 0.41 on a 200-record gate. `agnews_yn` (yes/no) 0.84,
`banking77` (77-way) 0.013 with NLL 4.36 = ln 77, `agnews` (4-way) NLL 1.38 = ln 4. The model answered uniform on
every Choice question.

## Ruled out

- **The backbone.** Coherent generation on MPS; calling the text model directly matches the full multimodal model's
  hidden states bit for bit; a token's final state changes with its context (cosine 0.29 across contexts).
- **Gradient flow.** One record overfits to zero loss in five steps.
- **Attention kernels.** SDPA and eager agree to bf16 noise.
- **Sequence structure.** Delimiter positions verified; Gemma 4's sliding-window layers are handled by running each
  question as its own causal row (Kev's row form), which is exact by construction.

## Cause 1: collapse of the option representations

After training, the `</opt>` states of the 77 Banking77 options had pairwise cosine 0.99 (base model: 0.86) and norm
54 (base: 217). The adapter had shrunk and homogenized them. Gemma's final hidden states have norms in the hundreds,
so Kev's untrained head starts at logits of ±12; with a random head, "make every option identical" (uniform output,
loss ln K) is the steepest descent from confidently-wrong, and the LoRA finds it within 50 steps.

Fix: layer-normalize the head's inputs (`PointerHead(norm=True)`). Necessary, not sufficient: the third pilot
collapsed again with cosine 1.000 and the norm grown to 400. Normalization removes the shrink path, not the direction
path.

## Cause 2: the missing `<bos>`

Kev's encoder starts a sequence with a reserved token. Gemma forms its attention sink on `<bos>` and reads a document
badly without it. On frozen final-layer features of 320 agnews records, a pointer head reaches 0.39 held-out without
`<bos>` and 0.61 with it (layer 12: 0.42 → 0.75); the article's topic is linearly decodable from the `<decide>` state at
0.45 without and 0.61 with. An earlier check had compared option-state cosines only and missed this.

Fix: `<bos>` is the state delimiter.

## Cause 3: the head must learn before the adapter moves

With `<bos>` and the normalized head, the pilot still sat at uniform: the adapter homogenized the options before the
head found the content signal. Fix: a head-only warm-up (linear probe, then fine-tune). Two implementation lessons on
MPS: a *forward-only* backbone (LoRA frozen) leaks about 49 GB of non-pool device memory within 250 steps, with or
without shape bucketing, so the warm-up keeps the training graph and discards the LoRA gradients instead; and head lr
1e-3 diverges (loss 1.4 → 3.5 in 200 steps) while 2e-4 is stable.

Fifth pilot: 0.811 on the gate (agnews 0.80, banking77 0.76). Loss fell from 1.26 to 0.72 the moment the adapter
unfroze.

## After that: what moved the numbers

| change | measured effect |
|---|---|
| readout mixing layers 12, 24 with the final layer | +4.6 points on the pilot gate; the learned mix stayed uniform |
| Brier term in the loss | +1 point, marginal |
| targeted delta passes (rules over hundreds of random structures, dates, unknowable cases with uniform targets, an MCQ mix, paraphrase pairs) | +5.9 and +9.0 points on the base model; MMLU 0.41 → 0.575, above the backbone's 0.526 letter-logit ceiling |
| instruction-tuned base (`gemma-4-E4B-it`) with the same recipe | transfer 0.652 → 0.800, Brier 0.422 → 0.271; the delta passes then tie |
| 2,000 public records per source instead of 1,000 | calibration, not accuracy |
| LoRA rank 32, head lr 5e-4, lower LoRA lr, ordinal loss, permutation loss | diverged / collapsed / worse / worse / 5× slower |
| ensembling v9 with the earlier models | hurts out of domain (0.777); ensembles of equal-strength models only |

`docs/results.md` has every run per source with paired bootstraps.

## Practical notes for Gemma backbones

- Keep `<bos>` first. Train with `--head_norm 1`, `--head_warmup_steps 300+`, `--head_lr 2e-4`, LoRA `--lr 1e-4`.
- Judge learning by Choice-source accuracy on a few hundred development records, never by the average loss (yes/no
  and policy records dominate it). Nothing under ~500 optimizer steps is informative.
- Bucket sequence lengths on MPS in training too; a suite admitted under another tokenizer can push a few records
  past the branch limit, so overlong records are de-augmented or skipped, not fatal.
- Public records cost ~0.85 s each on an M3 Max in bf16; one epoch on 36k records is ~9 hours. Checkpoint every 100
  steps and resume.
