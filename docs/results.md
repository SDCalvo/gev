# Results

Every run, on the same frozen partitions: decision-v7 development (in distribution, 1,264 questions) and transfer-v4 development (out of domain: MMLU, Emotion, TweetEval, QNLI, PAWS, SciQ, held-out policy rule structures; 656 questions). Accuracy is unchanged by calibration; Brier, ECE and coverage are reported after the temperature fitted on development rows (`scripts/calibrate_checkpoint.py`). Coverage = the share of decisions automatable at a 5% error budget when accepting by confidence. Kev-4B and Jev numbers are from Kev's model card and its evaluation of Jev.

| run | what | dev acc | dev Brier | dev ECE | dev cov@5% | transfer acc | transfer Brier | transfer ECE | transfer cov@5% | T |
|---|---|---|---|---|---|---|---|---|---|---|
| e4b-v7 | Kev-4B recipe on google/gemma-4-E4B with the Gemma fixes (bos, normalized head, warm-up); 2 epochs on decision-v7 | 0.838 | 0.231 | 0.019 | 0.64 | 0.665 | 0.446 | 0.058 | 0.21 | 1.48 |
| e4b-v7-delta | + delta-v1 pass (lr 2e-5, 1 epoch, 2,000 replayed records) | 0.836 | 0.227 | 0.016 | 0.67 | 0.689 | 0.418 | 0.055 | 0.23 | 1.91 |
| e4b-v8 | google/gemma-4-E4B; readout layers 12,24 + Brier term; 1 epoch on decision-v7 + public-pool-2k + delta-v1 | 0.838 | 0.222 | 0.017 | 0.67 | 0.652 | 0.422 | 0.075 | 0.27 | 1.35 |
| e4b-v8-delta | + delta-v1 pass | 0.869 | 0.198 | 0.027 | 0.74 | 0.680 | 0.412 | 0.085 | 0.28 | 1.52 |
| e4b-v8-delta2 | + delta-v2 pass (MCQ mix, paraphrase, 200 rule structures, dates-v2) | 0.866 | 0.192 | 0.020 | 0.77 | 0.729 | 0.350 | 0.044 | 0.40 | 1.70 |
| e4b-v9 | **released Gev-E4B v1**: the v8 recipe on google/gemma-4-E4B-it (instruction-tuned) | 0.863 | 0.191 | 0.016 | 0.77 | 0.800 | 0.271 | 0.030 | 0.50 | 1.41 |
| e4b-v9-delta-v2 | + delta-v1 and delta-v2 passes | 0.860 | 0.190 | 0.011 | 0.79 | 0.796 | 0.270 | 0.043 | 0.52 | 2.09 |
| Kev-4B | Qwen3.5-4B-Base, Kev's recipe + delta | 0.872 | | 0.075 raw | 0.57 | 0.797 | 0.264 | 0.040 | 0.57 | 2.14 |
| Jev | TypeSafe's hosted model | 0.845 | | | | 0.857 | 0.211 | 0.049 | 0.70 | |

## Per source, transfer-v4 development (accuracy)

| source | e4b-v7 | e4b-v8-delta2 | **e4b-v9** | e4b-v9-delta-v2 | Kev-4B |
|---|---|---|---|---|---|
| composition_held_and_or | 0.656 | 0.875 | 0.906 | 0.906 |  |
| composition_held_conditional | 0.375 | 0.531 | 0.812 | 0.938 |  |
| composition_held_or_not | 0.500 | 0.844 | 0.969 | 0.906 |  |
| contrastive_authorization | 0.850 | 0.900 | 1.000 | 1.000 |  |
| contrastive_deadline | 0.450 | 0.750 | 0.950 | 0.900 | 0.6 |
| emotion | 0.562 | 0.500 | 0.550 | 0.537 | 0.56 |
| mmlu | 0.500 | 0.575 | 0.637 | 0.688 | 0.7 |
| paws | 0.525 | 0.637 | 0.688 | 0.662 | 0.74 |
| qnli | 0.887 | 0.863 | 0.912 | 0.887 | 0.91 |
| sciq | 0.963 | 0.975 | 0.975 | 0.975 | 0.97 |
| tweet_offensive | 0.750 | 0.700 | 0.750 | 0.725 | 0.74 |

## Per source, decision-v7 development (accuracy)

| source | e4b-v7 | e4b-v8-delta2 | **e4b-v9** | e4b-v9-delta-v2 |
|---|---|---|---|---|
| agnews | 0.825 | 0.838 | 0.863 | 0.850 |
| agnews_yn | 0.900 | 0.925 | 0.912 | 0.919 |
| amazon | 0.650 | 0.637 | 0.575 | 0.600 |
| banking77 | 0.850 | 0.863 | 0.912 | 0.900 |
| boolq | 0.900 | 0.863 | 0.863 | 0.875 |
| composition_atom | 1.000 | 1.000 | 1.000 | 1.000 |
| composition_conditional | 0.562 | 0.688 | 0.812 | 0.938 |
| composition_conjunction | 0.812 | 1.000 | 1.000 | 1.000 |
| composition_disjunction | 0.188 | 0.875 | 1.000 | 1.000 |
| composition_exception | 0.750 | 1.000 | 1.000 | 1.000 |
| composition_negation | 0.688 | 1.000 | 1.000 | 1.000 |
| composition_nested_and | 0.625 | 0.812 | 0.812 | 0.812 |
| composition_nested_or | 0.375 | 1.000 | 1.000 | 1.000 |
| contrastive_age_eligibility | 1.000 | 1.000 | 1.000 | 1.000 |
| contrastive_quantity_limit | 1.000 | 1.000 | 1.000 | 0.917 |
| contrastive_return_window | 0.875 | 0.958 | 0.833 | 0.917 |
| contrastive_spend_threshold | 1.000 | 1.000 | 1.000 | 1.000 |
| dbpedia14 | 0.963 | 0.963 | 0.963 | 0.963 |
| imdb | 0.938 | 0.950 | 0.950 | 0.963 |
| mnli | 0.838 | 0.825 | 0.863 | 0.863 |
| sst5 | 0.713 | 0.675 | 0.575 | 0.562 |
| trec | 0.950 | 0.925 | 0.938 | 0.925 |
| yelp | 0.725 | 0.688 | 0.725 | 0.675 |
| yelp_yn | 0.925 | 0.938 | 0.912 | 0.875 |

## Paired comparisons

Record-clustered bootstrap (1,000 resamples) over the same questions, `gev.compare`. Macro accuracy delta with its 95% interval.

| candidate | reference | split | acc delta | 95% CI | Brier delta |
|---|---|---|---|---|---|
| e4b-v7-delta | e4b-v7 | development | +0.018 | [-0.020, +0.055] | -0.017 |
| e4b-v7-delta | e4b-v7 | transfer | +0.044 | [-0.009, +0.094] | -0.023 |
| e4b-v8-delta | e4b-v7-delta | development | +0.086 | [+0.045, +0.123] | -0.105 |
| e4b-v8-delta | e4b-v8 | development | +0.090 | [+0.050, +0.126] | -0.087 |
| e4b-v8-delta | e4b-v7-delta | transfer | -0.000 | [-0.044, +0.045] | -0.026 |
| e4b-v8-delta | e4b-v8 | transfer | +0.054 | [+0.012, +0.102] | -0.001 |
| e4b-v8-delta2 | e4b-v8-delta | development | -0.005 | [-0.023, +0.010] | -0.015 |
| e4b-v8-delta2 | e4b-v8 | development | +0.085 | [+0.042, +0.122] | -0.101 |
| e4b-v8-delta2 | e4b-v8-delta | transfer | +0.059 | [+0.015, +0.101] | -0.097 |
| e4b-v8-delta2 | e4b-v8 | transfer | +0.112 | [+0.066, +0.164] | -0.098 |
| e4b-v8 | e4b-v7-delta | development | -0.004 | [-0.043, +0.037] | -0.018 |
| e4b-v8 | e4b-v7 | development | +0.014 | [-0.025, +0.057] | -0.035 |
| e4b-v8 | e4b-v7-delta | transfer | -0.054 | [-0.097, -0.010] | -0.025 |
| e4b-v8 | e4b-v7 | transfer | -0.010 | [-0.063, +0.041] | -0.047 |
| e4b-v9-delta-v2 | e4b-v8-delta | development | +0.000 | [-0.015, +0.014] | -0.021 |
| e4b-v9-delta-v2 | e4b-v8-delta | transfer | +0.147 | [+0.096, +0.197] | -0.207 |
| e4b-v9 | e4b-v8-delta | development | -0.002 | [-0.015, +0.010] | -0.026 |
| e4b-v9 | e4b-v8-delta | transfer | +0.149 | [+0.099, +0.201] | -0.220 |

## Pilot sweep (1 epoch on 4,576 records, first 400 development records)

| pilot | acc | agnews | banking77 | boolq | ECE | Brier |
|---|---|---|---|---|---|---|
| base | 0.688 | 0.787 | 0.688 | 0.700 | 0.041 | 0.397 |
| layers12-24 | 0.734 | 0.762 | 0.750 | 0.787 | 0.037 | 0.372 |
| headlr5e-4-w200 | 0.443 | 0.425 | 0.000 | 0.450 | 0.071 | 0.627 |
| loralr5e-5 | 0.618 | 0.787 | 0.500 | 0.613 | 0.036 | 0.499 |
| brierw0.5 | 0.698 | 0.800 | 0.762 | 0.725 | 0.036 | 0.403 |
| ordw0.5 | 0.559 | 0.688 | 0.338 | 0.600 | 0.085 | 0.533 |
| itbase | 0.721 | 0.825 | 0.675 | 0.800 | 0.073 | 0.392 |
| it-layers | 0.757 | 0.825 | 0.738 | 0.800 | 0.063 | 0.364 |

Not in the table: LoRA rank 32 diverged (non-finite loss), `--perm_kl 0.5` ran at 3 s/record and was stopped. Inference-time options measured on e4b-v7: date preprocessing neutral, 3-way rotation averaging +0.2 points in domain.
