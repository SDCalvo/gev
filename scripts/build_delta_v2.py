"""Second targeted delta set, aimed at the out-of-domain gaps of e4b-v8-delta (--data JSONL).

    uv run python scripts/build_delta_v2.py --out evals/delta-v2

  mcq         ARC-Challenge, OpenBookQA, CommonsenseQA train splits (trainable per Kev's policy; MMLU and SciQ stay
              eval-only): a multiple-choice mix so the pointer head learns to tap the backbone's knowledge
  paraphrase  QQP and MRPC (GLUE) as Noul "do these mean the same?" questions (PAWS stays eval-only)
  compositional  200 fresh random rule structures x 4 groups, training styles (held-out structures excluded)
  legacy_policy  contrastive minimal pairs, trainable families, new seed
  dates-v2    the date / unknowable / assertion files built by scripts/build_date_data.py with seed dates-v2
"""
import argparse, random, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from datasets import load_dataset                                                            # noqa: E402
from systemone.composition import canonical, check_group, generate as compose, sample_trees   # noqa: E402
from systemone.contrastive import FAMILIES                                                    # noqa: E402
from systemone.data import ALL_REPOS, ALL_SOURCES, build                                      # noqa: E402
from systemone.suite import digest, read_jsonl, write_json, write_jsonl                       # noqa: E402
sys.path.insert(0, str(ROOT / "scripts")); from build_delta_data import legacy, HELD_OUT_FAMILIES   # noqa: E402


def pairs(config, n, seed, question, fields, source):
    ds = load_dataset("nyu-mll/glue", config, split="train"); rng = random.Random(seed)
    out = []
    for i in rng.sample(range(len(ds)), n):
        row = ds[i]
        if row["label"] not in (0, 1): continue
        state = {fields[0]: row[fields[0]], fields[1]: row[fields[1]]}
        out.append({"state": state, "questions": {"same": {"type": "noul", "instructions": question, "label": bool(row["label"]), "src": source}},
                    "_meta": {"source": source, "id": f"{source}/train/{i}", "row": i, "split": "train"}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="delta-v2-20260925"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mcq = build(1000, "train", 2, sources={k: ALL_SOURCES[k] for k in ("arc", "openbookqa", "csqa")}, repos=ALL_REPOS)
    para = pairs("qqp", 1200, a.seed, "Do these two questions ask the same thing?", ("question1", "question2"), "qqp") \
         + pairs("mrpc", 800, a.seed, "Do these two sentences mean the same thing?", ("sentence1", "sentence2"), "mrpc")
    trees = {f"rand:{canonical(t)}": t for t in sample_trees(200, a.seed)}
    comp = compose(4, a.seed, styles=(0, 1), trees=trees)
    for i in range(0, len(comp), 4): check_group(comp[i:i + 4])
    leg = legacy(30, a.seed, [f for f in FAMILIES if f not in HELD_OUT_FAMILIES])
    dates = [r for f in ("dates", "unknowable", "assertion") for r in read_jsonl(ROOT / "evals/dates-v2" / f"{f}.jsonl")]
    records = mcq + para + comp + leg + dates
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", records)
    counts = Counter(r["_meta"]["source"] for r in records)
    write_json(out / "manifest.json", {"seed": a.seed, "sources": dict(counts), "files": {"train.jsonl": {"sha256": digest(out / "train.jsonl"), "records": len(records)}}})
    print(f"{len(records)} records -> {out}/train.jsonl", dict(counts))


if __name__ == "__main__":
    main()
