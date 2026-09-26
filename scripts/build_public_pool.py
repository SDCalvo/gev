"""A larger public training pool than the frozen suite's 1,000 records per source, in the --data JSONL format.

    uv run python scripts/build_public_pool.py --n_per_source 2000 --seed 1 --out evals/public-pool-2k

Same ten public sources, converters and dataset revisions as decision-v7's training partition (train splits only, so
nothing overlaps the development/test partitions, which come from the datasets' test splits); a different seed, so the
sample differs from the suite's. Train with:
    --data evals/public-pool-2k/train.jsonl --suite evals/v7/decision-v7 --replay 12576   (replay = the whole suite partition)
"""
import argparse, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from gev.data import ALL_REPOS, ALL_SOURCES, build          # noqa: E402
from gev.suite import digest, read_manifest, write_json, write_jsonl   # noqa: E402

PUBLIC = ("agnews", "amazon", "banking77", "boolq", "dbpedia14", "imdb", "mnli", "sst5", "trec", "yelp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_source", type=int, default=2000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--suite", default="evals/v7/decision-v7", help="manifest whose dataset revisions to pin")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    revisions = read_manifest(a.suite).get("dataset_revisions", {})
    records = build(a.n_per_source, "train", a.seed, sources={k: ALL_SOURCES[k] for k in PUBLIC}, repos=ALL_REPOS, revisions=revisions)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", records)
    counts = Counter(r["_meta"]["source"] for r in records)
    write_json(out / "manifest.json", {"n_per_source": a.n_per_source, "seed": a.seed, "sources": dict(counts), "dataset_revisions": revisions,
                                      "files": {"train.jsonl": {"sha256": digest(out / "train.jsonl"), "records": len(records)}}})
    print(f"{len(records)} records -> {out}/train.jsonl", dict(counts))


if __name__ == "__main__":
    main()
