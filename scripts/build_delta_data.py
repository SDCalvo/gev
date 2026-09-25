"""Targeted data for a delta fine-tune of a trained checkpoint (--data JSONL), aimed at the weakest sources of the first
Gemma model: compositional policy rules, date arithmetic, over-confidence on unknowable cases, statement-phrased Noul.

    uv run python scripts/build_delta_data.py --out evals/delta-v1

Contents: Kev's frozen night-2 files (evals/night2: 900 date-bearing policy cases rendered three ways, 525 unknowable
cases with uniform targets plus intact controls, 800 assertion-phrased Noul questions), plus fresh generated records from
our own generators with a new seed: compositional rules over 60 new random structures (held-out structures excluded by
sample_trees) in the two training rendering styles, and contrastive minimal pairs from the trainable policy families
(deadline and authorization stay held out). Train with:
    --data evals/delta-v1/train.jsonl --suite evals/v7/decision-v7 --replay 2000 --init_from runs/<run> --lr 2e-5 --epochs 1
"""
import argparse, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from systemone.composition import canonical, check_group, generate as compose, sample_trees   # noqa: E402
from systemone.contrastive import FAMILIES, generate as contrast                              # noqa: E402
from systemone.suite import digest, read_jsonl, semantic_hash, write_json, write_jsonl        # noqa: E402

HELD_OUT_FAMILIES = ("deadline", "authorization")


def legacy(pairs, seed, families, source="legacy_policy"):
    """Kev's study_v3.legacy: `pairs` unique contrastive minimal pairs per family."""
    candidates, _ = contrast(3 * pairs, seed, families)
    records, seen, counts = [], set(), Counter()
    for a, b in zip(candidates[::2], candidates[1::2]):
        family = a["_meta"]["family"]; hashes = {semantic_hash(a), semantic_hash(b)}
        if counts[family] >= pairs or hashes & seen: continue
        for r in (a, b):
            m = r["_meta"]; m.update(source=source, group_id=m["pair_id"], variant="clean"); m["text_sha256"] = semantic_hash(r)
        records.extend((a, b)); seen.update(hashes); counts[family] += 1
    if any(counts[f] != pairs for f in families): raise ValueError(f"insufficient unique pairs: {dict(counts)}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="delta-v1-20260924"); ap.add_argument("--structures", type=int, default=60)
    ap.add_argument("--groups_per_structure", type=int, default=4); ap.add_argument("--pairs_per_family", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trees = {f"rand:{canonical(t)}": t for t in sample_trees(a.structures, a.seed)}
    comp = compose(a.groups_per_structure, a.seed, styles=(0, 1), trees=trees)
    for i in range(0, len(comp), 4): check_group(comp[i:i + 4])
    families = [f for f in FAMILIES if f not in HELD_OUT_FAMILIES]
    leg = legacy(a.pairs_per_family, a.seed, families)
    night2 = [r for f in ("dates", "unknowable", "assertion") for r in read_jsonl(ROOT / "evals/night2" / f"{f}.jsonl")]
    records = night2 + comp + leg
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", records)
    counts = Counter(r["_meta"]["source"] for r in records)
    write_json(out / "manifest.json", {"seed": a.seed, "structures": a.structures, "sources": dict(counts),
                                      "files": {"train.jsonl": {"sha256": digest(out / "train.jsonl"), "records": len(records)}}})
    print(f"{len(records)} records -> {out}/train.jsonl", dict(counts))


if __name__ == "__main__":
    main()
