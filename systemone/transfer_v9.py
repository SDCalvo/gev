"""transfer-v9 (the plan draft called it transfer-v5; v5-v8 are taken by decision-data versions): transfer-v4 byte-for-byte, plus three eval-only additions (PLAN.md, "Qwen3.5 port" Phase 0).

    uv run python -m systemone.transfer_v9 --out evals/v9/transfer-v9

Additions, each partition (development, test):
  mmlu_pro            200 items from TIGER-Lab/MMLU-Pro (10-way Choice, revision pinned). Replaces the saturated 4-way MMLU as
                      the knowledge column; the 4-way items stay so v4 numbers remain a subset.
  buried              80 records from the public sources whose state is a string (paws, qnli, tweet_offensive, emotion), with
                      the state embedded among three unrelated records from the same source. Same label; tests whether the
                      model reads the primary record (SemIf "irrelevant context", localjev distraction condition).
  unknowable          90 programmatic policy records with the deciding evidence sentence removed, so the label is not
                      recoverable from the text (contrastive.label_of(item, drop=i) == UNDETERMINED). Scored on confidence,
                      not accuracy: a model that knows it does not know puts low probability on every option. Their intact
                      siblings are frozen alongside as unknowable_control so the contrast is paired.

v4 items keep their ids and bytes; v5 minus the new sources equals v4. Development and test are built with disjoint seeds.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

from huggingface_hub import HfApi

from . import contrastive
from .data import materialize
from .model import fits, load_tokenizer
from .suite import ENCODING, digest, read_jsonl, read_manifest, record_digest, write_json, write_jsonl

PARENT = Path("evals/v4/transfer-v4")
MMLU_PRO = "TIGER-Lab/MMLU-Pro"
BURIED_SOURCES = ("paws", "qnli", "tweet_offensive", "emotion")
QWEN35 = {"Qwen/Qwen3.5-4B-Base": "1001bb4d826a52d1f399e183466143f4da7b741b", "Qwen/Qwen3.5-9B-Base": "68c46c4b3498877f3ef123c856ecfde50c39f404",
          "Qwen/Qwen3.5-0.8B-Base": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"}


def mmlu_pro(n, seed, revision, tokenizers, exclude_ids=()):
    from datasets import load_dataset
    ds = load_dataset(MMLU_PRO, split="test", revision=revision)
    rng = random.Random(f"{seed}:mmlu_pro"); out = []
    for i in rng.sample(range(len(ds)), 4 * n):
        ex = ds[i]
        options = [o for o in ex["options"] if o and o != "N/A"]
        if not 4 <= len(options) <= 10 or ex["answer_index"] >= len(options) or f"mmlu_pro/test/{ex['question_id']}" in exclude_ids: continue
        keys = "abcdefghij"[: len(options)]
        text = " ".join(ex["question"].casefold().split())
        out.append({"state": {"category": ex["category"], "question": ex["question"]},
                    "questions": {"answer": {"type": "choice", "instructions": "Which option correctly answers the question?",
                                             "criteria": dict(zip(keys, options)), "label": keys[ex["answer_index"]], "src": "mmlu_pro"}},
                    "_meta": {"row": ex["question_id"], "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "source": "mmlu_pro", "repo": MMLU_PRO,
                              "revision": revision, "split": "test", "id": f"mmlu_pro/test/{ex['question_id']}", "group_id": f"mmlu_pro/test/{ex['question_id']}", "variant": "clean"}})
        out[-1]["_meta"]["row_sha256"] = record_digest({k: v for k, v in out[-1].items() if k != "_meta"})
        if not fits(materialize(out[-1]), *tokenizers): out.pop(); continue          # same context rule as every frozen suite
        if len(out) == n: break
    if len(out) < n: raise ValueError(f"mmlu_pro: only {len(out)}/{n} usable")
    return out


def buried(records, n, seed):
    """Embed a record's string state among three unrelated same-source states. The question is unchanged."""
    rng = random.Random(f"{seed}:buried"); out = []
    by_source = {s: [r for r in records if r["_meta"]["source"] == s and isinstance(r["state"], str)] for s in BURIED_SOURCES}
    per = -(-n // len(BURIED_SOURCES))
    for source, pool in by_source.items():
        for r in rng.sample(pool, min(per, len(pool))):
            others = rng.sample([o for o in pool if o["_meta"]["id"] != r["_meta"]["id"]], 3)
            slot = rng.randrange(4)
            background = [o["state"] for o in others]
            rec = copy.deepcopy(r)
            rec["state"] = {"records": background[:slot] + [r["state"]] + background[slot:], "primary_record": slot + 1,
                            "note": "Answer about the primary record only; the other records are unrelated."}
            for q in rec["questions"].values(): q["src"] = f"buried_{q['src']}"
            rec["_meta"] = {**r["_meta"], "source": "buried", "variant": "clean", "parent_id": r["_meta"]["id"], "parent_source": source,
                            "id": f"buried/{r['_meta']['id']}", "group_id": f"buried/{r['_meta']['id']}", "primary_slot": slot,
                            "text_sha256": hashlib.sha256(json.dumps(rec["state"], sort_keys=True).encode()).hexdigest()}
            rec["_meta"]["row_sha256"] = record_digest({k: v for k, v in rec.items() if k != "_meta"})
            out.append(rec)
    return out[:n]


def unknowable(pairs_per_family, seed):
    """For every contrastive record, the version with its deciding evidence sentence dropped (label UNDETERMINED by the
    family's own evaluate) plus the intact control. The frozen label on the unknowable record is the *intact* label, kept
    only so the record validates; accuracy on it is meaningless by construction, confidence is what is scored."""
    out = []
    for family in contrastive.FAMILIES:
        rng = random.Random(f"{seed}:{family}"); kept, attempts = 0, 0
        while kept < pairs_per_family and attempts < 50 * pairs_per_family:
            attempts += 1
            a, b = contrastive.FAMILIES[family](rng)
            if contrastive.check_pair(a, b): continue
            pair_id = f"{seed}-{family}-{kept:04d}"; order_seed = rng.getrandbits(64)
            for sibling, item in (("a", a), ("b", b)):
                control = contrastive.to_request(item, family, pair_id, sibling, random.Random(order_seed))
                drop = next(i for i, (_, facts) in enumerate(item["sentences"]) if facts and contrastive.label_of(item, drop=i) == contrastive.UNDETERMINED)
                stripped = {**item, "sentences": [s for i, s in enumerate(item["sentences"]) if i != drop]}
                unk = contrastive.to_request(stripped, family, pair_id, sibling, random.Random(order_seed))
                for q in unk["questions"].values(): q["label"] = control["questions"]["decision"]["label"]; q["src"] = f"unknowable_{family}"
                for q in control["questions"].values(): q["src"] = f"unknowable_control_{family}"
                unk["_meta"].update(source="unknowable", variant="clean", dropped_sentence=item["sentences"][drop][0], intact_label=control["questions"]["decision"]["label"],
                                    id=f"unknowable/{family}/{pair_id}/{sibling}", group_id=f"unknowable/{family}/{pair_id}", control_id=f"unknowable_control/{family}/{pair_id}/{sibling}")
                unk["_meta"].pop("pair_id", None); unk["_meta"].pop("sibling", None)
                control["_meta"].update(source="unknowable_control", variant="clean", id=f"unknowable_control/{family}/{pair_id}/{sibling}", group_id=f"unknowable_control/{family}/{pair_id}")
                control["_meta"].pop("pair_id", None); control["_meta"].pop("sibling", None)
                for r in (unk, control): r["_meta"]["row_sha256"] = record_digest({k: v for k, v in r.items() if k != "_meta"})
                out += [unk, control]
            kept += 1
        if kept < pairs_per_family: raise ValueError(f"{family}: only {kept} pairs")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evals/v9/transfer-v9")
    ap.add_argument("--parent", default=str(PARENT))
    ap.add_argument("--mmlu_pro", type=int, default=200)
    ap.add_argument("--buried", type=int, default=80)
    ap.add_argument("--unknowable_pairs", type=int, default=5, help="pairs per family per partition (x2 siblings, x2 unknowable/control)")
    a = ap.parse_args()
    out, parent = Path(a.out), Path(a.parent)
    if out.exists(): raise FileExistsError(out)
    pm = read_manifest(parent)
    revision = HfApi().dataset_info(MMLU_PRO).sha
    tokenizers = [load_tokenizer("Qwen/Qwen3-4B-Base", revision=pm["base_revisions"].get("Qwen/Qwen3-4B-Base")), load_tokenizer("Qwen/Qwen3.5-4B-Base", revision=QWEN35["Qwen/Qwen3.5-4B-Base"])]
    files, counts = {}, {}
    out.mkdir(parents=True)
    used = set()
    for split, seed in (("development", "v5-dev-20260920"), ("test", "v5-test-20260920")):
        base = read_jsonl(parent / f"{split}.jsonl")
        assert digest(parent / f"{split}.jsonl") == pm["files"][f"{split}.jsonl"]["sha256"]
        mp = mmlu_pro(a.mmlu_pro, seed, revision, tokenizers, exclude_ids=used); used |= {r["_meta"]["id"] for r in mp}
        records = base + mp + buried(base, a.buried, seed) + unknowable(a.unknowable_pairs, seed)
        ids = [r["_meta"]["id"] for r in records]
        if len(ids) != len(set(ids)): raise ValueError("duplicate ids")
        path = out / f"{split}.jsonl"
        write_jsonl(path, records)
        files[path.name] = {"sha256": digest(path), "records": len(records)}
        counts[split] = {s: sum(r["_meta"]["source"] == s for r in records) for s in sorted({r["_meta"]["source"] for r in records})}
    for name in ("train.jsonl", "calibration.jsonl"):
        (out / name).write_text("", encoding=ENCODING); files[name] = {"sha256": digest(out / name), "records": 0}
    manifest = {"version": 5, "parent": str(parent), "parent_files": pm["files"], "base_revisions": {**pm["base_revisions"], **QWEN35},
                "dataset_revisions": {**pm.get("dataset_revisions", {}), MMLU_PRO: revision}, "holdout_sources": pm["holdout_sources"], "trainable_sources": [],
                "eval_only_sources": pm["eval_only_sources"] + ["mmlu_pro", "buried", "unknowable", "unknowable_control"], "context": pm["context"],
                "protocol": {**pm.get("protocol", {}), "v5": "v4 items byte-identical (same ids); mmlu_pro 10-way knowledge; buried = state among three unrelated same-source records; "
                             "unknowable = deciding evidence removed, scored on confidence (mean max-probability, share >= 0.9) against the intact control"},
                "files": files, "counts": counts, "eval_only": True, "code_hashes": {p.name: digest(p) for p in [Path(__file__), Path("kev/contrastive.py"), Path("kev/data.py")]}}
    write_json(out / "manifest.json", manifest)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
