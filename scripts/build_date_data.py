"""Training records for the 2026-09-20 overnight deltas (PLAN.md, "Tonight's autoresearch"). Three files under evals/night2/,
each in the --data JSONL format (systemone.data.load_records), plus a manifest with hashes. Frozen: rerunning must reproduce the bytes.

  dates.jsonl       date-bearing policy families (trainable ones; `deadline` stays held out) rendered three ways at random:
                    plain (as decision-v7), with a relational day-count sentence that names the roles ("The return request was
                    submitted 23 days after the purchase."), or with a `date_facts` field identical to what systemone.api.with_date_facts
                    produces at serving time. Teaches the model to consume a stated day count and to bind a generic date fact to
                    the policy's roles (issue #8).
  unknowable.jsonl  contrastive records with the deciding evidence sentence removed, soft target = uniform over the options, plus
                    their intact controls with hard labels. Trainable families only. Teaches "no evidence -> no confidence".
  assertion.jsonl   Noul questions phrased as statements ("This article is about business.") built from public choice/noul records
                    in decision-v7's training partition, balanced true/false. scienthoon's "The customer sounds angry." failure.

    uv run python scripts/build_night2_data.py
"""
import hashlib, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from systemone import contrastive                              # noqa: E402
from systemone.api import date_facts, question_keys, render    # noqa: E402
from systemone.suite import digest, load_split, write_json     # noqa: E402
from systemone.transfer_v9 import unknowable                   # noqa: E402

import os
OUT = ROOT / os.environ.get("DATES_OUT", "evals/night2")
SEED = os.environ.get("DATES_SEED", "night2-20260920")
DATE_FAMILIES = ("return_window", "warranty_claim", "shipping_delay")          # trainable families whose evidence is two absolute dates
UNKNOWABLE_FAMILIES = [f for f in contrastive.FAMILIES if f not in ("deadline", "authorization")]   # held-out families stay out of training
ROLE = {"purchase": "purchase", "request": "return request", "claim": "warranty claim", "promised": "promised delivery", "delivered": "delivery"}
STATEMENTS = {"Would this reviewer recommend the business?": "This reviewer recommends the business.", "Is this movie review positive?": "This movie review is positive.",
              "Is this article about science and technology?": "This article is about science and technology.", "Is this article about world news?": "This article is about world news.",
              "Is this article about sports?": "This article is about sports.", "Is this article about business?": "This article is about business."}
ASSERTION_TEMPLATES = ["This text is about {x}.", "The right category for this is {x}.", "This should be filed under {x}.", "The topic here is {x}."]


def relational_sentence(item):
    facts = {}
    for _, f in item["sentences"]: facts.update(f)
    dates = [(k, v) for k, v in facts.items() if hasattr(v, "toordinal")]
    if len(dates) != 2: return None
    (ka, a), (kb, b) = sorted(dates, key=lambda kv: kv[1])
    n = (b - a).days
    return f"The {ROLE.get(kb, kb)} date was {n} day{'s' if n != 1 else ''} after the {ROLE.get(ka, ka)} date."


def dates_file(pairs_per_family):
    out = []
    for family in DATE_FAMILIES:
        rng = random.Random(f"{SEED}:dates:{family}"); kept, attempts = 0, 0
        while kept < pairs_per_family and attempts < 50 * pairs_per_family:
            attempts += 1
            a, b = contrastive.FAMILIES[family](rng)
            if contrastive.check_pair(a, b): continue
            pair_id = f"{SEED}-{family}-{kept:04d}"; order_seed = rng.getrandbits(64); style = rng.choice(["plain", "relational", "facts"])
            for sibling, item in (("a", a), ("b", b)):
                if style == "relational":
                    sent = relational_sentence(item)
                    item = {**item, "sentences": item["sentences"] + [(sent, {})]} if sent else item
                req = contrastive.to_request(item, family, pair_id, sibling, random.Random(order_seed))
                if style == "facts":
                    facts = date_facts(render(req["state"]))
                    if facts: req["state"] = {**req["state"], "date_facts": facts}
                req["_meta"].update(source="night2_dates", rendering=style, id=f"night2_dates/{family}/{pair_id}/{sibling}", group_id=f"night2_dates/{family}/{pair_id}")
                out.append(req)
            kept += 1
        if kept < pairs_per_family: raise ValueError(f"{family}: only {kept} pairs")
    return out


def unknowable_file(pairs_per_family):
    saved = contrastive.FAMILIES
    try:
        contrastive.FAMILIES = {k: v for k, v in saved.items() if k in UNKNOWABLE_FAMILIES}
        recs = unknowable(pairs_per_family, f"{SEED}:unknowable")
    finally:
        contrastive.FAMILIES = saved
    for r in recs:
        if r["_meta"]["source"] == "unknowable":
            for q in r["questions"].values():
                keys = question_keys(q["type"], q.get("criteria"))
                q["target"] = {k: 1.0 / len(keys) for k in keys}
            r["_meta"]["source"] = "night2_unknowable"
        else:
            r["_meta"]["source"] = "night2_unknowable_control"
        r["_meta"]["id"] = "night2_" + r["_meta"]["id"]; r["_meta"]["group_id"] = "night2_" + r["_meta"]["group_id"]
    return recs


def assertion_file(n):
    pool = [r for r in load_split(ROOT / "evals/v7/decision-v7", "train") if r["_meta"]["source"] in ("agnews", "dbpedia14", "trec", "banking77", "yelp", "imdb", "boolq")]
    rng = random.Random(f"{SEED}:assertion"); rng.shuffle(pool); out = []
    for r in pool:
        if len(out) >= n: break
        qs = {}
        for qid, q in r["questions"].items():
            if q["type"] == "choice" and len(q["criteria"]) >= 3:
                truth = q["label"]; want_true = rng.random() < 0.5
                key = truth if want_true else rng.choice([k for k in q["criteria"] if k != truth])
                desc = q["criteria"].get(key) or key.replace("_", " ")
                if not isinstance(desc, str): continue                      # structured option descriptions do not read as a statement
                desc = desc.rstrip("?. ")
                qs[f"{qid}_is"] = {"type": "noul", "instructions": rng.choice(ASSERTION_TEMPLATES).format(x=desc), "label": key == truth, "src": f"night2_assertion_{r['_meta']['source']}"}
            elif q["type"] == "noul" and isinstance(q["instructions"], str):
                stmt = STATEMENTS.get(q["instructions"])
                if stmt: qs[f"{qid}_stmt"] = {"type": "noul", "instructions": stmt, "label": q["label"], "src": f"night2_assertion_{r['_meta']['source']}"}
        if qs:
            out.append({"state": r["state"], "questions": qs, "_meta": {**r["_meta"], "source": "night2_assertion", "id": "night2_assertion/" + r["_meta"]["id"], "group_id": "night2_assertion/" + r["_meta"]["id"]}})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = {r["_meta"]["text_sha256"] for split in ("development", "test") for suite in ("evals/v4/transfer-v4", "evals/v7/decision-v7") for r in load_split(ROOT / suite, split, allow_test=True)}
    files = {"dates.jsonl": dates_file(int(os.environ.get("DATES_N", "150"))), "unknowable.jsonl": unknowable_file(int(os.environ.get("UNKNOWABLE_N", "15"))), "assertion.jsonl": assertion_file(int(os.environ.get("ASSERTION_N", "800")))}
    manifest = {"seed": SEED, "files": {}}
    for name, recs in files.items():
        before = len(recs); recs = [r for r in recs if r["_meta"].get("text_sha256") not in dev]   # never train on an evaluation state
        body = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in recs)   # default=str for date objects: not write_jsonl
        (OUT / name).write_text(body, encoding="utf-8")
        manifest["files"][name] = {"records": len(recs), "dropped_eval_overlap": before - len(recs), "sha256": digest(OUT / name),
                                   "sources": sorted({r["_meta"]["source"] for r in recs})}
        print(name, manifest["files"][name])
    write_json(OUT / "manifest.json", manifest)


if __name__ == "__main__":
    main()
