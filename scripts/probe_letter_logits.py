"""Zero-shot letter-logit accuracy of the raw backbone on a suite partition's Choice questions (K <= 8): the knowledge
ceiling the pointer head could reach. Prompt: state, question, lettered options, "Answer:"; the answer is the letter
with the highest next-token logit.

    uv run python scripts/probe_letter_logits.py --suite evals/v4/transfer-v4 --sources mmlu,sciq,emotion
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
from systemone.data import materialize                         # noqa: E402
from systemone.suite import load_split                         # noqa: E402

LETTERS = "ABCDEFGH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E4B"); ap.add_argument("--suite", default="evals/v4/transfer-v4")
    ap.add_argument("--split", default="development"); ap.add_argument("--sources", default="mmlu,sciq,emotion,paws,qnli")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to("mps").eval()
    letter_ids = [tok(f" {L}", add_special_tokens=False).input_ids[-1] for L in LETTERS]
    by = defaultdict(lambda: [0, 0])
    for r in load_split(a.suite, a.split):
        src = r["_meta"]["source"]
        if src not in a.sources.split(","): continue
        rec = materialize(r)
        for q in rec["questions"]:
            K = len(q["options"])
            if K > len(LETTERS): continue
            opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
            prompt = f"{rec['state']}\n\nQuestion: {q['instr']}\n{opts}\nAnswer:"
            ids = tok(prompt, return_tensors="pt").input_ids.to("mps")
            with torch.no_grad():
                logits = model(input_ids=ids).logits[0, -1].float()
            pred = int(torch.argmax(logits[letter_ids[:K]]))
            by[src][0] += int(pred == q["label"]); by[src][1] += 1
    for src, (ok, n) in sorted(by.items()):
        print(f"{src:12} n={n:3} letter-logit acc {ok / n:.3f}")


if __name__ == "__main__":
    main()
