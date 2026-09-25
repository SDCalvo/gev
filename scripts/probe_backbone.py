"""Load the backbone with LoRA on this machine, check what got adapted, and time forward + backward on real records.

    uv run python scripts/probe_backbone.py --n 12 --weights_dtype bf16

Prints: backbone class and layer mix, trainable parameter count and the LoRA'd module names, then per-record wall time
(forward, forward+backward) and peak device memory on --n development records of the smoke suite / decision-v7, which
is what a training run would see. Use the numbers to size a run before starting it.
"""
import argparse, statistics, time
from pathlib import Path
import torch
from systemone.checkpoint import Meta  # noqa: F401  (import check)
from systemone.data import materialize
from systemone.device import allocated_bytes, default_device, empty_cache, sync
from systemone.model import DecisionModel, load_tokenizer, training_context
from systemone.suite import load_split
from systemone.train import question_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E4B")
    ap.add_argument("--suite", default="evals/v7/decision-v7")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--weights_dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or default_device()
    tok = load_tokenizer(a.base)
    t0 = time.time()
    model = DecisionModel(a.base, tok, dev, lora=a.lora, dtype=torch.bfloat16 if a.weights_dtype == "bf16" else torch.float32)
    print(f"loaded in {time.time() - t0:.0f}s on {dev}: {type(model.lm.get_base_model() if hasattr(model.lm, 'get_base_model') else model.lm).__name__}, "
          f"hidden {model.lm.config.hidden_size}, layers {model.lm.config.num_hidden_layers}, rows_only={model.rows_only}, hybrid={model.hybrid}")
    trainable = sum(p.numel() for p in model.trainable_parameters())
    lora_modules = sorted({n.split(".")[-1] for n, m in model.lm.named_modules() if hasattr(m, "lora_A")})
    print(f"trainable params {trainable / 1e6:.1f}M; LoRA on {lora_modules}")
    print(f"device memory after load: {allocated_bytes(dev) / 1e9:.1f} GB")

    recs = [materialize(r) for r in load_split(a.suite, "development")[: a.n]]
    ctx = training_context()
    encs = [model.encode(tok, r, strict=True, max_state=ctx["max_state"], max_branch=ctx["max_branch"]) for r in recs]
    print(f"{len(encs)} records, {statistics.mean(len(e['ids']) for e in encs):.0f} packed tokens on average, {sum(len(e['decide_idx']) for e in encs)} questions")

    model.train()
    fwd, both, peak = [], [], 0
    for i, (rec, enc) in enumerate(zip(recs, encs)):
        sync(dev); t = time.perf_counter()
        logits = model.forward_batch([enc])[0]
        sync(dev); f = time.perf_counter() - t
        loss = sum(question_loss(z.float(), q, dev, 0.0) for z, q in zip(logits, rec["questions"])) / len(logits)
        loss.backward()
        sync(dev); b = time.perf_counter() - t
        if i == 0:
            assert all(torch.isfinite(z).all() for z in logits), "non-finite logits"
            print(f"first record: loss {loss.item():.3f}, logits {[z.detach().float().cpu().numpy().round(2).tolist() for z in logits]}")
        peak = max(peak, allocated_bytes(dev))
        model.zero_grad(set_to_none=True)
        if dev == "mps": empty_cache(dev)
        fwd.append(f); both.append(b)
        print(f"  record {i}: {len(enc['ids'])} tokens, forward {f:.2f}s, forward+backward {b:.2f}s", flush=True)
    warm_f, warm_b = fwd[1:] or fwd, both[1:] or both
    print(f"\nper record (excluding the first): forward {statistics.median(warm_f):.2f}s median, forward+backward {statistics.median(warm_b):.2f}s median, "
          f"{statistics.mean(warm_b):.2f}s mean; peak device memory {peak / 1e9:.1f} GB")
    n_v7 = 12576
    print(f"decision-v7 estimate at this rate: {n_v7 * statistics.mean(warm_b) / 3600:.1f} h per epoch")


if __name__ == "__main__":
    main()
