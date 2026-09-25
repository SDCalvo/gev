"""Decision model: causal LM backbone + block-causal branch mask (or one causal row per question) + pointer readout.

Derived from Kev (github.com/jaredpalmer/kev, Apache-2.0); ported to Google's Gemma 4 (Apache-2.0) as the backbone.
"""
import copy, functools, math, os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Existing single-token, never-used vocabulary entries serve as the five delimiters (state, q, opt, /opt, decide), so
# no embedding rows need to be added or trained; LoRA adapts their meaning. Gemma tokenizers reserve thousands of
# `<unusedN>` tokens (ids 6.. on Gemma 4); Qwen's rarely used FIM/box tokens are kept so Kev checkpoints still encode.
DELIMITERS = {"gemma": ["<unused0>", "<unused1>", "<unused2>", "<unused3>", "<unused4>"],
              "qwen": ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]}
SPECIAL = DELIMITERS["gemma"]   # the default backbone's set; encode() resolves the right one per tokenizer (delimiters)


def delimiters(tok):
    """The delimiter token names this tokenizer has as single tokens (the first matching set in DELIMITERS)."""
    for names in DELIMITERS.values():
        ids = tok.convert_tokens_to_ids(names)
        if all(isinstance(i, int) and i >= 0 and i != tok.unk_token_id for i in ids) and len(set(ids)) == len(ids):
            return names
    raise ValueError("tokenizer has none of the known delimiter token sets")
# training context: state tokens, tokens per question branch, and the whole packed record. Frozen suites are admitted with
# this rule (systemone.suite) and training applies it to records built on the fly, so train and eval see the same population.
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
# serving context (systemone.serve): per-branch cap mirrors Jev's ~32k, bounded by the base model window; longer than training, so untested there
SERVE_MAX_STATE, SERVE_MAX_BRANCH = 8192, 8192
SERVE_MAX_PACKED = SERVE_MAX_STATE + SERVE_MAX_BRANCH   # one row at most: a packed request longer than this runs in the row form (the block-causal mask is L x L)
# the longest state a checkpoint may be trained on (systemone.train --max_state) and still leave every question its training
# branch budget when served: serving's row limit is SERVE_MAX_BRANCH = state + branch
MAX_TRAIN_STATE = SERVE_MAX_BRANCH - (MAX_BRANCH - MAX_STATE)


def training_context(max_state=MAX_STATE):
    """The encoder limits for training with the state limit lifted to `max_state`: the row (state + one branch) and
    packed limits grow by the same amount, so every question keeps its token budget. training_context() is the default
    training context (systemone.suite.CONTEXT without `truncate`)."""
    if not MAX_STATE <= max_state <= MAX_TRAIN_STATE:
        raise ValueError(f"max_state must be in [{MAX_STATE}, {MAX_TRAIN_STATE}]")
    extra = max_state - MAX_STATE
    return {"max_state": max_state, "max_branch": MAX_BRANCH + extra, "max_packed": MAX_PACKED + extra}


def rows_per_pass(rows, prefix_len=0, budget=SERVE_MAX_PACKED):
    """How many causal rows one inference forward pass takes: as many as fit `budget` tokens counting the cached state
    each row carries (prefix_len) plus its own tokens, at least one. Memory per pass is bounded by one maximal row however
    many questions a request has, and the rows are independent, so the answers do not depend on the split. Kev-4B on MLX,
    a 4.8k-token state with 64 questions: 24.7 GB peak in one pass, 8.9 GB one row at a time (and faster: the cost was
    64 copies of the state cache)."""
    return max(1, budget // (prefix_len + max(len(r) for r in rows)))


class ContextOverflow(ValueError):
    """A record does not encode within its context (state, branch or packed limit). Serving turns it into a 422; the
    benchmark counts it as a rejected record for suites scored as published (skip_overlong)."""


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


def pad_id(tok):
    """The id used to right-pad token rows (never attended to); Qwen tokenizers define one, others fall back to 0."""
    return tok.pad_token_id if tok.pad_token_id is not None else 0


def is_hybrid(config):
    """Whether a (text) config has Gated DeltaNet layers (Qwen3.5). Such backbones cannot honour the block-causal mask and
    run the row form."""
    return "linear_attention" in set(getattr(config, "layer_types", None) or [])


def has_sliding_window(config):
    """Whether a (text) config has sliding-window attention layers (Gemma 4: 35 of 42 layers, window 512). A custom 4D
    mask would replace the window rather than intersect it, so such backbones run the row form too, where transformers
    builds the causal + sliding masks itself from the padding mask."""
    return "sliding_attention" in set(getattr(config, "layer_types", None) or [])


def rows_only(config):
    """Backbones that must run one causal row per question instead of the packed block-causal mask."""
    return is_hybrid(config) or has_sliding_window(config)


def text_config(name, revision=None):
    return AutoConfig.from_pretrained(name, revision=revision).get_text_config()


def load_backbone(name, revision=None, dtype=torch.float32, attn=None):
    """The text backbone without a vocab head (we never generate text). Multimodal checkpoints (Gemma 4 ships as
    Gemma4ForConditionalGeneration) contribute their language model alone; the vision/audio towers are dropped. Every
    text weight must load: a silently missing tensor would train and report a loss on random weights."""
    lm, info = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation=attn, output_loading_info=True)
    inner = lm.model
    backbone = getattr(inner, "language_model", inner)
    missing = [k for k in info.get("missing_keys", []) if "language_model" in k or not hasattr(inner, "language_model")]
    if missing:
        raise ValueError(f"{name}: {len(missing)} backbone tensors did not load (e.g. {missing[:3]})")
    return backbone


@functools.lru_cache(maxsize=8)
def _forbidden_re(special_tokens):
    """Every literal the tokenizer turns into a control token: its own special tokens (Gemma 4: <eos>, <turn|>, <|turn>,
    <mask>, ...), Qwen-style <|name|> markers and the <unusedN> delimiter names (Gemma's tokenizer already splits those
    in plain text, but the rewrite keeps the guarantee independent of tokenizer behaviour)."""
    literal = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    return re.compile(r"<\|([A-Za-z0-9_]+)\|>|<(unused\d+)>" + (rf"|({literal})" if literal else ""))


def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens (option boundaries are unforgeable).
    The fast tokenizer ignores split_special_tokens, so every such literal is rewritten to `<¦name¦>` before tokenizing."""
    rx = _forbidden_re(tuple(tok.all_special_tokens))
    return tok(rx.sub(lambda m: f"<¦{m.group(1) or m.group(2) or m.group(3).strip('<>|')}¦>", text), add_special_tokens=False).input_ids


OPT_NONE, OPT_DECIDE = -1, -2   # values of enc["opt"]: instruction/state tokens, and the <decide> token


READOUTS = ("delimiter", "content")


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False, option_isolation=False, readout="delimiter"):
    """Pack one record: [<state> ...] then per-question [<q> instr <opt> o </opt>... <decide>].

    Returns ids, seg (0 = state, k = question k), pos (branch positions restart after state),
    decide_idx [Q], opt_idx [Q][K] (the token the pointer head reads for each option), opt (per-token option index
    within its question: OPT_NONE for state/instruction, 0..K-1 for option spans, OPT_DECIDE for <decide>).

    readout: which token of each option span the head reads. "delimiter" = its </opt> token (Kev's choice: Qwen's
    delimiters are pretrained FIM tokens whose states summarize the span). "content" = the last token of the option
    text (the </opt> token when the text is empty). On Gemma 4 the delimiters are never-pretrained <unusedN> tokens and
    their final-layer states barely identify the option they follow (nearest-neighbour accuracy across option orders
    0.05 on Banking77 against chance 0.013), while the last content token's do (0.29): the first Gemma pilots trained
    on the delimiter readout stayed at chance on every Choice source.

    option_isolation=True: every option span is its own sub-branch (it sees state + instruction + itself only), all
    option spans share the same position ids, and <decide> sits at one fixed position after the longest span. Then the
    per-option representations and <decide>'s attention over them are permutation-invariant by construction.
    """
    special = delimiters(tok)
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ContextOverflow(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}")
    S = [tok.convert_tokens_to_ids(special[0])] + state_tokens[: max_state - 1]
    ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in special[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(S):
            raise ContextOverflow(f"branch too long: {len(br)} tokens with a {len(S)}-token state (row limit {max_branch})")
        base = len(ids); p0 = len(S)
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        if option_isolation:
            longest = max(len(sp) for sp in spans)
            br_pos = list(range(p0, p0 + len(instr))) + [p0 + len(instr) + i for sp in spans for i in range(len(sp))] + [p0 + len(instr) + longest]
        else:
            br_pos = list(range(p0, p0 + len(br)))
        if readout not in READOUTS: raise ValueError(f"readout must be one of {READOUTS}")
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp); ends.append(cursor - 1 - (1 if readout == "content" and len(sp) > 2 else 0))
        ids += br; seg += [k] * len(br); pos += br_pos; opt += br_opt
        decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt, "option_isolation": option_isolation, "readout": readout, "decide_idx": decide_idx, "opt_idx": opt_idx,
            "labels": [q["label"] for q in rec["questions"]], "state_truncated": len(state_tokens) + 1 > max_state}


def fits(rec, *tokenizers, max_state=MAX_STATE, max_branch=MAX_BRANCH, max_packed=MAX_PACKED):
    """True when the internal record encodes strictly (no truncation) within the training context under every tokenizer
    given (frozen suites are admitted against the tokenizers of all their pinned bases)."""
    try:
        return all(len(encode(tok, rec, max_state=max_state, max_branch=max_branch, strict=True)["ids"]) <= max_packed for tok in tokenizers)
    except ValueError:
        return False


def branch_mask(seg, device, dtype=torch.float32):
    """attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i]). Returns additive [1,1,L,L]."""
    return branch_mask_batch([seg], device, dtype)


def branch_mask_batch(segs, device, dtype=torch.float32, opts=None, length=None):
    """Batched block-causal mask, additive [B,1,L,L], right-padded to the longest sequence.

    Padded key positions are masked for every query; padded query rows keep the diagonal so no row is fully
    masked (finfo.min, not -inf, so softmax stays finite either way). Real tokens never see pads because pads sit
    after them (causal) and belong to no segment (-1).

    opts (option isolation): within a question, an option-span token may attend to state, the instruction, and its own
    span only; <decide> attends to everything in its question. Instruction tokens never see option spans (causal)."""
    L = max(max(len(s) for s in segs), length or 0)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    if opts is not None:
        o = torch.full((len(segs), L), OPT_NONE, device=device)
        for b, op in enumerate(opts):
            o[b, : len(op)] = torch.tensor(op, device=device)
        key_is_option = (o[:, None, :] >= 0)
        query_is_decide = (o[:, :, None] == OPT_DECIDE)
        same_option = o[:, None, :] == o[:, :, None]
        allow = allow & (~key_is_option | query_is_decide | same_option)
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def rows_of(enc):
    """Split a packed encoding into its state and per-question branch rows.

    Returns (state_ids, state_pos, rows) with rows[k] = {"ids", "pos", "decide", "opts"}: the branch tokens of question
    k with their (already state-continuing) positions, and the readout offsets *within the branch*. Feeding
    state + rows[k] as one causal row is equivalent to the packed block-causal form for that question, on any
    architecture: the row contains exactly the tokens question k may attend to, in the same positions."""
    seg = enc["seg"]; Ls = seg.count(0)
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1                                    # <decide> is the last token of its branch
        if seg[start] != k or seg[end - 1] != k: raise ValueError("branch layout mismatch")
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end], "decide": d - start, "opts": [o - start for o in oi]})
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


class PointerHead(nn.Module):
    def __init__(self, d, dp=256, norm=False):
        """dp = pointer dimension (head capacity knob). norm: layer-normalize (no affine) the hidden states before the
        projections. Gemma 4's final hidden states have norms in the hundreds (Qwen's are far smaller), so without it the
        untrained head starts at extreme logits and LoRA training finds a shortcut: shrink and homogenize the option
        states so every Choice question answers uniform (the first Gemma pilot collapsed exactly so: option-state cosine
        0.85 -> 0.99, norm 217 -> 54, Choice accuracy at chance). Normalized inputs make the head scale-invariant, so
        that shortcut no longer lowers the loss."""
        super().__init__()
        self.d, self.norm = d, norm
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        # calibration: logits are divided by this at inference (eval mode) only. 1.0 = raw. A checkpoint carries the value fitted on
        # its in-distribution development rows (scripts/calibrate_checkpoint.py -> head.pt["temperature"]); training always sees T=1 so
        # a fitted value stays meaningful, and the argmax is unchanged by construction.
        self.temperature = 1.0

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K]
        if self.norm:
            h_decide, h_opts = F.layer_norm(h_decide, (self.d,)), F.layer_norm(h_opts, (self.d,))
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature


# What a loaded model exposes to systemone.serve and systemone.predictors: the scoring interface (Kev's MLX backend
# implemented the same list; a second backend for this project would too).
SCORING_INTERFACE = ("encode", "forward", "probs", "probs_and_prefix", "probs_with_prefix", "eval",
                     "head", "backend", "dtype", "device", "hybrid", "option_isolation", "prefix_min_tokens")


class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256, option_isolation=False, special_embeddings=False, lora_targets="all", dtype=torch.float32, head_norm=False, readout="delimiter"):
        super().__init__()
        cfg = text_config(name, revision)
        # hybrid backbones (Qwen3.5: Gated DeltaNet layers, recurrent) cannot honour the block-causal mask, and
        # sliding-window backbones (Gemma 4) would lose their window under a custom 4D mask, so on both every question
        # runs as its own causal row continuing from the state (rows_of), under a plain padding mask. Attention-only
        # global-attention backbones keep the packed form; the two agree to fp32 noise (tests/test_model.py::test_rows_match_packed).
        self.hybrid, self.rows_only = is_hybrid(cfg), rows_only(cfg)
        # backbone only (no vocab head): we never generate text.
        # SDPA on CUDA (accepts arbitrary additive masks) and for row-form backbones anywhere (2D padding masks only);
        # eager on MPS/CPU for the packed float 4D mask (known-good there).
        attn = attn or ("sdpa" if str(device).startswith("cuda") or self.rows_only else "eager")
        # dtype: fp32 for training and exact evaluation; bf16 for the frozen backbone when memory is the limit (Gemma 4
        # E4B on a Mac: --weights_dtype bf16 keeps the 8B-parameter backbone at 16 GB; LoRA and the head stay fp32)
        self.lm = load_backbone(name, revision=revision, dtype=dtype, attn=attn)
        self.pad_id = pad_id(tok)
        if self.rows_only and option_isolation: raise ValueError("option_isolation needs the packed mask; not available on row-form backbones")
        self.option_isolation = option_isolation
        if readout not in READOUTS: raise ValueError(f"readout must be one of {READOUTS}")
        self.readout = readout
        if lora:
            from peft import LoraConfig, get_peft_model
            extra = {"trainable_token_indices": {"embed_tokens": [tok.convert_tokens_to_ids(t) for t in delimiters(tok)]}} if special_embeddings else {}
            targets = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       "dense": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],   # "all" minus the DeltaNet projections on hybrids (retention ablation)
                       "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "qv": ["q_proj", "v_proj"]}[lora_targets]
            if self.hybrid and lora_targets in ("all", "attn"):
                # Gated DeltaNet projections (transformers 5 names, verified on Qwen3_5TextModel); the mixer's out_proj too
                targets = targets + ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05, target_modules=targets, **extra)
            self.lm = get_peft_model(self.lm, cfg)
        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim, norm=head_norm)
        self.device = device
        self.to(device)

    backend = "torch"           # the only backend here (see checkpoint.LoadOptions)

    @property
    def prefix_min_tokens(self):
        """systemone.serve caches the state prefix from this many state tokens. Packed-form backbones: 384, below which the
        branch-only pass is not faster than one packed pass on MPS (per-op overhead). Row-form backbones: always, because
        their miss path otherwise recomputes the state once per question (Kev-0.8B bf16 on MPS, 5 questions: 1011 -> 413 ms)."""
        return 0 if self.rows_only else 384

    @property
    def dtype(self):
        return str(next(self.lm.parameters()).dtype).removeprefix("torch.")

    def encode(self, tok, rec, **kw):
        """encode() with this model's option-isolation and readout settings; use this from serving/eval code."""
        return encode(tok, rec, option_isolation=self.option_isolation, readout=self.readout, **kw)

    def hidden(self, enc):
        return self.hidden_batch([enc])[0, : len(enc["ids"])]

    SHAPE_BUCKET = int(os.environ.get("SYSTEMONE_SHAPE_BUCKET", "64"))   # MPS: pad the sequence to a multiple of this (per-shape kernel warm-up); 1 disables

    def _pad_rows(self, rows):
        """Right-pad (ids, pos) token rows into [N, L] id / position tensors and a [N, L] attention mask (1 = real token).
        Pads sit after every real token and are masked keys, so they never change a real token's hidden state (parity
        measured exact). On MPS in eval mode L is rounded up to a SHAPE_BUCKET multiple so kernels are warmed per bucket."""
        L = max(len(ids) for ids, _ in rows)
        if str(self.device) == "mps" and not self.training: L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        ids = torch.full((len(rows), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, (rid, rpos) in enumerate(rows):
            ids[i, : len(rid)] = torch.tensor(rid, device=self.device); pos[i, : len(rpos)] = torch.tensor(rpos, device=self.device); att[i, : len(rid)] = 1
        return ids, pos, att

    def hidden_batch(self, encs):
        """[B, L_max, d] hidden states for a right-padded batch of encoded records under the packed block-causal mask."""
        ids, pos, _ = self._pad_rows([(e["ids"], e["pos"]) for e in encs])
        isolate = any(e.get("option_isolation") for e in encs)
        if isolate and not all(e.get("option_isolation") for e in encs):
            raise ValueError("cannot mix option-isolated and plain encodings in one batch")
        lm_dtype = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([e["seg"] for e in encs], self.device, dtype=lm_dtype, opts=[e["opt"] for e in encs] if isolate else None, length=ids.shape[1])
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state.float()   # head stays fp32

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)]) for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    def rows_form(self, encs):
        """Whether these records run as causal rows: hybrid and sliding-window backbones always (see __init__), packed
        ones when the packed sequence would exceed one serving row (its L x L mask grows without bound with the number
        of questions). The two forms agree (tests/test_model.py::test_rows_match_packed)."""
        return self.rows_only or any(len(e["ids"]) > SERVE_MAX_PACKED for e in encs)

    def _rows_hidden(self, rows, cache=None, prefix_len=0):
        """Hidden states of causal token rows, one [L_i, d] tensor per row. In eval mode the rows go through the backbone
        rows_per_pass at a time; training keeps one batch (its batches are small and autograd needs the whole graph anyway).
        With `cache`, the rows are branches continuing the cached state: the cache is replicated once per chunk (a copy,
        so the caller's prefix stays pristine) and the cached tokens are marked real in the attention mask."""
        chunk = len(rows) if self.training else rows_per_pass([ids for ids, _ in rows], prefix_len)
        out = []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            ids, pos, att = self._pad_rows(part)
            past = {}
            if cache is not None:
                replica = copy.deepcopy(cache); replica.reorder_cache(torch.zeros(len(part), dtype=torch.long, device=self.device))
                att = torch.cat([torch.ones((len(part), prefix_len), dtype=torch.long, device=self.device), att], 1)
                past = {"past_key_values": replica, "use_cache": True}
            h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att, **past).last_hidden_state.float()
            out += [h[i, : len(row_ids)] for i, (row_ids, _) in enumerate(part)]
        return out

    def forward_rows_batch(self, encs):
        """Row form: every question of every record is one causal row = state tokens + its branch tokens. Returns the same
        nested logits as forward_batch. Exact isolation by construction (rows are independent); the state is recomputed
        per row (Q x state tokens), which training accepts; serving uses the prefix cache instead."""
        rows, readouts = [], []   # one causal row per question; readouts[i] = (record, <decide> offset, option offsets)
        for b, e in enumerate(encs):
            S, Sp, brs = rows_of(e)
            for r in brs:
                rows.append((S + r["ids"], Sp + r["pos"])); readouts.append((b, len(S) + r["decide"], [len(S) + o for o in r["opts"]]))
        out = [[] for _ in encs]
        for h, (b, d, oi) in zip(self._rows_hidden(rows), readouts):
            out[b].append(self.head(h[d], h[torch.tensor(oi, device=self.device)]))
        return out

    def forward(self, enc):
        """Returns list of logits tensors, one per question."""
        return self.forward_batch([enc])[0]

    def forward_batch(self, encs):
        """List (per record) of lists (per question) of logits. Row form or packed block-causal mask, see rows_form."""
        if self.rows_form(encs): return self.forward_rows_batch(encs)
        hs = self.hidden_batch(encs)
        return [self._readout(hs[b], e) for b, e in enumerate(encs)]

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    # --- state-prefix reuse (serving): the state is encoded once, question branches attend to its cached keys/values.
    # Exact by construction: branch tokens never attend to each other across questions (block-causal mask) and the state
    # never sees the branches (causal), so the state's hidden states and KV are identical with or without the branches.

    def _branch_rows_from_prefix(self, enc, cache):
        """Row-form serving: the branches run as causal rows continuing the cached state (exactly the forward_rows_batch
        layout, minus the recomputed state)."""
        S, _, rows = rows_of(enc)
        hs = self._rows_hidden([(r["ids"], r["pos"]) for r in rows], cache=cache, prefix_len=len(S))
        return [F.softmax(self.head(h[r["decide"]], h[torch.tensor(r["opts"], device=self.device)]), -1).cpu() for h, r in zip(hs, rows)]

    @torch.no_grad()
    def prefix(self, enc):
        """Run the state tokens only. Returns (n_state_tokens, kv cache, state hidden states [Ls, d])."""
        Ls = enc["seg"].count(0)
        ids = torch.tensor([enc["ids"][:Ls]], device=self.device); pos = torch.tensor([enc["pos"][:Ls]], device=self.device)
        # the cache must know the layer types (hybrid backbones keep recurrent + conv states per DeltaNet layer)
        out = self.lm(input_ids=ids, position_ids=pos, past_key_values=DynamicCache(config=self.lm.config), use_cache=True)
        return Ls, out.past_key_values, out.last_hidden_state[0].float()

    @torch.no_grad()
    def probs_and_prefix(self, enc):
        """One full pass that also returns the state prefix (KV cropped to the state, state hidden states): a cache miss
        costs a single forward pass, not two."""
        Ls = enc["seg"].count(0)
        if self.rows_form([enc]):
            # recurrent layers cannot be cropped back to the state (and an over-long packed pass is what the row form avoids),
            # so a miss here is a state pass (kept as the prefix) plus the branch rows
            Ls, cache, h_state = self.prefix(enc)
            return self._branch_rows_from_prefix(enc, cache), (Ls, cache, h_state)
        ids = torch.tensor([enc["ids"]], device=self.device); pos = torch.tensor([enc["pos"]], device=self.device)
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype=dt, opts=[enc["opt"]] if enc.get("option_isolation") else None)
        out = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask, past_key_values=DynamicCache(config=self.lm.config), use_cache=True)
        h = out.last_hidden_state[0].float()
        out.past_key_values.crop(-(len(enc["ids"]) - Ls))     # keep the state only (negative = drop that many trailing tokens; positive form deprecated in transformers 5)
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)], (Ls, out.past_key_values, h[:Ls].clone())

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix):
        """probs() for a record whose state tokens equal the cached prefix's; only the branches run. The cache is cropped
        back to the state afterwards so it can be reused."""
        Ls, cache, h_state = prefix
        if enc["seg"].count(0) != Ls: raise ValueError("prefix does not match this record's state")
        if self.rows_form([enc]):
            return self._branch_rows_from_prefix(enc, cache)
        ids = torch.tensor([enc["ids"][Ls:]], device=self.device); pos = torch.tensor([enc["pos"][Ls:]], device=self.device)
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype=dt, opts=[enc["opt"]] if enc.get("option_isolation") else None)[:, :, Ls:, :]
        try:
            out = self.lm(input_ids=ids, position_ids=pos, past_key_values=cache, attention_mask=mask, use_cache=True)
            h = torch.cat([h_state, out.last_hidden_state[0].float()], 0)
        finally:
            cache.crop(-(len(enc["ids"]) - Ls))
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
