"""Utilities for hunting a 'conspiracy narrative' latent in the Qwen3.5-2B SAEs.

Fixes vs. the notebook version:
  * encoder subtracts b_dec and applies ReLU before TopK (pick_encoder() verifies
    which variant actually reconstructs the residual stream);
  * scoring uses AUC + coverage instead of a mean difference, so a latent that
    fires hugely on two texts cannot win;
  * latents are validated on a *generic* corpus (max-activating examples) before
    being trusted -- that is what separates "narrative of concealment" from
    "the token ' secret'";
  * steering coefficient is calibrated in units of the latent's own activation.
"""
import contextlib, os, random, re, urllib.request
from contextlib import contextmanager
from pathlib import Path

import torch, torch.nn.functional as F

D_MODEL, N_FEATS, K = 2048, 32768, 50
DATA = Path(__file__).resolve().parent / "data"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
SUB_BDEC = True          # set by pick_encoder(); default for every encode() call


# ── corpora ──────────────────────────────────────────────────────────────────
def load_corpus(name, strip_title=True, limit=None):
    """Read data/<name>.txt. Lines are 'Title — claim'; strip_title drops the title,
    which removes the topic word but also the shared surface form of the two corpora.
    Run the hunt both ways: a feature that only survives with titles is a topic detector."""
    lines = [l.strip() for l in (DATA / f"{name}.txt").read_text().splitlines() if l.strip()]
    if strip_title:
        lines = [l.split(" \u2014 ", 1)[-1] for l in lines]
    lines = [l.replace("\\n", "\n") for l in lines]
    return lines[:limit]


def ensure_sae(layer):
    f = f"layer{layer}.sae.pt"
    if not os.path.exists(f):
        url = f"https://huggingface.co/{SAE_REPO}/resolve/main/{f}?download=true"
        print(f"downloading {f} ...")
        urllib.request.urlretrieve(url, f)
    return f


# ── SAE math ─────────────────────────────────────────────────────────────────
def load_sae(layer, device="cpu", dtype=torch.float32):
    sae = torch.load(ensure_sae(layer), map_location="cpu", weights_only=False)
    return {k: v.to(device=device, dtype=dtype) for k, v in sae.items()}


def encode(x, sae, sub_bdec=None, k=K):
    """x: (..., 2048) residual -> (..., 32768) sparse acts (TopK, non-negative)."""
    sub_bdec = SUB_BDEC if sub_bdec is None else sub_bdec
    x = x.to(sae["W_enc"].dtype).to(sae["W_enc"].device)
    z = (x - sae["b_dec"]) if sub_bdec else x
    pre = F.relu(z @ sae["W_enc"].T + sae["b_enc"])
    vals, idx = pre.topk(k, dim=-1)
    out = torch.zeros_like(pre)
    return out.scatter_(-1, idx, vals)


def decode(acts, sae):
    return acts @ sae["W_dec"].T + sae["b_dec"]


def pick_encoder(x, sae):
    """Return ('sub_bdec' flag, fvu) for whichever encoder variant reconstructs best.
    Run this once on real residuals -- if both FVUs are ~1.0 you are hooking the
    wrong tensor (e.g. layer input vs. output) and every feature hunt below is noise."""
    report = {}
    for flag in (True, False):
        xh = decode(encode(x, sae, sub_bdec=flag), sae)
        fvu = ((x - xh).pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()
        report[flag] = fvu
    global SUB_BDEC
    best = min(report, key=report.get)
    SUB_BDEC = best
    print(f"FVU  sub_b_dec=True: {report[True]:.3f}   False: {report[False]:.3f}  -> use {best}")
    return best, report


# ── activation collection ────────────────────────────────────────────────────
class ResidHook:
    """Captures the residual stream after `layer` (keeps it on GPU, no .cpu() sync)."""
    def __init__(self, model, layer):
        self.h = model.model.layers[layer].register_forward_hook(self._fn)
        self.out = None

    def _fn(self, mod, inp, out):
        self.out = (out[0] if isinstance(out, tuple) else out).detach()

    def remove(self):
        self.h.remove()


@torch.no_grad()
def text_acts(texts, model, tok, hook, sae, sub_bdec=None, skip_bos=1, keep_tokens=False):
    """-> (n_texts, 32768) max activation per text. keep_tokens=True also returns
    per-token acts, which is ~130 MB per 1000 tokens -- only for a handful of texts."""
    maxes, per_tok = [], []
    dev = next(model.parameters()).device
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256).to(dev)
        model(**ids)
        a = encode(hook.out[0].float(), sae, sub_bdec)[skip_bos:]      # (seq, 32768)
        maxes.append(a.max(0).values.cpu())
        if keep_tokens:
            per_tok.append((ids["input_ids"][0, skip_bos:].cpu(), a.cpu()))
    m = torch.stack(maxes)
    return (m, per_tok) if keep_tokens else m


@torch.no_grad()
def collect_resid(texts, model, tok, hook, skip_bos=1):
    """-> (n_tokens, 2048) stacked residuals, for diff-of-means directions."""
    dev, out = next(model.parameters()).device, []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256).to(dev)
        model(**ids)
        out.append(hook.out[0, skip_bos:].float().cpu())
    return torch.cat(out, 0)


# ── scoring ──────────────────────────────────────────────────────────────────
def auc(pos, neg):
    """Per-feature ROC-AUC of pos vs neg. pos/neg: (n, 32768) max-act matrices."""
    n_p, n_n = pos.shape[0], neg.shape[0]
    allv = torch.cat([pos, neg], 0)                       # (n_p+n_n, F)
    rank = allv.argsort(0).argsort(0).float() + 1          # average-tie-free ranks
    r_pos = rank[:n_p].sum(0)
    return (r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n)


def rank_features(pos, neg, min_cov=0.5, thresh=0.5, top=25):
    """AUC + coverage. Coverage kills lexical detectors that only hit a few texts."""
    a = auc(pos, neg)
    cov_p = (pos > thresh).float().mean(0)
    cov_n = (neg > thresh).float().mean(0)
    score = torch.where(cov_p >= min_cov, a, torch.zeros_like(a))
    vals, idx = score.topk(top)
    rows = [(int(i), float(a[i]), float(cov_p[i]), float(cov_n[i]),
             float(pos[:, i].mean()), float(neg[:, i].mean())) for i in idx]
    print(f"{'feat':>7} {'auc':>6} {'cov+':>6} {'cov-':>6} {'mean+':>7} {'mean-':>7}")
    for r in rows:
        print(f"{r[0]:>7} {r[1]:>6.3f} {r[2]:>6.0%} {r[3]:>6.0%} {r[4]:>7.2f} {r[5]:>7.2f}")
    return rows


# ── validation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def max_activating(feature, corpus, model, tok, hook, sae, sub_bdec=None, n=15, window=8):
    """Top-n snippets from a GENERIC corpus. If these are all about one topic or one
    word, the latent is not the abstract feature you want."""
    dev, hits = next(model.parameters()).device, []
    for t in corpus:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256).to(dev)
        model(**ids)
        a = encode(hook.out[0].float(), sae, sub_bdec)[:, feature].cpu()
        p = int(a.argmax())
        if a[p] > 0:
            toks = ids["input_ids"][0].cpu()
            lo, hi = max(0, p - window), min(len(toks), p + window)
            hits.append((float(a[p]), tok.decode(toks[lo:p]) + " «" +
                         tok.decode(toks[p:p + 1]) + "» " + tok.decode(toks[p + 1:hi])))
    hits.sort(reverse=True)
    for v, s in hits[:n]:
        print(f"{v:6.2f}  {s}")
    return hits


@torch.no_grad()
def logit_lens(feature, model, sae, tok, n=20):
    """Tokens promoted by writing this latent's decoder direction into the residual."""
    d = sae["W_dec"][:, feature].to(model.lm_head.weight.dtype).to(model.lm_head.weight.device)
    d = model.model.norm(d.unsqueeze(0))            # final RMSNorm
    logits = (d @ model.lm_head.weight.T)[0].float()
    v, i = logits.topk(n)
    print("promotes:", [tok.decode([t]) for t in i.tolist()])
    v, i = (-logits).topk(n)
    print("suppresses:", [tok.decode([t]) for t in i.tolist()])


# ── direction building ───────────────────────────────────────────────────────
def diff_of_means(pos_resid, neg_resid):
    """Mean-difference direction in residual space (usually steers better than one
    latent). pos_resid/neg_resid: (n_tokens, 2048) stacked residuals."""
    d = pos_resid.mean(0) - neg_resid.mean(0)
    return d / d.norm()


def explain_direction(direction, sae, top=15):
    """Which SAE latents compose an arbitrary direction -> interpretable decomposition."""
    cos = F.normalize(sae["W_dec"], dim=0).T @ F.normalize(direction.float(), dim=0)
    v, i = cos.topk(top)
    for a, b in zip(v.tolist(), i.tolist()):
        print(f"  feature {b:>6}: cos={a:.3f}")
    return list(zip(i.tolist(), v.tolist()))


def feature_direction(sae, feats, weights=None):
    """Single latent or a weighted bundle of latents -> one unit steering vector."""
    if isinstance(feats, int):
        feats = [feats]
    w = torch.ones(len(feats)) if weights is None else torch.tensor(weights, dtype=torch.float32)
    d = (sae["W_dec"][:, feats].float() * w.to(sae["W_dec"].device)).sum(1)
    return d / d.norm()


# ── scale calibration ────────────────────────────────────────────────────────
def act_scale(acts, q=0.99):
    """Typical size of a FIRING activation. Do NOT take a percentile over the whole
    TopK tensor: 50 of 32768 entries are nonzero, so .quantile(0.99) is 0 and every
    strength derived from it silently becomes zero."""
    nz = acts[acts > 0]
    return float(nz.quantile(q)) if nz.numel() else 0.0


def resid_scale(R):
    """Mean residual norm at this layer. The right unit for a diff-of-means direction,
    which is not a dictionary column and has no activation scale of its own."""
    return float(R.norm(dim=-1).mean())


@torch.no_grad()
def steering_selftest(model, tok, direction, layers, strength, **kw):
    """Assert the hook actually changes the forward pass before trusting any result."""
    dev = next(model.parameters()).device
    ids = tok("The report was", return_tensors="pt").to(dev)
    base = model(**ids).logits[0, -1].float().clone()
    with steering(model, direction, layers, strength, **kw):
        got = model(**ids).logits[0, -1].float()
    shift = (got - base).abs().max().item()
    flag = "  <-- NO-OP: strength or direction is zero" if shift < 1e-4 else ""
    print(f"self-test: max logit shift {shift:.4f} at strength {strength:.3g}{flag}")
    assert shift > 1e-4, "steering hook had no effect"
    return shift


# ── steering ─────────────────────────────────────────────────────────────────
@contextmanager
def steering(model, direction, layers, strength, mode="add", feature=None, sae=None,
             preserve_norm=True, split=True, positions="all"):
    """mode='add'   : h += strength * direction   (strength in the direction's own units)
       mode='clamp' : raise this latent's activation to `strength` where it is lower.

    preserve_norm : rescale h back to its original norm after the push. An unconstrained
        addition inflates ||h||, later layers see off-distribution input, and the model
        degrades long before the concept takes over. This is what widens the usable window.
    split   : divide strength by len(layers), so `strength` means the TOTAL push rather
        than the push per layer -- five layers at 0.25 is really 1.25.
    positions : 'all', or 'new' to leave the prompt untouched and steer only generated
        tokens (the prefill pass is the one with seq_len > 1).
    Position 0 is never touched -- the BOS residual is an attention sink.
    """
    handles, eff = [], strength / max(len(list(layers)), 1) if split else strength
    try:
        for L in layers:
            blk = model.model.layers[L]
            p = next(blk.parameters())
            d = direction.to(dtype=p.dtype, device=p.device)

            def hook(mod, inp, out, d=d):
                tup = isinstance(out, tuple)
                h = out[0] if tup else out
                if positions == "new" and h.shape[1] > 1:
                    return out
                if mode == "add":
                    delta = (eff * d).expand_as(h).clone()
                else:
                    cur = encode(h.float(), sae)[..., feature].unsqueeze(-1)
                    delta = ((eff - cur).clamp(min=0).to(h.dtype) * d).clone()
                if h.shape[1] > 1:          # prefill: position 0 is BOS (attention sink)
                    delta[:, 0] = 0         # decode steps: position 0 is the new token
                h2 = h + delta
                if preserve_norm:
                    n0 = h.norm(dim=-1, keepdim=True)
                    h2 = h2 * (n0 / h2.norm(dim=-1, keepdim=True).clamp(min=1e-6))
                return (h2,) + out[1:] if tup else h2

            handles.append(blk.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def fluency(model, tok, texts):
    """Mean token log-prob on ordinary text -- the cost side of the steering trade-off.
    `plain` in the gap eval is not a fluency proxy: suppressing it is the intended effect."""
    dev, tot, n = next(model.parameters()).device, 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=128).input_ids.to(dev)
        if ids.shape[1] < 2:
            continue
        lg = model(ids).logits[0, :-1].float().log_softmax(-1)
        tot += lg.gather(-1, ids[0, 1:].unsqueeze(-1)).mean().item()
        n += 1
    return tot / max(n, 1)


def steer(model, tok, prompt, direction, layers, strength, mode="add",
          feature=None, sae=None, preserve_norm=True, split=True, positions="all", **gen):
    ids = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with steering(model, direction, layers, strength, mode, feature, sae,
                  preserve_norm, split, positions):
        with torch.no_grad():
            o = model.generate(**ids, **{"max_new_tokens": 120, "do_sample": True,
                                         "temperature": 0.8, "top_p": 0.9, **gen})
    return tok.decode(o[0], skip_special_tokens=True)


def sweep(model, tok, prompt, direction, layers, strengths=(0, 2, 4, 8, 16, 32, 64), **kw):
    """Coefficient sweep -- the useful window is usually narrow and ends in word salad."""
    for s in strengths:
        print(f"\n===== strength {s} =====")
        print(steer(model, tok, prompt, direction, layers, s, **kw))


@torch.no_grad()
def continuation_logprob(model, tok, prompt, continuation):
    """Mean log-prob per token of `continuation` given `prompt`. Steer around this
    call with `with steering(...)` to measure an effect instead of eyeballing samples."""
    dev = next(model.parameters()).device
    p = tok(prompt, return_tensors="pt").input_ids.to(dev)
    c = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    ids = torch.cat([p, c], 1)
    lg = model(ids).logits[0, p.shape[1] - 1:-1].float().log_softmax(-1)
    return lg.gather(-1, c[0].unsqueeze(-1)).mean().item()


# ── contrastive activation addition (CAA) ───────────────────────────────────
# diff_of_means() above averages over whole *texts*, so part of what it captures is
# topic (1970s covert operations, alien abductions, ...) rather than stance. CAA fixes
# this by anchoring both sides to the SAME question and taking the residual over just
# the answer tokens -- the only thing that differs is whether the answer is
# conspiratorial or not. It steers considerably harder for the same fluency cost
# (roughly 2x the gap at matched Δfluency in the runs this was built for).
_STOPWORDS = set("the a an and or of to in on for by with was were is are be as at from "
                  "that this it its their his her they he she which who not no but has "
                  "have had into over after before than then there these those all any "
                  "some more most other such only own same so too very can will just "
                  "about also per".split())


def _content_words(line):
    return {w for w in re.findall(r"[a-z0-9]+", line.lower()) if w not in _STOPWORDS and len(w) > 2}


def align_by_topic(a_lines, b_lines):
    """Monotone alignment (Needleman-Wunsch, free gaps) of two 'Title — claim' corpora,
    maximising summed Jaccard similarity of each FULL LINE's content words (title and claim
    both -- the claim text usually shares more wording with its matched claim than the two
    titles share with each other, e.g. "shot down", "cover-up", proper nouns). -> [(i, j,
    score), ...] in a_lines order. conspiracy.txt and the two control corpora are NOT the same
    length or in the same order, so this is what turns them into topic-matched pairs (used by
    build_topic_pairs() below) instead of pairing by raw line index."""
    ta = [_content_words(l) for l in a_lines]
    tb = [_content_words(l) for l in b_lines]
    sim = [[len(x & y) / max(len(x | y), 1) for y in tb] for x in ta]
    n, m = len(a_lines), len(b_lines)
    S = [[0.0] * (m + 1) for _ in range(n + 1)]
    P = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S[i][j], P[i][j] = max((S[i - 1][j - 1] + sim[i - 1][j - 1], "d"),
                                    (S[i - 1][j], "u"), (S[i][j - 1], "l"))
    i, j, out = n, m, []
    while i and j:
        p = P[i][j]
        if p == "d":
            out.append((i - 1, j - 1, sim[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif p == "u":
            i -= 1
        else:
            j -= 1
    return out[::-1]


ANSWER_TEMPLATE = "Question: What is the truth about {}?\nAnswer:"


def build_topic_pairs():
    """Pair every conspiracy.txt row with its topic-matched row in control_debunk.txt and
    control_assertive.txt. -> [{"topic", "consp", "ctrl", "src", "ci"}, ...], where "ci" is
    the conspiracy.txt row index -- hold out by "ci", not by pair, so a topic's debunk AND
    assertive pairs never end up split across train/test."""
    raw = lambda f: [l for l in (DATA / f"{f}.txt").read_text().splitlines() if l.strip()]
    split = lambda l: [s.strip() for s in l.split(" — ", 1)]
    C = raw("conspiracy")
    pairs = []
    for name in ("control_debunk", "control_assertive"):
        B = raw(name)
        for i, j, _ in align_by_topic(C, B):
            _, cc = split(C[i])
            bt, bc = split(B[j])
            pairs.append(dict(topic=bt, consp=cc, ctrl=bc, src=name, ci=i))
    return pairs


@torch.no_grad()
def _answer_mean(model, tok, hooks, prompt, answer):
    """Mean residual over the ANSWER tokens only, at every hooked layer -- not the question,
    which is identical between the two sides of a pair and would just dilute the direction."""
    n_p = len(tok(prompt).input_ids)
    dev = next(model.parameters()).device
    model(**tok(prompt + " " + answer, return_tensors="pt").to(dev))
    return {L: h.out[0, n_p:].float().mean(0) for L, h in hooks.items()}


def caa_vectors(model, tok, pairs, layers, template=ANSWER_TEMPLATE):
    """For each pair, run 'question + consp answer' and 'question + ctrl answer', and average
    (consp - ctrl) over the answer-token residual at each layer. -> {layer: raw direction},
    NOT unit-normalised -- its own norm is a meaningful scale (see caa_layer_scan's |v| column)."""
    hooks = {L: ResidHook(model, L) for L in layers}
    diffs = {L: [] for L in layers}
    for p in pairs:
        q = template.format(p["topic"])
        a = _answer_mean(model, tok, hooks, q, p["consp"])
        b = _answer_mean(model, tok, hooks, q, p["ctrl"])
        for L in layers:
            diffs[L].append(a[L] - b[L])
    for h in hooks.values():
        h.remove()
    return {L: torch.stack(diffs[L]).mean(0) for L in layers}


def caa_layer_scan(model, tok, pairs, layers, template=ANSWER_TEMPLATE, holdout_frac=0.2, seed=0):
    """Split pairs by conspiracy-row id, build CAA vectors on the training split, and report
    how often each layer's raw vector separates a held-out TOPIC's own consp/ctrl answers
    (projection sign, not magnitude). Run this before a full heldout_gap() sweep to see which
    layers are worth steering at all. -> (vecs, test_pairs, {layer: held-out separation acc})."""
    cis = sorted({p["ci"] for p in pairs})
    random.Random(seed).shuffle(cis)
    test_ci = set(cis[:max(1, int(len(cis) * holdout_frac))])
    train = [p for p in pairs if p["ci"] not in test_ci]
    test = [p for p in pairs if p["ci"] in test_ci]
    vecs = caa_vectors(model, tok, train, layers, template)
    hooks = {L: ResidHook(model, L) for L in layers}
    acc = {L: 0 for L in layers}
    for p in test:
        q = template.format(p["topic"])
        a = _answer_mean(model, tok, hooks, q, p["consp"])
        b = _answer_mean(model, tok, hooks, q, p["ctrl"])
        for L in layers:
            acc[L] += int(((a[L] - b[L]) @ vecs[L]) > 0)
    for h in hooks.values():
        h.remove()
    return vecs, test, {L: acc[L] / len(test) for L in layers}


@contextmanager
def steering_multi(model, vecs, alpha, ablate=False, positions="all"):
    """Like steering(), but takes a {layer: direction} dict with a DIFFERENT vector per
    layer (what caa_vectors() returns) instead of one direction shared across a layer range.
    ablate=True projects each layer's own unit direction OUT of h instead of adding it
    (alpha is then ignored) -- the strongest test that a direction is causal: if removing it
    collapses the held-out effect (see heldout_gap), it isn't merely correlated with it.
    Position 0 (BOS) is protected on the prefill pass only, same as steering()."""
    handles = []
    for L, v in vecs.items():
        blk = model.model.layers[L]
        p = next(blk.parameters())
        v = v.to(dtype=p.dtype, device=p.device)
        u = v / v.norm()

        def hook(mod, inp, out, v=v, u=u):
            tup = isinstance(out, tuple)
            h = out[0] if tup else out
            if positions == "new" and h.shape[1] > 1:
                return out
            delta = -(h @ u).unsqueeze(-1) * u if ablate else (alpha * v).expand_as(h).clone()
            if h.shape[1] > 1:
                delta[:, 0] = 0
            h2 = h + delta
            return (h2,) + out[1:] if tup else h2

        handles.append(blk.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def heldout_gap(model, tok, test_pairs, ctx=None, template=ANSWER_TEMPLATE, fluency_corpus=None):
    """Score a steering context (from steering() or steering_multi(), or None for baseline)
    on held-out topic pairs: mean log-prob gap (consp answer - ctrl answer), the fraction of
    topics where the conspiratorial answer wins, and fluency on generic text as the cost side.
    -> {"gap", "consp_wins", "fluency"}."""
    ctx = ctx or contextlib.nullcontext()
    gaps = []
    with ctx:
        for p in test_pairs:
            q = template.format(p["topic"])
            gaps.append(continuation_logprob(model, tok, q, " " + p["consp"]) -
                        continuation_logprob(model, tok, q, " " + p["ctrl"]))
        flu = fluency(model, tok, fluency_corpus if fluency_corpus is not None else load_corpus("generic")[:40])
    g = torch.tensor(gaps)
    return dict(gap=float(g.mean()), consp_wins=float((g > 0).float().mean()), fluency=flu)
