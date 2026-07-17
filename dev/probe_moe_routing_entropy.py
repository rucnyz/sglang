"""Probe early-layer MoE router degeneracy.

Hypothesis: in early MoE layers the router decision is dominated by token id
(because hidden_state ≈ token embedding before attention has had a chance to
mix in context), so the same token id always lands on the same expert and
Zipf-frequent tokens overload a few experts.

To test it, for each MoE layer L and each token id t we estimate
    P(expert | t, L) = histogram of which expert(s) token t was routed to
    H(L, t)          = entropy of that distribution

Aggregates:
  * per-layer mean H, weighted by token occurrence count (this is the number
    that matters for real load imbalance)
  * per-layer H for the top-K most frequent tokens (the ones doing the damage)

If the hypothesis holds: early layers have H near 0, deep layers approach
log(num_experts).

Usage (from sglang repo root):
    python dev/probe_moe_routing_entropy.py
    python dev/probe_moe_routing_entropy.py --model Qwen/Qwen3-30B-A3B
    python dev/probe_moe_routing_entropy.py --model allenai/OLMoE-1B-7B-0924
"""

from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FALLBACK_TEXT = """The mixture-of-experts (MoE) architecture sparsely activates a subset of
expert sub-networks for each token, allowing the total parameter count to grow
without a proportional increase in compute per token. The router is typically a
linear projection of the hidden state followed by a softmax and a top-k
selection. In practice, the early layers of an MoE model often suffer from a
load-imbalance problem: a small number of high-frequency tokens dominate the
input distribution, and because their hidden states are still very close to
their raw embeddings (attention has not yet mixed in much context), the router
maps them to the same few experts every time. This effect attenuates with
depth, as attention progressively diffuses context across positions and the
router begins to see contextually distinct hidden states even for identical
tokens. Common mitigations include leaving the first layer as a dense MLP
(DeepSeek-V2/V3 does this), adding a shared expert that every token must pass
through, applying a z-loss to suppress router logit blow-up, and using a
larger top-k near the input."""


def find_moe_blocks(model) -> List[Tuple[int, torch.nn.Module, torch.nn.Module, int, int]]:
    """Return list of (layer_idx, mlp_module, gate_module, num_experts, top_k)."""
    out = []
    layers = model.model.layers if hasattr(model, "model") else model.layers
    cfg = model.config
    cfg_topk = (
        getattr(cfg, "num_experts_per_tok", None)
        or getattr(cfg, "moe_topk", None)
        or getattr(cfg, "top_k", None)
    )
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if mlp is None:
            continue
        gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
        experts = getattr(mlp, "experts", None)
        if gate is None or experts is None:
            continue
        if hasattr(experts, "__len__"):
            num_experts = len(experts)
        else:
            num_experts = (
                getattr(mlp, "num_experts", None)
                or getattr(cfg, "num_experts", None)
                or getattr(cfg, "n_routed_experts", None)
            )
        top_k = (
            getattr(mlp, "top_k", None)
            or getattr(mlp, "num_experts_per_tok", None)
            or cfg_topk
        )
        assert num_experts and top_k, f"layer {i}: cannot infer num_experts/top_k"
        out.append((i, mlp, gate, num_experts, top_k))
    return out


@torch.no_grad()
def probe(model, tokenizer, texts: List[str], seq_len: int, device: str):
    blocks = find_moe_blocks(model)
    if not blocks:
        raise RuntimeError("No MoE blocks found in this model")

    num_experts = blocks[0][3]
    top_k = blocks[0][4]
    # Use the *embedding* size, not the tokenizer's vocab_size — Qwen3.5-MoE
    # pads its embedding (248320) past the tokenizer's base vocab, and special
    # tokens can have ids > tokenizer.vocab_size, which would scatter OOB.
    embed = model.get_input_embeddings()
    vocab_size = int(embed.num_embeddings)
    tok_vocab = len(tokenizer)
    print(
        f"Found {len(blocks)} MoE layers at indices "
        f"{[b[0] for b in blocks[:3]]}...{[b[0] for b in blocks[-3:]]}, "
        f"num_experts={num_experts}, top_k={top_k}, embed_vocab={vocab_size}, tok_vocab={tok_vocab}"
    )
    counts_bytes = len(blocks) * vocab_size * num_experts * 4
    print(f"Counts tensor: {counts_bytes / 1024**3:.2f} GiB on {device}")
    free, total = torch.cuda.mem_get_info(device)
    print(f"GPU mem (post-model-load): free={free/1024**3:.1f} GiB / total={total/1024**3:.1f} GiB")
    if counts_bytes > free * 0.8:
        raise RuntimeError(
            f"counts tensor ({counts_bytes/1024**3:.1f} GiB) would not fit in "
            f"free GPU memory ({free/1024**3:.1f} GiB). Try a smaller --model "
            f"or shrink the workload."
        )

    # counts[layer, token_id, expert_id]  — dense, lives on GPU for fast scatter.
    # int32 to avoid worrying about overflow even with --num-batches=128+.
    # Memory: num_layers * vocab * num_experts * 4 bytes
    #   Qwen3.5-35B-A3B: 40 * 248k * 256 * 4 ≈ 10 GB (fine on 96 GB cards)
    counts = torch.zeros(
        (len(blocks), vocab_size, num_experts), dtype=torch.int32, device=device
    )
    token_freq = torch.zeros(vocab_size, dtype=torch.int64, device=device)

    current_ids: List[torch.Tensor] = []

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            _, topk_idx = logits.topk(top_k, dim=-1)  # [N, K] long
            ids = current_ids[0].to(topk_idx.device).long()
            assert ids.shape[0] == topk_idx.shape[0], (
                f"shape mismatch: ids={ids.shape}, topk={topk_idx.shape}"
            )
            N = ids.shape[0]
            flat = (ids.unsqueeze(1) * num_experts + topk_idx).flatten()
            ones = torch.ones(N * top_k, dtype=torch.int32, device=device)
            counts[layer_idx].view(-1).scatter_add_(0, flat, ones)
        return hook

    handles = [b[2].register_forward_hook(make_hook(li)) for li, b in enumerate(blocks)]
    try:
        for bi, text in enumerate(texts):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            ids = enc.input_ids.to(device)
            flat = ids.flatten()
            current_ids.clear()
            current_ids.append(flat)
            token_freq.scatter_add_(
                0, flat.long(), torch.ones_like(flat, dtype=torch.int64)
            )
            model(ids)
            print(f"  batch {bi+1}/{len(texts)}  tokens={flat.numel()}")
    finally:
        for h in handles:
            h.remove()

    return blocks, counts, token_freq, num_experts, top_k


def per_token_entropy(counts_layer: torch.Tensor) -> torch.Tensor:
    """H(expert | token) for each token id; counts_layer is [vocab, num_experts]."""
    totals = counts_layer.sum(dim=-1, keepdim=True).clamp(min=1)
    p = counts_layer.to(torch.float64) / totals.to(torch.float64)
    log_p = torch.where(p > 0, p.log(), torch.zeros_like(p))
    return -(p * log_p).sum(dim=-1)  # [vocab]


def load_corpus(args) -> List[str]:
    """Pull `--num-batches` text chunks roughly `--seq-len`-tokens worth each."""
    chunks: List[str] = []
    if args.dataset:
        try:
            from datasets import load_dataset

            print(f"Loading {args.dataset} / {args.dataset_config} / {args.split} ...")
            ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
            cur = ""
            char_budget = args.seq_len * 4
            for row in ds:
                txt = row.get("text") or row.get("content") or ""
                if not txt:
                    continue
                cur += txt + "\n"
                if len(cur) >= char_budget:
                    chunks.append(cur)
                    cur = ""
                    if len(chunks) >= args.num_batches:
                        break
            if cur and len(chunks) < args.num_batches:
                chunks.append(cur)
        except Exception as e:
            print(f"  dataset load failed ({e}); falling back to built-in text")

    if len(chunks) < args.num_batches:
        need = args.num_batches - len(chunks)
        chunks.extend([FALLBACK_TEXT] * need)
    print(f"Prepared {len(chunks)} text chunks")
    return chunks[: args.num_batches]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--top-tokens", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": args.device},
    )
    model.eval()

    texts = load_corpus(args)
    blocks, counts, token_freq, num_experts, _top_k = probe(
        model, tok, texts, args.seq_len, args.device
    )

    max_h = math.log(num_experts)
    print()
    print("=" * 82)
    print(f"Uniform-routing entropy ceiling = log({num_experts}) = {max_h:.4f} nats")
    print()
    print("Mean H(expert | token), weighted by token occurrence count:")
    print(f"  {'layer':>6} | {'mean H':>10} | {'H / log(E)':>11} | {'unique tokens':>13}")
    freq_cpu = token_freq.cpu().to(torch.float64)
    layer_summary = []
    for li, block in enumerate(blocks):
        lidx = block[0]
        h_per_token = per_token_entropy(counts[li]).cpu()  # [vocab]
        weight = freq_cpu
        mass = weight.sum().item()
        mean_h = (h_per_token * weight).sum().item() / max(mass, 1.0)
        unique = int((weight > 0).sum().item())
        print(
            f"  {lidx:>6d} | {mean_h:>10.4f} | "
            f"{mean_h/max_h:>11.3f} | {unique:>13d}"
        )
        layer_summary.append((lidx, mean_h))

    # Per-layer entropy for top-K most-frequent tokens. Sample at most ~10
    # layers (first 4 + evenly-spaced rest) so 40-layer models still fit in
    # a terminal width.
    top_ids = torch.topk(token_freq, args.top_tokens).indices.tolist()
    n = len(blocks)
    if n <= 12:
        sampled = list(range(n))
    else:
        sampled = sorted(set(list(range(4)) + [round(i * (n - 1) / 7) for i in range(8)]))
    print()
    print(f"Top-{args.top_tokens} most frequent tokens (the ones causing imbalance):")
    header = (
        f"  {'token':>20s} | {'id':>6s} | {'freq':>6s} | "
        + " ".join(f"L{blocks[li][0]:>2d}" for li in sampled)
    )
    print(header)
    for tid in top_ids:
        s = tok.decode([tid]).replace("\n", "\\n")
        if len(s) > 18:
            s = s[:18]
        freq = int(freq_cpu[tid].item())
        row_h = [per_token_entropy(counts[li][tid : tid + 1])[0].item() for li in sampled]
        print(
            f"  {repr(s):>20s} | {tid:>6d} | {freq:>6d} | "
            + " ".join(f"{h:>4.2f}" for h in row_h)
        )

    if layer_summary:
        first_h = layer_summary[0][1]
        last_h = layer_summary[-1][1]
        print()
        print(
            f"Summary: layer {layer_summary[0][0]} mean H = {first_h:.3f}, "
            f"layer {layer_summary[-1][0]} mean H = {last_h:.3f}, "
            f"ratio last/first = {last_h/max(first_h, 1e-6):.2f}x  "
            f"(ceiling = {max_h:.3f})"
        )

    # Per-layer expert load distribution. Entropy measures per-token routing
    # diversity; this measures the orthogonal axis -- across-expert load skew,
    # which is what "load imbalance" actually refers to.
    print()
    print("Per-layer expert load distribution (across all routed token-expert pairs):")
    perfect_mean = 100.0 / num_experts  # perfect-balance per-expert share, percent
    top_k_show = min(10, num_experts)
    perfect_topk = top_k_show * perfect_mean
    print(
        f"  {'layer':>6} | {'min%':>6} | {'max%':>6} | {'max/mean':>9} | "
        f"{'CV':>5} | {f'top{top_k_show}%':>7} | {'dead':>5}"
    )
    for li, block in enumerate(blocks):
        lidx = block[0]
        load = counts[li].sum(dim=0).to(torch.float64)  # [num_experts]
        total = load.sum().item()
        if total == 0:
            continue
        pct = (load / total) * 100.0
        pct_min = pct.min().item()
        pct_mean = pct.mean().item()
        pct_max = pct.max().item()
        cv = pct.std(unbiased=False).item() / max(pct_mean, 1e-12)
        topk_share = pct.topk(top_k_show).values.sum().item()
        dead = int((load == 0).sum().item())
        print(
            f"  {lidx:>6d} | {pct_min:>5.2f}% | {pct_max:>5.2f}% | "
            f"{pct_max/pct_mean:>8.2f}x | {cv:>5.2f} | {topk_share:>6.2f}% | {dead:>5d}"
        )
    print(
        f"  (perfect balance: min=max=mean={perfect_mean:.2f}%, max/mean=1.00x, "
        f"CV=0, top{top_k_show}%={perfect_topk:.2f}%, dead=0)"
    )


if __name__ == "__main__":
    main()
