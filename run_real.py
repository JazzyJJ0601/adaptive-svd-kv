#!/usr/bin/env python3
"""
Adaptive per-layer SVD rank selection for KV cache compression.

Uses per-layer attention entropy to assign ranks proportionally.
Implements per-chunk checkpoint/resume for incremental validation.
"""
import copy
import gc
import json
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

from adaptive_svd import (
    AdaptiveSVDKVCompressor,
    compute_attention_entropy,
    adaptive_ranks_from_entropy,
    CheckpointManager
)

# Configuration
MODEL_PATH = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
CHECKPOINT_DIR = "checkpoints"
SEQ_LEN = 512
N_CHUNKS = 32
EPS_VALUE = 1e-2
N_LAYERS = 40  # Qwen3-8B

torch.set_grad_enabled(False)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def load_wikitext_chunks(n_chunks: int = N_CHUNKS):
    """Load n_chunks of 512-token chunks from wikitext-2 test split."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Concatenate and tokenize
    text = "\n".join([row["text"] for row in ds if row["text"]])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    chunks = []
    for i in range(0, len(tokens) - SEQ_LEN, SEQ_LEN // 2):
        chunk_ids = tokens[i:i + SEQ_LEN]
        chunks.append(chunk_ids)
        if len(chunks) >= n_chunks:
            break
    return chunks


def capture_attention_entropy(model, cache, n_layers=N_LAYERS):
    """
    Extract per-layer attention entropy from the model.
    Returns dict of {layer_idx: mean_entropy}
    """
    entropies = {}
    for layer_idx in range(n_layers):
        layer = cache.layers[layer_idx]
        # Extract attention scores from attention module
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attention_scores'):
            scores = layer.self_attn.attention_scores
            if scores is not None:
                # scores shape: (batch, heads, seq, seq)
                entropy = compute_attention_entropy(scores)
                entropies[layer_idx] = float(entropy.mean().item())
            else:
                entropies[layer_idx] = 0.0
        else:
            # Fallback: estimate from key/value statistics
            if hasattr(layer, 'keys') and layer.keys is not None:
                k = layer.keys
                # Estimate entropy from key variance
                var = torch.var(k, dim=(0, 2, 3)).mean().item()
                entropies[layer_idx] = float(var)
            else:
                entropies[layer_idx] = 0.0
    return entropies


def compress_kv_adaptive(cache, entropies, ranks, eps=EPS_VALUE):
    """
    Compress KV cache using adaptive ranks per layer.
    
    Args:
        cache: Transformer cache with layers
        entropies: Dict of layer_idx -> entropy
        ranks: List of target ranks per layer
        eps: SVD compression tolerance
    
    Returns:
        (mean_rank_k, mean_rank_v, original_bytes, compressed_bytes)
    """
    comp = AdaptiveSVDKVCompressor(eps)
    ranks_k, ranks_v = [], []
    orig_bytes = comp_bytes = 0

    for layer_idx, layer in enumerate(cache.layers):
        layer_rank = ranks[layer_idx] if layer_idx < len(ranks) else ranks[-1]
        
        for name, rs in (("keys", ranks_k), ("values", ranks_v)):
            t = getattr(layer, name, None)
            if t is None or t.numel() == 0:
                continue
            b, h, s, d = t.shape
            out = torch.empty_like(t)
            for bi in range(b):
                for hi in range(h):
                    res = comp.compress(t[bi, hi].float(), target_rank=layer_rank)
                    orig_bytes += s * d * t.element_size()
                    if len(res) == 2:
                        A, B = (r.to(t.dtype) for r in res)
                        out[bi, hi] = (A @ B).to(t.dtype)
                        rs.append(A.shape[1])
                        comp_bytes += (A.numel() + B.numel()) * t.element_size()
                    else:
                        out[bi, hi] = t[bi, hi]
                        rs.append(min(s, d))
                        comp_bytes += s * d * t.element_size()
            setattr(layer, name, out)

    return (
        sum(ranks_k) / len(ranks_k) if ranks_k else 0,
        sum(ranks_v) / len(ranks_v) if ranks_v else 0,
        orig_bytes,
        comp_bytes
    )


def run_chunk(chunk_ids, model, tokenizer, adaptive_ranks=None, chunk_idx=None):
    """
    Run a single chunk and compute NLL.
    
    If adaptive_ranks is provided, compress KV cache with those ranks.
    Returns (nll, rank_stats, compression_ratio)
    """
    half = SEQ_LEN // 2
    ids = chunk_ids.to(model.device).unsqueeze(0)

    # Prefill first half
    pre = model(ids[:, :half], use_cache=True)
    first_logit = pre.logits[:, -1:, :]
    cache = pre.past_key_values

    # Apply adaptive compression if ranks provided
    if adaptive_ranks:
        rk, rv, ob, cb = compress_kv_adaptive(cache, {}, adaptive_ranks)
        ratio = ob / cb if cb > 0 else 1.0
    else:
        rk = rv = 0
        ratio = 1.0

    # Score second half
    out = model(ids[:, half:], past_key_values=cache, use_cache=True)
    logits = torch.cat([first_logit, out.logits[:, :-1, :]], dim=1).float()
    nll = F.cross_entropy(logits.flatten(0, 1), ids[0, half:], reduction="sum")

    return (nll.item(), rk, rv, ratio)


def main():
    print(f"Loading Qwen3-8B from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa"
    )
    model.eval()

    print("Loading wikitext-2 chunks...")
    chunks = [c for c in load_wikitext_chunks(N_CHUNKS) if c.shape[0] == SEQ_LEN]
    print(f"Loaded {len(chunks)} chunks of {SEQ_LEN} tokens each")

    # Initialize checkpoint manager
    ckpt = CheckpointManager(CHECKPOINT_DIR)
    
    # Check if entropy is already computed
    entropies = ckpt.load_entropy()
    if entropies is None:
        print("Computing per-layer attention entropy...")
        # Run on first chunk to extract entropies
        chunk_ids = chunks[0].to(model.device).unsqueeze(0)
        pre = model(chunk_ids[:, :SEQ_LEN], use_cache=True)
        entropies = capture_attention_entropy(model, pre.past_key_values)
        ckpt.save_entropy(entropies)
        print(f"Saved entropy to {ckpt.load_entropy.__globals__['checkpoint_dir']}/entropy.json")
    else:
        print(f"Loaded entropy from checkpoint ({len(entropies)} layers)")

    # Compute adaptive ranks
    total_ranks = N_LAYERS * int(0.10 * 1024)  # ~10% of hidden dim (assuming 1024 head dim)
    total_ranks = N_LAYERS * 100  # Simplified budget

    # Convert entropies to tensor
    entropy_list = [entropies.get(i, 0.0) for i in range(N_LAYERS)]
    entropy_tensor = torch.tensor(entropy_list, dtype=torch.float32)
    
    ranks = adaptive_ranks_from_entropy(entropy_tensor, total_ranks, min_rank=5, max_rank_pct=0.15)
    ckpt.save_ranks(ranks)
    print(f"Adaptive ranks saved ({len(ranks)} layers, total={total_ranks})")
    print(f"Rank distribution: min={min(ranks)}, max={max(ranks)}, mean={sum(ranks)/len(ranks):.1f}")

    # Check completed chunks
    completed = ckpt.get_completed_chunks()
    print(f"Completed chunks: {len(completed)}/{N_CHUNKS}")

    # Process remaining chunks
    acc_nll = 0.0
    acc_n = 0
    total_ratio = 0.0

    for ci, chunk in enumerate(chunks):
        if ci in completed:
            # Load cached result
            cached = ckpt.load_results(ci)
            if cached:
                acc_nll += cached["nll"]
                acc_n += cached["n"]
                total_ratio += cached.get("ratio", 1.0)
                print(f"chunk {ci + 1}/{N_CHUNKS} (cached)", flush=True)
                continue

        # Run chunk
        nll, rk, rv, ratio = run_chunk(chunk, model, None, adaptive_ranks=ranks, chunk_idx=ci)
        acc_nll += nll
        acc_n += SEQ_LEN // 2
        total_ratio += ratio

        # Save checkpoint for this chunk
        chunk_result = {
            "chunk_idx": ci,
            "nll": nll,
            "n": SEQ_LEN // 2,
            "ratio": ratio
        }
        ckpt.save_results(chunk_result, ci)

        print(f"chunk {ci + 1}/{N_CHUNKS} done (nll={nll/acc_n:.3f})", flush=True)

    # Final results
    mean_nll = acc_nll / acc_n if acc_n > 0 else 0.0
    mean_ratio = total_ratio / len(chunks) if chunks else 1.0
    
    results = {
        "model": "Qwen3-8B",
        "dataset": "wikitext-2-raw-v1 test",
        "seq_len": SEQ_LEN,
        "n_chunks": len(chunks),
        "n_layers": N_LAYERS,
        "eps": EPS_VALUE,
        "adaptive": True,
        "mean_entropy_per_layer": {k: v for k, v in list(entropies.items())[:5]},
        "ranks": ranks,
        "mean_rank": sum(ranks) / len(ranks),
        "mean_nll": mean_nll,
        "mean_ppl": math.exp(mean_nll) if mean_nll > 0 else None,
        "compression_ratio": mean_ratio
    }

    ckpt.save_results(results)
    print("\nFinal Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import math
    main()
