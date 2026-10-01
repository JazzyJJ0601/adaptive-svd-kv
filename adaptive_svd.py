"""
Adaptive per-layer SVD rank selection for KV cache compression.

Based on attention entropy to assign ranks proportionally:
- High entropy layers -> higher rank (less compressible)
- Low entropy layers -> lower rank (more compressible)

Total parameter budget matches fixed-rank baseline.
"""
import torch
import math
import json
import os
from typing import Optional


class AdaptiveSVDKVCompressor:
    def __init__(self, eps: float = 1e-2):
        self.eps = eps

    def compress(self, x: torch.Tensor, target_rank: Optional[int] = None) -> tuple:
        """
        Compress a single head's matrix (seq, d) using SVD.
        If target_rank is given, use that rank directly.
        Otherwise, choose smallest rank such that relative error <= eps.
        Returns (A, B) low-rank factors if compression saves space, else returns (x,).
        """
        s, d = x.shape
        fro_norm = torch.linalg.norm(x)

        if fro_norm == 0 or target_rank == 0:
            return (x,)

        # Full SVD
        U, S, Vt = torch.linalg.svd(x, full_matrices=False)

        if target_rank is not None:
            # Use specified rank directly
            rank = min(target_rank, len(S), s, d)
        else:
            # Find smallest rank such that relative error <= eps
            target = 1 - self.eps * self.eps
            cumsum_sq = torch.cumsum(S ** 2, dim=0)
            ratios = cumsum_sq / (fro_norm ** 2)

            if not (ratios >= target).any():
                rank = len(S)
            else:
                rank = (ratios >= target).to(torch.long).argmax().item() + 1

        # Check if compression saves space
        if rank * (s + d) >= s * d:
            return (x,)

        # Return low-rank factors
        A = U[:, :rank] @ torch.diag(S[:rank])
        B = Vt[:rank, :]
        return (A, B)

    def decompress(self, A: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
        """Decompress low-rank factors back to original shape."""
        if B is None:
            return A
        return A @ B


def compute_attention_entropy(attn_scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute attention entropy for given scores.
    Entropy = -sum(p * log(p)) where p = softmax(scores)
    Higher entropy = more uniform attention = less compressible
    """
    attn_probs = torch.softmax(attn_scores, dim=dim)
    # Avoid log(0)
    attn_probs = torch.clamp(attn_probs, min=1e-10)
    entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=dim)
    return entropy


def adaptive_ranks_from_entropy(
    entropies: torch.Tensor,
    total_ranks: int,
    min_rank: int = 5,
    max_rank_pct: float = 0.15
) -> list:
    """
    Assign SVD ranks proportionally to entropy values.
    
    Args:
        entropies: Tensor of per-layer entropy values (40 for Qwen3-8B)
        total_ranks: Total rank budget (same as fixed-rank baseline)
        min_rank: Minimum rank for any layer
        max_rank_pct: Maximum rank as fraction of hidden dimension
    
    Returns:
        List of ranks per layer
    """
    # Normalize entropies to sum to 1
    entropies = torch.clamp(entropies, min=1e-10)
    probs = entropies / entropies.sum()
    
    # Assign ranks proportionally
    ranks = torch.round(probs * total_ranks).int()
    
    # Ensure minimum rank and total budget
    ranks = torch.clamp(ranks, min=min_rank)
    
    # Adjust to match total budget exactly
    current_total = ranks.sum().item()
    diff = total_ranks - current_total
    
    if diff != 0:
        # Sort layers by entropy (descending) and adjust top layers
        sorted_idx = torch.argsort(entropies, descending=True)
        for i in range(abs(diff)):
            idx = sorted_idx[i]
            if diff > 0:
                ranks[idx] += 1
            else:
                ranks[idx] = max(ranks[idx].item() - 1, min_rank)
    
    return ranks.tolist()


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_entropy(self, entropies: dict):
        """Save per-layer entropy values."""
        path = os.path.join(self.checkpoint_dir, "entropy.json")
        with open(path, "w") as f:
            json.dump(entropies, f, indent=2)
        return path

    def load_entropy(self) -> Optional[dict]:
        """Load per-layer entropy values if checkpoint exists."""
        path = os.path.join(self.checkpoint_dir, "entropy.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def save_ranks(self, ranks: list):
        """Save per-layer rank assignments."""
        path = os.path.join(self.checkpoint_dir, "ranks.json")
        with open(path, "w") as f:
            json.dump({"ranks": ranks}, f, indent=2)
        return path

    def load_ranks(self) -> Optional[list]:
        """Load per-layer rank assignments if checkpoint exists."""
        path = os.path.join(self.checkpoint_dir, "ranks.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f).get("ranks")
        return None

    def save_results(self, results: dict, chunk_idx: int = None):
        """Save per-chunk results with checkpointing."""
        if chunk_idx is not None:
            path = os.path.join(self.checkpoint_dir, f"adaptive_results_{chunk_idx}.json")
        else:
            path = os.path.join(self.checkpoint_dir, "adaptive_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        return path

    def load_results(self, chunk_idx: int = None) -> Optional[dict]:
        """Load per-chunk results if checkpoint exists."""
        if chunk_idx is not None:
            path = os.path.join(self.checkpoint_dir, f"adaptive_results_{chunk_idx}.json")
        else:
            path = os.path.join(self.checkpoint_dir, "adaptive_results.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def get_completed_chunks(self) -> set:
        """Get set of completed chunk indices."""
        completed = set()
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith("adaptive_results_") and f.endswith(".json"):
                try:
                    idx = int(f.replace("adaptive_results_", "").replace(".json", ""))
                    completed.add(idx)
                except ValueError:
                    pass
        return completed
