"""Tests for adaptive SVD KV compression."""
import torch
import json
import os
import pytest
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from adaptive_svd import (
    AdaptiveSVDKVCompressor,
    compute_attention_entropy,
    adaptive_ranks_from_entropy,
    CheckpointManager
)


class TestAdaptiveSVDKVCompressor:
    def test_basic_compression(self):
        """Test basic SVD compression with error tolerance."""
        torch.manual_seed(42)
        tokens, dim = 100, 50
        low_rank = 10
        base = torch.randn(tokens, low_rank) @ torch.randn(low_rank, dim)
        noise = torch.randn(tokens, dim) * 0.01
        x = base + noise

        comp = AdaptiveSVDKVCompressor(eps=1e-2)
        result = comp.compress(x)

        if len(result) == 2:
            A, B = result
            recon = comp.decompress(A, B)
        else:
            recon = result[0]

        rel_err = torch.linalg.norm(x - recon) / torch.linalg.norm(x)
        assert rel_err <= 1e-2, f"Relative error {rel_err} exceeds tolerance"

    def test_compression_saves_space(self):
        """Test that compression reduces memory for low-rank inputs."""
        torch.manual_seed(42)
        tokens, dim = 100, 50
        low_rank = 5
        x = torch.randn(tokens, low_rank) @ torch.randn(low_rank, dim)

        comp = AdaptiveSVDKVCompressor(eps=1e-2)
        result = comp.compress(x)

        if len(result) == 2:
            A, B = result
            compressed_bytes = (A.numel() + B.numel()) * 4
            raw_bytes = x.numel() * 4
            assert compressed_bytes < raw_bytes
        else:
            assert False, "Expected compression to save space for low-rank input"

    def test_zero_matrix(self):
        """Test handling of zero matrices."""
        x = torch.zeros(10, 20)
        comp = AdaptiveSVDKVCompressor(eps=1e-2)
        result = comp.compress(x)
        assert len(result) == 1  # Should return raw tensor


class TestAttentionEntropy:
    def test_entropy_computation(self):
        """Test attention entropy calculation."""
        torch.manual_seed(42)
        batch, heads, seq = 1, 8, 64
        scores = torch.randn(batch, heads, seq, seq)
        entropy = compute_attention_entropy(scores)

        assert entropy.dim() == 3
        assert entropy.shape == (batch, heads, seq)
        assert (entropy > 0).all()  # Entropy should be positive for softmax

    def test_high_entropy_uniform(self):
        """Test that uniform attention has high entropy."""
        seq = 64
        # Uniform probabilities
        probs = torch.ones(1, 1, seq) / seq
        scores = torch.log(probs + 1e-10)  # Convert to logit-like scores
        
        # Manual entropy calculation
        expected_entropy = -torch.sum(probs * torch.log(probs + 1e-10))

        torch.manual_seed(42)
        entropy = compute_attention_entropy(scores)
        # Should be close to log(seq) for uniform distribution
        assert abs(entropy.mean() - torch.log(torch.tensor(seq))).item() < 1.0


class TestAdaptiveRanksFromEntropy:
    def test_ranks_sum_to_budget(self):
        """Test that assigned ranks sum to total budget."""
        torch.manual_seed(42)
        n_layers = 40
        total_budget = 4000  # 100 per layer average

        entropies = torch.rand(n_layers) + 0.1  # Positive values
        ranks = adaptive_ranks_from_entropy(entropies, total_budget, min_rank=5)

        assert sum(ranks) == total_budget
        assert all(r >= 5 for r in ranks)

    def test_high_entropy_gets_higher_rank(self):
        """Test that high entropy layers get higher ranks."""
        torch.manual_seed(42)
        entropies = torch.tensor([0.1, 0.2, 0.5, 1.0])  # Increasing
        ranks = adaptive_ranks_from_entropy(entropies, 100, min_rank=5)

        # Higher entropy layers should have higher ranks
        assert ranks[3] >= ranks[2] >= ranks[1] >= ranks[0]

    def test_min_rank_respected(self):
        """Test that minimum rank is respected for all layers."""
        entropies = torch.ones(40) * 0.01  # Very low entropy
        ranks = adaptive_ranks_from_entropy(entropies, 400, min_rank=10)

        assert all(r >= 10 for r in ranks)


class TestCheckpointManager:
    def test_save_and_load_entropy(self):
        """Test entropy checkpoint save/load."""
        ckpt = CheckpointManager("/tmp/test_ckpt_adaptive")
        entropies = {str(i): float(i * 0.1) for i in range(40)}  # JSON keys are strings

        path = ckpt.save_entropy(entropies)
        assert os.path.exists(path)

        loaded = ckpt.load_entropy()
        assert loaded == entropies

    def test_save_and_load_ranks(self):
        """Test ranks checkpoint save/load."""
        ckpt = CheckpointManager("/tmp/test_ckpt_adaptive")
        ranks = [i % 20 + 5 for i in range(40)]

        path = ckpt.save_ranks(ranks)
        assert os.path.exists(path)

        loaded = ckpt.load_ranks()
        assert loaded == ranks

    def test_get_completed_chunks(self):
        """Test completed chunk tracking."""
        ckpt = CheckpointManager("/tmp/test_ckpt_adaptive")
        
        # Save some chunk results
        for ci in [0, 1, 3, 5]:
            ckpt.save_results({"chunk": ci, "nll": 1.0}, ci)

        completed = ckpt.get_completed_chunks()
        assert completed == {0, 1, 3, 5}

    def test_missing_checkpoint_returns_none(self):
        """Test that missing checkpoints return None."""
        ckpt = CheckpointManager("/tmp/test_ckpt_adaptive")
        assert ckpt.load_entropy() is not None  # We saved above
        assert ckpt.load_results(999) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
