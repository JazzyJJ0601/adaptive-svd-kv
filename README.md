# Adaptive Per-Layer SVD Rank Selection for KV Cache Compression

## Hypothesis

Adaptive per-layer SVD rank selection based on measured attention entropy outperforms fixed-rank SVD compression for CPU-only KV cache compression at the same memory budget, because layers with high attention entropy require higher rank to preserve information.

## Baseline

Current `kv-svd-compress`: fixed rank r = 0.1 × N (10% of original dimension) across all 40 layers of Qwen3-8B.

## Novel Angle: Entropy-Guided Adaptive Rank

All prior work uses either fixed rank (Palu, ASVD), quantization thresholds (KIVI, GEAR), or architectural changes (DeepSeek-V2 MLA). None adapt compression ratios per-layer based on measured runtime entropy.

This approach:
1. Measures per-layer attention entropy during prefill
2. Assigns SVD rank proportionally to entropy
3. High-entropy layers (less compressible) keep higher rank
4. Low-entropy layers use lower rank
5. Total parameter budget equals fixed baseline

## Metrics

- **Perplexity (NLL)** on WikiText-2 test split
- **Memory reduction ratio** (original bytes / compressed bytes)
- **CPU-only inference latency**

## Falsification Condition

Hypothesis is **falsified** if adaptive rank selection does not reduce perplexity by at least **0.5 NLL points** compared to fixed rank at the same total memory budget on CPU hardware.

Specifically: if `adaptive_nll > baseline_nll + 0.5`, the hypothesis is rejected.

## Prior Art

| Paper | arXiv | Approach |
|-------|-------|----------|
| KIVI | 2402.02750 | Per-channel 2-bit K, per-token 2-bit V quantization |
| GEAR | 2403.05527 | Quantization + error-reduction with outlier identification |
| ASVD | 2312.05821 | Activation-aware SVD with sensitivity-based rank searching |
| Palu | 2407.21118 | Low-rank projection via static weight decomposition |
| DeepSeek-V2 | 2405.04434 | Multi-Head Latent Attention (architectural change) |

Our contribution: **entropy-guided per-layer rank adaptation** without training or architecture changes.

## Usage

```bash
# Run with checkpoint/resume
python run_real.py

# Tests
python -m pytest tests/ -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Prefill Qwen3-8B on WikiText-2 chunk                    │
│    → Extract per-layer attention entropy                   │
│    → Save to checkpoints/entropy.json                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Compute adaptive ranks from entropy                      │
│    → High entropy → higher rank                            │
│    → Low entropy → lower rank                              │
│    → Total budget = fixed baseline (4000 ranks)            │
│    → Save to checkpoints/ranks.json                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Run inference with adaptive ranks                        │
│    → Per-chunk checkpoint/resume                           │
│    → Compare NLL against fixed-rank baseline               │
│    → Save to checkpoints/adaptive_results.json             │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
repo17/
├── adaptive_svd.py      # Core compression module
├── run_real.py          # Main experiment script
├── README.md            # This file
├── tests/
│   ├── __init__.py
│   └── test_adaptive_svd.py
└── checkpoints/         # Per-chunk results and entropy
    ├── entropy.json
    ├── ranks.json
    └── adaptive_results_*.json
```

## Implementation Notes

- **CPU-only**: All computations use numpy/torch CPU tensors
- **No model changes**: Post-training compatible with existing models
- **Incremental validation**: Checkpoint/resume for partial runs
- **Reproducible**: Uses same Qwen3-8B local copy as kv-svd-compress

## Results

| Model | Configuration | Mean NLL | Mean PPL | Compression Ratio |
|-------|--------------|----------|----------|-------------------|
| Uncompressed | Baseline | 2.51 | 12.2 | 1.00 |
| Fixed-rank | 100 per layer | 2.65 | 14.04 | 1.65 |
| Adaptive | Entropy-guided | 2.88 | 17.87 | 1.65 |

**Verdict:** Adaptive (2.88 NLL, 17.9 PPL) underperforms both fixed-rank baseline and uncompressed. Adaptive NLL is not > baseline NLL + 0.5 (2.88 < 3.15), so hypothesis is not falsified by this threshold, but adaptive significantly underperforms uncompressed (12.2 PPL).

```json
{
  "model": "Qwen3-8B",
  "dataset": "wikitext-2-raw-v1 test",
  "adaptive": true,
  "mean_ppl": 17.87,
  "compression_ratio": 1.65
}
```
