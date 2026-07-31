# burn-sct - Spectral Compact Training

[![CI](https://github.com/sehaxe/burn-sct/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-sct/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-sct)](https://crates.io/crates/burn-sct)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)

Drop-in `nn.Linear` replacement for [Burn](https://burn.dev). Weights stored as
**permanent truncated SVD**: `W = U * diag(s) * V^T`. The dense matrix is never
materialized. After each optimizer step, U and V are retracted to the Stiefel
manifold via QR decomposition.

> Based on [Spectral Compact Training](https://arxiv.org/abs/2604.00733) (Kohlberger, 2026).
> Up to **199× memory reduction** per MLP layer at rank 32.

## Quick start

```rust
use burn_sct::{SctConfig, SctLinear};

let device = Default::default();
let cfg = SctConfig::new(512, 2048, 64);  // in=512, out=2048, rank=64
let mut layer = SctLinear::<NdArray>::new(&cfg, &device);

// Forward pass - three small matmuls, no dense matrix
let y = layer.forward(x);  // [batch, 512] → [batch, 2048]

// After optimizer.step(), maintain orthonormality
layer.retract();
```

## How it works

```
Dense:   y = x @ W                          [m×n matrix, O(b·m·n) FLOPs]
SCT:     y = (x @ U) * s @ V^T              [three small matmuls, O(b·k·(m+n)) FLOPs]
```

Where `U ∈ ℝ^{m×k}`, `V ∈ ℝ^{n×k}` have orthonormal columns, `s ∈ ℝ^k`.

## Memory savings (Adam, rank 32)

| Model | Dense MLP | SCT MLP | Compression |
|-------|-----------|---------|-------------|
| SmolLM2-135M | 14.2 MB | 1.1 MB | 13× |
| SmolLM2-1.7B | 268.4 MB | 5.2 MB | 51× |
| LLaMA-7B | 721.4 MB | 7.7 MB | 93× |
| LLaMA-70B | 3,758 MB | 18.9 MB | **199×** |

## License

AGPL-3.0
