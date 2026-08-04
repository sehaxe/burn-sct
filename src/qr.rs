//! Householder QR decomposition.
//!
//! Two implementations of the same Householder-with-sign algorithm:
//!
//! - [`qr_cpu`]: native, single-allocation pass over `&[f32]` (LAPACK
//!   `dgeqrf`/`dorgqr` scheme). This is what [`crate::orthogonalize`] uses:
//!   a tensor-op QR costs 100-300x more than LAPACK on CPU (per-step
//!   allocations and dispatches dominate), and retraction runs outside the
//!   autodiff graph so the device sync is free.
//! - [`qr`]: batched tensor-op version, adapted from burn-rs/burn,
//!   `crates/burn-tensor/src/tensor/linalg/qr.rs` (main branch, authored by
//!   the burn-rs maintainers, MIT/Apache-2.0). Reduced to O(m k^2) for
//!   SCT's tall-skinny factors (m >> k): reflection vectors are stored in
//!   the R pass and Q is built back-to-front (LAPACK `orgqr` scheme), so
//!   the m x m Q is never materialized. Kept as the reference/GPU path
//!   until a fused kernel replaces it.
//!
//! Combined with the `sign(diag(R))` correction in [`crate::orthogonalize`]
//! both reproduce the paper's `safe_qr` (PyTorch `torch.linalg.qr` + sign
//! flip), verified by `tests/cmp_reference.rs`.

use burn::tensor::backend::Backend;
use burn::tensor::{s, Tensor};

/// Native CPU Householder QR of a row-major `m x n` matrix.
///
/// Returns `(q, r)` where `q` is the reduced `m x min(m,n)` orthonormal
/// factor and `r` the upper-triangular `min(m,n) x n` factor (row-major).
/// One allocation per pass; numerically identical to [`qr`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

/// SIMD (AVX2/FMA) dot product of two equal-length slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_pair(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(av, bv, acc);
        i += 8;
    }
    let mut s = hsum(acc);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// SIMD (AVX2/FMA) `a -= f * b` for two equal-length slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_pair(a: &mut [f32], b: &[f32], f: f32) {
    use std::arch::x86_64::*;
    let n = a.len();
    let fv = _mm256_set1_ps(f);
    let mut i = 0;
    while i + 8 <= n {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let r = _mm256_fnmadd_ps(fv, bv, av);
        _mm256_storeu_ps(a.as_mut_ptr().add(i), r);
        i += 8;
    }
    while i < n {
        a[i] -= f * b[i];
        i += 1;
    }
}

/// Full Q-pass column (all i steps) under AVX2/FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q_pass_col_avx(col: &mut [f32], r: &[f32], tau: &[f32], m: usize, k: usize) {
    for i in (0..k).rev() {
        let rows = m - i;
        let t = tau[i];
        if t == 0.0 {
            continue;
        }
        let w = &r[i * m + i + 1..(i + 1) * m];
        let dot = col[i] + dot_pair(&col[i + 1..i + rows], w);
        if dot != 0.0 {
            let f = t * dot;
            col[i] -= f;
            apply_pair(&mut col[i + 1..i + rows], w, f);
        }
    }
}

/// SIMD (AVX2/FMA) R-pass column updates for a set of disjoint columns.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn r_cols_avx(cols: &mut [&mut [f32]], w: &[f32], i: usize, rows: usize, t: f32) {
    for col in cols.iter_mut() {
        let dot = col[i] + dot_pair(&col[i + 1..i + rows], &w[1..rows]);
        if dot != 0.0 {
            let f = t * dot;
            col[i] -= f * w[0];
            apply_pair(&mut col[i + 1..i + rows], &w[1..rows], f);
        }
    }
}

pub fn qr_cpu(a: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let k = m.min(n);
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get().min(4))
        .unwrap_or(1);
    let parallel = n_threads > 1 && k >= 64 && m >= 1024;
    #[cfg(target_arch = "x86_64")]
    let avx = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let avx = false;
    // Column-major storage: reflectors live in the lower part of column i
    // of R; every column update is a contiguous slice, so the per-step
    // column loops can run in parallel without aliasing.
    let mut r = vec![0.0f32; k * m];
    for c in 0..k {
        for rr in 0..m {
            r[c * m + rr] = a[rr * n + c];
        }
    }
    let mut tau = vec![0.0f32; k];
    for i in 0..k {
        let rows = m - i;
        // build w from column i (LAPACK dgeqrf): w = v/u0, w[0] = 1
        let mut w = vec![0.0f32; rows];
        let col_i = &r[i * m + i..(i + 1) * m];
        let mut norm2 = 0.0f32;
        for rr in 0..rows {
            norm2 += col_i[rr] * col_i[rr];
        }
        let norm = norm2.sqrt();
        let v0 = col_i[0];
        let sign = if v0 < 0.0 { 1.0 } else { -1.0 };
        let u0 = v0 - sign * norm;
        let t = if norm2 > 0.0 { -u0 / norm * sign } else { 0.0 };

        tau[i] = t;
        w[0] = 1.0;
        if u0 != 0.0 {
            for rr in 1..rows {
                w[rr] = col_i[rr] / u0;
            }
        }
        // store the reflector (w[1..]) and the R diagonal (beta, LAPACK
        // dgeqrf layout: w[0] = 1 is implicit, R[i][i] = sign * norm)
        r[i * m + i] = sign * norm;
        r[i * m + i + 1..(i + 1) * m].copy_from_slice(&w[1..]);
        // R[i.., i..k] -= tau * w (w^T R[i.., i..k]); sequential (per-step
        // thread spawns cost more than the columns they cover), SIMD when
        // available.
        if t != 0.0 && k - i - 1 > 0 {
            let mut cols: Vec<&mut [f32]> = r[(i + 1) * m..k * m].chunks_mut(m).collect();
            if avx {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    r_cols_avx(cols.as_mut_slice(), &w, i, rows, t);
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let _ = &cols;
                    let _ = &w;
                    let _ = &rows;
                    let _ = &t;
                }
            } else {
                for col in cols.iter_mut() {
                    let mut dot = col[i];
                    for rr in 1..rows {
                        dot += w[rr] * col[i + rr];
                    }
                    if dot != 0.0 {
                        let f = t * dot;
                        col[i] -= f * w[0];
                        for rr in 1..rows {
                            col[i + rr] -= f * w[rr];
                        }
                    }
                }
            }
        }
    }
    // Q pass: Q = H_1..H_k I_k, back-to-front; H_i touches rows i.. only.
    // Columns of Q are independent, so in the parallel case each thread owns
    // a column range and walks all i steps itself (one spawn per thread).
    let mut q = vec![0.0f32; m * k];
    for c in 0..k {
        q[c * m + c] = 1.0; // transposed storage: q[c][row]
    }
    if parallel {
        let chunk = k.div_ceil(n_threads);
        let tau = &tau;
        let r = &r;
        std::thread::scope(|s| {
            for (ci, qc) in q.chunks_mut(chunk * m).enumerate() {
                s.spawn(move || {
                    let lo = ci * chunk;
                    let hi = ((ci + 1) * chunk).min(k);
                    for c in lo..hi {
                        let col = &mut qc[(c - lo) * m..(c - lo + 1) * m];
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            q_pass_col_avx(col, r, tau, m, k);
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            for i in (0..k).rev() {
                                let rows = m - i;
                                let t = tau[i];
                                if t == 0.0 {
                                    continue;
                                }
                                let w = &r[i * m + i + 1..(i + 1) * m];
                                let mut dot = col[i];
                                for rr in 1..rows {
                                    dot += w[rr - 1] * col[i + rr];
                                }
                                if dot != 0.0 {
                                    let f = t * dot;
                                    col[i] -= f;
                                    for rr in 1..rows {
                                        col[i + rr] -= f * w[rr - 1];
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });
    } else if avx {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            for c in 0..k {
                q_pass_col_avx(&mut q[c * m..(c + 1) * m], &r, &tau, m, k);
            }
        }
    } else {
        for c in 0..k {
            let col = &mut q[c * m..(c + 1) * m];
            for i in (0..k).rev() {
                let rows = m - i;
                let t = tau[i];
                if t == 0.0 {
                    continue;
                }
                let w = &r[i * m + i + 1..(i + 1) * m];
                let mut dot = col[i];
                for rr in 1..rows {
                    dot += w[rr - 1] * col[i + rr];
                }
                if dot != 0.0 {
                    let f = t * dot;
                    col[i] -= f;
                    for rr in 1..rows {
                        col[i + rr] -= f * w[rr - 1];
                    }
                }
            }
        }
    }
    // sign(diag(R)) correction: columns of Q flip to make R's diagonal >= 0
    for i in 0..k {
        if r[i * m + i] < 0.0 {
            let col = &mut q[i * m..(i + 1) * m];
            for x in col.iter_mut() {
                *x = -*x;
            }
        }
    }
    // convert q to row-major [m, k]
    let mut q_rm = vec![0.0f32; m * k];
    for c in 0..k {
        for rr in 0..m {
            q_rm[rr * k + c] = q[c * m + rr];
        }
    }
    let mut r_k = vec![0.0f32; k * k];
    for i in 0..k {
        let flip = r[i * m + i] < 0.0;
        for c in i..k {
            let v = r[i * m + c];
            r_k[i * k + c] = if flip { -v } else { v };
        }
    }
    (q_rm, r_k)
}



/// QR decomposition of a matrix `[n_rows, n_cols]`.
///
/// Returns `(Q, R)` with `A = QR`. `Q` has `min(n_rows, n_cols)` columns;
/// `R` is `[n_cols, n_cols]` when `reduced && n_rows > n_cols`, otherwise
/// `[n_rows, n_cols]`.
pub fn qr<B: Backend>(tensor: Tensor<B, 2>, reduced: bool) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let dims = tensor.dims();
    let device = tensor.device();
    let (n_rows, n_cols) = (dims[0], dims[1]);

    let max_iters = n_rows.min(n_cols);
    let mut r = tensor.clone();

    // Pass 1: reduce A to upper-triangular R, remembering (w, tau).
    let mut reflectors: Vec<(Tensor<B, 2>, Tensor<B, 2>)> = Vec::with_capacity(max_iters);
    for i in 0..max_iters {
        let sub_tensor = r.clone().slice_dim(0, s![i..]).slice_dim(1, s![i..]);
        let v = sub_tensor.clone().slice_dim(1, 0..1);
        let v0 = v.clone().slice_dim(0, 0..1);
        let zeros = v0.clone().zeros_like();
        let norm_v = v
            .clone()
            .slice_dim(0, s![..])
            .powf_scalar(2.0)
            .sum_dim(0)
            .sqrt();

        // removing zeros from the sign
        let sign = -v0.clone().sign();
        let mask = sign.clone().is_close(zeros.clone(), None, None);
        let sign = sign.mask_fill(mask, -1.0);

        // if norm_v==0, the vector w has to be zero and no reflection is applied
        let u0 = v0.clone().sub(norm_v.clone().mul(sign.clone()));

        let mask_norm = norm_v.clone().is_close(zeros, None, None);
        let mut tau = -u0.clone().div(norm_v.clone()).mul(sign.clone());
        tau = tau.clone().mask_fill(mask_norm.clone(), 0.0);

        let e0 = v0.clone().mul_scalar(0.0).add_scalar(1.0);
        let mut w = v.clone().div(u0.clone());
        w = w.slice_assign([s![0], s![..]], e0);
        w = w.clone().mask_fill(mask_norm, 0.0);

        // H_i * sub = sub - tau * w * (w^T sub)
        let wt_sub = w.clone().transpose().matmul(sub_tensor.clone());
        let upd = w.clone().matmul(wt_sub).mul(tau.clone());
        r = r.slice_assign([s![i..], s![i..]], sub_tensor - upd);

        reflectors.push((w, tau));
    }

    // Pass 2: Q = H_1..H_k I_k, built back-to-front; H_i only touches rows i..
    // so the reduced Q is exact and costs O(m k^2) total.
    let mut q: Tensor<B, 2> = Tensor::eye(n_rows, &device).slice([s![..], s![0..max_iters]]);
    for i in (0..max_iters).rev() {
        let (w, tau) = &reflectors[i];
        let q_sub = q.clone().slice([s![i..], s![..]]);
        let wt_q = w.clone().transpose().matmul(q_sub.clone());
        let upd = w.clone().matmul(wt_q).mul(tau.clone());
        q = q.slice_assign([s![i..], s![..]], q_sub - upd);
    }

    if reduced & (n_rows > n_cols) {
        let result_r = r.slice([s![0..n_cols], s![0..n_cols]]);
        return (q, result_r);
    }
    (q, r)
}
