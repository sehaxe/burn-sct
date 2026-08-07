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

use burn::tensor::Tensor;

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
pub(crate) unsafe fn dot_pair(a: &[f32], b: &[f32]) -> f32 {
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

/// SIMD (AVX2/FMA) fused (|a|^2, |b|^2, a·b) of two equal-length f64 slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot3_avx_f64(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut aa = _mm256_setzero_pd();
    let mut bb = _mm256_setzero_pd();
    let mut ab = _mm256_setzero_pd();
    let mut i = 0;
    while i + 4 <= n {
        let av = _mm256_loadu_pd(a.as_ptr().add(i));
        let bv = _mm256_loadu_pd(b.as_ptr().add(i));
        aa = _mm256_fmadd_pd(av, av, aa);
        bb = _mm256_fmadd_pd(bv, bv, bb);
        ab = _mm256_fmadd_pd(av, bv, ab);
        i += 4;
    }
    let mut s1 = hsum_pd(aa);
    let mut s2 = hsum_pd(bb);
    let mut s3 = hsum_pd(ab);
    while i < n {
        s1 += a[i] * a[i];
        s2 += b[i] * b[i];
        s3 += a[i] * b[i];
        i += 1;
    }
    (s1, s2, s3)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_pd(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_castpd256_pd128(v);
    let s = _mm_add_pd(lo, hi);
    let s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
    _mm_cvtsd_f64(s)
}

/// SIMD (AVX2/FMA) Jacobi rotation on f64 slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn rotate2_avx_f64(a: &mut [f64], b: &mut [f64], c: f64, s: f64) {
    use std::arch::x86_64::*;
    let n = a.len();
    let cv = _mm256_set1_pd(c);
    let sv = _mm256_set1_pd(s);
    let mut i = 0;
    while i + 4 <= n {
        let av = _mm256_loadu_pd(a.as_ptr().add(i));
        let bv = _mm256_loadu_pd(b.as_ptr().add(i));
        let a2 = _mm256_fmadd_pd(sv, bv, _mm256_mul_pd(cv, av));
        let b2 = _mm256_fnmadd_pd(sv, av, _mm256_mul_pd(cv, bv));
        _mm256_storeu_pd(a.as_mut_ptr().add(i), a2);
        _mm256_storeu_pd(b.as_mut_ptr().add(i), b2);
        i += 4;
    }
    while i < n {
        let av = a[i];
        let bv = b[i];
        a[i] = c * av + s * bv;
        b[i] = c * bv - s * av;
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

/// SIMD (AVX2/FMA) R-pass column update: `col = col - t·(refl^T·col)·refl`,
/// reading the reflector from R storage (`refl` = `r[i·m + i+1..]`). The
/// arithmetic is identical to the sequential `r_cols_avx` path (same
/// product order, `w[0]` is always 1.0 so `col[0] -= f·1.0` == `-= f`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn r_update_col_avx(col: &mut [f32], refl: &[f32], t: f32) {
    let dot = col[0] + dot_pair(&col[1..], refl);
    if dot != 0.0 {
        let f = t * dot;
        col[0] -= f;
        apply_pair(&mut col[1..], refl, f);
    }
}

fn r_update_col_scalar(col: &mut [f32], refl: &[f32], t: f32) {
    let mut dot = col[0];
    for rr in 0..refl.len() {
        dot += refl[rr] * col[1 + rr];
    }
    if dot != 0.0 {
        let f = t * dot;
        col[0] -= f;
        for rr in 0..refl.len() {
            col[1 + rr] -= f * refl[rr];
        }
    }
}

/// Thread budget: the Q pass is compute-bound, so up to 12 hardware
/// threads; the retract runs its two QRs sequentially (each gets the full
/// budget instead of two thread-starved halves).
pub(crate) fn thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get().min(12))
        .unwrap_or(1)
}

/// Householder R pass: transposes `a` (row-major `m x n`) into the
/// column-major work buffer `r` (reflectors below the diagonal, R[i][i] on
/// it) and returns `(r, tau, k)`. Column-major storage makes every column
/// update a contiguous slice; the transpose is done in 32-row blocks (a
/// reads coalesced, r writes complete full 128B cache lines). The
/// wavefront/sequential split mirrors the old single qr_cpu.
pub(crate) fn r_pass(
    a: &[f32],
    m: usize,
    n: usize,
    n_threads: usize,
) -> (Vec<f32>, Vec<f32>, usize) {
    let k = m.min(n);
    let parallel = n_threads > 1 && k >= 64 && m >= 1024;
    #[cfg(target_arch = "x86_64")]
    let avx = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let avx = false;
    let mut r = vec![0.0f32; k * m];
    let mut bb = 0;
    while bb < m {
        let r1 = (bb + 32).min(m);
        for c in 0..k {
            let dst = &mut r[c * m + bb..c * m + r1];
            let mut i = 0;
            let mut rr = bb;
            while rr < r1 {
                dst[i] = a[rr * n + c];
                i += 1;
                rr += 1;
            }
        }
        bb += 32;
    }
    let mut tau = vec![0.0f32; k];
    if parallel {
        // Wavefront R pass: thread t owns the contiguous column range
        // [t·chunk, (t+1)·chunk). At step i the owner computes the reflector
        // (identical arithmetic to the sequential path), a barrier publishes
        // it, then every thread updates its own columns j > i. Column j is
        // only ever touched by its owner, so the per-column operation order
        // is unchanged and the result is bit-identical to sequential.
        let chunk = k.div_ceil(n_threads);
        let n_chunks = k.div_ceil(chunk);
        let barrier = std::sync::Barrier::new(n_chunks);
        // SAFETY: raw access is ordered by `barrier` (a full fence between
        // the owner's reflector write and every cross-thread read); each
        // thread only writes its own disjoint column range.
        let r_ptr = r.as_mut_ptr() as usize;
        let tau_ptr = tau.as_mut_ptr() as usize;
        std::thread::scope(|s| {
            for (t, cols) in r.chunks_mut(chunk * m).enumerate().take(n_chunks) {
                let lo = t * chunk;
                let hi = ((t + 1) * chunk).min(k);
                let barrier = &barrier;
                s.spawn(move || {
                    // SAFETY: valid for the lifetime of `r`/`tau` (both live
                    // until after the scope); access is barrier-ordered.
                    let r_ptr = r_ptr as *mut f32;
                    let tau_ptr = tau_ptr as *mut f32;
                    for i in 0..k {
                        let rows = m - i;
                        if i >= lo && i < hi {
                            // col_i starts at the diagonal (row i), exactly
                            // like the sequential path.
                            let col_i = &mut cols[(i - lo) * m + i..(i - lo + 1) * m];
                            let mut norm2 = 0.0f32;
                            for rr in 0..rows {
                                norm2 += col_i[rr] * col_i[rr];
                            }
                            let norm = norm2.sqrt();
                            let v0 = col_i[0];
                            let sign = if v0 < 0.0 { 1.0 } else { -1.0 };
                            let u0 = v0 - sign * norm;
                            let tv = if norm2 > 0.0 { -u0 / norm * sign } else { 0.0 };
                            // SAFETY: barrier-ordered, single writer.
                            unsafe {
                                *tau_ptr.add(i) = tv;
                            }
                            if u0 != 0.0 {
                                for rr in 1..rows {
                                    col_i[rr] = col_i[rr] / u0;
                                }
                            }
                            col_i[0] = sign * norm;
                        }
                        barrier.wait();
                        // SAFETY: reflector written by the owner before the
                        // barrier; read only after.
                        let tv = unsafe { *tau_ptr.add(i) };
                        if tv != 0.0 && i + 1 < hi {
                            let refl =
                                unsafe { std::slice::from_raw_parts(r_ptr.add(i * m + i + 1), rows - 1) };
                            let from = (i + 1).max(lo);
                            for cc in from - lo..hi - lo {
                                let col = &mut cols[cc * m + i..cc * m + m];
                                #[cfg(target_arch = "x86_64")]
                                if avx {
                                    unsafe {
                                        r_update_col_avx(col, refl, tv);
                                    }
                                    continue;
                                }
                                r_update_col_scalar(col, refl, tv);
                            }
                        }
                    }
                });
            }
        });
    } else {
        let mut w = vec![0.0f32; m];
        for i in 0..k {
            let rows = m - i;
            // build w from column i (LAPACK dgeqrf): w = v/u0, w[0] = 1
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
            r[i * m + i + 1..(i + 1) * m].copy_from_slice(&w[1..rows]);
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
    }
    (r, tau, k)
}

/// Extracts the upper-triangular factor `R` (`[k x k]` row-major) from the
/// column-major reflector storage, applying the same sign(diag(R)) flip
/// that `q_pass` applies to Q so that `A = Q·R` is preserved.
pub(crate) fn r_k_from_r(r: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut r_k = vec![0.0f32; k * k];
    for i in 0..k {
        let flip = r[i * m + i] < 0.0;
        for c in i..k {
            r_k[i * k + c] = if flip { -r[c * m + i] } else { r[c * m + i] };
        }
    }
    r_k
}

/// Q pass: builds the reduced `Q` (with the sign(diag(R)) correction) from
/// the reflector storage and returns it row-major `[m x k]`. Q = H_1..H_k I_k
/// back-to-front; H_i touches rows i.. only, so columns are independent and
/// the parallel case splits them across threads (one spawn per thread).
pub(crate) fn q_pass(
    r: &[f32],
    tau: &[f32],
    m: usize,
    k: usize,
    n_threads: usize,
) -> Vec<f32> {
    let parallel = n_threads > 1 && k >= 64 && m >= 1024;
    #[cfg(target_arch = "x86_64")]
    let avx = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let avx = false;
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
    // convert q to row-major [m, k]: coalesced writes to q_rm, strided
    // reads from q (which sits in L2 for the ranks SCT uses)
    let mut q_rm = vec![0.0f32; m * k];
    for rr in 0..m {
        for c in 0..k {
            q_rm[rr * k + c] = q[c * m + rr];
        }
    }
    q_rm
}

/// Host-side Cholesky of a symmetric positive-definite `[k, k]` row-major
/// matrix; returns the upper-triangular factor `R` (row-major `[k, k]`) with
/// `G = R^T·R`. Used by the CUDA retraction path: the Gram matrix is small
/// (k <= 1024, typically 128-256), so a single host thread finishes in
/// microseconds, far faster than a GPU kernel would (cubecl 0.11 serializes
/// launches and its barriers/atomics cost ~ms).
pub(crate) fn cholesky_host(g: &[f32], k: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; k * k];
    for j in 0..k {
        let mut acc = g[j * k + j];
        for l in 0..j {
            acc -= r[l * k + j] * r[l * k + j];
        }
        // clamp: G may be rank-deficient at f32 rounding; retraction inputs
        // are near-orthonormal, so the clamp only fires on numerical noise
        let d = acc.max(1e-30f32).sqrt();
        r[j * k + j] = d;
        for i in (j + 1)..k {
            let mut acc2 = g[j * k + i];
            for l in 0..j {
                acc2 -= r[l * k + j] * r[l * k + i];
            }
            r[j * k + i] = acc2 / d;
        }
    }
    r
}

pub fn qr_cpu(a: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let n_threads = thread_count();
    let (r, tau, k) = r_pass(a, m, n, n_threads);
    let r_k = r_k_from_r(&r, m, k);
    let q_rm = q_pass(&r, &tau, m, k, n_threads);
    (q_rm, r_k)
}



/// QR decomposition of a matrix `[n_rows, n_cols]`.
///
/// Returns `(Q, R)` with `A = QR`. `Q` has `min(n_rows, n_cols)` columns;
/// `R` is `[n_cols, n_cols]` when `reduced && n_rows > n_cols`, otherwise
/// `[n_rows, n_cols]`.
pub fn qr(tensor: Tensor<2>, reduced: bool) -> (Tensor<2>, Tensor<2>) {
    let dims = tensor.dims();
    let device = tensor.device();
    let (n_rows, n_cols) = (dims[0], dims[1]);

    let max_iters = n_rows.min(n_cols);
    let mut r = tensor.clone();

    // Pass 1: reduce A to upper-triangular R, remembering (w, tau).
    let mut reflectors: Vec<(Tensor<2>, Tensor<2>)> = Vec::with_capacity(max_iters);
    for i in 0..max_iters {
        let sub_tensor = r.clone().slice_dim(0, i..).slice_dim(1, i..);
        let v = sub_tensor.clone().slice_dim(1, 0..1);
        let v0 = v.clone().slice_dim(0, 0..1);
        let zeros = v0.clone().zeros_like();
        let norm_v = v
            .clone()
            .slice_dim(0, ..)
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
        w = w.slice_assign(0..1, e0);
        w = w.clone().mask_fill(mask_norm, 0.0);

        // H_i * sub = sub - tau * w * (w^T sub)
        let wt_sub = w.clone().transpose().matmul(sub_tensor.clone());
        let upd = w.clone().matmul(wt_sub).mul(tau.clone());
        r = r.slice_assign([i.., i..], sub_tensor - upd);

        reflectors.push((w, tau));
    }

    // Pass 2: Q = H_1..H_k I_k, built back-to-front; H_i only touches rows i..
    // so the reduced Q is exact and costs O(m k^2) total.
    let mut q: Tensor<2> = Tensor::eye(n_rows, &device).slice([0..n_rows, 0..max_iters]);
    for i in (0..max_iters).rev() {
        let (w, tau) = &reflectors[i];
        let q_sub = q.clone().slice([i.., 0..]);
        let wt_q = w.clone().transpose().matmul(q_sub.clone());
        let upd = w.clone().matmul(wt_q).mul(tau.clone());
        q = q.slice_assign([i.., 0..], q_sub - upd);
    }

    if reduced & (n_rows > n_cols) {
        let result_r = r.slice([0..n_cols, 0..n_cols]);
        return (q, result_r);
    }
    (q, r)
}
