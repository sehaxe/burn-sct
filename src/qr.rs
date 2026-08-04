//! Householder QR decomposition.
//!
//! Adapted from burn-rs/burn, `crates/burn-tensor/src/tensor/linalg/qr.rs`
//! (main branch, post-0.21; authored by the burn-rs maintainers, MIT/Apache-2.0).
//! Two adaptations to the 0.21 tensor API and to the SCT use-case:
//! - 0.21 API: `SliceArg`/`s!` macro, manual l2-norm, rank-preserving slices.
//! - Reduced Q: burn's version accumulates the full `[n_rows, n_rows]` Q,
//!   which is O(m^2 k) and prohibitive for SCT's tall-skinny factors
//!   (m >> k, e.g. 4096 x 128). Here the reflection vectors are stored
//!   during the R pass and Q is built back-to-front (LAPACK `orgqr`
//!   scheme): Q = H_1..H_k I_k, each H_i acting on rows i.. only, so the
//!   whole thing is O(m k^2) and memory O(m k).
//!
//! The algorithm is the same Householder-with-sign; combined with the
//! `sign(diag(R))` correction in [`crate::orthogonalize`] it reproduces the
//! paper's `safe_qr` (PyTorch `torch.linalg.qr` + sign flip), verified by
//! `tests/cmp_reference.rs`.

use burn::tensor::backend::Backend;
use burn::tensor::{s, Tensor};

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
