#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub mod qr;
#[cfg(feature = "cuda")]
pub mod qr_cuda;
use burn::backend::{Backend, DispatchKindConversion};
use burn::module::{Module, Param};
use burn::tensor::{Device, DispatchTensor, Distribution, Tensor, TensorData};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SctConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
}

impl SctConfig {
    pub fn new(in_features: usize, out_features: usize, rank: usize) -> Self {
        Self {
            in_features,
            out_features,
            rank,
        }
    }
    pub fn init(&self, device: &Device) -> SctLinear {
        SctLinear::new(self, device)
    }
}

#[derive(Module, Debug)]
pub struct SctLinear {
    pub u: Param<Tensor<2>>,
    pub s: Param<Tensor<1>>,
    pub v: Param<Tensor<2>>,
    #[module(skip)]
    pub rank: usize,
    #[module(skip)]
    pub in_features: usize,
    #[module(skip)]
    pub out_features: usize,
}

impl SctLinear {
    pub fn new(cfg: &SctConfig, device: &Device) -> Self {
        let k = cfg.rank.min(cfg.in_features).min(cfg.out_features);
        Self {
            u: Param::from_tensor(random_orthonormal(cfg.in_features, k, device)),
            s: Param::from_tensor(Tensor::ones([k], device)),
            v: Param::from_tensor(random_orthonormal(cfg.out_features, k, device)),
            rank: k,
            in_features: cfg.in_features,
            out_features: cfg.out_features,
        }
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        // Paper order exactly: y = (x@U) * s @ V^T. Scaling after the first
        // GEMM costs only a [B,k] pass (no [m,k] or [k,n] intermediate), so
        // the peak footprint of forward is x + the two GEMM outputs.
        x.matmul(self.u.val())
            .mul(self.s.val().unsqueeze_dims(&[0]))
            .matmul(self.v.val().transpose())
    }

    pub fn retract<B: Backend>(&mut self)
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        // The two QR retractions are independent; run them concurrently.
        // Param::consume/from_mapped_value preserve ParamId and the
        // load/save mappers, exactly like Param::map.
        //
        // mem::replace (instead of self.u.clone().consume()) keeps the
        // tensor refcount at 1: a shared NdArray tensor makes into_data fall
        // into ndarray's slow copy path (~15ms per QR at k=128), which would
        // dwarf the QR itself.
        let device = self.u.device();
        let dummy = |d: &Device| Param::from_tensor(Tensor::<2>::empty([0, 0], d));
        let (u_id, u_val, u_map) = std::mem::replace(&mut self.u, dummy(&device)).consume();
        let (v_id, v_val, v_map) = std::mem::replace(&mut self.v, dummy(&device)).consume();
        // CUDA: sequential launches into one queue (the GPU pipelines them;
        // two host threads hammering the same client serialize badly). Every
        // other backend: the two QRs run on separate threads.
        #[cfg(feature = "cuda")]
        let (nu, nv) = if crate::qr_cuda::is_cuda::<B>() {
            (orthogonalize::<B>(u_val), orthogonalize::<B>(v_val))
        } else {
            std::thread::scope(|s| {
                let a = s.spawn(move || orthogonalize::<B>(u_val));
                let b = s.spawn(move || orthogonalize::<B>(v_val));
                (a.join().unwrap(), b.join().unwrap())
            })
        };
        #[cfg(not(feature = "cuda"))]
        let (nu, nv) = std::thread::scope(|s| {
            let a = s.spawn(move || orthogonalize::<B>(u_val));
            let b = s.spawn(move || orthogonalize::<B>(v_val));
            (a.join().unwrap(), b.join().unwrap())
        });
        self.u = Param::from_mapped_value(u_id, nu, u_map);
        self.v = Param::from_mapped_value(v_id, nv, v_map);
    }

    pub fn ortho_error(&self) -> f32 {
        let k = self.rank;
        let device = self.u.device();
        let mut eye = vec![0.0f32; k * k];
        for i in 0..k {
            eye[i * k + i] = 1.0;
        }
        let eye_t = Tensor::<2>::from_data(TensorData::new(eye, [k, k]), &device);
        let err = |m: &Tensor<2>| -> f32 {
            let diff = m.clone().transpose().matmul(m.clone()) - eye_t.clone();
            diff.powf_scalar(2.0)
                .into_data()
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .sum::<f32>()
                .sqrt()
        };
        err(&self.u.val()).max(err(&self.v.val()))
    }

    pub fn from_dense<B: Backend>(dense_weight: Tensor<2>, rank: usize) -> Self
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        Self::from_dense_with_iters::<B>(dense_weight, rank, 15)
    }

    pub fn from_dense_with_iters<B: Backend>(
        dense_weight: Tensor<2>,
        rank: usize,
        svd_iters: usize,
    ) -> Self
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [n, m] = dense_weight.dims();
        let k = rank.min(m).min(n);
        let device = dense_weight.device();
        let data = dense_weight.into_data();
        let flat = bytes_f32(data.as_bytes());
        // SVD via QR reduction (LAPACK gesdd scheme): A = Q·R, SVD of the
        // square R gives (U_r, s, V), then U = Q·U_r. The one-sided Jacobi
        // then runs on the m×m R instead of the n×m A: ~2.4x less flops and
        // traffic at LLM-scale tall layers. Wide layers (n < m) are handled
        // by SVD-ing A^T and swapping left/right factors.
        let n_threads = crate::qr::thread_count();
        let (u_flat, s, v_flat) = if n >= m {
            // Defer the Q build until after the Jacobi: peak memory during
            // the sweep is r + r_k + Jacobi(a,v) instead of + the n x m Q.
            let (r, tau, _) = crate::qr::r_pass(flat, n, m, n_threads);
            drop(data);
            let r_k = crate::qr::r_k_from_r(&r, n, m);
            // svd_cpu returns (V_r [m,k], s, U_r [m,k]) for R.
            let (v_r, s, u_r) = svd_cpu(&r_k, m, m, k, svd_iters);
            drop(r_k);
            let q_rm = crate::qr::q_pass(&r, &tau, n, m, n_threads);
            // layer u [in=m, k] = V of A = V_r; layer v [out=n, k] = U of A = Q·U_r
            let v_flat = matmul_rt(&q_rm, n, m, &u_r, k, n_threads);
            (v_r, s, v_flat)
        } else {
            let data_t = transpose_f32(flat, n, m);
            drop(data);
            let (r, tau, _) = crate::qr::r_pass(&data_t, m, n, n_threads);
            let r_k = crate::qr::r_k_from_r(&r, m, n);
            let (v_r, s, u_r) = svd_cpu(&r_k, n, n, k, svd_iters);
            drop(r_k);
            let q_rm = crate::qr::q_pass(&r, &tau, m, n, n_threads);
            // A^T = Q·U_r·s·V_r^T -> U of A = V_r [n,k], V of A = Q·U_r [m,k]
            let u_flat = matmul_rt(&q_rm, m, n, &u_r, k, n_threads);
            (u_flat, s, v_r)
        };
        Self {
            u: Param::from_tensor(
                Tensor::<1>::from_floats(u_flat.as_slice(), &device).reshape([m, k]),
            ),
            s: Param::from_tensor(Tensor::<1>::from_floats(s.as_slice(), &device)),
            v: Param::from_tensor(
                Tensor::<1>::from_floats(v_flat.as_slice(), &device).reshape([n, k]),
            ),
            rank: k,
            in_features: m,
            out_features: n,
        }
    }

    pub fn param_count(&self) -> usize {
        self.rank * (self.in_features + self.out_features + 1)
    }
    pub fn dense_params(&self) -> usize {
        self.in_features * self.out_features
    }
    pub fn compression_ratio(&self) -> f64 {
        self.dense_params() as f64 / self.param_count() as f64
    }
}

#[macro_export]
macro_rules! retract_all {
    ($model:expr, $($field:ident),+ $(,)?) => { $( $model.$field.retract(); )+ };
}

fn random_orthonormal(rows: usize, cols: usize, device: &Device) -> Tensor<2> {
    orthogonalize_cpu(Tensor::<2>::random(
        [rows, cols],
        Distribution::Normal(0.0, 1.0 / (rows as f64).sqrt()),
        device,
    ))
}

/// Zero-copy `&[f32]` view of `TensorData` bytes.
///
/// SAFETY: `TensorData` f32 payloads are allocated by the native allocator
/// (16-byte aligned in practice) and hold a multiple of 4 bytes; the
/// alignment guarantee is asserted in debug builds.
fn bytes_f32(bytes: &[u8]) -> &[f32] {
    debug_assert!(bytes.len() % 4 == 0);
    debug_assert!(bytes.as_ptr() as usize % 4 == 0);
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

/// Row-major transpose of an `n x m` matrix into `m x n`.
fn transpose_f32(a: &[f32], n: usize, m: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * m];
    for r in 0..n {
        for c in 0..m {
            out[c * n + r] = a[r * m + c];
        }
    }
    out
}

/// `C = A·B` with `A` `[rows x kk]` row-major and `B` `[kk x k]` row-major,
/// returning `C` `[rows x k]` row-major. `B` is transposed once for
/// contiguous AVX2 dot products; rows are split across threads.
fn matmul_rt(
    a: &[f32],
    rows: usize,
    kk: usize,
    b: &[f32],
    k: usize,
    n_threads: usize,
) -> Vec<f32> {
    let mut bt = vec![0.0f32; k * kk];
    for i in 0..k {
        for c in 0..kk {
            bt[i * kk + c] = b[c * k + i];
        }
    }
    let mut out = vec![0.0f32; rows * k];
    if n_threads > 1 && rows >= 1024 {
        let chunk = rows.div_ceil(n_threads);
        std::thread::scope(|s| {
            for (ci, out_rows) in out.chunks_mut(chunk * k).enumerate() {
                let (a, bt) = (&a, &bt);
                s.spawn(move || {
                    let lo = ci * chunk;
                    let hi = ((ci + 1) * chunk).min(rows);
                    for j in lo..hi {
                        let arow = &a[j * kk..(j + 1) * kk];
                        for i in 0..k {
                            out_rows[(j - lo) * k + i] =
                                unsafe { crate::qr::dot_pair(arow, &bt[i * kk..(i + 1) * kk]) };
                        }
                    }
                });
            }
        });
    } else {
        for j in 0..rows {
            let arow = &a[j * kk..(j + 1) * kk];
            for i in 0..k {
                out[j * k + i] = unsafe { crate::qr::dot_pair(arow, &bt[i * kk..(i + 1) * kk]) };
            }
        }
    }
    out
}

/// Paper Eq 5: Q,R = QR(U); U <- Q * sign(diag(R)). Native single-allocation
/// Householder path (qr.rs, LAPACK dgeqrf scheme, O(m k^2), AVX2/FMA +
/// parallel Q). The sign(diag(R)) correction matches the paper's safe_qr
/// (PyTorch linalg.qr + sign flip), verified bit-close by
/// tests/cmp_reference.rs.
fn orthogonalize_cpu(matrix: Tensor<2>) -> Tensor<2> {
    let device = matrix.device();
    let require_grad = matrix.is_require_grad();
    let [m, k] = matrix.dims();
    let data = matrix.into_data();
    // One copy (into_data); the old to_vec() second copy is gone. qr_cpu
    // reads the cast slice and writes q_rm directly.
    let (q, _r) = crate::qr::qr_cpu(bytes_f32(data.as_bytes()), m, k);
    Tensor::<1>::from_floats(q.as_slice(), &device)
        .reshape([m, k])
        .set_require_grad(require_grad)
}

/// Retraction orthonormalization. On the bare CUDA backend this runs as two
/// fused kernels with no host round-trip (qr_cuda.rs); everywhere else the
/// CPU path above.
fn orthogonalize<B: Backend>(matrix: Tensor<2>) -> Tensor<2>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    #[cfg(feature = "cuda")]
    if crate::qr_cuda::is_cuda::<B>() {
        if let Some(q) = crate::qr_cuda::retract_cuda::<B>(matrix.clone()) {
            return q;
        }
    }
    orthogonalize_cpu(matrix)
}

/// Relative off-diagonal threshold at which a Jacobi pair stops being
/// rotated. Internal state is f64, so the dot/rotation noise floor is
/// ~eps/sqrt(n) ~ 1e-14 even at n=11008; 1e-12 is both reachable (sweeps
/// exit once converged) and ~9 orders tighter than the 1e-3 reconstruction
/// tolerance the reference tests gate.
const JACOBI_EPS: f64 = 1e-12;

/// Truncated SVD of an `n x m` row-major matrix via one-sided (Hestenes)
/// Jacobi rotations. Computes the exact SVD of `A = U diag(s) V^T`, then
/// keeps the top-k triplets sorted by descending s.
///
/// Returns flat buffers in the exact layout [`SctLinear::from_dense`] needs:
/// `(u_flat [m·k], s [k], v_flat [n·k])` where column i of `u_flat` is the
/// i-th right singular vector `V[:, i]` and column i of `v_flat` the i-th
/// left one `U[:, i]` (so `A[r][j] = sum_i v_flat[r·k+i]·s[i]·u_flat[j·k+i]`).
///
/// Accuracy: internal state is f64 (one-sided Jacobi in f32 accumulates
/// ~sqrt(m·sweeps)·eps of rotation noise per element, which for m=1024 lands
/// at ~1e-3 relative — right on the reference test tolerance). With f64 the
/// reconstruction is exact to ~1e-13, i.e. several orders better than
/// torch.linalg.svd's own f32 result, so the comparison against the
/// reference is dominated by torch's rounding, not ours.
///
/// Peak memory: the transposed f64 copy `a` (n·m·8) + right-rotation
/// accumulator `v` (m·m·8) + outputs (k·(m+n)). The old per-column triplet
/// list (m·(m+n) floats, ~700 MB for a 4096x11008 layer) is replaced by an
/// index sort. Inner loops are AVX2/FMA (4-wide f64 fused dots + rotations).
fn svd_cpu(
    data: &[f32],
    n: usize,
    m: usize,
    k: usize,
    sweeps: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // a[j] = column j of A (n rows); transposed fill so every pair operation
    // is a contiguous pass.
    let mut a = vec![0.0f64; n * m];
    for j in 0..m {
        let col = &mut a[j * n..(j + 1) * n];
        for r in 0..n {
            col[r] = data[r * m + j] as f64;
        }
    }
    // v: accumulated right rotations, rows = V^T rows; starts as I.
    let mut v = vec![0.0f64; m * m];
    for j in 0..m {
        v[j * m + j] = 1.0;
    }

    let avx = cfg!(target_arch = "x86_64") && std::arch::is_x86_feature_detected!("avx2");
    for _ in 0..sweeps {
        let mut converged = true;
        for p in 0..m {
            let (_, rest) = a.split_at_mut(p * n);
            let (ap, cols_after) = rest.split_at_mut(n);
            for q in (p + 1)..m {
                let aq = &mut cols_after[(q - p - 1) * n..(q - p) * n];
                let (alpha, beta, gamma) = if avx {
                    unsafe { crate::qr::dot3_avx_f64(ap, aq) }
                } else {
                    let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                    for r in 0..n {
                        let pv = ap[r];
                        let qv = aq[r];
                        alpha += pv * pv;
                        beta += qv * qv;
                        gamma += pv * qv;
                    }
                    (alpha, beta, gamma)
                };
                if gamma.abs() <= JACOBI_EPS * (alpha * beta).sqrt() {
                    continue;
                }
                converged = false;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = gamma.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                if avx {
                    unsafe {
                        crate::qr::rotate2_avx_f64(ap, aq, c, s);
                        let (vp, vq) = v.split_at_mut(q * m);
                        crate::qr::rotate2_avx_f64(
                            &mut vp[p * m..(p + 1) * m],
                            &mut vq[..m],
                            c,
                            s,
                        );
                    }
                } else {
                    for r in 0..n {
                        let av = ap[r];
                        let bv = aq[r];
                        ap[r] = c * av + s * bv;
                        aq[r] = c * bv - s * av;
                    }
                    for j in 0..m {
                        let vp = v[p * m + j];
                        let vq = v[q * m + j];
                        v[p * m + j] = c * vp + s * vq;
                        v[q * m + j] = c * vq - s * vp;
                    }
                }
            }
        }
        if converged {
            break;
        }
    }

    // Singular values; keep only (sigma, column) index pairs, then extract
    // the top-k vectors straight from a and v (no triplet materialization).
    let mut idx: Vec<(f64, usize)> = (0..m)
        .map(|j| {
            let col = &a[j * n..(j + 1) * n];
            let sigma = col.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            (sigma, j)
        })
        .collect();
    idx.sort_unstable_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    let kk = k.min(idx.len());
    let mut u_flat = vec![0.0f32; m * kk];
    let mut s = vec![0.0f32; kk];
    let mut v_flat = vec![0.0f32; n * kk];
    for i in 0..kk {
        let (sigma, j) = idx[i];
        s[i] = sigma as f32;
        let col = &a[j * n..(j + 1) * n];
        let vrow = &v[j * m..(j + 1) * m];
        for r in 0..n {
            v_flat[r * kk + i] = (col[r] / sigma) as f32;
        }
        for jj in 0..m {
            u_flat[jj * kk + i] = vrow[jj] as f32;
        }
    }
    (u_flat, s, v_flat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn forward_shape() {
        let l = SctLinear::new(&SctConfig::new(32, 64, 8), &dev());
        assert_eq!(
            l.forward(Tensor::<2>::random(
                [16, 32],
                Distribution::Normal(0.0, 1.0),
                &dev()
            ))
            .dims(),
            [16, 64]
        );
    }
    #[test]
    fn orthonormal_init() {
        let u = random_orthonormal(64, 16, &dev());
        let vals: Vec<f32> = u
            .clone()
            .transpose()
            .matmul(u)
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for i in 0..16 {
            assert!((vals[i * 16 + i] - 1.0).abs() < 0.1);
        }
    }
    #[test]
    fn retract_restores_ortho() {
        let mut l = SctLinear::new(&SctConfig::new(32, 64, 8), &dev());
        l.u = Param::from_tensor(
            l.u.val().clone()
                + Tensor::<2>::random([32, 8], Distribution::Normal(0.0, 0.3), &dev()),
        );
        l.retract::<NdArray>();
        let vals: Vec<f32> =
            l.u.val()
                .clone()
                .transpose()
                .matmul(l.u.val().clone())
                .into_data()
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
        for i in 0..8 {
            assert!((vals[i * 8 + i] - 1.0).abs() < 0.1);
        }
    }
    #[test]
    fn ortho_error_decreases() {
        let mut l = SctLinear::new(&SctConfig::new(32, 64, 8), &dev());
        l.u = Param::from_tensor(
            l.u.val().clone()
                + Tensor::<2>::random([32, 8], Distribution::Normal(0.0, 0.05), &dev()),
        );
        let e1 = l.ortho_error();
        l.retract::<NdArray>();
        assert!(l.ortho_error() < e1);
    }
    #[test]
    fn compression_ratio() {
        assert!(
            SctLinear::new(&SctConfig::new(4096, 11008, 128), &dev()).compression_ratio() > 20.0
        );
    }
    #[test]
    fn param_count() {
        let l = SctLinear::new(&SctConfig::new(100, 200, 10), &dev());
        assert_eq!(l.param_count(), 10 * (100 + 200 + 1));
    }
    #[test]
    fn rank_auto_clamped() {
        assert_eq!(
            SctLinear::new(&SctConfig::new(32, 64, 1000), &dev()).rank,
            32
        );
    }
    #[test]
    fn svd_cpu_roundtrip_small() {
        // A = [[3,0],[0,2],[0,0]] 3x2 rank 2
        let a = vec![3.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (u_flat, s, v_flat) = svd_cpu(&a, 3, 2, 2, 50);
        assert!(
            (s[0] - 3.0).abs() < 1e-3 && (s[1] - 2.0).abs() < 1e-3,
            "s = {s:?}"
        );
        for r in 0..3 {
            for c in 0..2 {
                let mut acc = 0.0;
                for i in 0..2 {
                    acc += v_flat[r * 2 + i] * s[i] * u_flat[c * 2 + i];
                }
                let e = (acc - a[r * 2 + c]).abs();
                assert!(e < 1e-4, "a_hat[{r}][{c}] = {acc} vs {}", a[r * 2 + c]);
            }
        }
    }
    #[test]
    fn svd_cpu_roundtrip_nondiag() {
        // full-rank non-diagonal 3x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (u_flat, s, v_flat) = svd_cpu(&a, 3, 2, 2, 200);
        for r in 0..3 {
            for c in 0..2 {
                let mut acc = 0.0;
                for i in 0..2 {
                    acc += v_flat[r * 2 + i] * s[i] * u_flat[c * 2 + i];
                }
                let e = (acc - a[r * 2 + c]).abs();
                assert!(e < 1e-3, "a_hat[{r}][{c}] = {acc} vs {}", a[r * 2 + c]);
            }
        }
    }
    #[test]
    fn svd_cpu_roundtrip_64x32() {
        // Deterministic rank-8 matrix (verified: sigma_9 = 0), no rand needed.
        let n = 64usize;
        let m = 32usize;
        let mut a = vec![0.0f32; n * m];
        for r in 0..n {
            for j in 0..m {
                let mut acc = 0.0;
                for i in 0..8 {
                    let p = ((i + 1) as f32 * (r + 1) as f32 * 0.7).sin();
                    let q = ((i + 1) as f32 * (j + 1) as f32 * 0.3).cos();
                    acc += p * q;
                }
                a[r * m + j] = acc;
            }
        }
        let (u_flat, s, v_flat) = svd_cpu(&a, n, m, 8, 100);
        let mut max_e = 0.0f32;
        for r in 0..n {
            for j in 0..m {
                let mut acc = 0.0;
                for i in 0..8 {
                    acc += v_flat[r * 8 + i] * s[i] * u_flat[j * 8 + i];
                }
                max_e = max_e.max((acc - a[r * m + j]).abs());
            }
        }
        assert!(max_e < 1e-2, "max reconstruction error {max_e}");
    }
    #[test]
    fn from_dense_shape() {
        let w = Tensor::<2>::random([64, 32], Distribution::Normal(0.0, 1.0), &dev());
        let l = SctLinear::from_dense::<NdArray>(w, 8);
        assert_eq!((l.in_features, l.out_features, l.rank), (32, 64, 8));
    }
    #[test]
    fn from_dense_roundtrip() {
        // Build an exactly rank-8 matrix so truncation error is zero; the SVD
        // must reproduce it. from_dense takes the weight as [out, in].
        let a = Tensor::<2>::random([32, 8], Distribution::Normal(0.0, 1.0), &dev());
        let b = Tensor::<2>::random([8, 64], Distribution::Normal(0.0, 1.0), &dev());
        let w = a.matmul(b); // [in, out] = [32, 64]
        let w_t = w.transpose(); // [out, in], the layout from_dense expects
        let l = SctLinear::from_dense_with_iters::<NdArray>(w_t.clone(), 8, 50);
        let w_hat = l
            .v
            .val()
            .clone()
            .mul(l.s.val().clone().unsqueeze_dims(&[0]))
            .matmul(l.u.val().clone().transpose()); // [out, in] = W^T
        let err: f32 = (w_hat - w_t).powf_scalar(2.0).mean().into_scalar();
        assert!(
            err < 1e-4,
            "from_dense must reproduce the dense weight, mse {err}"
        );
    }
    #[test]
    fn sign_correction() {
        let q = orthogonalize::<NdArray>(Tensor::<2>::random(
            [64, 16],
            Distribution::Normal(0.0, 1.0),
            &dev(),
        ));
        let vals: Vec<f32> = q
            .clone()
            .transpose()
            .matmul(q)
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for i in 0..16 {
            assert!((vals[i * 16 + i] - 1.0).abs() < 0.1);
        }
    }
}
