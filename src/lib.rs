#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub mod qr;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor, TensorData};

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
    pub fn init<B: Backend>(&self, device: &B::Device) -> SctLinear<B> {
        SctLinear::new(self, device)
    }
}

#[derive(Module, Debug)]
pub struct SctLinear<B: Backend> {
    pub u: Param<Tensor<B, 2>>,
    pub s: Param<Tensor<B, 1>>,
    pub v: Param<Tensor<B, 2>>,
    #[module(skip)]
    pub rank: usize,
    #[module(skip)]
    pub in_features: usize,
    #[module(skip)]
    pub out_features: usize,
}

impl<B: Backend> SctLinear<B> {
    pub fn new(cfg: &SctConfig, device: &B::Device) -> Self {
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

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let s_b = self.s.val().clone().unsqueeze_dims(&[0]);
        x.matmul(self.u.val().clone())
            .mul(s_b)
            .matmul(self.v.val().clone().transpose())
    }

    pub fn retract(&mut self) {
        self.u = Param::from_tensor(orthogonalize(self.u.val().clone()));
        self.v = Param::from_tensor(orthogonalize(self.v.val().clone()));
    }

    pub fn ortho_error(&self) -> f32 {
        let k = self.rank;
        let device = self.u.device();
        let mut eye = vec![0.0f32; k * k];
        for i in 0..k {
            eye[i * k + i] = 1.0;
        }
        let eye_t = Tensor::<B, 2>::from_data(TensorData::new(eye, [k, k]), &device);
        let err = |m: &Tensor<B, 2>| -> f32 {
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

    pub fn from_dense(dense_weight: Tensor<B, 2>, rank: usize) -> Self {
        Self::from_dense_with_iters(dense_weight, rank, 15)
    }

    pub fn from_dense_with_iters(
        dense_weight: Tensor<B, 2>,
        rank: usize,
        svd_iters: usize,
    ) -> Self {
        let [n, m] = dense_weight.dims();
        let k = rank.min(m).min(n);
        let device = dense_weight.device();
        let data = dense_weight.into_data();
        let vals: Vec<f32> = data
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let (u_svd, s_svd, vt_svd) = svd_cpu(&vals, n, m, k, svd_iters);
        // u_svd is k rows of length n (left singular vectors = V of A^T);
        // vt_svd is k rows of length m (right singular vectors = U of A^T).
        // forward needs u [in=m, k] = vt_svd^T and v [out=n, k] = u_svd^T, so
        // fill the flat arrays in [m|n][k] row-major order directly.
        let mut u_flat = vec![0.0f32; m * k];
        for i in 0..k {
            for j in 0..m {
                u_flat[j * k + i] = vt_svd[i][j];
            }
        }
        let mut v_flat = vec![0.0f32; n * k];
        for i in 0..k {
            for r in 0..n {
                v_flat[r * k + i] = u_svd[i][r];
            }
        }
        Self {
            u: Param::from_tensor(
                Tensor::<B, 1>::from_floats(u_flat.as_slice(), &device).reshape([m, k]),
            ),
            s: Param::from_tensor(Tensor::<B, 1>::from_floats(s_svd.as_slice(), &device)),
            v: Param::from_tensor(
                Tensor::<B, 1>::from_floats(v_flat.as_slice(), &device).reshape([n, k]),
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

fn random_orthonormal<B: Backend>(rows: usize, cols: usize, device: &B::Device) -> Tensor<B, 2> {
    orthogonalize(Tensor::<B, 2>::random(
        [rows, cols],
        Distribution::Normal(0.0, 1.0 / (rows as f64).sqrt()),
        device,
    ))
}

fn orthogonalize<B: Backend>(matrix: Tensor<B, 2>) -> Tensor<B, 2> {
    // Paper Eq 5: Q,R = QR(U); U <- Q * sign(diag(R)). Runs through the
    // native single-allocation Householder path (qr.rs, LAPACK dgeqrf
    // scheme, O(m k^2)); a tensor-op QR is 100-300x slower on CPU and
    // retraction sits outside the autodiff graph, so the sync is free.
    // The sign(diag(R)) correction is applied inside qr_cpu, matching the
    // paper's safe_qr (PyTorch linalg.qr + sign flip), verified bit-close
    // by tests/cmp_reference.rs.
    let device = matrix.device();
    let [m, k] = matrix.dims();
    let data: Vec<f32> = matrix.into_data().to_vec().unwrap();
    let (q, _r) = crate::qr::qr_cpu(&data, m, k);
    Tensor::<B, 1>::from_floats(q.as_slice(), &device).reshape([m, k])
}

/// Truncated SVD of an `n x m` row-major matrix via one-sided (Hestenes)
/// Jacobi rotations. Computes the exact SVD of `A = U diag(s) V^T` (accurate
/// to f32 rounding), then keeps the top-k triplets sorted by descending s.
///
/// Returns `(u, s, vt)` where `u[i]`/`vt[i]` are the i-th left/right singular
/// vectors (rows of length n/m) paired with `s[i]`.
fn svd_cpu(
    data: &[f32],
    n: usize,
    m: usize,
    k: usize,
    sweeps: usize,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>) {
    let mut a: Vec<Vec<f32>> = vec![vec![0.0f32; n]; m];
    for j in 0..m {
        for r in 0..n {
            a[j][r] = data[r * m + j];
        }
    }
    // accumulated right rotations: V starts as identity, columns rotated with A
    let mut v: Vec<Vec<f32>> = (0..m)
        .map(|j| {
            let mut row = vec![0.0f32; m];
            row[j] = 1.0;
            row
        })
        .collect();

    for _ in 0..sweeps {
        let mut converged = true;
        for p in 0..m {
            for q in (p + 1)..m {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for r in 0..n {
                    alpha += a[p][r] * a[p][r];
                    beta += a[q][r] * a[q][r];
                    gamma += a[p][r] * a[q][r];
                }
                if gamma.abs() <= 1e-12 * (alpha * beta).sqrt() {
                    continue;
                }
                converged = false;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = gamma.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for r in 0..n {
                    let ap = a[p][r];
                    let aq = a[q][r];
                    a[p][r] = c * ap + s * aq;
                    a[q][r] = c * aq - s * ap;
                }
                for j in 0..m {
                    let vp = v[p][j];
                    let vq = v[q][j];
                    v[p][j] = c * vp + s * vq;
                    v[q][j] = c * vq - s * vp;
                }
            }
        }
        if converged {
            break;
        }
    }

    let mut cols: Vec<(f32, Vec<f32>, Vec<f32>)> = (0..m)
        .map(|j| {
            let sigma = a[j].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let u_col: Vec<f32> = a[j].iter().map(|x| x / sigma).collect();
            (sigma, u_col, v[j].clone())
        })
        .collect();
    cols.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());

    let kk = k.min(cols.len());
    let mut u = vec![vec![0.0f32; n]; kk];
    let mut s = vec![0.0f32; kk];
    let mut vt = vec![vec![0.0f32; m]; kk];
    for i in 0..kk {
        s[i] = cols[i].0;
        u[i].copy_from_slice(&cols[i].1);
        vt[i].copy_from_slice(&cols[i].2);
    }
    (u, s, vt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};
    type B = NdArray;
    fn dev() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    #[test]
    fn forward_shape() {
        let l = SctLinear::<B>::new(&SctConfig::new(32, 64, 8), &dev());
        assert_eq!(
            l.forward(Tensor::<B, 2>::random(
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
        let u = random_orthonormal::<B>(64, 16, &dev());
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
        let mut l = SctLinear::<B>::new(&SctConfig::new(32, 64, 8), &dev());
        l.u = Param::from_tensor(
            l.u.val().clone()
                + Tensor::<B, 2>::random([32, 8], Distribution::Normal(0.0, 0.3), &dev()),
        );
        l.retract();
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
        let mut l = SctLinear::<B>::new(&SctConfig::new(32, 64, 8), &dev());
        l.u = Param::from_tensor(
            l.u.val().clone()
                + Tensor::<B, 2>::random([32, 8], Distribution::Normal(0.0, 0.05), &dev()),
        );
        let e1 = l.ortho_error();
        l.retract();
        assert!(l.ortho_error() < e1);
    }
    #[test]
    fn compression_ratio() {
        assert!(
            SctLinear::<B>::new(&SctConfig::new(4096, 11008, 128), &dev()).compression_ratio()
                > 20.0
        );
    }
    #[test]
    fn param_count() {
        let l = SctLinear::<B>::new(&SctConfig::new(100, 200, 10), &dev());
        assert_eq!(l.param_count(), 10 * (100 + 200 + 1));
    }
    #[test]
    fn rank_auto_clamped() {
        assert_eq!(
            SctLinear::<B>::new(&SctConfig::new(32, 64, 1000), &dev()).rank,
            32
        );
    }
    #[test]
    fn svd_cpu_roundtrip_small() {
        // A = [[3,0],[0,2],[0,0]] 3x2 rank 2
        let a = vec![3.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (u, s, vt) = svd_cpu(&a, 3, 2, 2, 50);
        assert!(
            (s[0] - 3.0).abs() < 1e-3 && (s[1] - 2.0).abs() < 1e-3,
            "s = {s:?}"
        );
        for r in 0..3 {
            for c in 0..2 {
                let mut acc = 0.0;
                for i in 0..2 {
                    acc += u[i][r] * s[i] * vt[i][c];
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
        let (u, s, vt) = svd_cpu(&a, 3, 2, 2, 200);
        for r in 0..3 {
            for c in 0..2 {
                let mut acc = 0.0;
                for i in 0..2 {
                    acc += u[i][r] * s[i] * vt[i][c];
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
        let (u, s, vt) = svd_cpu(&a, n, m, 8, 100);
        let mut max_e = 0.0f32;
        for r in 0..n {
            for j in 0..m {
                let mut acc = 0.0;
                for i in 0..8 {
                    acc += u[i][r] * s[i] * vt[i][j];
                }
                max_e = max_e.max((acc - a[r * m + j]).abs());
            }
        }
        assert!(max_e < 1e-2, "max reconstruction error {max_e}");
    }
    #[test]
    fn from_dense_shape() {
        let w = Tensor::<B, 2>::random([64, 32], Distribution::Normal(0.0, 1.0), &dev());
        let l = SctLinear::<B>::from_dense(w, 8);
        assert_eq!((l.in_features, l.out_features, l.rank), (32, 64, 8));
    }
    #[test]
    fn from_dense_roundtrip() {
        // Build an exactly rank-8 matrix so truncation error is zero; the SVD
        // must reproduce it. from_dense takes the weight as [out, in].
        let a = Tensor::<B, 2>::random([32, 8], Distribution::Normal(0.0, 1.0), &dev());
        let b = Tensor::<B, 2>::random([8, 64], Distribution::Normal(0.0, 1.0), &dev());
        let w = a.matmul(b); // [in, out] = [32, 64]
        let w_t = w.transpose(); // [out, in], the layout from_dense expects
        let l = SctLinear::<B>::from_dense_with_iters(w_t.clone(), 8, 50);
        let w_hat =
            l.v.val()
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
        let q = orthogonalize::<B>(Tensor::<B, 2>::random(
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
