#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
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
        Self { in_features, out_features, rank }
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
    #[module(skip)] pub rank: usize,
    #[module(skip)] pub in_features: usize,
    #[module(skip)] pub out_features: usize,
}

impl<B: Backend> SctLinear<B> {
    pub fn new(cfg: &SctConfig, device: &B::Device) -> Self {
        let k = cfg.rank.min(cfg.in_features).min(cfg.out_features);
        Self {
            u: Param::from_tensor(random_orthonormal(cfg.in_features, k, device)),
            s: Param::from_tensor(Tensor::ones([k], device)),
            v: Param::from_tensor(random_orthonormal(cfg.out_features, k, device)),
            rank: k, in_features: cfg.in_features, out_features: cfg.out_features,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let s_b = self.s.val().clone().unsqueeze_dims(&[0]);
        x.matmul(self.u.val().clone()).mul(s_b).matmul(self.v.val().clone().transpose())
    }

    pub fn retract(&mut self) {
        self.u = Param::from_tensor(orthogonalize(self.u.val().clone()));
        self.v = Param::from_tensor(orthogonalize(self.v.val().clone()));
    }

    pub fn ortho_error(&self) -> f32 {
        let k = self.rank;
        let device = self.u.device();
        let mut eye = vec![0.0f32; k * k];
        for i in 0..k { eye[i * k + i] = 1.0; }
        let eye_t = Tensor::<B, 2>::from_data(TensorData::new(eye, [k, k]), &device);
        let err = |m: &Tensor<B, 2>| -> f32 {
            let diff = m.clone().transpose().matmul(m.clone()) - eye_t.clone();
            diff.powf_scalar(2.0).into_data().bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())).sum::<f32>().sqrt()
        };
        err(&self.u.val()).max(err(&self.v.val()))
    }

    pub fn from_dense(dense_weight: Tensor<B, 2>, rank: usize) -> Self {
        Self::from_dense_with_iters(dense_weight, rank, 5)
    }

    pub fn from_dense_with_iters(dense_weight: Tensor<B, 2>, rank: usize, svd_iters: usize) -> Self {
        let [n, m] = dense_weight.dims();
        let k = rank.min(m).min(n);
        let device = dense_weight.device();
        let data = dense_weight.into_data();
        let vals: Vec<f32> = data.bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let (u_svd, s_svd, vt_svd) = svd_cpu(&vals, n, m, k, svd_iters);
        let u_flat: Vec<f32> = vt_svd.iter().flat_map(|r| r.iter()).copied().collect();
        let v_flat: Vec<f32> = u_svd.iter().flat_map(|r| r.iter()).copied().collect();
        Self {
            u: Param::from_tensor(Tensor::<B, 1>::from_floats(u_flat.as_slice(), &device).reshape([m, k])),
            s: Param::from_tensor(Tensor::<B, 1>::from_floats(s_svd.as_slice(), &device)),
            v: Param::from_tensor(Tensor::<B, 1>::from_floats(v_flat.as_slice(), &device).reshape([n, k])),
            rank: k, in_features: m, out_features: n,
        }
    }

    pub fn param_count(&self) -> usize { self.rank * (self.in_features + self.out_features + 1) }
    pub fn dense_params(&self) -> usize { self.in_features * self.out_features }
    pub fn compression_ratio(&self) -> f64 { self.dense_params() as f64 / self.param_count() as f64 }
}

#[macro_export]
macro_rules! retract_all {
    ($model:expr, $($field:ident),+ $(,)?) => { $( $model.$field.retract(); )+ };
}

fn random_orthonormal<B: Backend>(rows: usize, cols: usize, device: &B::Device) -> Tensor<B, 2> {
    orthogonalize(Tensor::<B, 2>::random(
        [rows, cols], Distribution::Normal(0.0, 1.0 / (rows as f64).sqrt()), device,
    ))
}

fn orthogonalize<B: Backend>(matrix: Tensor<B, 2>) -> Tensor<B, 2> {
    let [m, k] = matrix.dims();
    let device = matrix.device();
    let mut q_cols: Vec<Tensor<B, 2>> = Vec::with_capacity(k);
    let mut r_diag = vec![0.0f32; k];
    for i in 0..k {
        let mut col = matrix.clone().slice([0..m, i..i + 1]);
        for q_j in q_cols.iter() {
            let dot = q_j.clone().transpose().matmul(col.clone());
            col = col.clone() - q_j.clone().matmul(dot);
        }
        let norm = col.clone().powf_scalar(2.0).sum_dim(0).sqrt();
        let nf = f32::from_le_bytes(norm.clone().into_data().bytes[..4].try_into().unwrap());
        r_diag[i] = nf;
        col = if nf > 1e-8 { col.div(norm) } else {
            Tensor::<B, 2>::random([m, 1], Distribution::Normal(0.0, 1.0), &device)
        };
        q_cols.push(col);
    }
    let q = Tensor::cat(q_cols, 1);
    let signs: Vec<f32> = r_diag.iter().map(|&d| if d >= 0.0 { 1.0 } else { -1.0 }).collect();
    let sign_t = Tensor::<B, 1>::from_floats(signs.as_slice(), &device).unsqueeze_dims(&[0]);
    q.mul(sign_t)
}

fn svd_cpu(data: &[f32], n: usize, m: usize, k: usize, iters: usize) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>) {
    let frob_sq: f32 = data.iter().map(|x| x * x).sum();
    let avg = (frob_sq / (n.min(m) as f32)).sqrt();
    let mut s_vals = vec![avg; k];
    let mut u = vec![vec![0.0f32; n]; k];
    let mut vt = vec![vec![0.0f32; m]; k];
    for i in 0..k { u[i][i % n] = 1.0; vt[i][i % m] = 1.0; }
    for _iter in 0..iters {
        for i in 0..k {
            let inv_s = 1.0 / s_vals[i].max(1e-8);
            for j in 0..m { let mut s = 0.0; for r in 0..n { s += u[i][r] * data[r * m + j]; } vt[i][j] = s * inv_s; }
            for j in 0..n { let mut s = 0.0; for c in 0..m { s += data[j * m + c] * vt[i][c]; } u[i][j] = s * inv_s; }
            s_vals[i] = (0..n).map(|j| u[i][j] * u[i][j]).sum::<f32>().sqrt();
            let inv_n = 1.0 / s_vals[i].max(1e-8);
            for j in 0..n { u[i][j] *= inv_n; } for j in 0..m { vt[i][j] *= inv_n; }
        }
    }
    (u, s_vals, vt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};
    type B = NdArray;
    fn dev() -> NdArrayDevice { NdArrayDevice::default() }

    #[test] fn forward_shape() {
        let l = SctLinear::<B>::new(&SctConfig::new(32,64,8),&dev());
        assert_eq!(l.forward(Tensor::<B,2>::random([16,32],Distribution::Normal(0.0,1.0),&dev())).dims(),[16,64]);
    }
    #[test] fn orthonormal_init() {
        let u = random_orthonormal::<B>(64,16,&dev());
        let vals: Vec<f32> = u.clone().transpose().matmul(u).into_data().bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        for i in 0..16 { assert!((vals[i*16+i]-1.0).abs()<0.1); }
    }
    #[test] fn retract_restores_ortho() {
        let mut l = SctLinear::<B>::new(&SctConfig::new(32,64,8),&dev());
        l.u = Param::from_tensor(l.u.val().clone()+Tensor::<B,2>::random([32,8],Distribution::Normal(0.0,0.3),&dev()));
        l.retract();
        let vals: Vec<f32> = l.u.val().clone().transpose().matmul(l.u.val().clone()).into_data().bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        for i in 0..8 { assert!((vals[i*8+i]-1.0).abs()<0.1); }
    }
    #[test] fn ortho_error_decreases() {
        let mut l = SctLinear::<B>::new(&SctConfig::new(32,64,8),&dev());
        l.u = Param::from_tensor(l.u.val().clone()+Tensor::<B,2>::random([32,8],Distribution::Normal(0.0,0.05),&dev()));
        let e1 = l.ortho_error(); l.retract();
        assert!(l.ortho_error() < e1);
    }
    #[test] fn compression_ratio() {
        assert!(SctLinear::<B>::new(&SctConfig::new(4096,11008,128),&dev()).compression_ratio()>20.0);
    }
    #[test] fn param_count() {
        let l = SctLinear::<B>::new(&SctConfig::new(100,200,10),&dev());
        assert_eq!(l.param_count(), 10*(100+200+1));
    }
    #[test] fn rank_auto_clamped() {
        assert_eq!(SctLinear::<B>::new(&SctConfig::new(32,64,1000),&dev()).rank, 32);
    }
    #[test] fn from_dense_shape() {
        let w = Tensor::<B,2>::random([64,32],Distribution::Normal(0.0,1.0),&dev());
        let l = SctLinear::<B>::from_dense(w,8);
        assert_eq!((l.in_features,l.out_features,l.rank),(32,64,8));
    }
    #[test] fn sign_correction() {
        let q = orthogonalize::<B>(Tensor::<B,2>::random([64,16],Distribution::Normal(0.0,1.0),&dev()));
        let vals: Vec<f32> = q.clone().transpose().matmul(q).into_data().bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        for i in 0..16 { assert!((vals[i*16+i]-1.0).abs()<0.1); }
    }
}
