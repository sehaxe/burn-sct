//! CUDA QR retraction kernels (feature `cuda`).
//!
//! Same math as [`crate::qr::qr_cpu`] (safe_qr of the paper: orthonormal Q
//! with non-negative R diagonal), computed as:
//!   1. Gram matrix G = A^T·A via the backend's matmul (cuBLAS-class, ~0.1
//!      ms for LLM shapes; a hand-written scalar kernel reached only
//!      ~40 GFLOP/s on this cubecl stack).
//!   2. Cholesky G = R^T·R on the host: k x k is tiny (64 KB at k=128), one
//!      thread finishes in microseconds. R's diagonal is positive by
//!      construction, which is exactly torch's sign(diag(R)) convention.
//!   3. `sct_qr_qsolve_kernel`: Q = A·R^-1 by forward substitution with a
//!      four-wide column block (independent FMAs hide latency); rows of Q
//!      are independent, so m threads each walk their own row with no
//!      cross-thread synchronization, and Q is written row-major directly.
//!
//! Accuracy: G = A^T·A squares the condition number; for retraction inputs
//! (near-orthonormal, kappa ~ 1.1-2) the f32 error stays ~1e-6, far inside
//! the 1e-4 reference tolerance. Results agree with the CPU Householder
//! path to ~1e-6 (verified by tests/cuda_retract.rs).
//!
//! On the bare CUDA `CubeBackend` this replaces the CPU path entirely
//! (no host round-trip for the data itself); every other backend falls back
//! to `qr_cpu`.

use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, Tensor};
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeBackend;
use cubecl::prelude::*;
use std::any::Any;
use std::any::TypeId;

/// The bare (non-fusion) CUDA backend the kernels target.
pub type CudaBare = CubeBackend<cubecl::cuda::CudaRuntime>;

pub fn is_cuda<B: Backend>() -> bool {
    TypeId::of::<B>() == TypeId::of::<CudaBare>()
}

/// Owned copy of the underlying `CubeTensor` of `t` (see burn-gdn2's
/// `cube_of`). `None` when `B` is not the bare CUDA backend, or the buffer
/// is non-contiguous: the caller then falls back to the CPU path.
pub fn cube_of<B: Backend>(t: &Tensor<2>) -> Option<CubeTensor<cubecl::cuda::CudaRuntime>>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    if !is_cuda::<B>() {
        return None;
    }
    let prim = t.clone().try_into_primitive::<B>().ok()?;
    let cube = (&prim as &dyn Any).downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
    let shape = cube.meta.shape().dims::<2>();
    let strides = cube.meta.strides().to_vec();
    let mut expected = 1usize;
    for i in (0..2).rev() {
        if shape[i] > 1 && strides[i] != expected {
            return None; // non-contiguous buffer, use the CPU path
        }
        expected *= shape[i];
    }
    Some(cube.clone())
}

/// Q = A·R^-1 by forward substitution, four columns at a time (the four
/// independent FMAs hide memory latency and give nvcc ILP to play with).
/// Each thread owns one row of Q, so no cross-thread synchronization is
/// needed; Q is written row-major directly.
#[cube(launch_unchecked)]
fn sct_qr_qsolve_kernel<F: Float>(
    a: &[F],     // [m, k] row-major input
    r: &[F],     // [k, k] upper-triangular factor
    q: &mut [F], // [m, k] row-major Q
    #[comptime] m: u32,
    #[comptime] k: u32,
) {
    let row = (CUBE_POS_Y * 256 + UNIT_POS_Y) as usize;
    let kk = k as usize;
    let mm = m as usize;
    if row < mm {
        let base = row * kk;
        let mut j = 0;
        while j < kk {
            let j1 = j + 1;
            let j2 = j + 2;
            let j3 = j + 3;
            let w1 = j1 < kk;
            let w2 = j2 < kk;
            let w3 = j3 < kk;
            let mut acc0 = a[base + j];
            let mut acc1 = if w1 { a[base + j1] } else { F::new(0.0_f32) };
            let mut acc2 = if w2 { a[base + j2] } else { F::new(0.0_f32) };
            let mut acc3 = if w3 { a[base + j3] } else { F::new(0.0_f32) };
            let mut i = 0;
            while i < j {
                let qi = q[base + i];
                acc0 -= r[i * kk + j] * qi;
                if w1 {
                    acc1 -= r[i * kk + j1] * qi;
                }
                if w2 {
                    acc2 -= r[i * kk + j2] * qi;
                }
                if w3 {
                    acc3 -= r[i * kk + j3] * qi;
                }
                i += 1;
            }
            // intra-block corrections, sequentially
            let q0 = acc0 / r[j * kk + j];
            q[base + j] = q0;
            if w1 {
                let q1 = (acc1 - r[j * kk + j1] * q0) / r[j1 * kk + j1];
                q[base + j1] = q1;
                if w2 {
                    let q2 = (acc2 - r[j * kk + j2] * q0 - r[j1 * kk + j2] * q1) / r[j2 * kk + j2];
                    q[base + j2] = q2;
                    if w3 {
                        let q3 = (acc3 - r[j * kk + j3] * q0 - r[j1 * kk + j3] * q1
                            - r[j2 * kk + j3] * q2)
                            / r[j3 * kk + j3];
                        q[base + j3] = q3;
                    }
                }
            }
            j += 4;
        }
    }
}

/// GPU retraction of `matrix` on the bare CUDA backend. `None` when the
/// backend is not CUDA or the shape is outside the kernel limits (then the
/// CPU path in [`crate::orthogonalize`] is used).
pub fn retract_cuda<B: Backend>(matrix: Tensor<2>) -> Option<Tensor<2>>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let a_cube = cube_of::<B>(&matrix)?;
    let [m, k] = matrix.dims();
    if m == 0 || k == 0 || k > 1024 || m < 256 || k < 16 {
        return None; // too small: the host round trip costs more than qr_cpu
    }
    let client = a_cube.client.clone();
    let device = matrix.device();

    // 1. Gram matrix via the backend matmul (lazy tensor op; into_data
    //    materializes it and copies the 64 KB result to the host).
    let g = matrix.clone().transpose().matmul(matrix.clone());
    let g_data = g.into_data();
    let g_flat = g_data.to_vec::<f32>().ok()?;

    // 2. Cholesky on the host (microseconds for k <= 256).
    let r_flat = crate::qr::cholesky_host(&g_flat, k);
    let r_tensor = Tensor::<1>::from_floats(r_flat.as_slice(), &device).reshape([k, k]);

    // 3. Q = A·R^-1, one thread per row, four columns at a time.
    let r_cube = cube_of::<B>(&r_tensor)?;
    let q = Tensor::<2>::zeros([m, k], &device);
    let q_cube = cube_of::<B>(&q)?;
    unsafe {
        sct_qr_qsolve_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static(1, m.div_ceil(256) as u32, 1),
            CubeDim {
                x: 1,
                y: 256,
                z: 1,
            },
            BufferArg::from_raw_parts(a_cube.handle, m * k),
            BufferArg::from_raw_parts(r_cube.handle.clone(), k * k),
            BufferArg::from_raw_parts(q_cube.handle.clone(), m * k),
            m as u32,
            k as u32,
        );
    }
    // Block on the server queue: burn 0.22 tensors are lazy, and a
    // raw-handle launch would otherwise be dropped or deferred past the
    // caller's next read. One sync per retract is acceptable: retraction
    // runs once per training step anyway.
    let _ = futures_lite::future::block_on(client.sync());

    Some(q)
}
