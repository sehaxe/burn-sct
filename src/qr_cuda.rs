//! CUDA QR retraction kernels (feature `cuda`).
//!
//! Same Householder algorithm as [`crate::qr::qr_cpu`], split into two
//! launches with no device sync in between:
//!   - `sct_qr_r_kernel`: one thread per column, sequential over reflectors
//!     (column i owns the w computation, columns > i update in parallel).
//!   - `sct_qr_q_kernel`: one cube per Q column, block-wide dot reduction.
//!
//! On the bare CUDA `CubeBackend` this replaces the CPU path entirely
//! (no host round-trip); every other backend falls back to `qr_cpu`.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_cubecl::CubeBackend;
use cubecl::prelude::*;
use std::any::TypeId;

/// The bare (non-fusion) CUDA backend the kernels target.
pub type CudaBare = CubeBackend<cubecl::cuda::CudaRuntime, f32, i32, u8>;

fn is_cuda<B: Backend>() -> bool {
    TypeId::of::<B>() == TypeId::of::<CudaBare>()
}

/// R pass: reduce `a` (row-major `[m, k]`) to the column-major work buffer
/// `r` (reflectors below the diagonal, `R[i][i] = sign*norm` on it) plus
/// `tau`. One thread per column; the reflector owner computes w, everyone
/// else updates their column after a barrier.
#[cube(launch_unchecked)]
fn sct_qr_r_kernel<F: Float>(
    a: &Array<F>,      // [m, k] row-major input
    r: &mut Array<F>,  // [k*m] column-major: reflectors + R diagonal
    tau: &mut Array<F>,// [k]
    #[comptime] m: u32,
    #[comptime] k: u32,
) {
    let c = UNIT_POS_Y as usize;
    let kk = k as usize;
    let mm = m as usize;
    // copy column c: a (row-major) -> r (column-major). All counters must
    // start from a runtime value: cubecl 0.10 panics on `const +=`.
    let mut idx = c - c;
    while idx < mm {
        r[c * mm + idx] = a[idx * kk + c];
        idx += 1;
    }
    sync_storage();

    let mut i = c - c;
    while i < kk {
        if c == i {
            // w from column i (accumulators must start from a runtime
            // value: cubecl 0.10 panics on `const +=`)
            let mut norm2 = r[i * mm + i] - r[i * mm + i];
            let mut rr = i + (c - c);
            while rr < mm {
                norm2 += r[i * mm + rr] * r[i * mm + rr];
                rr += 1;
            }
            let norm = norm2.sqrt();
            let v0 = r[i * mm + i];
            let sign = if v0 < F::new(0.0) {
                F::new(1.0)
            } else {
                F::new(-1.0)
            };
            let u0 = v0 - sign * norm;
            let t = if norm2 > F::new(0.0) {
                -u0 / norm * sign
            } else {
                F::new(0.0)
            };
            tau[i] = t;
            r[i * mm + i] = sign * norm;
            rr = i + (c - c) + 1;
            while rr < mm {
                if u0 != F::new(0.0) {
                    let v = r[i * mm + rr];
                    r[i * mm + rr] = v / u0;
                }
                rr += 1;
            }
        }
        sync_storage();
        if c > i {
            let rows = mm - i;
            let mut dot = r[c * mm + i];
            let mut rr = 1 + (c - c);
            while rr < rows {
                dot += r[i * mm + i + rr] * r[c * mm + i + rr];
                rr += 1;
            }
            if dot != F::new(0.0) {
                let f = tau[i] * dot;
                let v = r[c * mm + i];
                r[c * mm + i] = v - f;
                rr = 1 + (c - c);
                while rr < rows {
                    let wv = r[i * mm + i + rr];
                    let v2 = r[c * mm + i + rr];
                    r[c * mm + i + rr] = v2 - f * wv;
                    rr += 1;
                }
            }
        }
        sync_storage();
        i += 1;
    }
}

/// Q pass: one cube per column c; builds `q = H_1..H_k I_k` back-to-front
/// with a block-wide dot reduction, applies the sign(diag(R)) correction
/// and writes the final row-major `[m, k]` result.
#[cube(launch_unchecked)]
fn sct_qr_q_kernel<F: Float>(
    r: &Array<F>,       // [k*m] column-major: reflectors + R diagonal
    tau: &Array<F>,     // [k]
    q: &mut Array<F>,   // [k*m] column-major Q accumulator
    q_rm: &mut Array<F>,// [m*k] row-major final Q
    #[comptime] m: u32,
    #[comptime] k: u32,
) {
    let c = CUBE_POS_X as usize;
    let t = UNIT_POS_X as usize;
    let nthr = CUBE_DIM as usize;
    let kk = k as usize;
    let mm = m as usize;
    let flip = r[c * mm + c] < F::new(0.0);

    // init column c of Q = I_k
    let mut idx = t;
    while idx < mm {
        q[c * mm + idx] = if idx == c {
            F::new(1.0)
        } else {
            F::new(0.0)
        };
        idx += nthr;
    }
    sync_storage();

    let mut red = SharedMemory::<F>::new(256usize);
    let mut i = kk + (t - t);
    while i > 0 {
        i -= 1;
        let rows = mm - i;
        let tv = tau[i];
        if tv != F::new(0.0) {
            // dot = q[c*m + i] + sum_{rr>=1} w[rr-1] * q[c*m + i+rr]
            let mut acc = q[c * mm + i + 1] - q[c * mm + i + 1];
            let mut rr = t + (t - t);
            while rr < rows - 1 {
                acc += r[i * mm + i + 1 + rr] * q[c * mm + i + 1 + rr];
                rr += nthr;
            }
            red[t] = acc;
            sync_storage();
            let mut step = nthr / 2;
            while step > 0 {
                if t < step {
                    let v = red[t] + red[t + step];
                    red[t] = v;
                }
                sync_storage();
                step /= 2;
            }
            let dot = q[c * mm + i] + red[0];
            let f = tv * dot;
            rr = t;
            while rr < rows - 1 {
                let wv = r[i * mm + i + 1 + rr];
                let v2 = q[c * mm + i + 1 + rr];
                q[c * mm + i + 1 + rr] = v2 - f * wv;
                rr += nthr;
            }
            if t == 0 {
                let v2 = q[c * mm + i];
                q[c * mm + i] = v2 - f;
            }
        }
        sync_storage();
    }

    // sign(diag(R)) correction + transpose to row-major q_rm
    idx = t;
    while idx < mm {
        let mut v = q[c * mm + idx];
        if flip {
            v = -v;
        }
        q_rm[idx * kk + c] = v;
        idx += nthr;
    }
}

/// GPU retraction of `matrix` on the bare CUDA backend. `None` when the
/// backend is not CUDA or the rank exceeds the kernel limit (then the CPU
/// path in [`crate::orthogonalize`] is used).
pub fn retract_cuda<B: Backend>(matrix: Tensor<B, 2>) -> Option<Tensor<B, 2>>
where
    B: 'static,
{
    if !is_cuda::<B>() {
        return None;
    }
    let [m, k] = matrix.dims();
    if m == 0 || k == 0 || k > 1024 {
        return None;
    }
    // SAFETY: guarded by is_cuda above: at runtime B is CudaBare, and
    // Tensor<B, 2> / Tensor<CudaBare, 2> have identical layout (one
    // primitive handle). Letting the compiler see CudaBare gives us the
    // CubeTensor accessors (0.21 has no backend-generic primitive API).
    // SAFETY: guarded by is_cuda above: at runtime B is CudaBare and the two
    // Tensor instantiations share the same layout; we only touch it through
    // the CudaBare reference while B stays alive.
    let matrix_ref: &Tensor<CudaBare, 2> =
        unsafe { &*(&matrix as *const Tensor<B, 2> as *const Tensor<CudaBare, 2>) };
    let a_cube = matrix_ref.clone().into_primitive().tensor();
    let client = a_cube.client.clone();
    // SAFETY: same guarded alias as matrix_ref; B::Device == CudaBare::Device
    // at runtime (is_cuda checked above).
    let device = matrix.device();
    let device_ref: &<CudaBare as burn::tensor::backend::BackendTypes>::Device =
        unsafe { &*(&device as *const B::Device as *const <CudaBare as burn::tensor::backend::BackendTypes>::Device) };

    // Work buffers, filled by the kernels.
    let r = Tensor::<CudaBare, 2>::empty([k, m], device_ref);
    let tau = Tensor::<CudaBare, 1>::empty([k], device_ref);
    let q = Tensor::<CudaBare, 2>::empty([k, m], device_ref);
    let q_rm = Tensor::<CudaBare, 2>::empty([m, k], device_ref);
    let r_cube = r.clone().into_primitive().tensor();
    let tau_cube = tau.clone().into_primitive().tensor();
    let q_cube = q.clone().into_primitive().tensor();
    let q_rm_cube = q_rm.clone().into_primitive().tensor();

    unsafe {
        sct_qr_r_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim {
                x: 1,
                y: k as u32,
                z: 1,
            },
            ArrayArg::from_raw_parts(a_cube.handle, m * k),
            ArrayArg::from_raw_parts(r_cube.handle.clone(), k * m),
            ArrayArg::from_raw_parts(tau_cube.handle.clone(), k),
            m as u32,
            k as u32,
        );
        sct_qr_q_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static(k as u32, 1, 1),
            CubeDim {
                x: 256,
                y: 1,
                z: 1,
            },
            ArrayArg::from_raw_parts(r_cube.handle.clone(), k * m),
            ArrayArg::from_raw_parts(tau_cube.handle.clone(), k),
            ArrayArg::from_raw_parts(q_cube.handle.clone(), k * m),
            ArrayArg::from_raw_parts(q_rm_cube.handle.clone(), m * k),
            m as u32,
            k as u32,
        );
    }
    // TEMP: measure R-only (skip Q)
    std::thread::sleep(std::time::Duration::from_millis(0));

    let out_cuda: Tensor<CudaBare, 2> = Tensor::from_primitive(q_rm.into_primitive());
    // SAFETY: symmetric with the input cast above (same layout, guarded by
    // is_cuda); ptr::read moves without dropping through the alias.
    let out: Tensor<B, 2> = unsafe {
        std::ptr::read(&out_cuda as *const Tensor<CudaBare, 2> as *const Tensor<B, 2>)
    };
    std::mem::forget(out_cuda);
    Some(out)
}
