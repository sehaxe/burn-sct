//! CUDA retraction must match the CPU reference exactly (feature "cuda").
#![cfg(feature = "cuda")]
use burn::module::Param;
use burn::tensor::{Device, Distribution, Tensor};
use burn_sct::qr::qr_cpu;
use burn_sct::qr_cuda::CudaBare;
use burn_sct::{SctConfig, SctLinear};
use std::time::Instant;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cuda_dev() -> Device {
    Device::cuda(0)
}

#[test]
fn gpu_retract_matches_cpu() {
    let dev = cuda_dev();
    let (m, k) = (1024usize, 128usize);
    let x = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev);

    // GPU path through the layer (kernels, no host round-trip in between).
    let mut layer = SctLinear::new(&SctConfig::new(m, m * 2, k), &dev);
    layer.u = Param::from_tensor(x.clone());
    layer.retract::<CudaBare>();
    let gpu: Vec<f32> = layer.u.val().clone().into_data().to_vec::<f32>().unwrap();

    // CPU reference on the same data.
    let data: Vec<f32> = x.clone().into_data().to_vec::<f32>().unwrap();
    let (cpu, _) = qr_cpu(&data, m, k);
    let diff = max_diff(&gpu, &cpu);
    assert!(diff < 1e-4, "GPU vs CPU retract max_diff {diff:.3e}");
    println!("gpu retract matches cpu: max_diff {diff:.3e}");
}

#[test]
fn gpu_retract_speed() {
    let dev = cuda_dev();
    let (m, n, k) = (2048usize, 8192usize, 128usize);
    let mut layer = SctLinear::new(&SctConfig::new(m, n, k), &dev);
    for _ in 0..10 {
        layer.retract::<CudaBare>(); // warm up the kernel JIT
    }
    let _ = layer.u.val().clone().into_data();
    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        layer.retract::<CudaBare>();
    }
    let _ = layer.u.val().clone().into_data(); // sync the queue once
    let t = t0.elapsed().as_secs_f64() / iters as f64;
    println!("gpu retract {m}x{n} k={k}: {:.4} ms/step", t * 1e3);
}
