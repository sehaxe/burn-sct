//! GPU retract benchmark (feature cuda). Compare vs CPU bench_ops.
use burn::module::Param;
use burn::tensor::{Device, Distribution, Tensor};
use burn_sct::qr_cuda::CudaBare;
use burn_sct::{SctConfig, SctLinear};
use std::time::Instant;

fn main() {
    let dev = Device::cuda(0);
    let cases = [
        ("tiny", 64usize, 128usize, 8usize),
        ("large", 1024, 2048, 64),
        ("llm_mid", 4096, 4096, 128),
        ("llm_big", 4096, 11008, 128),
        ("llm_huge", 8192, 8192, 256),
    ];
    println!("{:<10} {:>12}", "case", "ret_ms");
    for (name, m, n, k) in cases.iter() {
        let mut layer = SctLinear::new(&SctConfig::new(*m, *n, *k), &dev);
        layer.u = Param::from_tensor(
            layer.u.val()
                + Tensor::<2>::random([*m, *k], Distribution::Normal(0.0, 0.1), &dev),
        );
        layer.v = Param::from_tensor(
            layer.v.val()
                + Tensor::<2>::random([*n, *k], Distribution::Normal(0.0, 0.1), &dev),
        );
        for _ in 0..5 {
            layer.retract::<CudaBare>();
        }
        layer.retract::<CudaBare>();
        let oe = layer.ortho_error();
        let iters = 50;
        let t0 = Instant::now();
        for _ in 0..iters {
            layer.retract::<CudaBare>();
        }
        println!("{:<10} {:>12.4}  ortho_err={:.2e}", name, t0.elapsed().as_secs_f64() * 1000.0 / iters as f64, oe);
    }
}
