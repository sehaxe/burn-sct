//! Op benchmarks: forward / retract / from_dense against the SCT paper's ops.
//! Run: cargo run --release --example bench_ops [name-filter]
use burn::tensor::{Distribution, Tensor};
use burn_sct::SctConfig;
use std::time::Instant;


fn ms(d: std::time::Duration, iters: usize) -> f64 {
    d.as_secs_f64() * 1000.0 / iters as f64
}

fn bench_forward(m: usize, n: usize, k: usize, batch: usize) -> (f64, usize) {
    let dev = burn::tensor::Device::ndarray();
    let layer = burn_sct::SctLinear::new(&SctConfig::new(m, n, k), &dev);
    let x = Tensor::<2>::random([batch, m], Distribution::Normal(0.0, 1.0), &dev);
    for _ in 0..3 {
        layer.forward(x.clone());
    }
    let iters = 20;
    let t = Instant::now();
    for _ in 0..iters {
        layer.forward(x.clone());
    }
    let per = ms(t.elapsed(), iters);
    let flops = 2 * batch as u64 * (m * k + k * n) as u64;
    (per, flops as usize)
}

fn bench_retract(m: usize, n: usize, k: usize) -> f64 {
    let dev = burn::tensor::Device::ndarray();
    let mut layer = burn_sct::SctLinear::new(&SctConfig::new(m, n, k), &dev);
    let pert = || {
        Tensor::<2>::random([m, k], Distribution::Normal(0.0, 0.1), &dev)
    };
    for _ in 0..2 {
        layer.u = burn::module::Param::from_tensor(pert());
        layer.v = burn::module::Param::from_tensor(pert());
        layer.retract::<burn_ndarray::NdArray>();
    }
    let iters = 10;
    let t = Instant::now();
    for _ in 0..iters {
        layer.u = burn::module::Param::from_tensor(pert());
        layer.v = burn::module::Param::from_tensor(pert());
        layer.retract::<burn_ndarray::NdArray>();
    }
    ms(t.elapsed(), iters)
}

fn bench_from_dense(m: usize, n: usize, k: usize, sweeps: usize, iters: usize) -> f64 {
    let dev = burn::tensor::Device::ndarray();
    let w = Tensor::<2>::random([n, m], Distribution::Normal(0.0, 1.0), &dev);
    let t = Instant::now();
    for _ in 0..iters {
        let _ = burn_sct::SctLinear::from_dense_with_iters::<burn_ndarray::NdArray>(w.clone(), k, sweeps);
    }
    ms(t.elapsed(), iters)
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    // (name, m(in), n(out), k, batch, svd_sweeps, svd_iters)
    let cases: Vec<(&str, usize, usize, usize, usize, usize, usize)> = vec![
        ("tiny", 64, 128, 8, 4, 15, 3),
        ("small", 256, 512, 16, 2, 15, 3),
        ("med", 512, 1024, 32, 2, 15, 2),
        ("large", 1024, 2048, 64, 1, 15, 1),
        ("llm_mid", 4096, 4096, 128, 64, 15, 1),
        ("llm_big", 4096, 11008, 128, 64, 15, 1),
    ];
    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "case", "fwd_ms", "fwd_gf", "ret_ms", "svd_ms", "m", "n"
    );
    for (name, m, n, k, batch, sweeps, svd_iters) in cases.iter() {
        if !filter.is_empty() && !name.contains(&filter) {
            continue;
        }
        let (fwd, flops) = bench_forward(*m, *n, *k, *batch);
        let ret = bench_retract(*m, *n, *k);
        let svd = bench_from_dense(*m, *n, *k, *sweeps, *svd_iters);
        let gflops = flops as f64 / (fwd / 1000.0) / 1e9;
        println!(
            "{:<10} {:>10.3} {:>10.1} {:>10.3} {:>12.3} {:>10} {:>10}",
            name, fwd, gflops, ret, svd, m, n
        );
    }
}
