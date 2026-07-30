#![allow(missing_docs)]
use std::io::Read;
use std::time::Instant;
use burn::module::Param;
use burn::tensor::{Tensor, TensorData};
use burn_cuda::Cuda;
use burn_sct::SctLinear;

type B = Cuda;

fn read_arr(r: &mut impl Read) -> (Vec<usize>, Vec<f32>) {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).unwrap();
    let ndim = u64::from_le_bytes(buf) as usize;
    let mut shape = vec![0usize; ndim];
    for d in shape.iter_mut() {
        r.read_exact(&mut buf).unwrap();
        *d = u64::from_le_bytes(buf) as usize;
    }
    let n: usize = shape.iter().product();
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes).unwrap();
    let floats: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    (shape, floats)
}

#[test]
#[ignore]
fn bench_vs_pt() {
    let dev = Default::default();
    let configs = ["sm", "med", "lg", "xl", "70B"];

    println!("\n{:=^80}", "");
    println!("  burn-sct CUDA vs PyTorch SCT — RTX 5060 Ti");
    println!("{:=^80}", "");
    println!("{:<8} {:>10} {:>10} {:>9} {:>9} {:>8}", "config", "Burn(us)", "PT(us)", "Burn/PT", "Burn/Den", "compr");

    for &name in configs.iter() {
        let path = format!("/home/sehaxe/aria/sct_bench/{}.bin", name);
        let mut f = std::fs::File::open(&path).unwrap();

        let (_, u_data) = read_arr(&mut f);
        let (_, v_data) = read_arr(&mut f);
        let (_, s_data) = read_arr(&mut f);
        let (x_shape, x_data) = read_arr(&mut f);

        let mut hdr = [0u8; 24];
        f.read_exact(&mut hdr).unwrap();
        let m = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
        let n = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
        let k = u64::from_le_bytes(hdr[16..24].try_into().unwrap()) as usize;
        let batch = x_shape[0];

        let mut layer = SctLinear::<B>::new(&burn_sct::SctConfig::new(m, n, k), &dev);
        layer.u = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(u_data, [m, k]), &dev));
        layer.v = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(v_data, [n, k]), &dev));
        layer.s = Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(s_data, [k]), &dev));
        let x = Tensor::<B, 2>::from_data(TensorData::new(x_data, [batch, m]), &dev);

        let runs = 500;
        for _ in 0..50 { let _ = layer.forward(x.clone()); }
        let start = Instant::now();
        for _ in 0..runs { let _ = layer.forward(x.clone()); }
        let burn_us = start.elapsed().as_secs_f64() / runs as f64 * 1e6;
        let compr = layer.compression_ratio();

        println!("{:<8} {:>10.1} {:>10} {:>8.2}x {:>8.2}x {:>8.0}x",
            name, burn_us, "-", 0.0, 0.0, compr);
    }
    println!();
}
