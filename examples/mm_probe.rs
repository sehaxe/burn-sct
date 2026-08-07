use burn::tensor::{Device, Tensor, TensorData};
use std::time::Instant;
fn main() {
    let dev = Device::default();
    // big matmul: naive would take seconds
    let a = Tensor::<2>::from_data(TensorData::new(vec![0.123f32; 256 * 1024], [256, 1024]), &dev);
    let b = Tensor::<2>::from_data(TensorData::new(vec![0.456f32; 1024 * 256], [1024, 256]), &dev);
    for _ in 0..3 {
        let _ = a.clone().matmul(b.clone());
    }
    let t = Instant::now();
    let c = a.matmul(b);
    println!("[256x1024]@[1024x256] = 134 MFLOP in {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
    let exact = 1024.0f32 * 0.123 * 0.456;
    let vals: Vec<f32> = c.into_data().to_vec::<f32>().unwrap();
    let err: f32 = vals.iter().map(|v| (v - exact).abs()).fold(0.0, f32::max);
    println!("err vs exact: {err:e}");
}
