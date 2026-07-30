#![allow(missing_docs)]

#[cfg(feature = "binary-tests")]
#[cfg(test)]
mod cmp_tests {
    use std::io::Read;
    use burn::module::Param;
    use burn::tensor::{Tensor, TensorData};
    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn_sct::SctLinear;

    type B = NdArray;

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

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn cmp_vs_reference() {
        let dev = NdArrayDevice::default();
        let configs = ["tiny", "small", "med", "large"];

        for &name in configs.iter() {
            let path = format!("/home/sehaxe/aria/sct_cmp/{}.bin", name);
            let mut f = std::fs::File::open(&path)
                .unwrap_or_else(|_| panic!("missing {path}. Run bench_sct_pt.py first"));

            let (_, u_data) = read_arr(&mut f);
            let (_, v_data) = read_arr(&mut f);
            let (_, s_data) = read_arr(&mut f);
            let (x_shape, x_data) = read_arr(&mut f);
            let (y_shape, y_ref) = read_arr(&mut f);

            let m = x_shape[1];
            let n = y_shape[1];
            let k = s_data.len();
            let batch = x_shape[0];

            let mut layer = SctLinear::<B>::new(
                &burn_sct::SctConfig::new(m, n, k),
                &dev,
            );

            // Override params with reference weights
            let u_t = Tensor::<B, 2>::from_data(
                TensorData::new(u_data.clone(), [m, k]), &dev);
            let v_t = Tensor::<B, 2>::from_data(
                TensorData::new(v_data.clone(), [n, k]), &dev);
            let s_t = Tensor::<B, 1>::from_data(
                TensorData::new(s_data.clone(), [k]), &dev);
            layer.u = Param::from_tensor(u_t);
            layer.v = Param::from_tensor(v_t);
            layer.s = Param::from_tensor(s_t);

            let x = Tensor::<B, 2>::from_data(
                TensorData::new(x_data.clone(), [batch, m]), &dev);
            let y = layer.forward(x);

            let y_vals: Vec<f32> = y.into_data().bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();

            assert_eq!(y_vals.len(), y_ref.len());
            let diff = max_diff(&y_vals, &y_ref);

            println!("  {name}: m={m} n={n} k={k} batch={batch} max_diff={diff:.2e}");
            assert!(diff < 1e-4,
                "{name}: max_diff={diff:.2e} exceeds 1e-4. NOT bit-exact but should be within f32 tol.",
            );
        }
        println!("  All configs match reference within 1e-4");
    }
}
