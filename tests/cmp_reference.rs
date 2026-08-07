#![allow(missing_docs)]

#[cfg(feature = "binary-tests")]
#[cfg(test)]
mod cmp_tests {
    use burn::module::Param;
    use burn::tensor::{Device, Tensor, TensorData};
    
    use burn_sct::SctLinear;
    use std::io::Read;


    /// Binary layout per file (see gen_reference.py):
    /// u, v, s, x, y, u_pert, v_pert, u_ret, v_ret, w_dense, w_recon
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
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        (shape, floats)
    }

    fn tensor2<const D: usize>(shape: &[usize], data: &[f32], dev: &Device) -> Tensor<D> {
        let mut dims = [1usize; D];
        dims[..D].copy_from_slice(shape);
        Tensor::from_data(TensorData::new(data.to_vec(), dims), dev)
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn cmp_vs_reference() {
        let dev = Device::ndarray();
        let configs = ["tiny", "small", "med", "large"];

        for &name in configs.iter() {
            let path = format!("{}/tests/ref_data/{}.bin", env!("CARGO_MANIFEST_DIR"), name);
            let mut f = std::fs::File::open(&path)
                .unwrap_or_else(|_| panic!("missing {path}. Run python3 gen_reference.py first"));

            let (_, u_data) = read_arr(&mut f);
            let (_, v_data) = read_arr(&mut f);
            let (_, s_data) = read_arr(&mut f);
            let (x_shape, x_data) = read_arr(&mut f);
            let (y_shape, y_ref) = read_arr(&mut f);
            let (_, u_pert) = read_arr(&mut f);
            let (_, v_pert) = read_arr(&mut f);
            let (_, u_ret) = read_arr(&mut f);
            let (_, v_ret) = read_arr(&mut f);
            let (_, w_dense) = read_arr(&mut f);
            let (_, w_recon) = read_arr(&mut f);

            let m = x_shape[1];
            let n = y_shape[1];
            let k = s_data.len();
            let batch = x_shape[0];

            let mut layer = burn_sct::SctLinear::new(&burn_sct::SctConfig::new(m, n, k), &dev);
            layer.u = Param::from_tensor(tensor2::<2>(&[m, k], &u_data, &dev));
            layer.v = Param::from_tensor(tensor2::<2>(&[n, k], &v_data, &dev));
            layer.s = Param::from_tensor(tensor2::<1>(&[k], &s_data, &dev));

            // --- forward ---
            let x = tensor2::<2>(&[batch, m], &x_data, &dev);
            let y = layer.forward(x);
            let y_vals: Vec<f32> = y
                .into_data()
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let fwd_diff = max_diff(&y_vals, &y_ref);

            // --- retract ---
            layer.u = Param::from_tensor(tensor2::<2>(&[m, k], &u_pert, &dev));
            layer.v = Param::from_tensor(tensor2::<2>(&[n, k], &v_pert, &dev));
            layer.retract::<burn_ndarray::NdArray>();
            let u_ret_vals: Vec<f32> = layer
                .u
                .val()
                .clone()
                .into_data()
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let v_ret_vals: Vec<f32> = layer
                .v
                .val()
                .clone()
                .into_data()
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let ret_diff = max_diff(&u_ret_vals, &u_ret).max(max_diff(&v_ret_vals, &v_ret));

            // --- from_dense ---
            // Compare burn's rank-k reconstruction against torch's rank-k
            // reconstruction of the same weight (sign-invariant). The raw
            // dense W is not comparable: a truncated SVD of a full-rank
            // matrix has intrinsic truncation error.
            let conv = burn_sct::SctLinear::from_dense::<burn_ndarray::NdArray>(tensor2::<2>(&[n, m], &w_dense, &dev), k);
            let recon = conv
                .v
                .val()
                .clone()
                .mul(conv.s.val().clone().unsqueeze_dims(&[0]))
                .matmul(conv.u.val().clone().transpose()); // [n, m]
            let recon_vals: Vec<f32> = recon
                .into_data()
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let svd_diff = max_diff(&recon_vals, &w_recon);

            println!(
                "  {name}: m={m} n={n} k={k} batch={batch} fwd={fwd_diff:.2e} retract={ret_diff:.2e} from_dense={svd_diff:.2e}"
            );
            assert!(
                fwd_diff < 1e-4,
                "{name}: forward max_diff={fwd_diff:.2e} exceeds 1e-4"
            );
            assert!(
                ret_diff < 1e-4,
                "{name}: retract max_diff={ret_diff:.2e} exceeds 1e-4"
            );
            assert!(
                svd_diff < 1e-3,
                "{name}: from_dense max_diff={svd_diff:.2e} exceeds 1e-3"
            );
        }
        println!("  All configs match the paper's reference within tolerance");
    }
}
