use mopac_core::export::cube::{
    ANGSTROM_TO_BOHR, CubeGridConfig, evaluate_density_at_point, generate_density_cube,
};
use mopac_core::gradients::nuclear_gradients::{
    GradientWorkspace, compute_cartesian_gradients_with_options,
};
use mopac_core::integrals::multipoles::{
    assemble_nddo_two_center_fock, precompute_diatomic_pairs,
};
use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::scf::eigensolver::diagonalize_symmetric;
use mopac_core::scf::scf_loop::{ScfOptions, run_rhf_scf_with_options};
use mopac_core::solvation::cosmo::{CosmoParams, CosmoState};
use mopac_core::types::{AlignedMatrix, AlignedVec64, MolecularBatch, ScfWorkspace};
use std::hint::black_box;
use std::time::Instant;

fn make_alkane(n_carbons: usize) -> (Vec<u8>, Vec<[f64; 3]>) {
    let mut z = Vec::with_capacity(3 * n_carbons + 2);
    let mut coords = Vec::with_capacity(3 * n_carbons + 2);
    for i in 0..n_carbons {
        let cx = i as f64 * 1.25;
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let cy = sign * 0.44;
        z.push(6u8);
        coords.push([cx, cy, 0.0]);
        z.push(1u8);
        coords.push([cx, cy + sign * 0.64, 0.89]);
        z.push(1u8);
        coords.push([cx, cy + sign * 0.64, -0.89]);
        if i == 0 {
            z.push(1u8);
            coords.push([cx - 0.90, cy - sign * 0.63, 0.0]);
        }
        if i == n_carbons - 1 {
            z.push(1u8);
            coords.push([cx + 0.90, cy - sign * 0.63, 0.0]);
        }
    }
    (z, coords)
}

fn make_benzene() -> (Vec<u8>, Vec<[f64; 3]>) {
    let mut z = Vec::with_capacity(12);
    let mut coords = Vec::with_capacity(12);
    let r_c = 1.395;
    let r_h = 2.485;
    for i in 0..6 {
        let ang = (i as f64) * std::f64::consts::PI / 3.0;
        z.push(6u8);
        coords.push([r_c * ang.cos(), r_c * ang.sin(), 0.0]);
    }
    for i in 0..6 {
        let ang = (i as f64) * std::f64::consts::PI / 3.0;
        z.push(1u8);
        coords.push([r_h * ang.cos(), r_h * ang.sin(), 0.0]);
    }
    (z, coords)
}

fn median_us<F: FnMut()>(warmups: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmups {
        f();
    }
    let mut samples_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples_us[samples_us.len() / 2]
}

fn build_symmetric_test_matrix(n: usize) -> AlignedMatrix<f64> {
    let mut a = AlignedMatrix::zeroed(n, n);
    for i in 0..n {
        for j in 0..=i {
            let val = if i == j {
                (i as f64) * 0.13 + ((i % 7) as f64) * 0.41 - 2.0
            } else {
                0.35 * ((i as f64 + 1.7) * (j as f64 + 0.9)).sin() / (1.0 + (i - j) as f64)
            };
            a.set(i, j, val);
            a.set(j, i, val);
        }
    }
    a
}

fn main() {
    // 1. Symmetric eigensolver microbenchmarks & numerical invariants
    for &(n, iters) in &[(64usize, 15usize), (150usize, 7usize), (250usize, 5usize)] {
        let a = build_symmetric_test_matrix(n);
        let mut evals = AlignedVec64::zeroed(n);
        let mut evecs = AlignedMatrix::zeroed(n, n);
        let med = median_us(2, iters, || {
            let s = diagonalize_symmetric(black_box(&a), black_box(&mut evals), black_box(&mut evecs));
            black_box(s);
            black_box(evals[0]);
        });
        let _ = diagonalize_symmetric(&a, &mut evals, &mut evecs);
        let mut max_orth_err = 0.0f64;
        let mut max_res_err = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                let mut dot_orth = 0.0;
                let mut ac_ij = 0.0;
                for k in 0..n {
                    dot_orth += evecs.get(k, i) * evecs.get(k, j);
                    ac_ij += a.get(i, k) * evecs.get(k, j);
                }
                let target_orth = if i == j { 1.0 } else { 0.0 };
                max_orth_err = max_orth_err.max((dot_orth - target_orth).abs());
                let ce_ij = evecs.get(i, j) * evals[j];
                max_res_err = max_res_err.max((ac_ij - ce_ij).abs());
            }
        }
        println!(
            "BENCH_JSON:{{\"name\":\"eigensolver_n{}\",\"unit\":\"us\",\"value\":{:.3},\"orth_err\":{:.3e},\"res_err\":{:.3e},\"e0\":{:.12},\"emid\":{:.12},\"elast\":{:.12}}}",
            n, med, max_orth_err, max_res_err, evals[0], evals[n / 2], evals[n - 1]
        );
    }

    // 2. NDDO diatomic pair precomputation + Fock assembly on C14H30 (44 atoms, 946 pairs)
    let (z14, c14) = make_alkane(14);
    let pm6 = Pm6Model;
    let batch14 = MolecularBatch::new_for_model(z14, &c14, &pm6);
    let precompute_us = median_us(2, 11, || {
        let p = precompute_diatomic_pairs(black_box(&batch14), black_box(&pm6));
        black_box(p.len());
    });
    println!(
        "BENCH_JSON:{{\"name\":\"nddo_precompute_c14h30\",\"unit\":\"us\",\"value\":{:.3},\"natoms\":{},\"pairs\":{}}}",
        precompute_us,
        batch14.natoms,
        batch14.natoms * (batch14.natoms - 1) / 2
    );

    let pairs14 = precompute_diatomic_pairs(&batch14, &pm6);
    let mut det_density14 = AlignedMatrix::zeroed(batch14.norbs, batch14.norbs);
    for i in 0..batch14.norbs {
        for j in 0..=i {
            let val = if i == j {
                0.85
            } else {
                0.08 * ((i + j + 1) as f64 * 0.17).sin()
            };
            det_density14.set(i, j, val);
            det_density14.set(j, i, val);
        }
    }
    let mut fock_scratch = AlignedMatrix::zeroed(batch14.norbs, batch14.norbs);
    let fock_us = median_us(3, 15, || {
        fock_scratch.data.fill(0.0);
        assemble_nddo_two_center_fock(
            black_box(&pairs14),
            black_box(&det_density14),
            black_box(&mut fock_scratch),
        );
        black_box(fock_scratch.get(0, 0));
    });
    fock_scratch.data.fill(0.0);
    assemble_nddo_two_center_fock(&pairs14, &det_density14, &mut fock_scratch);
    let mut fock_fro_sq = 0.0f64;
    let mut fock_sum = 0.0f64;
    for r in 0..batch14.norbs {
        for c in 0..batch14.norbs {
            let v = fock_scratch.get(r, c);
            fock_fro_sq += v * v;
            fock_sum += v;
        }
    }
    println!(
        "BENCH_JSON:{{\"name\":\"nddo_fock_c14h30\",\"unit\":\"us\",\"value\":{:.3},\"norbs\":{},\"pairs\":{},\"fock_fro\":{:.12},\"fock_sum\":{:.12}}}",
        fock_us,
        batch14.norbs,
        pairs14.len(),
        fock_fro_sq.sqrt(),
        fock_sum
    );

    // 2b. Empirical scaling sweep across pair counts to locate Rayon vs serial crossover
    for &nc in &[6usize, 10, 14, 19, 24, 30, 38, 48] {
        let (zn, cn) = make_alkane(nc);
        let bn = MolecularBatch::new_for_model(zn, &cn, &pm6);
        let pn = precompute_diatomic_pairs(&bn, &pm6);
        let mut p_den = AlignedMatrix::zeroed(bn.norbs, bn.norbs);
        for i in 0..bn.norbs {
            for j in 0..=i {
                let val = 0.1 * ((i + j + 1) as f64 * 0.17).sin();
                p_den.set(i, j, val);
                p_den.set(j, i, val);
            }
        }
        let mut f_buf = AlignedMatrix::zeroed(bn.norbs, bn.norbs);
        let sweep_us = median_us(2, 11, || {
            f_buf.data.fill(0.0);
            assemble_nddo_two_center_fock(
                black_box(&pn),
                black_box(&p_den),
                black_box(&mut f_buf),
            );
            black_box(f_buf.get(0, 0));
        });
        println!(
            "BENCH_JSON:{{\"name\":\"fock_sweep_c{}\",\"unit\":\"us\",\"value\":{:.3},\"natoms\":{},\"pairs\":{},\"norbs\":{}}}",
            nc,
            sweep_us,
            bn.natoms,
            pn.len(),
            bn.norbs
        );
    }

    // 3. Analytical Cartesian nuclear gradients on C10H22 (32 atoms, 496 pairs)
    let scf_opts = ScfOptions {
        use_nddo: true,
        max_iter: 80,
        ..Default::default()
    };
    let (z10, c10) = make_alkane(10);
    let am1 = Am1Model;
    let mut batch10 = MolecularBatch::new_for_model(z10, &c10, &am1);
    let mut ws10 = ScfWorkspace::allocate(batch10.norbs);
    let _res10 = run_rhf_scf_with_options(&batch10, &am1, &mut ws10, &scf_opts);
    let mut grad_ws10 = GradientWorkspace::allocate(batch10.norbs);
    let mut grads10 = vec![[0.0f64; 3]; batch10.natoms];
    let grad_us = median_us(1, 5, || {
        compute_cartesian_gradients_with_options(
            black_box(&mut batch10),
            black_box(&am1),
            black_box(&ws10.density),
            black_box(&mut grad_ws10),
            black_box(&mut grads10),
            true,
        );
        black_box(grads10[0]);
    });
    println!(
        "BENCH_JSON:{{\"name\":\"nuclear_gradients_c10h22\",\"unit\":\"us\",\"value\":{:.3},\"natoms\":{},\"pairs\":{}}}",
        grad_us,
        batch10.natoms,
        batch10.natoms * (batch10.natoms - 1) / 2
    );

    // 4. COSMO initialization, b_matrix footprint, and reaction-field Fock evaluation on C14H30
    let cosmo_params = CosmoParams {
        epsilon: 78.4,
        rsolv: 1.3,
    };
    let cosmo_init_us = median_us(1, 5, || {
        let s = CosmoState::initialize(black_box(&batch14), black_box(&pm6), cosmo_params)
            .expect("COSMO init failed");
        black_box(s.e_nuc_diel_ev);
    });
    let mut cosmo_state =
        CosmoState::initialize(&batch14, &pm6, cosmo_params).expect("COSMO init failed");
    let b_rows = cosmo_state.b_matrix.rows;
    let b_cols = cosmo_state.b_matrix.cols;
    let b_bytes = b_rows * b_cols * std::mem::size_of::<f64>();
    let cosmo_fock_us = median_us(2, 9, || {
        fock_scratch.data.fill(0.0);
        let ediel = cosmo_state.apply_electronic_reaction_field_to_fock(
            black_box(&det_density14),
            black_box(&mut fock_scratch),
        );
        black_box(ediel);
    });
    fock_scratch.data.fill(0.0);
    let cosmo_ediel =
        cosmo_state.apply_electronic_reaction_field_to_fock(&det_density14, &mut fock_scratch);
    println!(
        "BENCH_JSON:{{\"name\":\"cosmo_c14h30\",\"init_us\":{:.3},\"fock_us\":{:.3},\"b_rows\":{},\"b_cols\":{},\"b_mb\":{:.3},\"ediel_ev\":{:.12},\"fock00\":{:.12}}}",
        cosmo_init_us,
        cosmo_fock_us,
        b_rows,
        b_cols,
        b_bytes as f64 / (1024.0 * 1024.0),
        cosmo_ediel,
        fock_scratch.get(0, 0)
    );

    // 5. CPU Gaussian Cube density grid + ASCII serialization on Benzene (resolution = 0.14 A)
    let (zbz, cbz) = make_benzene();
    let batch_bz = MolecularBatch::new_for_model(zbz, &cbz, &pm6);
    let mut ws_bz = ScfWorkspace::allocate(batch_bz.norbs);
    let _res_bz = run_rhf_scf_with_options(&batch_bz, &pm6, &mut ws_bz, &scf_opts);
    let cube_cfg = CubeGridConfig {
        padding_angstrom: 3.0,
        resolution_angstrom: 0.14,
        n_threads: None,
    };
    let cube_us = median_us(1, 3, || {
        let s = generate_density_cube(
            black_box(&batch_bz),
            black_box(&pm6),
            black_box(&ws_bz.density),
            black_box(&cube_cfg),
        );
        black_box(s.len());
    });
    println!(
        "BENCH_JSON:{{\"name\":\"cube_cpu_benzene_fine\",\"unit\":\"us\",\"value\":{:.3}}}",
        cube_us
    );

    // 6. Vulkan GPU >64 active atoms density grid accuracy & timing
    if let Ok(ctx) = mopac_gpu::VulkanContext::new()
        && let Ok(gpu_cube) = mopac_gpu::GpuCubeEvaluator::new(std::sync::Arc::new(ctx))
    {
        let natoms = 72usize;
        let atomic_numbers = vec![1u8; natoms];
        let mut coords = Vec::with_capacity(natoms);
        let golden_angle = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        for i in 0..natoms {
            let t = (i as f64 + 0.5) / natoms as f64;
            let y = 1.0 - 2.0 * t;
            let r_xy = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f64;
            let radius = 1.85;
            coords.push([
                radius * r_xy * theta.cos(),
                radius * y,
                radius * r_xy * theta.sin(),
            ]);
        }
        let batch_72 = MolecularBatch::new_for_model(atomic_numbers, &coords, &pm6);
        let mut density_72 = AlignedMatrix::zeroed(natoms, natoms);
        for i in 0..natoms {
            density_72.set(i, i, if i >= 64 { 1.5 } else { 0.4 });
        }
        for i in 0..natoms {
            let j = (i + 64) % natoms;
            if i != j {
                density_72.set(i, j, 0.12);
                density_72.set(j, i, 0.12);
            }
        }
        let dims = (8usize, 8usize, 8usize);
        let origin_bohr = [-1.4f64, -1.4f64, -1.4f64];
        let step_bohr = 0.4f64;

        let gpu_72_us = median_us(1, 5, || {
            let grid = gpu_cube
                .compute_density_grid(
                    black_box(&batch_72),
                    black_box(&pm6),
                    black_box(&density_72),
                    origin_bohr,
                    step_bohr,
                    dims,
                )
                .expect("GPU grid failed");
            black_box(grid[0]);
        });
        let gpu_grid = gpu_cube
            .compute_density_grid(&batch_72, &pm6, &density_72, origin_bohr, step_bohr, dims)
            .expect("GPU grid failed");
        let mut max_abs_err = 0.0f64;
        for ix in 0..dims.0 {
            for iy in 0..dims.1 {
                for iz in 0..dims.2 {
                    let idx = ix * dims.1 * dims.2 + iy * dims.2 + iz;
                    let pt_a = [
                        (origin_bohr[0] + ix as f64 * step_bohr) / ANGSTROM_TO_BOHR,
                        (origin_bohr[1] + iy as f64 * step_bohr) / ANGSTROM_TO_BOHR,
                        (origin_bohr[2] + iz as f64 * step_bohr) / ANGSTROM_TO_BOHR,
                    ];
                    let cpu_val =
                        evaluate_density_at_point(&batch_72, &pm6, &density_72, pt_a);
                    max_abs_err = max_abs_err.max((gpu_grid[idx] - cpu_val).abs());
                }
            }
        }
        println!(
            "BENCH_JSON:{{\"name\":\"gpu_dense_cluster_72_atoms\",\"unit\":\"us\",\"value\":{:.3},\"max_abs_err_vs_cpu\":{:.8}}}",
            gpu_72_us, max_abs_err
        );
    }
}
