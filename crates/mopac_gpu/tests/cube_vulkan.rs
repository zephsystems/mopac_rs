//! Empirical Parity and Stress Verification Suite for Vulkan GPU Gaussian Cube Evaluator.
//!
//! Evaluates numerical parity between CPU Rayon multithreaded grid evaluation
//! and direct Vulkan compute shader kernels for Molecular Orbitals and Total Density.

use mopac_core::export::cube::{
    generate_density_cube, generate_molecular_orbital_cube, CubeGridConfig,
};
use mopac_core::parameters::{Pm6Model, Pm7Model};
use mopac_core::scf::scf_loop::run_rhf_scf;
use mopac_core::types::{MolecularBatch, ScfWorkspace};
use mopac_gpu::{GpuCubeEvaluator, VulkanContext};
use std::sync::Arc;

fn build_water() -> (MolecularBatch, Pm7Model) {
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.469],
        [0.0, -0.757, -0.469],
    ];
    (MolecularBatch::new(atomic_numbers, &coords), Pm7Model)
}

fn build_benzene() -> (MolecularBatch, Pm6Model) {
    let atomic_numbers = vec![6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1];
    let coords = vec![
        [1.397, 0.0, 0.0],
        [0.698, 1.210, 0.0],
        [-0.698, 1.210, 0.0],
        [-1.397, 0.0, 0.0],
        [-0.698, -1.210, 0.0],
        [0.698, -1.210, 0.0],
        [2.481, 0.0, 0.0],
        [1.240, 2.148, 0.0],
        [-1.240, 2.148, 0.0],
        [-2.481, 0.0, 0.0],
        [-1.240, -2.148, 0.0],
        [1.240, -2.148, 0.0],
    ];
    (MolecularBatch::new(atomic_numbers, &coords), Pm6Model)
}

#[test]
fn test_vulkan_gpu_water_orbital_cube_parity() {
    let ctx = match VulkanContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping Vulkan test: {}", e);
            return;
        }
    };

    let gpu_cube =
        GpuCubeEvaluator::new(Arc::clone(&ctx)).expect("Failed to create GpuCubeEvaluator");
    let (batch, model) = build_water();

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res = run_rhf_scf(&batch, &model, &mut ws, 100, 1e-6, 1e-5);
    assert!(scf_res.converged);

    let config = CubeGridConfig {
        padding_angstrom: 2.0,
        resolution_angstrom: 0.3,
        n_threads: Some(1),
    };

    let homo_idx = batch.norbs / 2 - 1;
    let homo_energy = ws.eigenvalues[homo_idx];
    let mo_coeffs = ws.eigenvectors.row(homo_idx);

    let cpu_cube = generate_molecular_orbital_cube(
        &batch,
        &model,
        mo_coeffs,
        homo_idx + 1,
        homo_energy,
        &config,
    );

    let gpu_cube_str = gpu_cube
        .generate_molecular_orbital_cube(
            &batch,
            &model,
            mo_coeffs,
            homo_idx + 1,
            homo_energy,
            &config,
        )
        .expect("GPU orbital cube generation failed");

    assert!(!cpu_cube.is_empty());
    assert!(!gpu_cube_str.is_empty());

    let cpu_lines: Vec<&str> = cpu_cube.lines().collect();
    let gpu_lines: Vec<&str> = gpu_cube_str.lines().collect();

    assert_eq!(
        cpu_lines.len(),
        gpu_lines.len(),
        "Line counts must match exactly"
    );

    let mut max_diff: f64 = 0.0;
    for i in 8..cpu_lines.len() {
        let cpu_vals: Vec<f64> = cpu_lines[i]
            .split_whitespace()
            .map(|s| s.parse::<f64>().expect("Float parse"))
            .collect();
        let gpu_vals: Vec<f64> = gpu_lines[i]
            .split_whitespace()
            .map(|s| s.parse::<f64>().expect("Float parse"))
            .collect();

        assert_eq!(cpu_vals.len(), gpu_vals.len());
        for (c, g) in cpu_vals.iter().zip(gpu_vals.iter()) {
            let diff = (c - g).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    println!(
        "Water HOMO Cube Max |CPU - GPU| Difference: {:.6E}",
        max_diff
    );
    assert!(
        max_diff < 5e-4,
        "Orbital cube CPU/GPU difference exceeded threshold: {:.6E}",
        max_diff
    );
}

#[test]
fn test_vulkan_gpu_benzene_density_cube_parity() {
    let ctx = match VulkanContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping Vulkan test: {}", e);
            return;
        }
    };

    let gpu_cube =
        GpuCubeEvaluator::new(Arc::clone(&ctx)).expect("Failed to create GpuCubeEvaluator");
    let (batch, model) = build_benzene();

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res = run_rhf_scf(&batch, &model, &mut ws, 100, 1e-6, 1e-5);
    assert!(scf_res.converged);

    let config = CubeGridConfig {
        padding_angstrom: 2.0,
        resolution_angstrom: 0.35,
        n_threads: Some(1),
    };

    let cpu_cube = generate_density_cube(&batch, &model, &ws.density, &config);
    let gpu_cube_str = gpu_cube
        .generate_density_cube(&batch, &model, &ws.density, &config)
        .expect("GPU density cube generation failed");

    assert!(!cpu_cube.is_empty());
    assert!(!gpu_cube_str.is_empty());

    let cpu_lines: Vec<&str> = cpu_cube.lines().collect();
    let gpu_lines: Vec<&str> = gpu_cube_str.lines().collect();

    assert_eq!(cpu_lines.len(), gpu_lines.len());

    let mut max_diff: f64 = 0.0;
    for i in 18..cpu_lines.len() {
        let cpu_vals: Vec<f64> = cpu_lines[i]
            .split_whitespace()
            .map(|s| s.parse::<f64>().expect("Float parse"))
            .collect();
        let gpu_vals: Vec<f64> = gpu_lines[i]
            .split_whitespace()
            .map(|s| s.parse::<f64>().expect("Float parse"))
            .collect();

        assert_eq!(cpu_vals.len(), gpu_vals.len());
        for (c, g) in cpu_vals.iter().zip(gpu_vals.iter()) {
            let diff = (c - g).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    println!(
        "Benzene Density Cube Max |CPU - GPU| Difference: {:.6E}",
        max_diff
    );
    assert!(
        max_diff < 1e-3,
        "Density cube CPU/GPU difference exceeded threshold: {:.6E}",
        max_diff
    );
}

#[test]
fn test_vulkan_gpu_cube_contratest_prime_dimensions() {
    let ctx = match VulkanContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping Vulkan test: {}", e);
            return;
        }
    };

    let gpu_cube =
        GpuCubeEvaluator::new(Arc::clone(&ctx)).expect("Failed to create GpuCubeEvaluator");
    let (batch, model) = build_water();

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res = run_rhf_scf(&batch, &model, &mut ws, 100, 1e-6, 1e-5);
    assert!(scf_res.converged);

    let dims = (17, 23, 31);
    let origin_bohr = [-5.0, -5.0, -5.0];
    let step_bohr = 0.5;

    let grid_gpu = gpu_cube
        .compute_density_grid(&batch, &model, &ws.density, origin_bohr, step_bohr, dims)
        .expect("GPU compute density grid failed for prime dimensions");

    assert_eq!(grid_gpu.len(), 17 * 23 * 31);
    assert!(grid_gpu.iter().all(|&v| !v.is_nan() && !v.is_infinite()));
}

#[test]
fn test_vulkan_gpu_cube_contratest_monoatomic_system() {
    let ctx = match VulkanContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping Vulkan test: {}", e);
            return;
        }
    };

    let gpu_cube =
        GpuCubeEvaluator::new(Arc::clone(&ctx)).expect("Failed to create GpuCubeEvaluator");
    let batch = MolecularBatch::new(vec![1], &[[0.0, 0.0, 0.0]]);
    let model = Pm7Model;

    let mo_coeffs = vec![1.0];
    let dims = (10, 10, 10);
    let origin_bohr = [-2.0, -2.0, -2.0];
    let step_bohr = 0.4;

    let grid = gpu_cube
        .compute_orbital_grid(&batch, &model, &mo_coeffs, origin_bohr, step_bohr, dims)
        .expect("GPU compute orbital grid failed for monoatomic H");

    assert_eq!(grid.len(), 1000);
    assert!(grid.iter().any(|&v| v > 0.01));
}

#[test]
fn test_vulkan_gpu_dense_grid_speedup() {
    let ctx = match VulkanContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping Vulkan test: {}", e);
            return;
        }
    };

    let gpu_cube =
        GpuCubeEvaluator::new(Arc::clone(&ctx)).expect("Failed to create GpuCubeEvaluator");
    let (batch, model) = build_benzene();

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res = run_rhf_scf(&batch, &model, &mut ws, 100, 1e-6, 1e-5);
    assert!(scf_res.converged);

    // Fine grid with 0.2 A resolution -> ~100,000+ voxels
    let config = CubeGridConfig {
        padding_angstrom: 2.5,
        resolution_angstrom: 0.2,
        n_threads: None,
    };

    let t0_cpu = std::time::Instant::now();
    let cpu_cube = generate_density_cube(&batch, &model, &ws.density, &config);
    let dur_cpu = t0_cpu.elapsed();

    let t0_gpu = std::time::Instant::now();
    let gpu_cube_str = gpu_cube
        .generate_density_cube(&batch, &model, &ws.density, &config)
        .expect("GPU dense grid evaluation failed");
    let dur_gpu = t0_gpu.elapsed();

    println!(
        "Dense Grid Benchmark (~100k voxels): CPU = {:.2?}, GPU = {:.2?} (Speedup = {:.2}x)",
        dur_cpu,
        dur_gpu,
        dur_cpu.as_secs_f64() / dur_gpu.as_secs_f64()
    );

    assert!(!cpu_cube.is_empty());
    assert!(!gpu_cube_str.is_empty());
}

/// Task GPU-3 / Test 1: Performance Throughput Benchmark for GPU Volumetric Grid with d-Orbitals.
#[test]
fn test_gpu_cube_d_orbitals_throughput_benchmark() {
    use mopac_core::export::cube::CubeGridConfig;
    use mopac_core::parameters::Pm7Model;
    use mopac_core::scf::scf_loop::run_rhf_scf;
    use mopac_core::types::{BasisType, MolecularBatch, ScfWorkspace};

    let ctx = match VulkanContext::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return, // Skip gracefully if Vulkan runtime not present
    };

    let gpu_cube =
        GpuCubeEvaluator::new(Arc::clone(&ctx)).expect("Failed to initialize GpuCubeEvaluator");

    // H2S: Sulfur Z=16 (norbs=9, BasisType::SPD), Hydrogen Z=1 (norbs=1) -> 11 orbitals total
    let atomic_numbers = vec![16, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.102],
        [0.0, 0.962, -0.817],
        [0.0, -0.962, -0.817],
    ];
    let pm7 = Pm7Model;
    let batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &pm7);
    assert_eq!(batch.basis_types[0], BasisType::SPD);

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res = run_rhf_scf(&batch, &pm7, &mut ws, 100, 1e-6, 1e-5);
    assert!(scf_res.converged);

    let dims = (32usize, 32usize, 32usize);
    let origin_bohr = [-4.0f64, -4.0f64, -4.0f64];
    let step_bohr = 0.25f64;

    // Warm-up and full-pipeline format sanity check
    let config = CubeGridConfig {
        padding_angstrom: 2.0,
        resolution_angstrom: 0.30,
        n_threads: None,
    };
    let cube_str = gpu_cube
        .generate_density_cube(&batch, &pm7, &ws.density, &config)
        .unwrap();
    assert!(!cube_str.is_empty());

    let start = std::time::Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        let grid = gpu_cube
            .compute_density_grid(&batch, &pm7, &ws.density, origin_bohr, step_bohr, dims)
            .unwrap();
        assert_eq!(grid.len(), dims.0 * dims.1 * dims.2);
    }
    let elapsed = start.elapsed();

    let approx_voxels = dims.0 * dims.1 * dims.2;
    let total_voxels = (approx_voxels * iterations) as f64;
    let throughput_mvox = (total_voxels / elapsed.as_secs_f64()) / 1_000_000.0;

    println!(
        "\n[BENCHMARK] GPU Volumetric Grid (d-orbitals / SPD) Throughput: {:.2} MVox/s ({:.2} ms per 32^3 density grid)",
        throughput_mvox,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64
    );

    let min_expected = if cfg!(debug_assertions) || !ctx.device_info.is_discrete {
        0.05
    } else {
        0.5
    };
    assert!(
        throughput_mvox > min_expected,
        "Throughput must exceed {} MVox/s on Vulkan GPU (achieved: {:.2})",
        min_expected,
        throughput_mvox
    );
}

/// Task GPU-3 / Test 2: Mathematical Parity Between GPU and CPU Volumetric Grid Evaluators for SPD Orbitals.
#[test]
fn test_gpu_cube_d_orbitals_exact_mathematical_parity() {
    use mopac_core::export::cube::ANGSTROM_TO_BOHR;
    use mopac_core::parameters::Pm7Model;
    use mopac_core::scf::scf_loop::run_rhf_scf;
    use mopac_core::types::{BasisType, MolecularBatch, ScfWorkspace};

    let ctx = match VulkanContext::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return, // Skip gracefully if Vulkan runtime not present
    };

    let gpu_cube =
        GpuCubeEvaluator::new(Arc::clone(&ctx)).expect("Failed to initialize GpuCubeEvaluator");

    // H2S: S (Z=16, 9 orbs, SPD), 2x H (Z=1)
    let atomic_numbers = vec![16, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.102],
        [0.0, 0.962, -0.817],
        [0.0, -0.962, -0.817],
    ];
    let pm7 = Pm7Model;
    let batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &pm7);
    assert_eq!(batch.basis_types[0], BasisType::SPD);

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res = run_rhf_scf(&batch, &pm7, &mut ws, 100, 1e-6, 1e-5);
    assert!(scf_res.converged);

    // Evaluate HOMO orbital amplitude on a 15x15x15 grid (3,375 points)
    let dims = (15, 15, 15);
    let origin_bohr = [-2.5, -2.5, -2.5];
    let step_bohr = 0.35f64;

    let homo_idx = batch.norbs / 2 - 1;
    let mo_coeffs = ws.eigenvectors.row(homo_idx);

    let gpu_grid = gpu_cube
        .compute_orbital_grid(&batch, &pm7, mo_coeffs, origin_bohr, step_bohr, dims)
        .expect("GPU orbital grid computation failed");

    // Compute CPU reference grid with d-orbitals active
    let mut max_diff = 0.0f64;
    for ix in 0..dims.0 {
        for iy in 0..dims.1 {
            for iz in 0..dims.2 {
                let idx = ix * dims.1 * dims.2 + iy * dims.2 + iz;
                let pt_angstrom = [
                    (origin_bohr[0] + (ix as f64) * step_bohr) / ANGSTROM_TO_BOHR,
                    (origin_bohr[1] + (iy as f64) * step_bohr) / ANGSTROM_TO_BOHR,
                    (origin_bohr[2] + (iz as f64) * step_bohr) / ANGSTROM_TO_BOHR,
                ];

                let cpu_val = mopac_core::export::cube::evaluate_molecular_orbital_at_point(
                    &batch,
                    &pm7,
                    mo_coeffs,
                    pt_angstrom,
                );

                let gpu_val = gpu_grid[idx];
                let diff = (gpu_val - cpu_val).abs();
                if diff > max_diff {
                    max_diff = diff;
                }

                assert!(
                    diff < 5e-4,
                    "Voxel ({}, {}, {}) mismatch: GPU = {:.6}, CPU = {:.6}, diff = {:e}",
                    ix,
                    iy,
                    iz,
                    gpu_val,
                    cpu_val,
                    diff
                );
            }
        }
    }

    println!(
        "[OK] H2S SPD d-orbital volumetric grid parity verified: max diff = {:e} across 3,375 voxels",
        max_diff
    );
}
