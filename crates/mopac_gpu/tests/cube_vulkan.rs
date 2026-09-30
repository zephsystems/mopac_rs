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
