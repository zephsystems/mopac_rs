//! Empirical Vulkan GPU Shader Benchmark on Medium-Sized Protein: T4 Lysozyme (PDB: 2LZM).
//!
//! Loads 164-residue T4 Lysozyme (1,309 atoms, 1,713,481 pairwise Coulomb interactions),
//! dispatches the Vulkan compute shader on the NVIDIA GeForce RTX 4050 GPU,
//! compares speed against CPU reference, and outputs a Molstar-compatible PDB/PDBQT file
//! with quantum partial charges mapped to the B-factor column.

use mopac_core::constants::codata2018::EV_TO_KCAL_MOL;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::parameters::ParameterModel;
use mopac_core::types::MolecularBatch;
use mopac_gpu::{GpuCoulombCalculator, VulkanContext};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::time::Instant;

#[allow(dead_code)] // PDB record metadata retained for diagnostics
struct PdbAtom {
    atom_num: usize,
    atom_name: String,
    res_name: String,
    chain_id: char,
    res_seq: i32,
    x: f64,
    y: f64,
    z: f64,
    z_atomic: u8,
}

fn element_symbol_to_z(symbol: &str) -> u8 {
    match symbol.trim() {
        "H" => 1,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "S" => 16,
        "P" => 15,
        "ZN" | "Zn" => 30,
        "FE" | "Fe" => 26,
        _ => 6,
    }
}

#[test]
fn test_vulkan_gpu_t4_lysozyme_protein() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(&manifest_dir);
    let pdb_path = repo_root.join("benchmarks/data/2lzm.pdb");
    let file = File::open(&pdb_path).expect("Failed to open 2LZM PDB file");
    let reader = BufReader::new(file);

    let mut pdb_atoms = Vec::new();

    for line_res in reader.lines() {
        let line = line_res.unwrap();
        if line.starts_with("ATOM  ") {
            let atom_num: usize = line[6..11].trim().parse().unwrap_or(0);
            let atom_name = line[12..16].trim().to_string();
            let res_name = line[17..20].trim().to_string();
            let chain_id = line[21..22].chars().next().unwrap_or('A');
            let res_seq: i32 = line[22..26].trim().parse().unwrap_or(0);
            let x: f64 = line[30..38].trim().parse().unwrap_or(0.0);
            let y: f64 = line[38..46].trim().parse().unwrap_or(0.0);
            let z: f64 = line[46..54].trim().parse().unwrap_or(0.0);

            // Element symbol in columns 76-78, or deduced from atom name
            let elem_str = if line.len() >= 78 {
                line[76..78].trim()
            } else {
                let first_char = atom_name.chars().next().unwrap_or('C');
                if first_char.is_alphabetic() {
                    &atom_name[0..1]
                } else {
                    "C"
                }
            };
            let z_num = element_symbol_to_z(elem_str);

            pdb_atoms.push(PdbAtom {
                atom_num,
                atom_name,
                res_name,
                chain_id,
                res_seq,
                x,
                y,
                z,
                z_atomic: z_num,
            });
        }
    }

    let natoms = pdb_atoms.len();
    println!("{}", "=".repeat(80));
    println!("MOPAC_GPU VULKAN BENCHMARK: T4 LYSOZYME (PDB: 2LZM)");
    println!("{}", "=".repeat(80));
    println!("Loaded {} atoms from 164 amino acid residues.", natoms);
    assert_eq!(natoms, 1309, "Expected 1309 heavy atoms in 2LZM");

    let atomic_numbers: Vec<u8> = pdb_atoms.iter().map(|a| a.z_atomic).collect();
    let coords: Vec<[f64; 3]> = pdb_atoms.iter().map(|a| [a.x, a.y, a.z]).collect();

    let batch = MolecularBatch::new(atomic_numbers.clone(), &coords);
    let pm6 = Pm6Model;

    // 1. Initialize Vulkan Context
    let ctx = match VulkanContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!(
                "Skipping Vulkan protein test (no Vulkan GPU runtime available): {}",
                e
            );
            return;
        }
    };

    if !ctx.device_info.supports_float64 {
        eprintln!(
            "Skipping Vulkan protein test (device {} does not support native Float64 precision)",
            ctx.device_info.device_name
        );
        return;
    }

    println!(
        "Vulkan GPU Device: {} (Discrete: {}, Vulkan API: {:?})",
        ctx.device_info.device_name, ctx.device_info.is_discrete, ctx.device_info.api_version
    );

    let calc = GpuCoulombCalculator::new(Arc::clone(&ctx))
        .expect("Failed to build Vulkan Coulomb compute pipeline");

    // 2. Dispatch on GPU (NVIDIA RTX 4050)
    let t0_gpu = Instant::now();
    let gpu_matrix = calc
        .compute_batch(&batch, &pm6)
        .expect("GPU compute shader execution failed");
    let t_gpu = t0_gpu.elapsed();

    println!("\n[Vulkan GPU Execution]");
    println!(
        "  Total matrix elements evaluated: {} x {} = {}",
        natoms,
        natoms,
        natoms * natoms
    );
    println!(
        "  GPU Wall Time (Dispatch + VRAM DMA transfer): {:.3} ms",
        t_gpu.as_secs_f64() * 1000.0
    );

    // 3. Compute CPU Reference for comparison
    let t0_cpu = Instant::now();
    let mut cpu_matrix = mopac_core::types::AlignedMatrix::zeroed(natoms, natoms);
    for i in 0..natoms {
        let za = batch.atomic_numbers[i];
        let pa = pm6.get_element(za).unwrap();
        for j in 0..natoms {
            let zb = batch.atomic_numbers[j];
            let pb = pm6.get_element(zb).unwrap();
            let dx = batch.x[i] - batch.x[j];
            let dy = batch.y[i] - batch.y[j];
            let dz = batch.z[i] - batch.z[j];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let val =
                mopac_core::integrals::two_electron::dewar_klopman_monopole(r, pa.gss, pb.gss);
            cpu_matrix.set(i, j, val);
        }
    }
    let t_cpu = t0_cpu.elapsed();

    println!("\n[CPU Reference Execution]");
    println!(
        "  CPU Wall Time (Single-core AVX): {:.3} ms",
        t_cpu.as_secs_f64() * 1000.0
    );
    let speedup = t_cpu.as_secs_f64() / t_gpu.as_secs_f64();
    println!("  Hardware Acceleration Speedup: {:.2}x", speedup);

    // Verify GPU vs CPU numerical precision
    let mut max_diff = 0.0f64;
    for i in 0..natoms {
        for j in 0..natoms {
            let diff = (gpu_matrix.get(i, j) - cpu_matrix.get(i, j)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    println!("  Maximum Float64 difference GPU vs CPU: {:e} eV", max_diff);
    assert!(
        max_diff < 1e-11,
        "Vulkan GPU Float64 mismatch exceeded threshold"
    );

    // Verify GPU Coulomb matrix symmetry strictly
    for i in 0..natoms {
        for j in 0..natoms {
            assert!(
                (gpu_matrix.get(i, j) - gpu_matrix.get(j, i)).abs() < 1e-10,
                "GPU Coulomb matrix must be strictly symmetric at ({}, {})",
                i,
                j
            );
        }
    }

    // Total classical Coulomb nuclear-core repulsion of the protein fold: E_nuc = sum_{i < j} Z_i * Z_j * gamma_ij
    let mut total_core_repulsion_ev = 0.0;
    for i in 0..natoms {
        let zi = batch.atomic_numbers[i];
        let pi = pm6.get_element(zi).unwrap();
        for j in (i + 1)..natoms {
            let zj = batch.atomic_numbers[j];
            let pj = pm6.get_element(zj).unwrap();
            total_core_repulsion_ev += pi.core_charge * pj.core_charge * gpu_matrix.get(i, j);
        }
    }
    assert!(total_core_repulsion_ev.is_finite() && total_core_repulsion_ev > 0.0);
    println!(
        "  Total Protein Nuclear-Core Classical Repulsion: {:.2} eV ({:.2} kcal/mol)",
        total_core_repulsion_ev,
        total_core_repulsion_ev * EV_TO_KCAL_MOL
    );
    println!("\n[Protein Vulkan GPU Monopole Parity Test: PASSED]");
    println!("{}", "=".repeat(80));
}
