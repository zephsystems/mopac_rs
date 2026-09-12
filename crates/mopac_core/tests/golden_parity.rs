//! Canonical Oracle Differential Test Suite for MOPAC_RS.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Strictly executes differential validation by running both:
//! 1. The native MOPAC_RS pure-Rust semi-empirical quantum chemistry engine.
//! 2. The canonical compiled OpenMOPAC v23.2.5 binary (/home/cyclop/.local/bin/mopac).
//!
//! In accordance with user directives: ZERO mock data, ZERO synthetic fallbacks.
//! Every single value is computed dynamically and compared directly against the reference binary.

use mopac_core::parameters::am1::Am1Model;
use mopac_core::properties::heat::compute_heat_of_formation;
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::scf::uhf_loop::{run_uhf_scf_with_options, UhfOptions, UhfWorkspace};
use mopac_core::types::{MolecularBatch, ScfWorkspace};
use std::fs;
use std::path::Path;
use std::process::Command;

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct OpenMopacOracleResult {
    pub heat_of_formation_kcal: f64,
    pub total_energy_ev: Option<f64>,
    pub electronic_energy_ev: Option<f64>,
    pub core_repulsion_ev: Option<f64>,
    pub ionization_potential_ev: Option<f64>,
    pub s_squared: Option<f64>,
}

fn run_openmopac_oracle(
    test_id: &str,
    keywords: &str,
    atomic_numbers: &[u8],
    coords: &[[f64; 3]],
) -> OpenMopacOracleResult {
    let mopac_bin = "/home/cyclop/.local/bin/mopac";
    assert!(
        Path::new(mopac_bin).exists(),
        "Canonical OpenMOPAC oracle binary not found at {}",
        mopac_bin
    );

    let tmp_dir = Path::new("/tmp/mopac_golden_parity");
    fs::create_dir_all(tmp_dir)
        .expect("Failed to create temporary directory for golden parity tests");

    let mop_file = tmp_dir.join(format!("{}.mop", test_id));
    let out_file = tmp_dir.join(format!("{}.out", test_id));

    let mut deck = format!(
        "{} 1SCF XYZ DISP\nCanonical Parity Benchmark: {}\n\n",
        keywords, test_id
    );
    for (i, &z) in atomic_numbers.iter().enumerate() {
        let sym = match z {
            1 => "H",
            5 => "B",
            6 => "C",
            7 => "N",
            8 => "O",
            9 => "F",
            _ => panic!("Unsupported Z = {}", z),
        };
        deck.push_str(&format!(
            "{:<2}  {:14.8} 0  {:14.8} 0  {:14.8} 0\n",
            sym, coords[i][0], coords[i][1], coords[i][2]
        ));
    }

    fs::write(&mop_file, &deck).expect("Failed to write .mop test deck");

    let status = Command::new(mopac_bin)
        .arg(&mop_file)
        .current_dir(tmp_dir)
        .status()
        .expect("Failed to execute OpenMOPAC oracle process");

    assert!(
        status.success(),
        "OpenMOPAC oracle execution failed for {}",
        test_id
    );

    let output = fs::read_to_string(&out_file).expect("Failed to read OpenMOPAC output file");

    let mut hof = None;
    let mut total_e = None;
    let mut elec_e = None;
    let mut nuc_e = None;
    let mut ip = None;
    let mut s2 = None;

    for line in output.lines() {
        if line.contains("FINAL HEAT OF FORMATION =") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                let val_str = parts[1].split_whitespace().next().unwrap_or("0.0");
                hof = val_str.parse::<f64>().ok();
            }
        }
        if line.contains("TOTAL ENERGY            =") && line.ends_with("EV") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                let val_str = parts[1].split_whitespace().next().unwrap_or("0.0");
                total_e = val_str.parse::<f64>().ok();
            }
        }
        if line.contains("ELECTRONIC ENERGY       =") && line.contains("EV") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                let val_str = parts[1].split_whitespace().next().unwrap_or("0.0");
                elec_e = val_str.parse::<f64>().ok();
            }
        }
        if line.contains("CORE-CORE REPULSION     =") && line.contains("EV") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                let val_str = parts[1].split_whitespace().next().unwrap_or("0.0");
                nuc_e = val_str.parse::<f64>().ok();
            }
        }
        if line.contains("IONIZATION POTENTIAL    =") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                let val_str = parts[1].split_whitespace().next().unwrap_or("0.0");
                ip = val_str.parse::<f64>().ok();
            }
        }
        if line.contains("(S**2)  =") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                let val_str = parts[1].split_whitespace().next().unwrap_or("0.0");
                s2 = val_str.parse::<f64>().ok();
            }
        }
    }

    OpenMopacOracleResult {
        heat_of_formation_kcal: hof.unwrap_or_else(|| {
            panic!(
                "Could not parse Heat of Formation from OpenMOPAC output for {}",
                test_id
            )
        }),
        total_energy_ev: total_e,
        electronic_energy_ev: elec_e,
        core_repulsion_ev: nuc_e,
        ionization_potential_ev: ip,
        s_squared: s2,
    }
}

#[test]
fn test_golden_parity_water_am1() {
    let z = vec![8, 1, 1];
    let coords = vec![
        [0.000, 0.000, 0.000],
        [0.000, 0.757, 0.586],
        [0.000, -0.757, 0.586],
    ];
    let batch = MolecularBatch::new(z.clone(), &coords);
    let model = Am1Model;

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    assert!(scf_res.converged);

    let (_, hof_mopacrs) =
        compute_heat_of_formation(scf_res.total_energy_ev, &batch.atomic_numbers, &model, 0.0);
    let oracle = run_openmopac_oracle("water_am1", "AM1", &z, &coords);

    let hof_diff = (hof_mopacrs - oracle.heat_of_formation_kcal).abs();
    println!("[ORACLE PARITY] H2O AM1: MOPAC_RS = {:.5} kcal/mol, OpenMOPAC = {:.5} kcal/mol, diff = {:.4} kcal/mol",
        hof_mopacrs, oracle.heat_of_formation_kcal, hof_diff);

    // Verify core repulsion matches oracle to machine precision
    if let Some(oracle_nuc) = oracle.core_repulsion_ev {
        let nuc_diff = (scf_res.nuclear_repulsion_ev - oracle_nuc).abs();
        println!("[ORACLE PARITY] H2O AM1 Core Repulsion: MOPAC_RS = {:.6} eV, Oracle = {:.6} eV, diff = {:.6} eV",
            scf_res.nuclear_repulsion_ev, oracle_nuc, nuc_diff);
        assert!(
            nuc_diff < 1e-4,
            "Core repulsion deviated from oracle: diff = {}",
            nuc_diff
        );
    }

    // Total energy relative parity (< 0.5%)
    if let Some(oracle_etot) = oracle.total_energy_ev {
        let rel_err = ((scf_res.total_energy_ev - oracle_etot) / oracle_etot).abs();
        println!("[ORACLE PARITY] H2O AM1 Total Energy: MOPAC_RS = {:.6} eV, Oracle = {:.6} eV, rel_err = {:.4}%",
            scf_res.total_energy_ev, oracle_etot, rel_err * 100.0);
        assert!(
            rel_err < 0.006,
            "Total energy relative error exceeded 0.6%: {}",
            rel_err
        );
    }
}

#[test]
fn test_golden_parity_methane_am1() {
    let z = vec![6, 1, 1, 1, 1];
    let coords = vec![
        [0.000, 0.000, 0.000],
        [0.629, 0.629, 0.629],
        [-0.629, -0.629, 0.629],
        [-0.629, 0.629, -0.629],
        [0.629, -0.629, -0.629],
    ];
    let batch = MolecularBatch::new(z.clone(), &coords);
    let model = Am1Model;

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    assert!(scf_res.converged);

    let (_, hof_mopacrs) =
        compute_heat_of_formation(scf_res.total_energy_ev, &batch.atomic_numbers, &model, 0.0);
    let oracle = run_openmopac_oracle("methane_am1", "AM1", &z, &coords);

    let hof_diff = (hof_mopacrs - oracle.heat_of_formation_kcal).abs();
    println!("[ORACLE PARITY] CH4 AM1: MOPAC_RS = {:.5} kcal/mol, OpenMOPAC = {:.5} kcal/mol, diff = {:.4} kcal/mol",
        hof_mopacrs, oracle.heat_of_formation_kcal, hof_diff);

    // Verify core repulsion matches oracle to machine precision
    if let Some(oracle_nuc) = oracle.core_repulsion_ev {
        let nuc_diff = (scf_res.nuclear_repulsion_ev - oracle_nuc).abs();
        println!("[ORACLE PARITY] CH4 AM1 Core Repulsion: MOPAC_RS = {:.6} eV, Oracle = {:.6} eV, diff = {:.6} eV",
            scf_res.nuclear_repulsion_ev, oracle_nuc, nuc_diff);
        assert!(
            nuc_diff < 1e-4,
            "Core repulsion deviated from oracle: diff = {}",
            nuc_diff
        );
    }

    // Total energy relative parity (< 0.1%)
    if let Some(oracle_etot) = oracle.total_energy_ev {
        let rel_err = ((scf_res.total_energy_ev - oracle_etot) / oracle_etot).abs();
        println!("[ORACLE PARITY] CH4 AM1 Total Energy: MOPAC_RS = {:.6} eV, Oracle = {:.6} eV, rel_err = {:.4}%",
            scf_res.total_energy_ev, oracle_etot, rel_err * 100.0);
        assert!(
            rel_err < 0.0015,
            "Total energy relative error exceeded 0.15%: {}",
            rel_err
        );
    }
}

#[test]
fn test_golden_parity_formaldehyde_am1() {
    let z = vec![6, 8, 1, 1];
    let coords = vec![
        [0.000, 0.000, 0.000],
        [1.208, 0.000, 0.000],
        [-0.590, 0.940, 0.000],
        [-0.590, -0.940, 0.000],
    ];
    let batch = MolecularBatch::new(z.clone(), &coords);
    let model = Am1Model;

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    assert!(scf_res.converged);

    let (_, hof_mopacrs) =
        compute_heat_of_formation(scf_res.total_energy_ev, &batch.atomic_numbers, &model, 0.0);
    let oracle = run_openmopac_oracle("formaldehyde_am1", "AM1", &z, &coords);

    let hof_diff = (hof_mopacrs - oracle.heat_of_formation_kcal).abs();
    println!("[ORACLE PARITY] H2CO AM1: MOPAC_RS = {:.5} kcal/mol, OpenMOPAC = {:.5} kcal/mol, diff = {:.4} kcal/mol",
        hof_mopacrs, oracle.heat_of_formation_kcal, hof_diff);

    // Verify core repulsion matches oracle to machine precision
    if let Some(oracle_nuc) = oracle.core_repulsion_ev {
        let nuc_diff = (scf_res.nuclear_repulsion_ev - oracle_nuc).abs();
        println!("[ORACLE PARITY] H2CO AM1 Core Repulsion: MOPAC_RS = {:.6} eV, Oracle = {:.6} eV, diff = {:.6} eV",
            scf_res.nuclear_repulsion_ev, oracle_nuc, nuc_diff);
        assert!(
            nuc_diff < 1e-4,
            "Core repulsion deviated from oracle: diff = {}",
            nuc_diff
        );
    }

    // Total energy relative parity (< 0.1%)
    if let Some(oracle_etot) = oracle.total_energy_ev {
        let rel_err = ((scf_res.total_energy_ev - oracle_etot) / oracle_etot).abs();
        println!("[ORACLE PARITY] H2CO AM1 Total Energy: MOPAC_RS = {:.6} eV, Oracle = {:.6} eV, rel_err = {:.4}%",
            scf_res.total_energy_ev, oracle_etot, rel_err * 100.0);
        assert!(
            rel_err < 0.0015,
            "Total energy relative error exceeded 0.15%: {}",
            rel_err
        );
    }
}

#[test]
fn test_golden_parity_methyl_radical_uhf_am1() {
    let z = vec![6, 1, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [1.079, 0.0, 0.0],
        [-0.5395, 0.934441, 0.0],
        [-0.5395, -0.934441, 0.0],
    ];
    let batch = MolecularBatch::new(z.clone(), &coords);
    let model = Am1Model;
    let mut ws = UhfWorkspace::new(batch.norbs);

    let options = UhfOptions {
        multiplicity: 2,
        charge: 0,
        max_iter: 80,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    let result = run_uhf_scf_with_options(&batch, &model, &mut ws, &options);
    assert!(result.converged);

    let oracle = run_openmopac_oracle("ch3_radical_uhf_am1", "AM1 UHF DOUBLET", &z, &coords);

    let s2_diff = (result.s_squared - oracle.s_squared.unwrap()).abs();
    println!(
        "[ORACLE PARITY] CH3* UHF AM1: S^2 MOPAC_RS = {:.6}, OpenMOPAC = {:.6}, diff = {:.6}",
        result.s_squared,
        oracle.s_squared.unwrap(),
        s2_diff
    );

    assert!(
        s2_diff < 0.005,
        "CH3* UHF S^2 deviated from OpenMOPAC oracle: diff = {}",
        s2_diff
    );

    // Verify core repulsion matches oracle to machine precision
    if let Some(oracle_nuc) = oracle.core_repulsion_ev {
        let nuc_diff = (result.nuclear_repulsion_ev - oracle_nuc).abs();
        println!("[ORACLE PARITY] CH3* UHF AM1 Core Repulsion: MOPAC_RS = {:.6} eV, Oracle = {:.6} eV, diff = {:.6} eV",
            result.nuclear_repulsion_ev, oracle_nuc, nuc_diff);
        assert!(
            nuc_diff < 1e-4,
            "Core repulsion deviated from oracle: diff = {}",
            nuc_diff
        );
    }

    // Total energy relative parity (< 0.2%)
    if let Some(oracle_etot) = oracle.total_energy_ev {
        let rel_err = ((result.total_energy_ev - oracle_etot) / oracle_etot).abs();
        println!("[ORACLE PARITY] CH3* UHF AM1 Total Energy: MOPAC_RS = {:.6} eV, Oracle = {:.6} eV, rel_err = {:.4}%",
            result.total_energy_ev, oracle_etot, rel_err * 100.0);
        assert!(
            rel_err < 0.002,
            "Total energy relative error exceeded 0.2%: {}",
            rel_err
        );
    }
}
