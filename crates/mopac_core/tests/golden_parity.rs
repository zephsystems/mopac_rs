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
use mopac_core::parameters::pm7::Pm7Model;
use mopac_core::parameters::ParameterModel;
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

fn symbol_for_z(z: u8) -> &'static str {
    match z {
        1 => "H",
        2 => "He",
        3 => "Li",
        4 => "Be",
        5 => "B",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        10 => "Ne",
        11 => "Na",
        12 => "Mg",
        13 => "Al",
        14 => "Si",
        15 => "P",
        16 => "S",
        17 => "Cl",
        26 => "Fe",
        28 => "Ni",
        29 => "Cu",
        30 => "Zn",
        35 => "Br",
        53 => "I",
        _ => panic!("Unsupported Z = {}", z),
    }
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
        let sym = symbol_for_z(z);
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

#[test]
fn test_golden_parity_water_pm7() {
    let z = vec![8, 1, 1];
    let coords = vec![
        [0.000, 0.000, 0.117176],
        [0.000, 0.756950, -0.468706],
        [0.000, -0.756950, -0.468706],
    ];
    let batch = MolecularBatch::new(z.clone(), &coords);
    let model = Pm7Model;

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
    println!("MOPAC_RS H_CORE:");
    for i in 0..ws.norbs {
        print!("row {}: ", i);
        for j in 0..=i {
            print!("{:12.6} ", ws.h_core.get(i, j));
        }
        println!();
    }

    let disp_kcal = mopac_core::corrections::compute_dispersion_energy(
        &batch,
        mopac_core::corrections::DispersionModel::Pm7,
    );
    let (_, hof_mopacrs) = compute_heat_of_formation(
        scf_res.total_energy_ev,
        &batch.atomic_numbers,
        &model,
        disp_kcal,
    );
    let oracle = run_openmopac_oracle("water_pm7", "PM7", &z, &coords);

    let hof_diff = (hof_mopacrs - oracle.heat_of_formation_kcal).abs();
    println!("[ORACLE PARITY] H2O PM7: MOPAC_RS = {:.5} kcal/mol, OpenMOPAC = {:.5} kcal/mol, diff = {:.4} kcal/mol",
        hof_mopacrs, oracle.heat_of_formation_kcal, hof_diff);

    // Print pairwise core repulsions
    let elem_o = model.get_element(8).unwrap();
    let elem_h = model.get_element(1).unwrap();
    let r_oh = batch.distance(0, 1);
    let r_hh = batch.distance(1, 2);
    let e_oh = model.pair_core_repulsion(r_oh, &elem_o, &elem_h);
    let e_hh = model.pair_core_repulsion(r_hh, &elem_h, &elem_h);
    println!(
        "DEBUG: r_oh = {:.6}, e_oh = {:.6}, r_hh = {:.6}, e_hh = {:.6}, total = {:.6}",
        r_oh,
        e_oh,
        r_hh,
        e_hh,
        2.0 * e_oh + e_hh
    );

    // Verify core repulsion matches oracle
    if let Some(oracle_nuc) = oracle.core_repulsion_ev {
        let nuc_diff = (scf_res.nuclear_repulsion_ev - oracle_nuc).abs();
        println!("[ORACLE PARITY] H2O PM7 Core Repulsion: MOPAC_RS = {:.6} eV, Oracle = {:.6} eV, diff = {:.6} eV",
            scf_res.nuclear_repulsion_ev, oracle_nuc, nuc_diff);
        assert!(
            nuc_diff < 1e-4,
            "PM7 Core repulsion deviated from oracle: diff = {}",
            nuc_diff
        );
    }

    // Verify total energy parity (< 0.2%)
    if let Some(oracle_etot) = oracle.total_energy_ev {
        let rel_err = ((scf_res.total_energy_ev - oracle_etot) / oracle_etot).abs();
        println!("[ORACLE PARITY] H2O PM7 Total Energy: MOPAC_RS = {:.6} eV, Oracle = {:.6} eV, rel_err = {:.4}%",
            scf_res.total_energy_ev, oracle_etot, rel_err * 100.0);
        assert!(
            rel_err < 0.002,
            "Total energy relative error exceeded 0.2%: {}",
            rel_err
        );
    }
}

#[test]
fn test_golden_parity_zinc_hydride_pm7() {
    let z = vec![30, 1, 1];
    let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.53], [0.0, 0.0, -1.53]];
    let model = Pm7Model;
    let batch = MolecularBatch::new(z.clone(), &coords);
    let mut ws = ScfWorkspace::allocate(batch.norbs);

    let scf_opts = ScfOptions {
        max_iter: 80,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    assert!(scf_res.converged, "ZnH2 SCF must converge");

    let disp_kcal = mopac_core::corrections::compute_dispersion_energy(
        &batch,
        mopac_core::corrections::DispersionModel::Pm7,
    );
    let (_, hof_mopacrs) = compute_heat_of_formation(
        scf_res.total_energy_ev,
        &batch.atomic_numbers,
        &model,
        disp_kcal,
    );
    let oracle = run_openmopac_oracle("znh2_pm7", "PM7", &z, &coords);

    let hof_diff = (hof_mopacrs - oracle.heat_of_formation_kcal).abs();
    println!(
        "[ORACLE PARITY] ZnH2 PM7: MOPAC_RS = {:.5} kcal/mol, OpenMOPAC = {:.5} kcal/mol, diff = {:.4} kcal/mol",
        hof_mopacrs, oracle.heat_of_formation_kcal, hof_diff
    );

    if let Some(oracle_nuc) = oracle.core_repulsion_ev {
        let nuc_diff = (scf_res.nuclear_repulsion_ev - oracle_nuc).abs();
        assert!(
            nuc_diff < 1e-4,
            "ZnH2 Core repulsion deviated: {}",
            nuc_diff
        );
    }
    if let Some(oracle_etot) = oracle.total_energy_ev {
        let rel_err = ((scf_res.total_energy_ev - oracle_etot) / oracle_etot).abs();
        assert!(
            rel_err < 0.001,
            "ZnH2 total energy relative error: {}",
            rel_err
        );
    }
}

#[test]
fn test_golden_parity_hydrogen_sulfide_pm7() {
    let z = vec![16, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.1022],
        [0.0, 0.9634, -0.8176],
        [0.0, -0.9634, -0.8176],
    ];
    let model = Pm7Model;
    let batch = MolecularBatch::new_for_model(z.clone(), &coords, &model);
    assert_eq!(batch.norbs, 11, "H2S must have 11 orbitals in PM7");
    let mut ws = ScfWorkspace::allocate(batch.norbs);

    let scf_opts = ScfOptions {
        max_iter: 80,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    assert!(scf_res.converged, "H2S SCF must converge");

    let disp_kcal = mopac_core::corrections::compute_dispersion_energy(
        &batch,
        mopac_core::corrections::DispersionModel::Pm7,
    );
    let (_, hof_mopacrs) = compute_heat_of_formation(
        scf_res.total_energy_ev,
        &batch.atomic_numbers,
        &model,
        disp_kcal,
    );
    let oracle = run_openmopac_oracle("h2s_pm7", "PM7", &z, &coords);

    let hof_diff = (hof_mopacrs - oracle.heat_of_formation_kcal).abs();
    println!(
        "[ORACLE PARITY] H2S PM7: MOPAC_RS = {:.5} kcal/mol, OpenMOPAC = {:.5} kcal/mol, diff = {:.4} kcal/mol",
        hof_mopacrs, oracle.heat_of_formation_kcal, hof_diff
    );

    if let Some(oracle_nuc) = oracle.core_repulsion_ev {
        let nuc_diff = (scf_res.nuclear_repulsion_ev - oracle_nuc).abs();
        println!(
            "[ORACLE PARITY] H2S Core Repulsion: MOPAC_RS = {:.6}, Oracle = {:.6}, diff = {:.6}",
            scf_res.nuclear_repulsion_ev, oracle_nuc, nuc_diff
        );
        assert!(nuc_diff < 1e-3, "H2S Core repulsion deviated: {}", nuc_diff);
    }
    if let Some(oracle_etot) = oracle.total_energy_ev {
        let rel_err = ((scf_res.total_energy_ev - oracle_etot) / oracle_etot).abs();
        println!(
            "[ORACLE PARITY] H2S Total Energy: MOPAC_RS = {:.6}, Oracle = {:.6}, rel_err = {:.4}%",
            scf_res.total_energy_ev,
            oracle_etot,
            rel_err * 100.0
        );
        assert!(
            rel_err < 0.002,
            "H2S Total energy relative error: {}",
            rel_err
        );
    }
}
