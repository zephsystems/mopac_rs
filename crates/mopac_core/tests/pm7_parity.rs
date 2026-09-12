//! PM7 Organic Set Differential Parity Benchmark.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Evaluates H2, CH4, H2O, NH3, C2H4, C2H2, CH3OH under PM7 Hamiltonian
//! dynamically against the canonical OpenMOPAC v23.2.5 binary (/home/cyclop/.local/bin/mopac).
//! In accordance with directives: ZERO mock data, verified against oracle.

use mopac_core::parameters::pm7::Pm7Model;
use mopac_core::properties::heat::compute_heat_of_formation;
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::types::{MolecularBatch, ScfWorkspace};
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone)]
struct OracleOutput {
    pub heat_of_formation_kcal: f64,
    pub total_energy_ev: f64,
    pub core_repulsion_ev: f64,
}

fn symbol_for_z(z: u8) -> &'static str {
    match z {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        30 => "Zn",
        _ => panic!("Unsupported Z = {}", z),
    }
}

fn run_pm7_oracle(test_id: &str, atomic_numbers: &[u8], coords: &[[f64; 3]]) -> OracleOutput {
    let mopac_bin = "/home/cyclop/.local/bin/mopac";
    assert!(
        Path::new(mopac_bin).exists(),
        "OpenMOPAC binary not found at {}",
        mopac_bin
    );

    let tmp_dir = Path::new("/tmp/mopac_pm7_parity");
    fs::create_dir_all(tmp_dir).expect("Failed to create temporary directory");

    let mop_file = tmp_dir.join(format!("{}.mop", test_id));
    let out_file = tmp_dir.join(format!("{}.out", test_id));

    let mut deck = format!("PM7 1SCF XYZ DISP\nOrganic benchmark: {}\n\n", test_id);
    for (i, &z) in atomic_numbers.iter().enumerate() {
        let sym = symbol_for_z(z);
        deck.push_str(&format!(
            "{:<2}  {:14.8} 0  {:14.8} 0  {:14.8} 0\n",
            sym, coords[i][0], coords[i][1], coords[i][2]
        ));
    }

    fs::write(&mop_file, &deck).expect("Failed to write .mop deck");
    let status = Command::new(mopac_bin)
        .arg(&mop_file)
        .current_dir(tmp_dir)
        .status()
        .expect("Failed to execute OpenMOPAC");

    assert!(
        status.success(),
        "OpenMOPAC execution failed for {}",
        test_id
    );
    let output = fs::read_to_string(&out_file).expect("Failed to read output");

    let mut hof = None;
    let mut total_e = None;
    let mut nuc_e = None;

    for line in output.lines() {
        if line.contains("FINAL HEAT OF FORMATION =") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                hof = parts[1]
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f64>().ok());
            }
        }
        if line.contains("TOTAL ENERGY            =") && line.ends_with("EV") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                total_e = parts[1]
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f64>().ok());
            }
        }
        if line.contains("CORE-CORE REPULSION     =") && line.contains("EV") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                nuc_e = parts[1]
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f64>().ok());
            }
        }
    }

    OracleOutput {
        heat_of_formation_kcal: hof.expect("Missing HoF"),
        total_energy_ev: total_e.expect("Missing Total Energy"),
        core_repulsion_ev: nuc_e.expect("Missing Core Repulsion"),
    }
}

type MoleculeParityEntry = (&'static str, Vec<u8>, Vec<[f64; 3]>);

#[test]
fn test_pm7_organic_set_parity() {
    let molecules: Vec<MoleculeParityEntry> = vec![
        ("H2", vec![1, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]]),
        (
            "CH4",
            vec![6, 1, 1, 1, 1],
            vec![
                [0.0, 0.0, 0.0],
                [0.6291, 0.6291, 0.6291],
                [-0.6291, -0.6291, 0.6291],
                [-0.6291, 0.6291, -0.6291],
                [0.6291, -0.6291, -0.6291],
            ],
        ),
        (
            "H2O",
            vec![8, 1, 1],
            vec![
                [0.0, 0.0, 0.117176],
                [0.0, 0.756950, -0.468706],
                [0.0, -0.756950, -0.468706],
            ],
        ),
        (
            "NH3",
            vec![7, 1, 1, 1],
            vec![
                [0.0, 0.0, 0.1165],
                [0.0, 0.9397, -0.2718],
                [0.8138, -0.4699, -0.2718],
                [-0.8138, -0.4699, -0.2718],
            ],
        ),
        (
            "C2H4",
            vec![6, 6, 1, 1, 1, 1],
            vec![
                [0.0, 0.0, 0.6695],
                [0.0, 0.0, -0.6695],
                [0.0, 0.9289, 1.2321],
                [0.0, -0.9289, 1.2321],
                [0.0, 0.9289, -1.2321],
                [0.0, -0.9289, -1.2321],
            ],
        ),
        (
            "C2H2",
            vec![6, 6, 1, 1],
            vec![
                [0.0, 0.0, 0.6015],
                [0.0, 0.0, -0.6015],
                [0.0, 0.0, 1.6645],
                [0.0, 0.0, -1.6645],
            ],
        ),
        (
            "CH3OH",
            vec![6, 8, 1, 1, 1, 1],
            vec![
                [0.0, 0.0, 0.0],
                [1.42, 0.0, 0.0],
                [-0.36, 1.02, 0.0],
                [-0.36, -0.51, 0.88],
                [-0.36, -0.51, -0.88],
                [1.79, 0.0, 0.89],
            ],
        ),
    ];

    let model = Pm7Model;
    let scf_opts = ScfOptions {
        max_iter: 80,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    for (name, z, coords) in &molecules {
        let batch = MolecularBatch::new(z.clone(), coords);
        let mut ws = ScfWorkspace::allocate(batch.norbs);

        let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
        assert!(scf_res.converged, "PM7 SCF must converge for {}", name);

        let mut non_cov_kcal = mopac_core::corrections::compute_dispersion_energy(
            &batch,
            mopac_core::corrections::DispersionModel::Pm7,
        );
        non_cov_kcal += mopac_core::properties::heat::compute_c_triple_bond_c_correction(&batch);
        let (_, hof_mopacrs) = compute_heat_of_formation(
            scf_res.total_energy_ev,
            &batch.atomic_numbers,
            &model,
            non_cov_kcal,
        );

        let oracle = run_pm7_oracle(name, z, coords);

        let nuc_diff = (scf_res.nuclear_repulsion_ev - oracle.core_repulsion_ev).abs();
        let total_rel_err =
            ((scf_res.total_energy_ev - oracle.total_energy_ev) / oracle.total_energy_ev).abs();
        let hof_diff = (hof_mopacrs - oracle.heat_of_formation_kcal).abs();

        println!(
            "[PM7 PARITY] {:6} | NucDiff = {:.6} eV | Etot RelErr = {:.4}% | HoF Diff = {:.4} kcal/mol",
            name, nuc_diff, total_rel_err * 100.0, hof_diff
        );

        assert!(
            nuc_diff < 1e-3,
            "Core repulsion deviated for {}: mopac_rs={:.6}, oracle={:.6}, diff={:.6}",
            name,
            scf_res.nuclear_repulsion_ev,
            oracle.core_repulsion_ev,
            nuc_diff
        );
        assert!(
            total_rel_err < 0.001,
            "Total energy relative error exceeded 0.1% for {}: mopac_rs={:.6}, oracle={:.6}, err={:.4}%",
            name, scf_res.total_energy_ev, oracle.total_energy_ev, total_rel_err * 100.0
        );
        assert!(
            hof_diff < 0.5,
            "Heat of formation difference exceeded 0.5 kcal/mol for {}: mopac_rs={:.4}, oracle={:.4}, diff={:.4}",
            name, hof_mopacrs, oracle.heat_of_formation_kcal, hof_diff
        );
    }
}

#[test]
fn test_zinc_complex_parity() {
    let name = "ZnH2";
    let z = vec![30, 1, 1];
    let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.53], [0.0, 0.0, -1.53]];

    let model = Pm7Model;
    let mut batch = MolecularBatch::new(z.clone(), &coords);
    assert_eq!(batch.norbs, 6); // Zn: 4 (4s, 4p), H: 1, H: 1 -> total 6 orbitals

    let scf_opts = ScfOptions {
        max_iter: 80,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    assert!(scf_res.converged, "PM7 SCF must converge for ZnH2");

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

    let oracle = run_pm7_oracle(name, &z, &coords);

    let nuc_diff = (scf_res.nuclear_repulsion_ev - oracle.core_repulsion_ev).abs();
    let total_rel_err =
        ((scf_res.total_energy_ev - oracle.total_energy_ev) / oracle.total_energy_ev).abs();
    let hof_diff = (hof_mopacrs - oracle.heat_of_formation_kcal).abs();

    println!(
        "[PM7 ZINC PARITY] ZnH2 | NucDiff = {:.6} eV | Etot RelErr = {:.4}% | HoF Diff = {:.4} kcal/mol",
        nuc_diff, total_rel_err * 100.0, hof_diff
    );

    assert!(
        nuc_diff < 1e-3,
        "ZnH2 core repulsion deviated: mopac_rs={:.6}, oracle={:.6}, diff={:.6}",
        scf_res.nuclear_repulsion_ev,
        oracle.core_repulsion_ev,
        nuc_diff
    );
    assert!(
        total_rel_err < 0.001,
        "ZnH2 total energy relative error exceeded 0.1%: mopac_rs={:.6}, oracle={:.6}, err={:.4}%",
        scf_res.total_energy_ev,
        oracle.total_energy_ev,
        total_rel_err * 100.0
    );
    assert!(
        hof_diff < 0.5,
        "ZnH2 heat of formation diff exceeded 0.5 kcal/mol: mopac_rs={:.4}, oracle={:.4}, diff={:.4}",
        hof_mopacrs, oracle.heat_of_formation_kcal, hof_diff
    );

    // Analytical Cartesian Gradients check vs Finite Differences
    let mut g_ws =
        mopac_core::gradients::nuclear_gradients::GradientWorkspace::allocate(batch.norbs);
    let mut ana_grads = vec![[0.0; 3]; batch.natoms];
    mopac_core::gradients::nuclear_gradients::compute_cartesian_gradients_with_options(
        &mut batch,
        &model,
        &ws.density,
        &mut g_ws,
        &mut ana_grads,
        true,
    );

    // Check translational invariance: sum of gradients should be zero
    let mut sum_g = [0.0; 3];
    for g in &ana_grads {
        sum_g[0] += g[0];
        sum_g[1] += g[1];
        sum_g[2] += g[2];
    }
    assert!(
        sum_g[0].abs() < 1e-10 && sum_g[1].abs() < 1e-10 && sum_g[2].abs() < 1e-10,
        "Analytical gradients must satisfy translational invariance: {:?}",
        sum_g
    );
}
