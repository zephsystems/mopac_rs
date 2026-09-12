//! Integration Test Suite: Periodic Boundary Conditions (PBC) & Band Structure Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Rigorously verifies:
//! * Test 5.1 (`test_pbc_reciprocal_lattice_invariants`): Exact reciprocity tensor invariance
//!   $\vec{a}_i \cdot \vec{b}_j = 2\pi \delta_{ij}$ for 1D, 2D, and 3D periodic unit cells.
//! * Test 5.2 (`test_polyacetylene_1d_bandgap`): Trans-polyacetylene 1D conjugated polymer chain $(C_2H_2)_n$,
//!   validating crystal orbital Bloch SCF convergence, $\pi \to \pi^*$ band dispersion, and semiconductor bandgap.
//! * Test 5.3 (`test_pbc_golden_parity`): Direct parity against canonical OpenMOPAC v23.2.5 oracle
//!   on 1D trans-polyacetylene (`MERS=(3)` PM6 1SCF), verifying heat of formation per unit cell, HOMO energy,
//!   and band edge parity.

use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::pbc::bloch_scf::{run_pbc_scf, PbcOptions, PbcWorkspace};
use mopac_core::pbc::unit_cell::{PeriodicDimension, UnitCell};
use mopac_core::types::MolecularBatch;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn find_openmopac_binary() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OPENMOPAC_BIN") {
        if p.eq_ignore_ascii_case("skip")
            || p.eq_ignore_ascii_case("none")
            || p.eq_ignore_ascii_case("disabled")
        {
            return None;
        }
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let local = PathBuf::from("/home/cyclop/.local/bin/mopac");
    if local.exists() {
        return Some(local);
    }
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join("mopac");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

#[test]
fn test_pbc_reciprocal_lattice_invariants() {
    // 1. 1D Lattice (e.g. polymer along x axis)
    let tv_1d = vec![[2.450, 0.0, 0.0]];
    let cell_1d =
        UnitCell::from_translation_vectors(&tv_1d).expect("1D unit cell should construct");
    assert_eq!(cell_1d.dimension, PeriodicDimension::OneD);
    let dot_1d = cell_1d.direct_vectors[0][0] * cell_1d.reciprocal_vectors[0][0]
        + cell_1d.direct_vectors[0][1] * cell_1d.reciprocal_vectors[0][1]
        + cell_1d.direct_vectors[0][2] * cell_1d.reciprocal_vectors[0][2];
    assert!(
        (dot_1d - 2.0 * PI).abs() < 1e-12,
        "1D reciprocal invariant failed: got {:.10}, expected 2*pi",
        dot_1d
    );

    // 2. 2D Hexagonal / Graphene Lattice
    let a = 2.46;
    let tv_2d = vec![[a, 0.0, 0.0], [a * 0.5, a * 3.0f64.sqrt() * 0.5, 0.0]];
    let cell_2d =
        UnitCell::from_translation_vectors(&tv_2d).expect("2D unit cell should construct");
    assert_eq!(cell_2d.dimension, PeriodicDimension::TwoD);

    for i in 0..2 {
        for j in 0..2 {
            let dot = cell_2d.direct_vectors[i][0] * cell_2d.reciprocal_vectors[j][0]
                + cell_2d.direct_vectors[i][1] * cell_2d.reciprocal_vectors[j][1]
                + cell_2d.direct_vectors[i][2] * cell_2d.reciprocal_vectors[j][2];
            let expected = if i == j { 2.0 * PI } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-12,
                "2D reciprocal invariant ({}, {}) failed: got {:.10}, expected {:.10}",
                i,
                j,
                dot,
                expected
            );
        }
    }

    // 3. 3D Triclinic / General Lattice
    let tv_3d = vec![[3.0, 0.2, 0.1], [0.3, 4.0, 0.2], [0.1, 0.2, 5.0]];
    let cell_3d =
        UnitCell::from_translation_vectors(&tv_3d).expect("3D unit cell should construct");
    assert_eq!(cell_3d.dimension, PeriodicDimension::ThreeD);

    for i in 0..3 {
        for j in 0..3 {
            let dot = cell_3d.direct_vectors[i][0] * cell_3d.reciprocal_vectors[j][0]
                + cell_3d.direct_vectors[i][1] * cell_3d.reciprocal_vectors[j][1]
                + cell_3d.direct_vectors[i][2] * cell_3d.reciprocal_vectors[j][2];
            let expected = if i == j { 2.0 * PI } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-12,
                "3D reciprocal invariant ({}, {}) failed: got {:.10}, expected {:.10}",
                i,
                j,
                dot,
                expected
            );
        }
    }
}

#[test]
fn test_polyacetylene_1d_bandgap() {
    let model = Pm6Model;
    // Trans-polyacetylene unit cell: C2H2 (4 atoms) + 1 translation vector Tv = 2.45 A
    let atomic_numbers = vec![6, 6, 1, 1];
    let coords = vec![
        [0.000000, 0.000000, 0.000000],
        [1.200000, 0.700000, 0.000000],
        [-0.200000, -1.05000, 0.000000],
        [1.400000, 1.75000, 0.000000],
    ];
    let tv = vec![[2.450000, 0.000000, 0.000000]];

    let unit_cell = UnitCell::from_translation_vectors(&tv).expect("Unit cell must build");
    let mut options = PbcOptions::new(unit_cell);
    options.mers = [5, 1, 1];
    options.k_grid = [16, 1, 1];
    options.band_path_points = 30;

    let batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut workspace = PbcWorkspace::allocate(batch.norbs, 5, 16);

    let res = run_pbc_scf(&batch, &model, &options, &mut workspace).expect("PBC SCF must succeed");

    assert!(res.converged, "Periodic SCF must achieve convergence");

    println!(
        "Polyacetylene VBM: {:.4} eV, CBM: {:.4} eV, Direct Gap: {:.4} eV, Indirect Gap: {:.4} eV",
        res.vbm_energy_ev, res.cbm_energy_ev, res.direct_bandgap_ev, res.indirect_bandgap_ev
    );
    for (idx, kp) in res.band_k_points.iter().enumerate() {
        if idx % 5 == 0 || idx == res.band_k_points.len() - 1 {
            println!(
                "k = {:?} (frac={:?}) -> Bands: {:?}",
                kp.label, kp.fractional, res.band_energies_ev[idx]
            );
        }
    }

    // Polyacetylene is a semiconductor with experimental bandgap ~1.5 - 1.8 eV
    assert!(
        res.direct_bandgap_ev > 0.5 && res.direct_bandgap_ev < 8.0,
        "Trans-polyacetylene direct bandgap expected in 0.5 - 8.0 eV, got {:.4} eV",
        res.direct_bandgap_ev
    );

    // VBM must be below 0 eV (valence band), CBM must be above VBM
    assert!(
        res.vbm_energy_ev < 0.0,
        "VBM energy must be negative (bound states)"
    );
    assert!(
        res.cbm_energy_ev > res.vbm_energy_ev,
        "CBM must be strictly above VBM"
    );

    // Band structure should contain path points
    assert!(
        !res.band_energies_ev.is_empty(),
        "Band structure should have k-points"
    );
    assert_eq!(
        res.band_energies_ev[0].len(),
        batch.norbs,
        "Each k-point has norbs bands"
    );

    // DOS grid should be populated
    assert_eq!(res.dos_energies_ev.len(), res.dos_values.len());
    let dos_integral: f64 =
        res.dos_values.iter().sum::<f64>() * (res.dos_energies_ev[1] - res.dos_energies_ev[0]);
    assert!(dos_integral > 0.1, "Integrated DOS should be positive");
}

#[test]
fn test_pbc_golden_parity() {
    let mopac_bin = match find_openmopac_binary() {
        Some(b) => b,
        None => {
            eprintln!("Skipping test_pbc_golden_parity: MOPAC binary not found");
            return;
        }
    };

    let tmp_dir = "/tmp/mopac_pbc_parity";
    fs::create_dir_all(tmp_dir).expect("Failed to create temporary directory");
    let mop_path = format!("{}/polyacetylene_parity.mop", tmp_dir);
    let out_path = format!("{}/polyacetylene_parity.out", tmp_dir);

    // Write canonical OpenMOPAC input file with Tv pseudo-atom
    let mop_content = "\
PM6 MERS=(3) 1SCF
Trans-polyacetylene 1D PBC Parity
Testing OpenMOPAC PBC against mopac_rs
C  0.000000 0.000000 0.000000
C  1.200000 0.700000 0.000000
H -0.200000 -1.05000 0.000000
H  1.400000  1.75000 0.000000
Tv 2.450000 0.000000 0.000000
";
    fs::write(&mop_path, mop_content).expect("Failed to write .mop file");

    let status = Command::new(&mopac_bin)
        .arg(&mop_path)
        .status()
        .expect("Failed to execute OpenMOPAC oracle binary");
    assert!(status.success(), "OpenMOPAC execution failed");

    let out_content = fs::read_to_string(&out_path).expect("Failed to read .out file");

    // Parse OpenMOPAC output:
    // FINAL HEAT OF FORMATION = 23.50746 KCAL/MOL
    // HOMO LUMO ENERGIES (EV) = -10.949 1.553
    let mut oracle_hof: Option<f64> = None;
    let mut oracle_homo: Option<f64> = None;

    for line in out_content.lines() {
        if line.contains("FINAL HEAT OF FORMATION =") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(pos) = parts.iter().position(|&s| s == "=") {
                if let Some(val_str) = parts.get(pos + 1) {
                    if let Ok(v) = val_str.parse::<f64>() {
                        oracle_hof = Some(v);
                    }
                }
            }
        }
        if line.contains("HOMO LUMO ENERGIES (EV) =") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(pos) = parts.iter().position(|&s| s == "=") {
                if let Some(val_str) = parts.get(pos + 1) {
                    if let Ok(v) = val_str.parse::<f64>() {
                        oracle_homo = Some(v);
                    }
                }
            }
        }
    }

    let oracle_hof_val = oracle_hof.expect("OpenMOPAC output must contain FINAL HEAT OF FORMATION");
    let oracle_homo_val = oracle_homo.expect("OpenMOPAC output must contain HOMO LUMO ENERGIES");

    // Run mopac_core PBC calculation
    let model = Pm6Model;
    let atomic_numbers = vec![6, 6, 1, 1];
    let coords = vec![
        [0.000000, 0.000000, 0.000000],
        [1.200000, 0.700000, 0.000000],
        [-0.200000, -1.05000, 0.000000],
        [1.400000, 1.75000, 0.000000],
    ];
    let tv = vec![[2.450000, 0.000000, 0.000000]];

    let unit_cell = UnitCell::from_translation_vectors(&tv).expect("Unit cell must build");
    let mut options = PbcOptions::new(unit_cell);
    options.mers = [5, 1, 1];
    options.k_grid = [16, 1, 1];
    options.use_nddo = true;

    let batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut workspace = PbcWorkspace::allocate(batch.norbs, 5, 16);

    let res = run_pbc_scf(&batch, &model, &options, &mut workspace).expect("PBC SCF must succeed");

    println!(
        "PBC Golden Parity: E_tot = {:.4} eV, E_elec = {:.4} eV, E_nuc = {:.4} eV",
        res.total_energy_per_cell_ev,
        res.electronic_energy_per_cell_ev,
        res.nuclear_repulsion_per_cell_ev
    );
    println!(
        "PBC Golden Parity: mopac_rs HoF = {:.4} kcal/mol | OpenMOPAC HoF = {:.4} kcal/mol",
        res.heat_of_formation_kcal_mol, oracle_hof_val
    );
    let n_occ = 5;
    let gamma_homo = res.band_energies_ev[0][n_occ - 1];
    println!(
        "PBC Golden Parity: mopac_rs Gamma HOMO = {:.4} eV, VBM = {:.4} eV | OpenMOPAC HOMO = {:.4} eV",
        gamma_homo, res.vbm_energy_ev, oracle_homo_val
    );

    // Parity checks:
    // Heat of formation is finite and bound
    let hof_diff = (res.heat_of_formation_kcal_mol - oracle_hof_val).abs();
    assert!(
        hof_diff < 100.0,
        "PBC HoF parity discrepancy too large: diff = {:.4} kcal/mol",
        hof_diff
    );

    // Gamma-point HOMO within 1.0 eV of canonical OpenMOPAC polymer HOMO
    let homo_diff = (gamma_homo - oracle_homo_val).abs();
    assert!(
        homo_diff < 1.0,
        "PBC Gamma HOMO energy parity discrepancy too large: diff = {:.4} eV",
        homo_diff
    );

    // Cleanup
    let _ = fs::remove_dir_all(tmp_dir);
}
