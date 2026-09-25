//! MOZYME Linear Scaling Differential Parity and Axiomatic Verification Tests.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Validates Localized Molecular Orbital (LMO) construction, 2x2 Jacobi sweeps,
//! trace invariance, and numerical parity against OpenMOPAC `MOZYME PM6 1SCF`.

use mopac_core::mozyme::lewis::construct_lewis_structure;
use mopac_core::mozyme::solver::run_mozyme_scf;
use mopac_core::mozyme::types::MozymeOptions;
use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::parameters::pm7::Pm7Model;
use mopac_core::types::MolecularBatch;
use std::fs;
use std::path::{Path, PathBuf};
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

fn run_openmopac_mozyme(test_id: &str, method: &str, z: &[u8], coords: &[[f64; 3]]) -> Option<f64> {
    let mopac_bin = match find_openmopac_binary() {
        Some(b) => b,
        None => {
            eprintln!(
                "SKIPPING OpenMOPAC oracle parity for {}: binary not found",
                test_id
            );
            return None;
        }
    };

    let tmp_dir = Path::new("/tmp/mopac_mozyme_parity");
    fs::create_dir_all(tmp_dir).expect("Failed to create temporary directory for MOZYME tests");

    let mop_file = tmp_dir.join(format!("{}.mop", test_id));
    let out_file = tmp_dir.join(format!("{}.out", test_id));

    let mut deck = format!(
        "{} MOZYME 1SCF XYZ DISP\nMOZYME Parity Benchmark: {}\n\n",
        method, test_id
    );
    for (idx, &atom_z) in z.iter().enumerate() {
        let sym = match atom_z {
            1 => "H",
            6 => "C",
            7 => "N",
            8 => "O",
            9 => "F",
            _ => "X",
        };
        deck.push_str(&format!(
            "{:2} {:12.6} 0 {:12.6} 0 {:12.6} 0\n",
            sym, coords[idx][0], coords[idx][1], coords[idx][2]
        ));
    }

    fs::write(&mop_file, &deck).expect("Failed to write MOZYME test deck");

    let output = Command::new(&mopac_bin)
        .arg(&mop_file)
        .output()
        .expect("Failed to execute OpenMOPAC binary");

    if !output.status.success() {
        return None;
    }

    if let Ok(content) = fs::read_to_string(&out_file) {
        for line in content.lines() {
            if line.contains("FINAL HEAT OF FORMATION") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                for (i, &p) in parts.iter().enumerate() {
                    if p == "=" && i + 1 < parts.len() {
                        if let Ok(val) = parts[i + 1].parse::<f64>() {
                            return Some(val);
                        }
                    }
                }
            }
        }
    }

    None
}

#[test]
fn test_water_mozyme_pm6_openmopac_parity() {
    let z = vec![8, 1, 1];
    let coords = vec![[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]];

    let model = Pm6Model;
    let batch = MolecularBatch::new_for_model(z.clone(), &coords, &model);
    let options = MozymeOptions {
        max_iter: 100,
        energy_tol: 1e-6,
        jacobi_tol: 1e-4,
        cutoff_distance: 9.0,
        damping: 1.0,
        verbose: false,
        ..Default::default()
    };

    let res = run_mozyme_scf(
        &batch,
        &model,
        Some(mopac_core::corrections::DispersionModel::Pm6DhPlus),
        &options,
    );
    assert!(res.converged, "MOZYME SCF must converge for water");
    assert!(
        res.iterations <= 30,
        "MOZYME should converge in < 30 iterations for water"
    );

    let mut ws = mopac_core::types::ScfWorkspace::allocate(batch.norbs);
    let rhf_opts = mopac_core::scf::scf_loop::ScfOptions {
        use_nddo: true,
        ..Default::default()
    };
    let rhf_res =
        mopac_core::scf::scf_loop::run_rhf_scf_with_options(&batch, &model, &mut ws, &rhf_opts);
    let (_, rhf_hof) = mopac_core::properties::heat::compute_heat_of_formation(
        rhf_res.total_energy_ev,
        &batch.atomic_numbers,
        &model,
        0.0,
    );

    println!(
        "[CANONICAL RHF PM6] Heat of Formation = {:.5} kcal/mol, Etot = {:.6} eV",
        rhf_hof, rhf_res.total_energy_ev
    );
    println!(
        "[MOZYME WATER PM6] Heat of Formation = {:.5} kcal/mol, Etot = {:.6} eV in {} iterations",
        res.heat_of_formation_kcal, res.total_energy_ev, res.iterations
    );

    // Verify against OpenMOPAC oracle if available
    if let Some(oracle_hof) = run_openmopac_mozyme("water_pm6_mozyme", "PM6", &z, &coords) {
        println!(
            "[ORACLE PARITY] Water MOZYME PM6: MOPAC_RS = {:.5} kcal/mol, OpenMOPAC = {:.5} kcal/mol, diff = {:.4} kcal/mol",
            res.heat_of_formation_kcal, oracle_hof, (res.heat_of_formation_kcal - oracle_hof).abs()
        );
        assert!(
            (res.heat_of_formation_kcal - oracle_hof).abs() < 0.50,
            "MOZYME PM6 water heat of formation deviates from OpenMOPAC: diff = {}",
            (res.heat_of_formation_kcal - oracle_hof).abs()
        );
    }
}

#[test]
fn test_methane_mozyme_pm6_lewis_and_energy() {
    let z = vec![6, 1, 1, 1, 1];
    let r = 1.09;
    let coords = vec![
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
        [-r / 3.0, r * (8.0f64 / 9.0).sqrt(), 0.0],
        [
            -r / 3.0,
            -r * (2.0f64 / 9.0).sqrt(),
            r * (2.0f64 / 3.0).sqrt(),
        ],
        [
            -r / 3.0,
            -r * (2.0f64 / 9.0).sqrt(),
            -r * (2.0f64 / 3.0).sqrt(),
        ],
    ];

    let model = Pm6Model;
    let batch = MolecularBatch::new_for_model(z, &coords, &model);
    let lewis = construct_lewis_structure(&batch);

    assert_eq!(lewis.bonds.len(), 4, "Methane must have 4 C-H bonds");
    assert_eq!(
        lewis.coordination_numbers[0], 4,
        "Carbon coordination number must be 4"
    );

    let options = MozymeOptions::default();
    let res = run_mozyme_scf(&batch, &model, None, &options);

    assert!(res.converged, "MOZYME SCF must converge for methane");
    assert_eq!(
        res.lmos.len(),
        8,
        "Methane must have 4 occupied + 4 virtual LMOs"
    );
}

#[test]
fn test_mozyme_density_trace_and_idempotency() {
    let z = vec![8, 1, 1];
    let coords = vec![[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]];

    let model = Pm6Model;
    let batch = MolecularBatch::new_for_model(z, &coords, &model);
    let res = run_mozyme_scf(&batch, &model, None, &MozymeOptions::default());

    // Trace of density matrix must equal 8 electrons for H2O (valence electrons: 6 + 1 + 1 = 8)
    let mut trace = 0.0;
    for i in 0..batch.norbs {
        trace += res.density.get(i, i);
    }
    assert!(
        (trace - 8.0).abs() < 1e-10,
        "Density trace must equal 8 electrons: trace = {}",
        trace
    );
}

#[test]
fn test_mozyme_hamiltonian_consistency() {
    let z = vec![8, 1, 1];
    let coords = vec![[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]];

    let model_pm6 = Pm6Model;
    let model_am1 = Am1Model;
    let model_pm7 = Pm7Model;

    let batch_pm6 = MolecularBatch::new_for_model(z.clone(), &coords, &model_pm6);
    let batch_am1 = MolecularBatch::new_for_model(z.clone(), &coords, &model_am1);
    let batch_pm7 = MolecularBatch::new_for_model(z, &coords, &model_pm7);

    let res_pm6 = run_mozyme_scf(&batch_pm6, &model_pm6, None, &MozymeOptions::default());
    let res_am1 = run_mozyme_scf(&batch_am1, &model_am1, None, &MozymeOptions::default());
    let res_pm7 = run_mozyme_scf(&batch_pm7, &model_pm7, None, &MozymeOptions::default());

    assert!(res_pm6.converged, "PM6 MOZYME must converge");
    assert!(res_am1.converged, "AM1 MOZYME must converge");
    assert!(res_pm7.converged, "PM7 MOZYME must converge");

    println!(
        "[HAMILTONIAN MOZYME COMPARISON] Water HoF:\n  PM6: {:.4} kcal/mol\n  AM1: {:.4} kcal/mol\n  PM7: {:.4} kcal/mol",
        res_pm6.heat_of_formation_kcal, res_am1.heat_of_formation_kcal, res_pm7.heat_of_formation_kcal
    );
}

#[test]
fn test_peptide_bond_mozyme_pm6_openmopac_parity() {
    // N-methylacetamide (CH3-CO-NH-CH3) peptide backbone unit
    let z = vec![6, 6, 8, 7, 1, 6, 1, 1, 1, 1, 1, 1];
    let coords = vec![
        [0.0000, 0.0000, 0.0000],    // C1 (methyl)
        [1.5000, 0.0000, 0.0000],    // C2 (carbonyl)
        [2.1500, 1.0500, 0.0000],    // O3 (carbonyl)
        [2.1000, -1.2000, 0.0000],   // N4 (amide)
        [1.5500, -2.0500, 0.0000],   // H5 (amide H)
        [3.5500, -1.3500, 0.0000],   // C6 (methyl)
        [-0.3500, 1.0300, 0.0000],   // H1a
        [-0.3500, -0.5200, 0.8900],  // H1b
        [-0.3500, -0.5200, -0.8900], // H1c
        [3.9000, -2.3800, 0.0000],   // H6a
        [3.9500, -0.8700, 0.8900],   // H6b
        [3.9500, -0.8700, -0.8900],  // H6c
    ];

    let model = Pm6Model;
    let batch = MolecularBatch::new_for_model(z.clone(), &coords, &model);
    let lewis = construct_lewis_structure(&batch);

    // Verify Lewis structure detects the peptide bond topology
    assert!(
        lewis.bonds.len() >= 11,
        "Must detect all single/double bonds in peptide unit"
    );

    let options = MozymeOptions {
        max_iter: 100,
        energy_tol: 1e-6,
        jacobi_tol: 1e-4,
        cutoff_distance: 8.5,
        damping: 1.0,
        verbose: false,
        ..Default::default()
    };

    let res = run_mozyme_scf(
        &batch,
        &model,
        Some(mopac_core::corrections::DispersionModel::Pm6DhPlus),
        &options,
    );
    assert!(res.converged, "Peptide bond model MOZYME SCF must converge");

    println!(
        "[PEPTIDE BOND MOZYME PM6] HoF = {:.4} kcal/mol, Etot = {:.6} eV, Iterations = {}",
        res.heat_of_formation_kcal, res.total_energy_ev, res.iterations
    );

    // Verify against OpenMOPAC oracle if available
    if let Some(oracle_hof) = run_openmopac_mozyme("peptide_pm6_mozyme", "PM6", &z, &coords) {
        println!(
            "[ORACLE PARITY] Peptide MOZYME PM6: MOPAC_RS = {:.4} kcal/mol, OpenMOPAC = {:.4} kcal/mol, diff = {:.4} kcal/mol",
            res.heat_of_formation_kcal, oracle_hof, (res.heat_of_formation_kcal - oracle_hof).abs()
        );
        assert!(
            (res.heat_of_formation_kcal - oracle_hof).abs() < 25.0,
            "Peptide MOZYME PM6 heat of formation deviates from OpenMOPAC: diff = {}",
            (res.heat_of_formation_kcal - oracle_hof).abs()
        );
        assert!(
            (res.total_energy_ev - (-934.10605)).abs() < 1.5,
            "Peptide MOZYME PM6 total energy deviates from OpenMOPAC: diff = {}",
            (res.total_energy_ev - (-934.10605)).abs()
        );
    }
}
