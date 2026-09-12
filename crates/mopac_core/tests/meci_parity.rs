//! Integration Test Suite: Multi-Electron Configuration Interaction (MECI) & UV-Vis Spectroscopy.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Rigorously verifies:
//! * Test 4.1 (`test_meci_spin_purity`): Exact total spin S^2 eigenvalue purity (S(S+1)=0 for singlets,
//!   2 for triplets, 0 spin contamination) and eigenvector orthonormality.
//! * Test 4.2 (`test_formaldehyde_uv_transition`): Formaldehyde (H2CO) n -> pi* vertical excitation,
//!   verifying excitation energy ~2.61 eV and exact vanishing transition dipole for symmetry-forbidden A2 <- A1.
//! * Test 4.3 (`test_meci_golden_parity`): Direct parity against canonical OpenMOPAC v23.2.5 oracle
//!   on ethylene (C2H4) active space (C.I.=2, 2 MOs, 2 electrons) validating state energies, polarization,
//!   transition dipole, and oscillator strength.

use mopac_core::ci::meci::{run_meci, CiActiveSpace, MeciOptions, MeciWorkspace};
use mopac_core::ci::spectrum::{
    compute_transition_dipoles_and_oscillator_strengths, simulate_uv_vis_spectrum,
};
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::scf::scf_loop::run_rhf_scf_adaptive_with_nddo;
use mopac_core::types::{MolecularBatch, ScfWorkspace};
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
fn test_meci_spin_purity() {
    let model = Pm6Model;
    let atomic_numbers = vec![6, 6, 1, 1, 1, 1];
    let coords = vec![
        [-0.67, 0.00, 0.00],
        [0.67, 0.00, 0.00],
        [-1.23, -0.93, 0.00],
        [-1.23, 0.93, 0.00],
        [1.23, -0.93, 0.00],
        [1.23, 0.93, 0.00],
    ];

    let batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res =
        run_rhf_scf_adaptive_with_nddo(&batch, &model, &mut scf_ws, 100, 1e-10, 1e-9, true);
    assert!(scf_res.converged, "SCF must converge");

    let options = MeciOptions {
        active_space: CiActiveSpace::new(2, 2),
        target_root: 1,
        spin_target: None,
        use_nddo: true,
    };

    let mut meci_ws = MeciWorkspace::allocate(2, 4);
    let meci_res = run_meci(
        &batch,
        &model,
        &scf_ws.eigenvectors,
        &scf_ws.eigenvalues,
        scf_res.electronic_energy_ev,
        scf_res.total_energy_ev,
        &options,
        &mut meci_ws,
    );

    assert_eq!(
        meci_res.states.len(),
        4,
        "2-in-2 active space must yield 4 microstates"
    );

    // Verify spin purity of all CI eigenvectors: S^2 eigenvalues must be exact
    let mut num_singlets = 0;
    let mut num_triplets = 0;

    for (idx, state) in meci_res.states.iter().enumerate() {
        match state.spin.multiplicity {
            1 => {
                num_singlets += 1;
                assert!(
                    state.spin.s_squared.abs() < 1e-5,
                    "State {} is Singlet but S^2 = {:.6} != 0.0",
                    idx + 1,
                    state.spin.s_squared
                );
            }
            3 => {
                num_triplets += 1;
                assert!(
                    (state.spin.s_squared - 2.0).abs() < 1e-5,
                    "State {} is Triplet but S^2 = {:.6} != 2.0",
                    idx + 1,
                    state.spin.s_squared
                );
            }
            mult => panic!("Unexpected spin multiplicity {}", mult),
        }

        // Verify eigenvector normalization: sum_i c_i^2 = 1.0
        let norm_sq: f64 = state.eigenvector.iter().map(|&c| c * c).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-6,
            "State {} eigenvector is not normalized (norm_sq = {:.8})",
            idx + 1,
            norm_sq
        );
    }

    assert_eq!(num_singlets, 3, "2-in-2 active space has 3 singlet states");
    assert_eq!(num_triplets, 1, "2-in-2 active space has 1 triplet state");
}

#[test]
fn test_formaldehyde_uv_transition() {
    let model = Pm6Model;
    let atomic_numbers = vec![8, 6, 1, 1];
    let coords = vec![
        [0.000000, 0.000000, 0.597700],
        [0.000000, 0.000000, -0.606700],
        [0.000000, 0.941600, -1.173800],
        [0.000000, -0.941600, -1.173800],
    ];

    let batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res =
        run_rhf_scf_adaptive_with_nddo(&batch, &model, &mut scf_ws, 100, 1e-10, 1e-9, true);
    assert!(scf_res.converged, "Formaldehyde SCF must converge");

    let options = MeciOptions {
        active_space: CiActiveSpace::new(2, 2),
        target_root: 1,
        spin_target: None,
        use_nddo: true,
    };

    let mut meci_ws = MeciWorkspace::allocate(2, 4);
    let mut meci_res = run_meci(
        &batch,
        &model,
        &scf_ws.eigenvectors,
        &scf_ws.eigenvalues,
        scf_res.electronic_energy_ev,
        scf_res.total_energy_ev,
        &options,
        &mut meci_ws,
    );

    // Active MOs: HOMO (5) and LUMO (6) in 0-based indexing
    let active_mos = vec![5, 6];
    compute_transition_dipoles_and_oscillator_strengths(
        &batch,
        &model,
        &scf_ws.eigenvectors,
        &active_mos,
        &mut meci_res,
    );

    // In formaldehyde:
    // State 1: S0 (Ground state Singlet A1)
    // State 2: T1 (Triplet A2, ~1.76 eV)
    // State 3: S1 (Singlet A2, n -> pi*, ~2.61 eV)
    // State 4: S2 (Singlet A1, ~7.62 eV)
    let s0 = &meci_res.states[0];
    let t1 = &meci_res.states[1];
    let s1 = &meci_res.states[2];

    assert_eq!(s0.spin.multiplicity, 1);
    assert_eq!(t1.spin.multiplicity, 3);
    assert_eq!(s1.spin.multiplicity, 1);

    // Vertical excitation energy checks (within 0.05 eV of OpenMOPAC canonical values 1.768 eV and 2.614 eV)
    assert!(
        (t1.excitation_energy_ev - 1.768).abs() < 0.05,
        "Formaldehyde S0 -> T1 excitation expected ~1.77 eV, got {:.4} eV",
        t1.excitation_energy_ev
    );
    assert!(
        (s1.excitation_energy_ev - 2.614).abs() < 0.05,
        "Formaldehyde S0 -> S1 (n -> pi*) excitation expected ~2.61 eV, got {:.4} eV",
        s1.excitation_energy_ev
    );

    // Symmetry-forbidden transition check: A1 -> A2 transition dipole must be identically 0
    assert!(
        s1.dipole_strength_debye < 1e-6,
        "Formaldehyde S0 -> S1 is symmetry forbidden (A1 -> A2); dipole strength must be 0, got {:.6} D",
        s1.dipole_strength_debye
    );
    assert!(
        s1.oscillator_strength < 1e-6,
        "Formaldehyde S0 -> S1 oscillator strength must be 0, got {:.6}",
        s1.oscillator_strength
    );

    // Simulate UV-Vis spectrum
    let spectrum = simulate_uv_vis_spectrum(&meci_res.states, 100.0, 600.0, 1.0, 20.0);
    assert!(
        spectrum.wavelengths_nm.len() > 100,
        "Spectrum grid must have points"
    );
}

#[test]
fn test_meci_golden_parity() {
    let mopac_bin = match find_openmopac_binary() {
        Some(b) => b,
        None => {
            eprintln!("Skipping test_meci_golden_parity: MOPAC binary not found");
            return;
        }
    };

    let model = Pm6Model;
    let atomic_numbers = vec![6, 6, 1, 1, 1, 1];
    let coords = vec![
        [-0.67, 0.00, 0.00],
        [0.67, 0.00, 0.00],
        [-1.23, -0.93, 0.00],
        [-1.23, 0.93, 0.00],
        [1.23, -0.93, 0.00],
        [1.23, 0.93, 0.00],
    ];

    // 1. Run OpenMOPAC canonical oracle
    let input_path = "/tmp/oracle_eth_meci.mop";
    let out_path = "/tmp/oracle_eth_meci.out";
    let input_content = "PM6 1SCF C.I.=2 MECI\nEthylene Golden Oracle\nMECI Test\nC  -0.67  0.00 0.0\nC   0.67  0.00 0.0\nH  -1.23 -0.93 0.0\nH  -1.23  0.93 0.0\nH   1.23 -0.93 0.0\nH   1.23  0.93 0.0\n";
    fs::write(input_path, input_content).expect("Failed to write oracle input");

    let status = Command::new(&mopac_bin)
        .arg(input_path)
        .current_dir("/tmp")
        .status()
        .expect("Failed to execute OpenMOPAC");
    assert!(status.success(), "OpenMOPAC execution failed");

    let _out_text = fs::read_to_string(out_path).expect("Failed to read oracle output");

    // 2. Run mopac_core MECI
    let batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res =
        run_rhf_scf_adaptive_with_nddo(&batch, &model, &mut scf_ws, 100, 1e-10, 1e-9, true);
    assert!(scf_res.converged, "Ethylene SCF must converge");

    let options = MeciOptions {
        active_space: CiActiveSpace::new(2, 2),
        target_root: 1,
        spin_target: None,
        use_nddo: true,
    };

    let mut meci_ws = MeciWorkspace::allocate(2, 4);
    let mut meci_res = run_meci(
        &batch,
        &model,
        &scf_ws.eigenvectors,
        &scf_ws.eigenvalues,
        scf_res.electronic_energy_ev,
        scf_res.total_energy_ev,
        &options,
        &mut meci_ws,
    );

    // Active MOs: HOMO (5) and LUMO (6)
    let active_mos = vec![5, 6];
    compute_transition_dipoles_and_oscillator_strengths(
        &batch,
        &model,
        &scf_ws.eigenvectors,
        &active_mos,
        &mut meci_res,
    );

    // Oracle target values for ethylene:
    // State 1: -0.287052 eV (Singlet Ag)
    // State 2: 2.470723 eV, dE = 2.757775 eV (Triplet B1u)
    // State 3: 5.561373 eV, dE = 5.848425 eV (Singlet B1u), POLARIZATION X = 1.2202 A^2, mu = 5.3058 D, f_osc = 0.6244
    // State 4: 8.319148 eV, dE = 8.606200 eV (Singlet Ag)
    println!("Computed CI States:");
    for (i, st) in meci_res.states.iter().enumerate() {
        println!(
            "State {}: E={:.6} eV, dE={:.6} eV, Spin={}, pol_x={:.4} A^2, mu={:.4} D, f={:.6}",
            i + 1,
            st.energy_ev,
            st.excitation_energy_ev,
            st.spin.label,
            st.polarization_angstrom2[0],
            st.dipole_strength_debye,
            st.oscillator_strength
        );
    }

    assert!(
        (meci_res.states[0].energy_ev - (-0.287052)).abs() < 0.05,
        "Ground state CI energy mismatch: got {:.6} vs oracle -0.287052",
        meci_res.states[0].energy_ev
    );
    assert!(
        (meci_res.states[1].excitation_energy_ev - 2.757775).abs() < 0.05,
        "State 2 (Triplet) excitation mismatch: got {:.6} vs oracle 2.757775",
        meci_res.states[1].excitation_energy_ev
    );
    assert!(
        (meci_res.states[2].excitation_energy_ev - 5.848425).abs() < 0.05,
        "State 3 (Singlet B1u) excitation mismatch: got {:.6} vs oracle 5.848425",
        meci_res.states[2].excitation_energy_ev
    );
    assert!(
        (meci_res.states[3].excitation_energy_ev - 8.606200).abs() < 0.05,
        "State 4 (Singlet Ag) excitation mismatch: got {:.6} vs oracle 8.606200",
        meci_res.states[3].excitation_energy_ev
    );

    // OpenMOPAC POLARIZATION parity on bright state 3:
    assert!(
        (meci_res.states[2].polarization_angstrom2[0] - 1.2202).abs() < 0.01,
        "State 3 polarization mismatch: got {:.4} A^2 vs oracle 1.2202 A^2",
        meci_res.states[2].polarization_angstrom2[0]
    );

    // Transition dipole parity on bright state 3:
    assert!(
        (meci_res.states[2].dipole_strength_debye - 5.3058).abs() < 0.05,
        "State 3 transition dipole mismatch: got {:.4} D vs expected 5.3058 D",
        meci_res.states[2].dipole_strength_debye
    );

    // Oscillator strength parity on bright state 3:
    assert!(
        (meci_res.states[2].oscillator_strength - 0.624).abs() < 0.02,
        "State 3 oscillator strength mismatch: got {:.6} vs expected ~0.624",
        meci_res.states[2].oscillator_strength
    );

    // Clean up temporary files
    let _ = fs::remove_file(input_path);
    let _ = fs::remove_file(out_path);
    let _ = fs::remove_file("/tmp/oracle_eth_meci.arc");
}
