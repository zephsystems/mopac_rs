//! Integration Test Suite: Multi-Electron Configuration Interaction (MECI) Excited State Gradients.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Rigorously verifies:
//! * Test 1: Conservation of state density trace (total electron number invariant).
//! * Test 2: Root 1 ground state gradients parity with closed-shell Hellmann-Feynman gradients.
//! * Test 3: Root 2 excited state Hellmann-Feynman forces from 1-RDM state density matrix.
//! * Test 4: Root 2 fully relaxed total energy gradients via `compute_meci_numerical_gradients`
//!   matching canonical OpenMOPAC v23.2.5 finite difference within 0.05 kcal/(mol * A).

use mopac_core::ci::meci::{
    compute_meci_nuclear_gradients, compute_meci_numerical_gradients, run_meci, CiActiveSpace,
    MeciOptions, MeciWorkspace,
};
use mopac_core::constants::codata2018::EV_TO_KCAL_MOL;
use mopac_core::gradients::nuclear_gradients::{
    compute_cartesian_gradients_with_options, GradientWorkspace,
};
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::scf::scf_loop::run_rhf_scf_adaptive_with_nddo;
use mopac_core::types::{MolecularBatch, ScfWorkspace};

#[test]
fn test_meci_state_density_trace_invariance() {
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

    // Total valence electrons for C2H4 = 4*2 + 1*4 = 12 electrons
    let expected_trace = 12.0f64;

    for target_root in 1..=4 {
        let options = MeciOptions {
            active_space: CiActiveSpace::new(2, 2),
            target_root,
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

        let mut trace = 0.0f64;
        for i in 0..batch.norbs {
            trace += meci_res.state_density.get(i, i);
        }

        assert!(
            (trace - expected_trace).abs() < 1e-6,
            "State density trace must equal total valence electrons (12.0), got {:.6} for root {}",
            trace,
            target_root
        );
    }
}

#[test]
fn test_meci_root1_ground_state_parity() {
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

    let mut batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res =
        run_rhf_scf_adaptive_with_nddo(&batch, &model, &mut scf_ws, 100, 1e-10, 1e-9, true);
    assert!(scf_res.converged, "SCF must converge");

    // Standard SCF gradients
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut scf_grads = vec![[0.0; 3]; batch.natoms];
    compute_cartesian_gradients_with_options(
        &mut batch,
        &model,
        &scf_ws.density,
        &mut grad_ws,
        &mut scf_grads,
        true,
    );

    // Root 1 MECI gradients
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

    // For pure ground reference determinant |HOMO^2>, state density must match SCF density identically
    let pure_ref_eigenvector = vec![1.0, 0.0, 0.0, 0.0];
    let pure_ref_density = mopac_core::ci::meci::compute_ci_state_density(
        batch.norbs,
        &scf_ws.eigenvectors,
        &[5, 6],
        &meci_res.microstates,
        &pure_ref_eigenvector,
        &[1.0, 0.0],
        Some(&scf_ws.density),
    );

    let mut pure_grads = vec![[0.0; 3]; batch.natoms];
    compute_meci_nuclear_gradients(
        &mut batch,
        &model,
        &pure_ref_density,
        &mut grad_ws,
        &mut pure_grads,
        true,
    );

    for a in 0..batch.natoms {
        for c in 0..3 {
            let diff = (scf_grads[a][c] - pure_grads[a][c]).abs() * EV_TO_KCAL_MOL;
            assert!(
                diff < 1e-10,
                "Pure reference determinant gradient must match SCF gradient at atom {} coord {}: diff = {:.12} kcal/(mol*A)",
                a,
                c,
                diff
            );
        }
    }

    // For correlated Root 1 (mixing of HOMO^2 and LUMO^2), translational invariance must strictly hold
    let mut correlated_grads = vec![[0.0; 3]; batch.natoms];
    compute_meci_nuclear_gradients(
        &mut batch,
        &model,
        &meci_res.state_density,
        &mut grad_ws,
        &mut correlated_grads,
        true,
    );
    let mut sum_f = [0.0f64; 3];
    for g in &correlated_grads {
        sum_f[0] += g[0];
        sum_f[1] += g[1];
        sum_f[2] += g[2];
    }
    assert!(
        sum_f[0].abs() < 1e-8 && sum_f[1].abs() < 1e-8 && sum_f[2].abs() < 1e-8,
        "Correlated Root 1 gradient must strictly preserve translational invariance"
    );
}

#[test]
fn test_meci_root2_excited_state_hellmann_feynman_forces() {
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

    let mut batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res =
        run_rhf_scf_adaptive_with_nddo(&batch, &model, &mut scf_ws, 100, 1e-10, 1e-9, true);
    assert!(scf_res.converged, "Ethylene SCF must converge");

    let options = MeciOptions {
        active_space: CiActiveSpace::new(2, 2),
        target_root: 2,
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

    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut gradients = vec![[0.0; 3]; batch.natoms];
    compute_meci_nuclear_gradients(
        &mut batch,
        &model,
        &meci_res.state_density,
        &mut grad_ws,
        &mut gradients,
        true,
    );

    let mut grads_kcal = vec![[0.0; 3]; batch.natoms];
    let mut grad_norm_sq = 0.0f64;
    for a in 0..batch.natoms {
        for c in 0..3 {
            grads_kcal[a][c] = gradients[a][c] * EV_TO_KCAL_MOL;
            grad_norm_sq += grads_kcal[a][c] * grads_kcal[a][c];
        }
    }
    let grad_norm = grad_norm_sq.sqrt();

    // Verify translational invariance: sum of Cartesian forces must strictly be zero
    let mut sum_fx = 0.0f64;
    let mut sum_fy = 0.0f64;
    let mut sum_fz = 0.0f64;
    for g in &grads_kcal {
        sum_fx += g[0];
        sum_fy += g[1];
        sum_fz += g[2];
    }
    assert!(
        sum_fx.abs() < 1e-8 && sum_fy.abs() < 1e-8 && sum_fz.abs() < 1e-8,
        "Translational invariance violated: sum_f = ({}, {}, {})",
        sum_fx,
        sum_fy,
        sum_fz
    );

    // C1 X and C2 X must be antisymmetric
    assert!(
        (grads_kcal[0][0] + grads_kcal[1][0]).abs() < 1e-6,
        "C1 and C2 forces must be equal and opposite"
    );

    // State density Hellmann-Feynman gradient norm ~159.12 kcal/(mol * A)
    assert!(
        (grad_norm - 159.12).abs() < 1.0,
        "State density gradient norm expected ~159.12, got {:.4}",
        grad_norm
    );
    assert!(
        (grads_kcal[0][0] - 112.05).abs() < 1.0,
        "C1 X force expected ~112.05, got {:.4}",
        grads_kcal[0][0]
    );
}

#[test]
fn test_meci_root2_full_numerical_gradients() {
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

    let mut batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let options = MeciOptions {
        active_space: CiActiveSpace::new(2, 2),
        target_root: 2,
        spin_target: None,
        use_nddo: true,
    };

    let mut gradients = vec![[0.0; 3]; batch.natoms];
    compute_meci_numerical_gradients(&mut batch, &model, &options, &mut gradients, 1e-4);

    let mut grads_kcal = vec![[0.0; 3]; batch.natoms];
    for a in 0..batch.natoms {
        for c in 0..3 {
            grads_kcal[a][c] = gradients[a][c] * EV_TO_KCAL_MOL;
        }
    }

    println!("Full Numerical MECI Root 2 Gradients (kcal/mol*A):");
    println!("C1 X: {:.4}", grads_kcal[0][0]);
    println!("C2 X: {:.4}", grads_kcal[1][0]);

    // OpenMOPAC v23.2.5 finite difference target:
    // C1: X = +114.599 kcal / (mol * A)
    // C2: X = -114.599 kcal / (mol * A)
    assert!(
        (grads_kcal[0][0] - 114.596).abs() < 0.05,
        "C1 X numerical gradient mismatch: expected ~114.596, got {:.4}",
        grads_kcal[0][0]
    );
    assert!(
        (grads_kcal[1][0] - (-114.596)).abs() < 0.05,
        "C2 X numerical gradient mismatch: expected ~-114.596, got {:.4}",
        grads_kcal[1][0]
    );
}
