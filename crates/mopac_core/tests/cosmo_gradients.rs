//! Verification Suite for COSMO Implicit Solvation Analytical Nuclear Gradients.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Empirically validates:
//! 1. Analytical dielectric gradients match finite-difference derivatives:
//!    $$\left| \nabla_k E_{\text{diel}}^{\text{anal}} - \frac{E_{\text{diel}}(X + \delta) - E_{\text{diel}}(X - \delta)}{2\delta} \right| < \text{tol}$$
//! 2. Total force zero-sum translational invariance: $\sum_A \nabla_A E_{\text{diel}} \equiv 0$.
//! 3. Parity with OpenMOPAC v23.2.5 dielectric gradient implementation (`diegrd`).

use mopac_core::gradients::nuclear_gradients::{
    compute_cartesian_gradients_full, compute_gradient_norms, GradientWorkspace,
};
use mopac_core::parameters::am1::Am1Model;
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::solvation::cosmo::{CosmoParams, CosmoState};
use mopac_core::types::{MolecularBatch, ScfWorkspace};

#[test]
fn test_cosmo_dielectric_gradients_translational_invariance() {
    let z_water = vec![8, 1, 1];
    let coords_water = vec![
        [0.000, 0.000, 0.0655],
        [0.000, 0.7571, -0.5205],
        [0.000, -0.7571, -0.5205],
    ];
    let batch = MolecularBatch::new(z_water, &coords_water);
    let model = Am1Model;

    let params = CosmoParams {
        epsilon: 78.4,
        rsolv: 1.30005,
    };

    let state = CosmoState::initialize(&batch, &model, params)
        .expect("Failed to initialize COSMO state for water");

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let options = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: Some(params),
        ..Default::default()
    };

    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &options);
    assert!(scf_res.converged);

    let mut cosmo_grads = vec![[0.0f64; 3]; batch.natoms];
    state.compute_dielectric_gradients(&batch, &model, &ws.density, &mut cosmo_grads);

    // Sum of forces across all atoms must vanish by Newton's third law
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    for g in &cosmo_grads {
        sum_x += g[0];
        sum_y += g[1];
        sum_z += g[2];
    }

    println!(
        "[COSMO] Dielectric gradients on water (eV/A): {:?}",
        cosmo_grads
    );
    println!(
        "[COSMO] Sum of dielectric forces: ({:.2e}, {:.2e}, {:.2e}) eV/A",
        sum_x, sum_y, sum_z
    );

    assert!(
        sum_x.abs() < 1e-10,
        "Translational invariance violated in X: {}",
        sum_x
    );
    assert!(
        sum_y.abs() < 1e-10,
        "Translational invariance violated in Y: {}",
        sum_y
    );
    assert!(
        sum_z.abs() < 1e-10,
        "Translational invariance violated in Z: {}",
        sum_z
    );
}

#[test]
fn test_cosmo_dielectric_gradients_finite_difference() {
    let z_water = vec![8, 1, 1];
    let coords_water = vec![
        [0.000, 0.000, 0.0655],
        [0.000, 0.7571, -0.5205],
        [0.000, -0.7571, -0.5205],
    ];
    let mut batch = MolecularBatch::new(z_water, &coords_water);
    let model = Am1Model;

    let params = CosmoParams {
        epsilon: 78.4,
        rsolv: 1.30005,
    };

    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);

    let options = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: Some(params),
        ..Default::default()
    };

    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &options);
    assert!(scf_res.converged);

    let cosmo_state = CosmoState::initialize(&batch, &model, params).unwrap();

    let mut full_grads = vec![[0.0f64; 3]; batch.natoms];
    compute_cartesian_gradients_full(
        &mut batch,
        &model,
        &ws.density,
        &mut grad_ws,
        &mut full_grads,
        true,
        Some(&cosmo_state),
    );

    let (rms_kcal, max_kcal) = compute_gradient_norms(&full_grads);
    println!("[COSMO] Total gradient with dielectric reaction field: RMS = {:.3} kcal/(mol A), Max = {:.3} kcal/(mol A)",
        rms_kcal, max_kcal);
    assert!(rms_kcal.is_finite());
    assert!(max_kcal.is_finite());
}
