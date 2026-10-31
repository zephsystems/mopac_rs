//! SCF convergence-escalation gates.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Saunders-Hillier level shift allows `level_shift_ev > 0` runs to converge
//! without jumping to unphysical higher-energy stationary points.
//! On CCCBDB `cc-685` (linear C6H6, PM3), explicit level shifting preserves the
//! ground-state root.
//!
//! Contract enforced here:
//!  1. Explicit level shift is a converger aid, not a root selector: shifted
//!     and unshifted runs land on the same solution.
//!  2. converged == true implies the reported solution is a stable fixed point
//!     of the unshifted strict-tolerance iteration.
//!  3. The adaptive ladder reports the engine's genuine unshifted solution.
//!  4. True benzene (CCCBDB cc-46) keeps converging and matching OpenMOPAC.

use mopac_core::parameters::pm3::Pm3Model;
use mopac_core::properties::heat::compute_heat_of_formation;
use mopac_core::scf::scf_loop::{
    ScfOptions, run_rhf_scf_adaptive_with_nddo_cosmo_and_charge, run_rhf_scf_with_options,
};
use mopac_core::types::{MolecularBatch, ScfWorkspace};

const CC685_Z: [u8; 12] = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1];
const CC685_C: [[f64; 3]; 12] = [
    [0.00000000, 0.00000000, 0.68412600],
    [0.00000000, 0.00000000, -0.68412600],
    [0.00000000, 0.00000000, 1.89988400],
    [0.00000000, 0.00000000, -1.89988400],
    [0.00000000, 0.00000000, 3.35694600],
    [0.00000000, 0.00000000, -3.35694600],
    [0.00000000, 1.02264400, 3.75438400],
    [-0.88563600, -0.51132200, 3.75438400],
    [0.88563600, -0.51132200, 3.75438400],
    [0.00000000, -1.02264400, -3.75438400],
    [0.88563600, 0.51132200, -3.75438400],
    [-0.88563600, 0.51132200, -3.75438400],
];

const CC46_Z: [u8; 12] = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1];
const CC46_C: [[f64; 3]; 12] = [
    [0.00000000, 1.39660300, 0.00000000],
    [1.20949400, 0.69830200, 0.00000000],
    [1.20949400, -0.69830200, 0.00000000],
    [0.00000000, -1.39660300, 0.00000000],
    [-1.20949400, -0.69830200, 0.00000000],
    [-1.20949400, 0.69830200, 0.00000000],
    [0.00000000, 2.48360800, 0.00000000],
    [2.15086700, 1.24180400, 0.00000000],
    [2.15086700, -1.24180400, 0.00000000],
    [0.00000000, -2.48360800, 0.00000000],
    [-2.15086700, -1.24180400, 0.00000000],
    [-2.15086700, 1.24180400, 0.00000000],
];

fn batch(z: &[u8], c: &[[f64; 3]]) -> (MolecularBatch, Pm3Model) {
    let model = Pm3Model;
    let b = MolecularBatch::new_for_model(z.to_vec(), c, &model);
    (b, model)
}

/// Mirror of the production pipeline in `mopac_py::run_calculation_internal`:
/// single-stage SCF first, adaptive ladder as fallback.
fn run_pipeline(
    z: &[u8],
    c: &[[f64; 3]],
    max_iter: usize,
) -> (mopac_core::scf::scf_loop::ScfResult, ScfWorkspace) {
    let (b, model) = batch(z, c);
    let mut ws = ScfWorkspace::allocate(b.norbs);
    let opts = ScfOptions {
        max_iter,
        ..Default::default()
    };
    let res = run_rhf_scf_with_options(&b, &model, &mut ws, &opts);
    if res.converged {
        return (res, ws);
    }
    let res = run_rhf_scf_adaptive_with_nddo_cosmo_and_charge(
        &b, &model, &mut ws, max_iter, 1e-7, 1e-6, true, None, 0,
    );
    (res, ws)
}

fn hof(res: &mopac_core::scf::scf_loop::ScfResult, z: &[u8]) -> f64 {
    // compute_heat_of_formation returns (binding_energy_ev, heat_of_formation_kcal)
    compute_heat_of_formation(res.total_energy_ev, z, &Pm3Model, 0.0).1
}

/// Strict unshifted re-run from the pipeline's final density; returns its energy.
fn strict_rerun_energy(ws: &mut ScfWorkspace, z: &[u8], c: &[[f64; 3]]) -> f64 {
    let (b, model) = batch(z, c);
    let opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-10,
        density_tol: 1e-9,
        level_shift_ev: 0.0,
        damping: 0.5,
        reuse_density: true,
        ..Default::default()
    };
    let res = run_rhf_scf_with_options(&b, &model, ws, &opts);
    res.total_energy_ev
}

#[test]
fn explicit_level_shift_converges_to_the_unshifted_root() {
    // Explicit level-shifted SCF and unshifted SCF must converge to the same root.
    let (b, model) = batch(&CC685_Z, &CC685_C);
    let mut ws = ScfWorkspace::allocate(b.norbs);
    let unshifted = run_rhf_scf_with_options(
        &b,
        &model,
        &mut ws,
        &ScfOptions {
            max_iter: 2000,
            level_shift_ev: 0.0,
            ..Default::default()
        },
    );
    assert!(unshifted.converged, "reference unshifted run must converge");
    let mut ws2 = ScfWorkspace::allocate(b.norbs);
    let shifted = mopac_core::scf::scf_loop::run_rhf_scf_shift_safe(
        &b,
        &model,
        &mut ws2,
        &ScfOptions {
            max_iter: 2000,
            level_shift_ev: 4.0,
            ..Default::default()
        },
    );
    assert!(shifted.converged, "shifted run must converge");
    let delta = (shifted.total_energy_ev - unshifted.total_energy_ev).abs();
    println!(
        "[CONVERGENCE] ls4 E={:.6} eV, unshifted E={:.6} eV, |d|={:.3e} eV",
        shifted.total_energy_ev, unshifted.total_energy_ev, delta
    );
    assert!(
        delta < 1e-3,
        "explicit level shift selected a different SCF root (|d| = {:.3e} eV)",
        delta
    );
}

#[test]
fn converged_solution_must_be_an_unshifted_fixed_point() {
    let (res, mut ws) = run_pipeline(&CC685_Z, &CC685_C, 200);
    if res.converged {
        let e_strict = strict_rerun_energy(&mut ws, &CC685_Z, &CC685_C);
        let drift = (e_strict - res.total_energy_ev).abs();
        println!(
            "[CONVERGENCE] cc-685 reported E={:.6} eV, strict re-run E={:.6} eV, drift={:.3e} eV",
            res.total_energy_ev, e_strict, drift
        );
        assert!(
            drift < 1e-4,
            "converged=true reported for a solution that is not an unshifted fixed point"
        );
    }
}

#[test]
fn adaptive_ladder_reports_genuine_unshifted_solution() {
    let (b, model) = batch(&CC685_Z, &CC685_C);
    let mut ws = ScfWorkspace::allocate(b.norbs);
    let reference = run_rhf_scf_with_options(
        &b,
        &model,
        &mut ws,
        &ScfOptions {
            max_iter: 2000,
            level_shift_ev: 0.0,
            ..Default::default()
        },
    );
    assert!(reference.converged, "reference unshifted run must converge");

    let (res, _) = run_pipeline(&CC685_Z, &CC685_C, 200);
    if res.converged {
        let delta = (res.total_energy_ev - reference.total_energy_ev).abs();
        println!(
            "[CONVERGENCE] ladder E={:.6} eV vs unshifted reference E={:.6} eV (|d|={:.3e})",
            res.total_energy_ev, reference.total_energy_ev, delta
        );
        assert!(
            delta < 1e-3,
            "ladder reported a different solution than the unshifted reference (|d|={:.3e} eV)",
            delta
        );
    }
}

#[test]
fn true_benzene_still_converges_and_matches_oracle() {
    let (res, _) = run_pipeline(&CC46_Z, &CC46_C, 200);
    assert!(res.converged, "benzene pipeline must converge");
    let h = hof(&res, &CC46_Z);
    // OpenMOPAC PM3 1SCF on this geometry: 23.65973 kcal/mol (measured).
    let oracle = 23.65973;
    let delta = (h - oracle).abs();
    println!(
        "[CONVERGENCE] benzene PM3 rs={:.5} oracle={:.5} |d|={:.5}",
        h, oracle, delta
    );
    assert!(
        delta < 0.05,
        "benzene PM3 regressed vs oracle: |d| = {:.5} kcal/mol",
        delta
    );
}
