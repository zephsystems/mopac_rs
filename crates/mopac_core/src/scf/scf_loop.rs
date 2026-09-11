//! Self-Consistent Field (SCF) Solver Loop.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Executes the Roothaan-Hall iteration cycle with 0 dynamic heap allocations.

use crate::fock::build_fock;
use crate::hamiltonian::build_hcore;
use crate::integrals::core_repulsion::compute_total_core_repulsion;
use crate::parameters::ParameterModel;
use crate::scf::density::{compute_density_matrix, compute_electronic_energy, max_density_diff};
use crate::scf::eigensolver::diagonalize_symmetric;
use crate::types::{MolecularBatch, ScfWorkspace};

/// Summary result of a completed Self-Consistent Field calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct ScfResult {
    pub converged: bool,
    pub iterations: usize,
    pub total_energy_ev: f64,
    pub electronic_energy_ev: f64,
    pub nuclear_repulsion_ev: f64,
    pub homo_energy_ev: f64,
    pub lumo_energy_ev: f64,
}

/// Run a complete closed-shell (RHF) Self-Consistent Field calculation.
///
/// Guaranteed to perform **zero heap allocations (`0 malloc`)** during the iterative loop.
pub fn run_rhf_scf(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    ws: &mut ScfWorkspace,
    max_iter: usize,
    energy_tol_ev: f64,
    density_tol: f64,
) -> ScfResult {
    assert_eq!(ws.norbs, batch.norbs);

    // 1. Calculate total valence electrons and number of occupied orbitals
    let mut total_valence_elecs = 0.0;
    for &z in &batch.atomic_numbers {
        if let Some(p) = model.get_element(z) {
            total_valence_elecs += p.core_charge;
        }
    }

    let nocc = (total_valence_elecs.round() as usize) / 2;
    assert!(nocc > 0, "System must have at least one electron pair for closed-shell RHF");

    // 2. Nuclear-nuclear core repulsion energy
    let enuc = compute_total_core_repulsion(batch, model);

    // 3. Build one-electron core Hamiltonian H_core
    build_hcore(batch, model, &mut ws.h_core);

    // 4. Initial guess: diagonalize H_core to generate initial density P^(0)
    diagonalize_symmetric(&ws.h_core, &mut ws.eigenvalues, &mut ws.eigenvectors);
    compute_density_matrix(&ws.eigenvectors, nocc, &mut ws.density);

    let mut prev_energy = 0.0f64;
    let mut converged = false;
    let mut iters_done = 0;
    let damping = 0.5; // Simple linear damping

    // 5. SCF Iteration Loop (ZERO dynamic heap allocations)
    for iter in 1..=max_iter {
        iters_done = iter;

        // Build Fock matrix F = H_core + G(P)
        build_fock(batch, model, &ws.h_core, &ws.density, &mut ws.fock);

        // Compute electronic energy
        let e_elec = compute_electronic_energy(&ws.density, &ws.h_core, &ws.fock);
        let e_total = e_elec + enuc;

        // Diagonalize Fock matrix: F C = C epsilon
        diagonalize_symmetric(&ws.fock, &mut ws.eigenvalues, &mut ws.eigenvectors);

        // Compute candidate new density into temporary buffer ws.tmp1
        compute_density_matrix(&ws.eigenvectors, nocc, &mut ws.tmp1);

        // Check convergence
        let delta_e = (e_total - prev_energy).abs();
        let delta_p = max_density_diff(&ws.tmp1, &ws.density);

        if iter > 1 && delta_e < energy_tol_ev && delta_p < density_tol {
            converged = true;
            break;
        }

        prev_energy = e_total;

        // Apply density damping: P = (1 - damping) * P_new + damping * P_old
        for i in 0..ws.norbs {
            for j in 0..ws.norbs {
                let p_new = ws.tmp1.get(i, j);
                let p_old = ws.density.get(i, j);
                ws.density.set(i, j, (1.0 - damping) * p_new + damping * p_old);
            }
        }
    }

    let homo = ws.eigenvalues[nocc - 1];
    let lumo = if nocc < ws.norbs {
        ws.eigenvalues[nocc]
    } else {
        0.0
    };

    let e_elec_final = compute_electronic_energy(&ws.density, &ws.h_core, &ws.fock);

    ScfResult {
        converged,
        iterations: iters_done,
        total_energy_ev: e_elec_final + enuc,
        electronic_energy_ev: e_elec_final,
        nuclear_repulsion_ev: enuc,
        homo_energy_ev: homo,
        lumo_energy_ev: lumo,
    }
}
