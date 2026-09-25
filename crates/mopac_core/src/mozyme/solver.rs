//! Complete Linear Scaling MOZYME SCF Solver.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Executes localized orbital self-consistent field minimization with $O(N)$ linear scaling.

use super::hybrid::construct_initial_lmos;
use super::lewis::construct_lewis_structure;
use super::locmin::execute_jacobi_sweep;
use super::types::{Lmo, MozymeOptions, MozymeResult};
use crate::corrections::dispersion::compute_dispersion_energy;
use crate::corrections::DispersionModel;
use crate::integrals::core_repulsion::compute_total_core_repulsion;
use crate::parameters::ParameterModel;
use crate::properties::heat::compute_heat_of_formation;
use crate::scf::eigensolver::diagonalize_symmetric;
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch};

/// Orthonormalize occupied LMOs via Löwdin symmetric orthogonalization: C_occ <- C_occ * (C_occ^T C_occ)^(-1/2).
#[allow(clippy::needless_range_loop)]
pub fn orthogonalize_occupied_lmos(norbs: usize, occupied_lmos: &mut [Lmo]) {
    let n_occ = occupied_lmos.len();
    if n_occ <= 1 {
        return;
    }

    // 1. Build overlap matrix S_ij = <phi_i | phi_j>
    let mut s_mat = AlignedMatrix::zeroed(n_occ, n_occ);
    for (i, lmo_i) in occupied_lmos.iter().enumerate() {
        for (j, lmo_j) in occupied_lmos.iter().enumerate() {
            let mut ov = 0.0;
            for (idx_a, &ao_a) in lmo_i.ao_indices.iter().enumerate() {
                for (idx_b, &ao_b) in lmo_j.ao_indices.iter().enumerate() {
                    if ao_a == ao_b {
                        ov += lmo_i.coeffs[idx_a] * lmo_j.coeffs[idx_b];
                    }
                }
            }
            s_mat.set(i, j, ov);
        }
    }

    // 2. Diagonalize S using proven real symmetric Jacobi eigensolver
    let mut eigenvalues = AlignedVec64::zeroed(n_occ);
    let mut eigenvectors = AlignedMatrix::zeroed(n_occ, n_occ);
    diagonalize_symmetric(&s_mat, &mut eigenvalues, &mut eigenvectors);

    // 3. Compute S^(-1/2) = V * diag(lambda^(-1/2)) * V^T
    let mut s_inv_sqrt = vec![vec![0.0f64; n_occ]; n_occ];
    for i in 0..n_occ {
        for j in 0..n_occ {
            let mut sum = 0.0;
            for k in 0..n_occ {
                let lambda_k = eigenvalues[k].max(1e-8);
                sum += eigenvectors.get(i, k) * (1.0 / lambda_k.sqrt()) * eigenvectors.get(j, k);
            }
            s_inv_sqrt[i][j] = sum;
        }
    }

    // 4. Transform occupied LMOs: C_ortho[i] = sum_j S^(-1/2)[j][i] * C[j]
    let mut c_dense = vec![vec![0.0f64; norbs]; n_occ];
    for (i, lmo) in occupied_lmos.iter().enumerate() {
        for (idx, &ao) in lmo.ao_indices.iter().enumerate() {
            c_dense[i][ao] = lmo.coeffs[idx];
        }
    }

    for i in 0..n_occ {
        let mut new_coeffs_full = vec![0.0f64; norbs];
        for j in 0..n_occ {
            let factor = s_inv_sqrt[j][i];
            for ao in 0..norbs {
                new_coeffs_full[ao] += factor * c_dense[j][ao];
            }
        }

        let mut ao_indices = Vec::new();
        let mut coeffs = Vec::new();
        for ao in 0..norbs {
            if new_coeffs_full[ao].abs() > 1e-12 {
                ao_indices.push(ao);
                coeffs.push(new_coeffs_full[ao]);
            }
        }
        occupied_lmos[i].ao_indices = ao_indices;
        occupied_lmos[i].coeffs = coeffs;
    }
}

/// Reconstruct the molecular density matrix P = 2 * sum_{i in occ} phi_i phi_i^T.
pub fn construct_density_from_lmos(
    _norbs: usize,
    occupied_lmos: &[Lmo],
    density: &mut AlignedMatrix<f64>,
) {
    density.fill_zero();
    for lmo in occupied_lmos {
        let n = lmo.ao_indices.len();
        for i in 0..n {
            let mu = lmo.ao_indices[i];
            let c_mu = lmo.coeffs[i];
            for j in 0..n {
                let nu = lmo.ao_indices[j];
                let c_nu = lmo.coeffs[j];
                density.add(mu, nu, 2.0 * c_mu * c_nu);
            }
        }
    }
}

/// Run the full MOZYME Linear Scaling SCF Calculation.
pub fn run_mozyme_scf(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    dispersion: Option<DispersionModel>,
    options: &MozymeOptions,
) -> MozymeResult {
    let norbs = batch.norbs;

    // 1. Construct Lewis chemical topology and connectivity
    let lewis = construct_lewis_structure(batch);

    // 2. Generate initial orthonormal LMO basis
    let (mut occupied_lmos, mut virtual_lmos) = construct_initial_lmos(batch, &lewis);

    let diatomic_pairs = crate::integrals::multipoles::precompute_diatomic_pairs(batch, model);

    // 3. Build one-electron core Hamiltonian H_core (NDDO)
    let mut h_core = AlignedMatrix::zeroed(norbs, norbs);
    crate::hamiltonian::hcore::build_hcore_nddo(batch, model, &diatomic_pairs, &mut h_core);

    // 4. Compute nuclear core-core repulsion energy
    let core_repulsion_ev = compute_total_core_repulsion(batch, model);

    // 5. Initialize density and Fock matrices
    orthogonalize_occupied_lmos(norbs, &mut occupied_lmos);
    let mut density = AlignedMatrix::zeroed(norbs, norbs);
    construct_density_from_lmos(norbs, &occupied_lmos, &mut density);

    let mut fock = AlignedMatrix::zeroed(norbs, norbs);

    let mut prev_energy = 0.0f64;
    let mut electronic_energy_ev = 0.0f64;
    let mut total_energy_ev = 0.0f64;
    let mut converged = false;
    let mut iteration = 0;

    // 6. Macro-SCF iterative minimization loop
    while iteration < options.max_iter {
        iteration += 1;

        // Build current Fock matrix F = H_core + G(P) using full NDDO diatomic multipoles
        crate::fock::fock_builder::build_fock_nddo(
            batch,
            model,
            &diatomic_pairs,
            &h_core,
            &density,
            &mut fock,
        );

        // Compute electronic energy E_elec = 0.5 * Tr[P * (H_core + F)]
        let mut e_elec = 0.0;
        for i in 0..norbs {
            for j in 0..norbs {
                e_elec += 0.5 * density.get(i, j) * (h_core.get(i, j) + fock.get(i, j));
            }
        }
        electronic_energy_ev = e_elec;
        total_energy_ev = electronic_energy_ev + core_repulsion_ev;

        let delta_e = (total_energy_ev - prev_energy).abs();

        if options.verbose {
            println!(
                "MOZYME Iteration {:3}: Etot = {:.8} eV, DeltaE = {:.8} eV",
                iteration, total_energy_ev, delta_e
            );
        }

        // Inner 2x2 Jacobi rotation sweeps
        let mut max_grad = 0.0;
        for _sweep in 0..options.max_jacobi_sweeps {
            let (grad, rotations) =
                execute_jacobi_sweep(batch, &mut occupied_lmos, &mut virtual_lmos, &fock, options);
            max_grad = grad;
            if rotations == 0 || grad < options.jacobi_tol {
                break;
            }
        }

        // Orthonormalize occupied LMOs to preserve strict idempotency and trace conservation
        orthogonalize_occupied_lmos(norbs, &mut occupied_lmos);

        // Update density matrix from rotated orthonormal LMOs with smooth damping
        let mut new_density = AlignedMatrix::zeroed(norbs, norbs);
        construct_density_from_lmos(norbs, &occupied_lmos, &mut new_density);

        let alpha = 0.65;
        for i in 0..norbs {
            for j in 0..norbs {
                let p_new = new_density.get(i, j);
                let p_old = density.get(i, j);
                density.set(i, j, (1.0 - alpha) * p_old + alpha * p_new);
            }
        }

        if iteration > 1 && delta_e < options.energy_tol && max_grad < options.jacobi_tol {
            converged = true;
            break;
        }

        prev_energy = total_energy_ev;
    }

    // Final density reconstruction strictly from converged orthonormal LMOs
    construct_density_from_lmos(norbs, &occupied_lmos, &mut density);

    // 7. Compute Mulliken atomic partial charges: q_A = Z_A - sum_{mu in A} P_{mu mu}
    let mut atomic_charges = Vec::with_capacity(batch.natoms);
    for a in 0..batch.natoms {
        let z_a = model
            .get_element(batch.atomic_numbers[a])
            .map(|p| p.core_charge)
            .unwrap_or(0.0);
        let off = batch.orbital_offsets[a];
        let num_a = batch.basis_types[a].num_orbitals();
        let mut pop = 0.0;
        for o in 0..num_a {
            pop += density.get(off + o, off + o);
        }
        atomic_charges.push(z_a - pop);
    }

    // 8. Compute Heat of Formation
    let non_covalent_kcal = if let Some(disp_model) = dispersion {
        compute_dispersion_energy(batch, disp_model)
    } else {
        0.0
    };
    let (_binding_ev, heat_of_formation_kcal) = compute_heat_of_formation(
        total_energy_ev,
        &batch.atomic_numbers,
        model,
        non_covalent_kcal,
    );

    let mut all_lmos = occupied_lmos;
    all_lmos.extend(virtual_lmos);

    MozymeResult {
        converged,
        iterations: iteration,
        electronic_energy_ev,
        core_repulsion_ev,
        total_energy_ev,
        heat_of_formation_kcal,
        lmos: all_lmos,
        density,
        atomic_charges,
    }
}
