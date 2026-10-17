//! Localized Orbital Minimization and 2x2 Jacobi Rotations for MOZYME.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Performs pairwise unitary Jacobi rotations between occupied and virtual LMOs
//! matching OpenMOPAC `diagg2.F90` and `locmin.F90`.

use super::types::{Lmo, MozymeOptions};
use crate::types::{AlignedMatrix, MolecularBatch};

/// Compute the expectation value and off-diagonal interaction:
/// F_ii = <phi_i | F | phi_i>, F_aa = <phi_a | F | phi_a>, F_ia = <phi_i | F | phi_a>
pub fn evaluate_fock_elements(
    lmo_i: &Lmo,
    lmo_a: &Lmo,
    fock: &AlignedMatrix<f64>,
) -> (f64, f64, f64) {
    let mut f_ii = 0.0;
    for (idx_mu, &mu) in lmo_i.ao_indices.iter().enumerate() {
        let c_mu = lmo_i.coeffs[idx_mu];
        for (idx_nu, &nu) in lmo_i.ao_indices.iter().enumerate() {
            let c_nu = lmo_i.coeffs[idx_nu];
            f_ii += c_mu * fock.get(mu, nu) * c_nu;
        }
    }

    let mut f_aa = 0.0;
    for (idx_mu, &mu) in lmo_a.ao_indices.iter().enumerate() {
        let c_mu = lmo_a.coeffs[idx_mu];
        for (idx_nu, &nu) in lmo_a.ao_indices.iter().enumerate() {
            let c_nu = lmo_a.coeffs[idx_nu];
            f_aa += c_mu * fock.get(mu, nu) * c_nu;
        }
    }

    let mut f_ia = 0.0;
    for (idx_mu, &mu) in lmo_i.ao_indices.iter().enumerate() {
        let c_mu = lmo_i.coeffs[idx_mu];
        for (idx_nu, &nu) in lmo_a.ao_indices.iter().enumerate() {
            let c_nu = lmo_a.coeffs[idx_nu];
            f_ia += c_mu * fock.get(mu, nu) * c_nu;
        }
    }

    (f_ii, f_aa, f_ia)
}

/// Perform a 2x2 unitary Jacobi rotation between occupied LMO phi_i and virtual LMO phi_a.
pub fn rotate_lmo_pair(
    batch: &MolecularBatch,
    lmo_i: &mut Lmo,
    lmo_a: &mut Lmo,
    f_ii: f64,
    f_aa: f64,
    f_ia: f64,
    damping: f64,
) {
    if f_ia.abs() < 1e-12 {
        return;
    }

    // tan(2 * theta) = 2 * F_ia / (F_ii - F_aa), requiring F_ii - F_aa < 0 for minimization
    let delta = if (f_ii - f_aa) < -0.05 {
        f_ii - f_aa
    } else {
        -0.05
    };
    let mut theta = 0.5 * (2.0 * f_ia / delta).atan();
    theta *= damping;

    let cos_t = theta.cos();
    let sin_t = theta.sin();

    // Union of participating AO indices
    let mut all_aos = lmo_i.ao_indices.clone();
    for &ao in &lmo_a.ao_indices {
        if !all_aos.contains(&ao) {
            all_aos.push(ao);
        }
    }
    all_aos.sort_unstable();

    // Reconstruct full coefficient vectors across the union
    let mut c_i_full = Vec::with_capacity(all_aos.len());
    let mut c_a_full = Vec::with_capacity(all_aos.len());

    for &ao in &all_aos {
        let ci = lmo_i
            .ao_indices
            .iter()
            .position(|&x| x == ao)
            .map(|idx| lmo_i.coeffs[idx])
            .unwrap_or(0.0);
        let ca = lmo_a
            .ao_indices
            .iter()
            .position(|&x| x == ao)
            .map(|idx| lmo_a.coeffs[idx])
            .unwrap_or(0.0);
        c_i_full.push(ci);
        c_a_full.push(ca);
    }

    // Normalize ci
    let norm_i: f64 = c_i_full.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm_i > 1e-14 {
        for val in &mut c_i_full {
            *val /= norm_i;
        }
    }

    // Project ca strictly orthogonal to ci
    let dot_ia: f64 = c_i_full
        .iter()
        .zip(c_a_full.iter())
        .map(|(&a, &b)| a * b)
        .sum();
    for k in 0..all_aos.len() {
        c_a_full[k] -= dot_ia * c_i_full[k];
    }
    let norm_a: f64 = c_a_full.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-12 {
        return;
    }
    for val in &mut c_a_full {
        *val /= norm_a;
    }

    // Exact 2D unitary rotation with insignificant coefficient pruning (< 1e-6)
    let mut pruned_i_aos = Vec::new();
    let mut pruned_i_coeffs = Vec::new();
    let mut pruned_a_aos = Vec::new();
    let mut pruned_a_coeffs = Vec::new();

    for k in 0..all_aos.len() {
        let ci = cos_t * c_i_full[k] + sin_t * c_a_full[k];
        let ca = -sin_t * c_i_full[k] + cos_t * c_a_full[k];
        if ci.abs() >= 1e-6 {
            pruned_i_aos.push(all_aos[k]);
            pruned_i_coeffs.push(ci);
        }
        if ca.abs() >= 1e-6 {
            pruned_a_aos.push(all_aos[k]);
            pruned_a_coeffs.push(ca);
        }
    }

    if pruned_i_aos.is_empty() {
        let (max_k, &max_val) = c_i_full
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap();
        pruned_i_aos.push(all_aos[max_k]);
        pruned_i_coeffs.push(max_val);
    }
    if pruned_a_aos.is_empty() {
        let (max_k, &max_val) = c_a_full
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap();
        pruned_a_aos.push(all_aos[max_k]);
        pruned_a_coeffs.push(max_val);
    }

    let norm_i_pruned: f64 = pruned_i_coeffs.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm_i_pruned > 1e-14 {
        for val in &mut pruned_i_coeffs {
            *val /= norm_i_pruned;
        }
    }
    let norm_a_pruned: f64 = pruned_a_coeffs.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm_a_pruned > 1e-14 {
        for val in &mut pruned_a_coeffs {
            *val /= norm_a_pruned;
        }
    }

    lmo_i.ao_indices = pruned_i_aos;
    lmo_i.coeffs = pruned_i_coeffs;
    lmo_i.update_atom_indices_from_aos(batch);

    lmo_a.ao_indices = pruned_a_aos;
    lmo_a.coeffs = pruned_a_coeffs;
    lmo_a.update_atom_indices_from_aos(batch);
}

/// Execute one complete 2x2 Jacobi rotation sweep across all interacting (occ, virt) LMO pairs.
pub fn execute_jacobi_sweep(
    batch: &MolecularBatch,
    occupied_lmos: &mut [Lmo],
    virtual_lmos: &mut [Lmo],
    fock: &AlignedMatrix<f64>,
    options: &MozymeOptions,
) -> (f64, usize) {
    let mut max_gradient = 0.0f64;
    let mut rotation_count = 0usize;

    for occ in occupied_lmos.iter_mut() {
        for virt in virtual_lmos.iter_mut() {
            // Check spatial cutoff distance between LMO centers
            let dist = occ.distance_to(virt, &batch.x, &batch.y, &batch.z);
            if dist > options.cutoff_distance && !occ.shares_atom(virt) {
                continue;
            }

            let (f_ii, f_aa, f_ia) = evaluate_fock_elements(occ, virt, fock);
            let grad = f_ia.abs();
            if grad > max_gradient {
                max_gradient = grad;
            }

            if grad > options.jacobi_tol {
                rotate_lmo_pair(batch, occ, virt, f_ii, f_aa, f_ia, options.damping);
                rotation_count += 1;
            }
        }
    }

    (max_gradient, rotation_count)
}
