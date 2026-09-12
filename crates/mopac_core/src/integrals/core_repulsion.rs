//! Core-Core Nuclear Repulsion Energy Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Calculates pairwise nuclear repulsion $E_{AB}^{\text{core-core}}$ with Klopman screening and Gaussian expansions.

use crate::parameters::SemiEmpiricalElementParams;
use super::two_electron::dewar_klopman_monopole;

/// Compute pairwise core-core nuclear repulsion energy between atom A and atom B in eV.
///
/// Formulation:
/// $$E_{AB} = Z_A Z_B \gamma_{AB} \left[ 1 + e^{-\alpha_A R} + e^{-\alpha_B R} \right] + \Delta E_{AB}^{\text{Gauss}}$$
///
/// where $\Delta E_{AB}^{\text{Gauss}}$ contains the element-specific Gaussian terms:
/// $$\Delta E_{AB}^{\text{Gauss}} = \frac{Z_A Z_B}{R} \left[ \sum_k a_k^A e^{-b_k^A (R - c_k^A)^2} + \sum_k a_k^B e^{-b_k^B (R - c_k^B)^2} \right]$$
pub fn compute_pair_core_repulsion(
    r_angstrom: f64,
    elem_a: &SemiEmpiricalElementParams,
    elem_b: &SemiEmpiricalElementParams,
) -> f64 {
    if r_angstrom < 1e-10 {
        return 0.0;
    }

    // Monopole two-electron integral (ss|ss) in eV
    let gab = dewar_klopman_monopole(r_angstrom, elem_a.gss, elem_b.gss);

    // Standard exponential screening term
    // For N-H and O-H pairs in AM1, MOPAC applies distance scaling on H
    let h_scale_a = if elem_a.z == 1 && (elem_b.z == 7 || elem_b.z == 8) {
        r_angstrom
    } else {
        1.0
    };
    let h_scale_b = if elem_b.z == 1 && (elem_a.z == 7 || elem_a.z == 8) {
        r_angstrom
    } else {
        1.0
    };

    let exp_a = h_scale_a * (-elem_a.alpha * r_angstrom).exp();
    let exp_b = h_scale_b * (-elem_b.alpha * r_angstrom).exp();
    let base_scale = 1.0 + exp_a + exp_b;

    let base_repulsion = elem_a.core_charge * elem_b.core_charge * gab * base_scale;

    // Gaussian core corrections
    let mut gaussian_sum = 0.0;

    for k in 0..elem_a.num_gaussians {
        let g = &elem_a.gaussians[k];
        let dr = r_angstrom - g.c;
        let exponent = g.b * dr * dr;
        if exponent < 25.0 {
            gaussian_sum += g.a * (-exponent).exp();
        }
    }

    for k in 0..elem_b.num_gaussians {
        let g = &elem_b.gaussians[k];
        let dr = r_angstrom - g.c;
        let exponent = g.b * dr * dr;
        if exponent < 25.0 {
            gaussian_sum += g.a * (-exponent).exp();
        }
    }

    let gaussian_repulsion = (elem_a.core_charge * elem_b.core_charge / r_angstrom) * gaussian_sum;

    base_repulsion + gaussian_repulsion
}

/// Compute pairwise core-core nuclear repulsion energy using PM6 formulation.
///
/// Direct port of OpenMOPAC `ccrep.F90` lines 70-120.
pub fn compute_pair_core_repulsion_pm6(
    r_angstrom: f64,
    elem_a: &SemiEmpiricalElementParams,
    elem_b: &SemiEmpiricalElementParams,
) -> f64 {
    if r_angstrom < 1e-10 {
        return 0.0;
    }
    let gab = dewar_klopman_monopole(r_angstrom, elem_a.gss, elem_b.gss);
    let (alpb, xfac) = crate::parameters::pm6::Pm6Model::get_pair_params(elem_a.z, elem_b.z);

    let z_min = elem_a.z.min(elem_b.z);
    let z_max = elem_a.z.max(elem_b.z);

    let scale = if z_min == 1 && (z_max == 6 || z_max == 7 || z_max == 8) {
        // C-H, N-H, O-H interaction uses r^2
        1.0 + 2.0 * xfac * (-alpb * r_angstrom * r_angstrom).exp()
    } else if xfac > 1e-5 {
        1.0 + 2.0 * xfac * (-alpb * (r_angstrom + 0.0003 * r_angstrom.powi(6))).exp()
    } else {
        1.0 + 10.0 * (-2.18 * r_angstrom).exp()
    };

    let base_repulsion = elem_a.core_charge * elem_b.core_charge * gab * scale;

    // PM6 VdW Gaussian terms
    let mut gaussian_sum = 0.0;
    for k in 0..elem_a.num_gaussians {
        let g = &elem_a.gaussians[k];
        let dr = r_angstrom - g.c;
        let exponent = g.b * dr * dr;
        if exponent < 25.0 {
            gaussian_sum += g.a * (-exponent).exp();
        }
    }
    for k in 0..elem_b.num_gaussians {
        let g = &elem_b.gaussians[k];
        let dr = r_angstrom - g.c;
        let exponent = g.b * dr * dr;
        if exponent < 25.0 {
            gaussian_sum += g.a * (-exponent).exp();
        }
    }

    let gaussian_repulsion = (elem_a.core_charge * elem_b.core_charge / r_angstrom) * gaussian_sum;
    base_repulsion + gaussian_repulsion
}

/// Compute the total core-core repulsion energy for an entire molecular system.
pub fn compute_total_core_repulsion(
    batch: &crate::types::MolecularBatch,
    model: &dyn crate::parameters::ParameterModel,
) -> f64 {
    let mut total_energy = 0.0;

    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let elem_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };

        for j in (i + 1)..batch.natoms {
            let zb = batch.atomic_numbers[j];
            let elem_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };

            let r = batch.distance(i, j);
            total_energy += model.pair_core_repulsion(r, &elem_a, &elem_b);
        }
    }

    total_energy
}
