//! Density Matrix Assembly & Quantum Electronic Energy Calculation.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").

use crate::types::AlignedMatrix;

/// Compute the closed-shell (RHF) density matrix from molecular orbital eigenvectors $C$.
///
/// Mathematical Formulation:
/// $$P_{\mu\nu} = 2 \sum_{i=1}^{N_{\text{occ}}} C_{\mu i} C_{\nu i}$$
pub fn compute_density_matrix(
    eigenvectors: &AlignedMatrix<f64>,
    nocc: usize,
    density: &mut AlignedMatrix<f64>,
) {
    let norbs = eigenvectors.rows;
    assert_eq!(eigenvectors.cols, norbs);
    assert_eq!(density.rows, norbs);
    assert_eq!(density.cols, norbs);
    assert!(
        nocc <= norbs,
        "Number of occupied orbitals cannot exceed basis size"
    );

    density.fill_zero();

    for mu in 0..norbs {
        for nu in mu..norbs {
            let mut p_val = 0.0;
            for i in 0..nocc {
                p_val += eigenvectors.get(mu, i) * eigenvectors.get(nu, i);
            }
            p_val *= 2.0; // Closed shell doubly-occupied orbitals
            density.set(mu, nu, p_val);
            density.set(nu, mu, p_val);
        }
    }
}

/// Compute total quantum electronic energy in eV.
///
/// Mathematical Formulation:
/// $$E_{\text{elec}} = \frac{1}{2} \sum_{\mu=1}^N \sum_{\nu=1}^N P_{\mu\nu} \left( H_{\mu\nu}^{\text{core}} + F_{\mu\nu} \right)$$
pub fn compute_electronic_energy(
    density: &AlignedMatrix<f64>,
    h_core: &AlignedMatrix<f64>,
    fock: &AlignedMatrix<f64>,
) -> f64 {
    let norbs = density.rows;
    let mut e_elec = 0.0;

    for mu in 0..norbs {
        for nu in 0..norbs {
            e_elec += density.get(mu, nu) * (h_core.get(mu, nu) + fock.get(mu, nu));
        }
    }

    0.5 * e_elec
}

/// Compute maximum absolute difference between two density matrices:
/// $$\Delta P = \max_{\mu,\nu} |P_{\mu\nu}^{\text{new}} - P_{\mu\nu}^{\text{old}}|$$
pub fn max_density_diff(p_new: &AlignedMatrix<f64>, p_old: &AlignedMatrix<f64>) -> f64 {
    let norbs = p_new.rows;
    let mut max_diff = 0.0f64;

    for mu in 0..norbs {
        for nu in 0..norbs {
            let diff = (p_new.get(mu, nu) - p_old.get(mu, nu)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    max_diff
}
