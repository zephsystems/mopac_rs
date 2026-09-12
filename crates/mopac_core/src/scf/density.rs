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

/// Compute spin density matrix from molecular orbital eigenvectors $C$ for a single spin channel (alpha or beta).
///
/// Mathematical Formulation:
/// $$P^\sigma_{\mu\nu} = \sum_{i=1}^{N_\sigma} C^\sigma_{\mu i} C^\sigma_{\nu i}$$
pub fn compute_spin_density_matrix(
    eigenvectors: &AlignedMatrix<f64>,
    n_elecs: usize,
    density: &mut AlignedMatrix<f64>,
) {
    let norbs = eigenvectors.rows;
    assert_eq!(eigenvectors.cols, norbs);
    assert_eq!(density.rows, norbs);
    assert_eq!(density.cols, norbs);
    assert!(
        n_elecs <= norbs,
        "Number of spin electrons cannot exceed basis size"
    );

    density.fill_zero();

    for mu in 0..norbs {
        for nu in mu..norbs {
            let mut p_val = 0.0;
            for i in 0..n_elecs {
                p_val += eigenvectors.get(mu, i) * eigenvectors.get(nu, i);
            }
            density.set(mu, nu, p_val);
            density.set(nu, mu, p_val);
        }
    }
}

/// Compute open-shell UHF electronic energy in eV.
///
/// $$E_{\text{elec}} = \frac{1}{2} \text{Tr}\left[ P^\alpha (H^{\text{core}} + F^\alpha) + P^\beta (H^{\text{core}} + F^\beta) \right]$$
pub fn compute_uhf_electronic_energy(
    p_alpha: &AlignedMatrix<f64>,
    p_beta: &AlignedMatrix<f64>,
    h_core: &AlignedMatrix<f64>,
    fock_alpha: &AlignedMatrix<f64>,
    fock_beta: &AlignedMatrix<f64>,
) -> f64 {
    let norbs = p_alpha.rows;
    let mut e_elec = 0.0;

    for mu in 0..norbs {
        for nu in 0..norbs {
            let pa = p_alpha.get(mu, nu);
            let pb = p_beta.get(mu, nu);
            let h = h_core.get(mu, nu);
            let fa = fock_alpha.get(mu, nu);
            let fb = fock_beta.get(mu, nu);
            e_elec += pa * (h + fa) + pb * (h + fb);
        }
    }

    0.5 * e_elec
}

/// Compute expectation value $\langle S^2 \rangle$ and spin contamination in UHF.
///
/// In ZDO basis ($S = I$):
/// $$\langle S^2 \rangle = S_z (S_z + 1) + N_\beta - \sum_{\mu, \nu} P^\alpha_{\mu\nu} P^\beta_{\nu\mu}$$
/// where $S_z = (N_\alpha - N_\beta)/2$.
pub fn compute_uhf_s_squared(
    p_alpha: &AlignedMatrix<f64>,
    p_beta: &AlignedMatrix<f64>,
    n_alpha: usize,
    n_beta: usize,
) -> f64 {
    let norbs = p_alpha.rows;
    let sz = (n_alpha as f64 - n_beta as f64) * 0.5;
    let mut overlap_trace = 0.0;
    for mu in 0..norbs {
        for nu in 0..norbs {
            overlap_trace += p_alpha.get(mu, nu) * p_beta.get(nu, mu);
        }
    }
    sz * (sz + 1.0) + (n_beta as f64) - overlap_trace
}
