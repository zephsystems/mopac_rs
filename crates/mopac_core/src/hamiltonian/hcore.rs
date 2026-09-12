//! Core One-Electron Hamiltonian Matrix Construction ($H^{\text{core}}$).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Evaluates kinetic energy, atomic core energies, and nuclear attraction integrals.

use crate::integrals::overlap::compute_diatomic_overlap_matrix_9x9;
use crate::integrals::two_electron::dewar_klopman_monopole;
use crate::parameters::ParameterModel;
use crate::types::{AlignedMatrix, BasisType, MolecularBatch};

/// Build the one-electron core Hamiltonian matrix $H^{\text{core}}$.
///
/// Mathematical Formulation:
/// 1. Diagonal one-center elements:
///    $$H_{\mu\mu} = U_{\mu\mu} - \sum_{B \neq A} Z_B^{\text{core}} \gamma_{AB}(R_{AB})$$
/// 2. Off-diagonal two-center elements (Resonance integrals):
///    $$H_{\mu\nu} = \frac{1}{2} (\beta_\mu^A + \beta_\nu^B) S_{\mu\nu}(\vec{R}_{AB})$$
pub fn build_hcore(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    h_core: &mut AlignedMatrix<f64>,
) {
    assert_eq!(h_core.rows, batch.norbs);
    assert_eq!(h_core.cols, batch.norbs);
    h_core.fill_zero();

    // 1. One-center diagonal elements and electron-nuclear attraction
    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };

        let orb_start = batch.orbital_offsets[i];
        let b_type = batch.basis_types[i];

        // Diagonal kinetic + nuclear core potential
        match b_type {
            BasisType::S => {
                h_core.set(orb_start, orb_start, p_a.uss);
            }
            BasisType::SP => {
                h_core.set(orb_start, orb_start, p_a.uss);
                h_core.set(orb_start + 1, orb_start + 1, p_a.upp);
                h_core.set(orb_start + 2, orb_start + 2, p_a.upp);
                h_core.set(orb_start + 3, orb_start + 3, p_a.upp);
            }
            BasisType::SPD => {
                h_core.set(orb_start, orb_start, p_a.uss);
                for k in 1..=3 {
                    h_core.set(orb_start + k, orb_start + k, p_a.upp);
                }
                for k in 4..=8 {
                    h_core.set(orb_start + k, orb_start + k, p_a.udd);
                }
            }
        }

        // Add electron-nuclear attraction from all other atoms B != A:
        // V_{mu mu, B} = -Z_B * gamma_{AB}
        for j in 0..batch.natoms {
            if i == j {
                continue;
            }
            let zb = batch.atomic_numbers[j];
            let p_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };

            let r_ab = batch.distance(i, j);
            let gamma_ab = dewar_klopman_monopole(r_ab, p_a.gss, p_b.gss);
            let v_nuc = -p_b.core_charge * gamma_ab;

            for orb in 0..b_type.num_orbitals() {
                let idx = orb_start + orb;
                let cur = h_core.get(idx, idx);
                h_core.set(idx, idx, cur + v_nuc);
            }
        }
    }

    // 2. Off-diagonal two-center resonance elements: H_mu nu = 1/2 (beta_mu + beta_nu) * S_mu nu
    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };
        let orb_a_start = batch.orbital_offsets[i];
        let norb_a = batch.basis_types[i].num_orbitals();

        for j in (i + 1)..batch.natoms {
            let zb = batch.atomic_numbers[j];
            let p_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };
            let orb_b_start = batch.orbital_offsets[j];
            let norb_b = batch.basis_types[j].num_orbitals();

            let r_ab = batch.distance(i, j);
            if r_ab < 1e-10 {
                continue;
            }

            let dx = batch.x[j] - batch.x[i];
            let dy = batch.y[j] - batch.y[i];
            let dz = batch.z[j] - batch.z[i];

            let mut s_mat = [[0.0f64; 9]; 9];
            compute_diatomic_overlap_matrix_9x9(
                za, zb, norb_a, norb_b, &p_a, &p_b, dx, dy, dz, r_ab, &mut s_mat,
            );

            let beta_a = [
                p_a.betas, p_a.betap, p_a.betap, p_a.betap, p_a.betad, p_a.betad, p_a.betad,
                p_a.betad, p_a.betad,
            ];
            let beta_b = [
                p_b.betas, p_b.betap, p_b.betap, p_b.betap, p_b.betad, p_b.betad, p_b.betad,
                p_b.betad, p_b.betad,
            ];

            for oa in 0..norb_a {
                let idx_a = orb_a_start + oa;
                for ob in 0..norb_b {
                    let idx_b = orb_b_start + ob;
                    let h_res = 0.5 * (beta_a[oa] + beta_b[ob]) * s_mat[oa][ob];
                    h_core.set(idx_a, idx_b, h_res);
                    h_core.set(idx_b, idx_a, h_res);
                }
            }
        }
    }
}

/// Build the one-electron core Hamiltonian matrix $H^{\text{core}}$ with full NDDO electron-nuclear attraction.
///
/// Axiomatic formulation matching OpenMOPAC `hcore.F90`:
/// 1. Diagonal one-center kinetic and nuclear potential $U_{\mu\mu}$.
/// 2. Off-diagonal two-center resonance elements: $H_{\mu\nu} = \frac{1}{2} (\beta_\mu + \beta_\nu) S_{\mu\nu}$.
/// 3. Rotated electron-nuclear attractions $E_{1B}$ and $E_{2A}$ from precomputed diatomic multipoles.
pub fn build_hcore_nddo(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    pairs: &[crate::integrals::multipoles::DiatomicPairIntegrals],
    h_core: &mut AlignedMatrix<f64>,
) {
    assert_eq!(h_core.rows, batch.norbs);
    assert_eq!(h_core.cols, batch.norbs);
    h_core.fill_zero();

    // 1. One-center diagonal elements
    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };

        let orb_start = batch.orbital_offsets[i];
        let b_type = batch.basis_types[i];

        match b_type {
            BasisType::S => {
                h_core.set(orb_start, orb_start, p_a.uss);
            }
            BasisType::SP => {
                h_core.set(orb_start, orb_start, p_a.uss);
                h_core.set(orb_start + 1, orb_start + 1, p_a.upp);
                h_core.set(orb_start + 2, orb_start + 2, p_a.upp);
                h_core.set(orb_start + 3, orb_start + 3, p_a.upp);
            }
            BasisType::SPD => {
                h_core.set(orb_start, orb_start, p_a.uss);
                for k in 1..=3 {
                    h_core.set(orb_start + k, orb_start + k, p_a.upp);
                }
                for k in 4..=8 {
                    h_core.set(orb_start + k, orb_start + k, p_a.udd);
                }
            }
        }
    }

    // 2. Add rotated electron-nuclear attractions E_1B and E_2A
    crate::integrals::multipoles::apply_electron_nuclear_attractions(pairs, h_core);

    // 3. Off-diagonal two-center resonance elements: H_mu nu = 1/2 (beta_mu + beta_nu) * S_mu nu
    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };
        let orb_a_start = batch.orbital_offsets[i];
        let norb_a = batch.basis_types[i].num_orbitals();

        for j in (i + 1)..batch.natoms {
            let zb = batch.atomic_numbers[j];
            let p_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };
            let orb_b_start = batch.orbital_offsets[j];
            let norb_b = batch.basis_types[j].num_orbitals();

            let r_ab = batch.distance(i, j);
            if r_ab < 1e-10 {
                continue;
            }

            let dx = batch.x[j] - batch.x[i];
            let dy = batch.y[j] - batch.y[i];
            let dz = batch.z[j] - batch.z[i];

            let mut s_mat = [[0.0f64; 9]; 9];
            compute_diatomic_overlap_matrix_9x9(
                za, zb, norb_a, norb_b, &p_a, &p_b, dx, dy, dz, r_ab, &mut s_mat,
            );

            let beta_a = [
                p_a.betas, p_a.betap, p_a.betap, p_a.betap, p_a.betad, p_a.betad, p_a.betad,
                p_a.betad, p_a.betad,
            ];
            let beta_b = [
                p_b.betas, p_b.betap, p_b.betap, p_b.betap, p_b.betad, p_b.betad, p_b.betad,
                p_b.betad, p_b.betad,
            ];

            for oa in 0..norb_a {
                let idx_a = orb_a_start + oa;
                for ob in 0..norb_b {
                    let idx_b = orb_b_start + ob;
                    let h_res = 0.5 * (beta_a[oa] + beta_b[ob]) * s_mat[oa][ob];
                    h_core.set(idx_a, idx_b, h_res);
                    h_core.set(idx_b, idx_a, h_res);
                }
            }
        }
    }
}
