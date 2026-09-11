//! Core One-Electron Hamiltonian Matrix Construction ($H^{\text{core}}$).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Evaluates kinetic energy, atomic core energies, and nuclear attraction integrals.

use crate::integrals::overlap::overlap_1s_1s;
use crate::integrals::two_electron::dewar_klopman_monopole;
use crate::parameters::ParameterModel;
use crate::types::{AlignedMatrix, BasisType, MolecularBatch};

/// Build the one-electron core Hamiltonian matrix $H^{\text{core}}$.
///
/// Mathematical Formulation:
/// 1. Diagonal one-center elements:
///    $$H_{\mu\mu} = U_{\mu\mu} - \sum_{B \neq A} Z_B^{\text{core}} \gamma_{AB}(R_{AB})$$
/// 2. Off-diagonal two-center elements (Resonance integrals):
///    $$H_{\mu\nu} = \frac{1}{2} (\beta_\mu^A + \beta_\nu^B) S_{\mu\nu}(R_{AB})$$
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

        for j in (i + 1)..batch.natoms {
            let zb = batch.atomic_numbers[j];
            let p_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };
            let orb_b_start = batch.orbital_offsets[j];

            let r_ab = batch.distance(i, j);

            // For s-s interactions (e.g. H-H or s-orbital pairs):
            let s_ss = overlap_1s_1s(r_ab, p_a.zs, p_b.zs);
            let h_ss = 0.5 * (p_a.betas + p_b.betas) * s_ss;

            h_core.set(orb_a_start, orb_b_start, h_ss);
            h_core.set(orb_b_start, orb_a_start, h_ss);
        }
    }
}
