//! High-Performance Data-Oriented Fock Matrix Builder.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Assembles $F = H^{\text{core}} + G(P)$ directly in contiguous 64-byte aligned memory.

use crate::integrals::two_electron::dewar_klopman_monopole;
use crate::parameters::ParameterModel;
use crate::types::{AlignedMatrix, BasisType, MolecularBatch};

/// Build the Fock matrix $F = H^{\text{core}} + G(P)$.
///
/// Mathematical Formulation:
/// 1. Copy $H^{\text{core}}$ baseline.
/// 2. One-center Coulomb & Exchange contributions ($F^{(1)}$):
///    $$F_{\mu\mu}^{(1)} += \frac{1}{2} P_{\mu\mu} g_{\mu\mu} + \sum_{\lambda \neq \mu} P_{\lambda\lambda} \left( g_{\mu\lambda} - \frac{1}{2} h_{\mu\lambda} \right)$$
/// 3. Two-center Coulomb & Exchange contributions ($F^{(2)}$):
///    $$F_{\mu\mu}^{(2)} += \sum_{B \neq A} q_B \gamma_{AB}(R_{AB})$$
///    $$F_{\mu\nu}^{(2)} -= \frac{1}{2} P_{\mu\nu} \gamma_{AB}(R_{AB}) \quad (\mu \in A, \nu \in B)$$
pub fn build_fock(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    h_core: &AlignedMatrix<f64>,
    density: &AlignedMatrix<f64>,
    fock: &mut AlignedMatrix<f64>,
) {
    assert_eq!(fock.rows, batch.norbs);
    assert_eq!(fock.cols, batch.norbs);

    // 1. Copy H_core into Fock matrix
    fock.data.copy_from_slice(&h_core.data);

    // Precompute total electronic population on each atom: q_A = sum_{mu in A} P_{mu mu}
    let mut atom_populations = vec![0.0; batch.natoms];
    for (i, pop) in atom_populations.iter_mut().enumerate().take(batch.natoms) {
        let orb_start = batch.orbital_offsets[i];
        let num_orbs = batch.basis_types[i].num_orbitals();
        let mut q = 0.0;
        for o in 0..num_orbs {
            q += density.get(orb_start + o, orb_start + o);
        }
        *pop = q;
    }

    // 2. One-center two-electron interactions
    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };

        let orb_start = batch.orbital_offsets[i];
        match batch.basis_types[i] {
            BasisType::S => {
                let p_ss = density.get(orb_start, orb_start);
                let cur = fock.get(orb_start, orb_start);
                // For closed-shell RHF, one-center Coulomb + Exchange: 1/2 * P_ss * g_ss
                fock.set(orb_start, orb_start, cur + 0.5 * p_ss * p_a.gss);
            }
            BasisType::SP => {
                let p_ss = density.get(orb_start, orb_start);
                let p_xx = density.get(orb_start + 1, orb_start + 1);
                let p_yy = density.get(orb_start + 2, orb_start + 2);
                let p_zz = density.get(orb_start + 3, orb_start + 3);

                // s orbital diagonal
                let f_ss = fock.get(orb_start, orb_start)
                    + 0.5 * p_ss * p_a.gss
                    + (p_xx + p_yy + p_zz) * (p_a.gsp - 0.5 * p_a.hsp);
                fock.set(orb_start, orb_start, f_ss);

                // p orbitals diagonal
                let p_p_sum = p_xx + p_yy + p_zz;
                for p_idx in 1..=3 {
                    let idx = orb_start + p_idx;
                    let p_ii = density.get(idx, idx);
                    let other_p = p_p_sum - p_ii;
                    let f_pp = fock.get(idx, idx)
                        + 0.5 * p_ii * p_a.gpp
                        + p_ss * (p_a.gsp - 0.5 * p_a.hsp)
                        + other_p * (p_a.gp2 - 0.25 * (p_a.gpp - p_a.gp2));
                    fock.set(idx, idx, f_pp);
                }
            }
            BasisType::SPD => {
                let p_ss = density.get(orb_start, orb_start);
                let cur = fock.get(orb_start, orb_start);
                fock.set(orb_start, orb_start, cur + 0.5 * p_ss * p_a.gss);
            }
        }
    }

    // 3. Two-center two-electron interactions (Coulomb and Exchange)
    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };
        let orb_a_start = batch.orbital_offsets[i];
        let num_a = batch.basis_types[i].num_orbitals();

        for (j, &q_b) in atom_populations.iter().enumerate().take(batch.natoms) {
            if i == j {
                continue;
            }
            let zb = batch.atomic_numbers[j];
            let p_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };
            let orb_b_start = batch.orbital_offsets[j];
            let num_b = batch.basis_types[j].num_orbitals();

            let r_ab = batch.distance(i, j);
            let gamma_ab = dewar_klopman_monopole(r_ab, p_a.gss, p_b.gss);

            // Two-center Coulomb: repulsion from total electronic cloud on atom B
            for oa in 0..num_a {
                let idx_a = orb_a_start + oa;
                let cur = fock.get(idx_a, idx_a);
                fock.set(idx_a, idx_a, cur + q_b * gamma_ab);
            }

            // Two-center Exchange: -1/2 * P_{mu nu} * gamma_{AB}
            for oa in 0..num_a {
                let idx_a = orb_a_start + oa;
                for ob in 0..num_b {
                    let idx_b = orb_b_start + ob;
                    let p_ab = density.get(idx_a, idx_b);
                    let cur = fock.get(idx_a, idx_b);
                    fock.set(idx_a, idx_b, cur - 0.5 * p_ab * gamma_ab);
                }
            }
        }
    }
}
