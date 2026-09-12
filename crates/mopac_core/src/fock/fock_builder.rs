//! High-Performance Data-Oriented Fock Matrix Builder.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Assembles $F = H^{\text{core}} + G(P)$ directly in contiguous 64-byte aligned memory.

use crate::integrals::two_electron::dewar_klopman_monopole;
use crate::parameters::ParameterModel;
use crate::types::{AlignedMatrix, BasisType, MolecularBatch};

#[inline(always)]
fn pair_idx(i: usize, j: usize) -> usize {
    let (r, c) = if i >= j { (i, j) } else { (j, i) };
    (r * (r + 1)) / 2 + c
}

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
    // Using stack buffer for natoms <= 256 to eliminate all heap allocations inside iterative SCF.
    let mut stack_populations = [0.0f64; 256];
    let mut heap_populations;
    let atom_populations: &mut [f64] = if batch.natoms <= 256 {
        &mut stack_populations[..batch.natoms]
    } else {
        heap_populations = vec![0.0; batch.natoms];
        &mut heap_populations[..]
    };
    for (i, pop) in atom_populations.iter_mut().enumerate().take(batch.natoms) {
        let orb_start = batch.orbital_offsets[i];
        let num_orbs = batch.basis_types[i].num_orbitals();
        let mut q = 0.0;
        for o in 0..num_orbs {
            q += density.get(orb_start + o, orb_start + o);
        }
        *pop = q;
    }

    // 2. One-center two-electron interactions (Coulomb and Exchange matching fock1.F90)
    add_one_center_fock_terms(batch, model, density, fock);

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

/// Accumulate one-center two-electron Coulomb and Exchange terms matching OpenMOPAC `fock1.F90`:
///
/// $$F_{ij} += \sum_{k, l \in A} \left[ P_{kl} (ij|kl) - \frac{1}{2} P_{kl} (ik|jl) \right]$$
///
/// Guaranteed to be strictly invariant under arbitrary 3D spatial rotations of the basis.
pub fn add_one_center_fock_terms(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    density: &AlignedMatrix<f64>,
    fock: &mut AlignedMatrix<f64>,
) {
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
                fock.set(orb_start, orb_start, cur + 0.5 * p_ss * p_a.gss);
            }
            BasisType::SP => {
                // Precompute 10x10 two-electron repulsion matrix W matching OpenMOPAC `wstore.F90`
                let mut w = [[0.0f64; 10]; 10];
                w[0][0] = p_a.gss;

                w[2][0] = p_a.gsp;
                w[0][2] = p_a.gsp;
                w[5][0] = p_a.gsp;
                w[0][5] = p_a.gsp;
                w[9][0] = p_a.gsp;
                w[0][9] = p_a.gsp;

                w[2][2] = p_a.gpp;
                w[5][5] = p_a.gpp;
                w[9][9] = p_a.gpp;

                w[5][2] = p_a.gp2;
                w[2][5] = p_a.gp2;
                w[9][2] = p_a.gp2;
                w[2][9] = p_a.gp2;
                w[9][5] = p_a.gp2;
                w[5][9] = p_a.gp2;

                w[1][1] = p_a.hsp;
                w[3][3] = p_a.hsp;
                w[6][6] = p_a.hsp;

                let g_exch_p = 0.5 * (p_a.gpp - p_a.gp2);
                w[4][4] = g_exch_p;
                w[7][7] = g_exch_p;
                w[8][8] = g_exch_p;

                for io in 0..4 {
                    for jo in 0..4 {
                        let ij = pair_idx(io, jo);
                        let mut sum = 0.0f64;
                        for ko in 0..4 {
                            for lo in 0..4 {
                                let kl = pair_idx(ko, lo);
                                let kj = pair_idx(ko, jo);
                                let li = pair_idx(lo, io);
                                let p_kl = density.get(orb_start + ko, orb_start + lo);
                                sum += p_kl * (w[ij][kl] - 0.5 * w[kj][li]);
                            }
                        }
                        let cur = fock.get(orb_start + io, orb_start + jo);
                        fock.set(orb_start + io, orb_start + jo, cur + sum);
                    }
                }
            }
            BasisType::SPD => {
                let d_params = model.get_d_element_params(za);
                let mut w = [0.0f64; 2025];
                crate::integrals::d_orbitals::wstore(
                    za,
                    9,
                    p_a.gss,
                    p_a.gsp,
                    p_a.gpp,
                    p_a.gp2,
                    p_a.hsp,
                    d_params.as_ref().map(|dp| &dp.repd),
                    &mut w,
                );

                for io in 0..9 {
                    for jo in 0..=io {
                        let ij = pair_idx(io, jo);
                        let mut sum = 0.0f64;
                        for ko in 0..9 {
                            for lo in 0..9 {
                                let kl = pair_idx(ko, lo);
                                let kj = pair_idx(ko, jo);
                                let li = pair_idx(lo, io);
                                let p_kl = density.get(orb_start + ko, orb_start + lo);
                                sum += p_kl * (w[ij * 45 + kl] - 0.5 * w[kj * 45 + li]);
                            }
                        }
                        let cur = fock.get(orb_start + io, orb_start + jo);
                        fock.set(orb_start + io, orb_start + jo, cur + sum);
                        if io != jo {
                            fock.set(orb_start + jo, orb_start + io, cur + sum);
                        }
                    }
                }
            }
        }
    }
}

/// Build the Fock matrix $F = H^{\text{core}} + G(P)$ using full NDDO 22 multipoles.
///
/// Axiomatic formulation matching OpenMOPAC `fock1.F90` and `fock2.F90`:
/// 1. Copy $H^{\text{core}}$.
/// 2. Add one-center two-electron Coulomb and Exchange.
/// 3. Assemble two-center two-electron Coulomb and Exchange via rotated 22 multipoles ($W$),
///    `contract_jab`, `contract_kab`, and heavy-light/light-light routines.
pub fn build_fock_nddo(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    pairs: &[crate::integrals::multipoles::DiatomicPairIntegrals],
    h_core: &AlignedMatrix<f64>,
    density: &AlignedMatrix<f64>,
    fock: &mut AlignedMatrix<f64>,
) {
    assert_eq!(fock.rows, batch.norbs);
    assert_eq!(fock.cols, batch.norbs);

    // 1. Copy H_core into Fock matrix
    fock.data.copy_from_slice(&h_core.data);

    // 2. One-center two-electron interactions (Coulomb and Exchange matching fock1.F90)
    add_one_center_fock_terms(batch, model, density, fock);

    // 3. Assemble two-center two-electron Coulomb & Exchange from precomputed pairs
    crate::integrals::multipoles::assemble_nddo_two_center_fock(pairs, density, fock);
}

/// Accumulate one-center two-electron Coulomb and Exchange terms for UHF matching OpenMOPAC `fock1.F90`:
///
/// $$F^\sigma_{ij} += \sum_{k, l \in A} \left[ P^{\text{tot}}_{kl} (ij|kl) - P^\sigma_{kl} (ik|jl) \right]$$
pub fn add_one_center_fock_terms_spin(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    p_tot: &AlignedMatrix<f64>,
    p_spin: &AlignedMatrix<f64>,
    fock: &mut AlignedMatrix<f64>,
) {
    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };

        let orb_start = batch.orbital_offsets[i];
        match batch.basis_types[i] {
            BasisType::S => {
                let p_ss_tot = p_tot.get(orb_start, orb_start);
                let p_ss_spin = p_spin.get(orb_start, orb_start);
                let cur = fock.get(orb_start, orb_start);
                fock.set(orb_start, orb_start, cur + (p_ss_tot - p_ss_spin) * p_a.gss);
            }
            BasisType::SP => {
                let mut w = [[0.0f64; 10]; 10];
                w[0][0] = p_a.gss;

                w[2][0] = p_a.gsp;
                w[0][2] = p_a.gsp;
                w[5][0] = p_a.gsp;
                w[0][5] = p_a.gsp;
                w[9][0] = p_a.gsp;
                w[0][9] = p_a.gsp;

                w[2][2] = p_a.gpp;
                w[5][5] = p_a.gpp;
                w[9][9] = p_a.gpp;

                w[5][2] = p_a.gp2;
                w[2][5] = p_a.gp2;
                w[9][2] = p_a.gp2;
                w[2][9] = p_a.gp2;
                w[9][5] = p_a.gp2;
                w[5][9] = p_a.gp2;

                w[1][1] = p_a.hsp;
                w[3][3] = p_a.hsp;
                w[6][6] = p_a.hsp;

                let g_exch_p = 0.5 * (p_a.gpp - p_a.gp2);
                w[4][4] = g_exch_p;
                w[7][7] = g_exch_p;
                w[8][8] = g_exch_p;

                for io in 0..4 {
                    for jo in 0..4 {
                        let ij = pair_idx(io, jo);
                        let mut sum = 0.0f64;
                        for ko in 0..4 {
                            for lo in 0..4 {
                                let kl = pair_idx(ko, lo);
                                let kj = pair_idx(ko, jo);
                                let li = pair_idx(lo, io);
                                let p_kl_tot = p_tot.get(orb_start + ko, orb_start + lo);
                                let p_kl_spin = p_spin.get(orb_start + ko, orb_start + lo);
                                sum += p_kl_tot * w[ij][kl] - p_kl_spin * w[kj][li];
                            }
                        }
                        let cur = fock.get(orb_start + io, orb_start + jo);
                        fock.set(orb_start + io, orb_start + jo, cur + sum);
                    }
                }
            }
            BasisType::SPD => {
                let d_params = model.get_d_element_params(za);
                let mut w = [0.0f64; 2025];
                crate::integrals::d_orbitals::wstore(
                    za,
                    9,
                    p_a.gss,
                    p_a.gsp,
                    p_a.gpp,
                    p_a.gp2,
                    p_a.hsp,
                    d_params.as_ref().map(|dp| &dp.repd),
                    &mut w,
                );

                for io in 0..9 {
                    for jo in 0..=io {
                        let ij = pair_idx(io, jo);
                        let mut sum = 0.0f64;
                        for ko in 0..9 {
                            for lo in 0..9 {
                                let kl = pair_idx(ko, lo);
                                let kj = pair_idx(ko, jo);
                                let li = pair_idx(lo, io);
                                let p_kl_tot = p_tot.get(orb_start + ko, orb_start + lo);
                                let p_kl_spin = p_spin.get(orb_start + ko, orb_start + lo);
                                sum += p_kl_tot * w[ij * 45 + kl] - p_kl_spin * w[kj * 45 + li];
                            }
                        }
                        let cur = fock.get(orb_start + io, orb_start + jo);
                        fock.set(orb_start + io, orb_start + jo, cur + sum);
                        if io != jo {
                            fock.set(orb_start + jo, orb_start + io, cur + sum);
                        }
                    }
                }
            }
        }
    }
}

/// Build UHF Fock matrix $F^\sigma = H^{\text{core}} + J(P^\text{tot}) - K(P^\sigma)$ using full NDDO 22 multipoles.
pub fn build_fock_uhf(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    pairs: &[crate::integrals::multipoles::DiatomicPairIntegrals],
    h_core: &AlignedMatrix<f64>,
    p_tot: &AlignedMatrix<f64>,
    p_spin: &AlignedMatrix<f64>,
    fock: &mut AlignedMatrix<f64>,
) {
    assert_eq!(fock.rows, batch.norbs);
    assert_eq!(fock.cols, batch.norbs);

    // 1. Copy H_core into Fock matrix
    fock.data.copy_from_slice(&h_core.data);

    // 2. One-center two-electron interactions
    add_one_center_fock_terms_spin(batch, model, p_tot, p_spin, fock);

    // 3. Assemble two-center two-electron Coulomb & Exchange
    crate::integrals::multipoles::assemble_nddo_two_center_fock_spin(pairs, p_tot, p_spin, fock);
}

/// Build UHF Fock matrix with monopole approximation (non-NDDO fallback).
pub fn build_fock_spin(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    h_core: &AlignedMatrix<f64>,
    p_tot: &AlignedMatrix<f64>,
    p_spin: &AlignedMatrix<f64>,
    fock: &mut AlignedMatrix<f64>,
) {
    assert_eq!(fock.rows, batch.norbs);
    assert_eq!(fock.cols, batch.norbs);

    // 1. Copy H_core into Fock matrix
    fock.data.copy_from_slice(&h_core.data);

    // Compute electronic atomic populations from total density
    let mut stack_populations = [0.0f64; 256];
    let mut heap_populations;
    let atom_populations: &mut [f64] = if batch.natoms <= 256 {
        &mut stack_populations[..batch.natoms]
    } else {
        heap_populations = vec![0.0; batch.natoms];
        &mut heap_populations[..]
    };
    for (i, pop) in atom_populations.iter_mut().enumerate().take(batch.natoms) {
        let orb_start = batch.orbital_offsets[i];
        let num_orbs = batch.basis_types[i].num_orbitals();
        let mut q = 0.0;
        for o in 0..num_orbs {
            q += p_tot.get(orb_start + o, orb_start + o);
        }
        *pop = q;
    }

    // 2. One-center two-electron interactions
    add_one_center_fock_terms_spin(batch, model, p_tot, p_spin, fock);

    // 3. Two-center two-electron interactions
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

            for oa in 0..num_a {
                let idx_a = orb_a_start + oa;
                let cur = fock.get(idx_a, idx_a);
                fock.set(idx_a, idx_a, cur + q_b * gamma_ab);
            }

            for oa in 0..num_a {
                let idx_a = orb_a_start + oa;
                for ob in 0..num_b {
                    let idx_b = orb_b_start + ob;
                    let p_ab = p_spin.get(idx_a, idx_b);
                    let cur = fock.get(idx_a, idx_b);
                    fock.set(idx_a, idx_b, cur - p_ab * gamma_ab);
                }
            }
        }
    }
}
