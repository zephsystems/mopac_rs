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

/// Apply external electric field perturbation to H_core and return nuclear interaction energy.
///
/// Direct port of OpenMOPAC electric field coupling in `static_polarizability.F90`.
///
/// $$H_{\mu\mu} \gets H_{\mu\mu} + \vec{E} \cdot (\vec{R}_A - \vec{R}_{\text{cm}})$$
/// $$H_{s, p_\alpha} \gets H_{s, p_\alpha} + E_\alpha D_{1, A}$$
/// $$E_{\text{nuc\_field}} = -\sum_A Z_A \vec{E} \cdot (\vec{R}_A - \vec{R}_{\text{cm}})$$
#[allow(clippy::needless_range_loop)]
pub fn apply_electric_field_to_hcore(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    h_core: &mut AlignedMatrix<f64>,
    efield: [f64; 3], // in eV / Angstrom
) -> f64 {
    let natoms = batch.natoms;
    if natoms == 0 {
        return 0.0;
    }

    // 1. Center of mass
    let mut com = [0.0; 3];
    let mut total_mass = 0.0;
    for a in 0..natoms {
        let m = crate::constants::standard_atomic_mass(batch.atomic_numbers[a]);
        total_mass += m;
        com[0] += m * batch.x[a];
        com[1] += m * batch.y[a];
        com[2] += m * batch.z[a];
    }
    if total_mass > 0.0 {
        let inv_m = 1.0 / total_mass;
        com[0] *= inv_m;
        com[1] *= inv_m;
        com[2] *= inv_m;
    }

    let mut e_nuc_field = 0.0f64;

    for a in 0..natoms {
        let z = batch.atomic_numbers[a];
        let elem = match model.get_element(z) {
            Some(e) => e,
            None => continue,
        };
        let core_charge = elem.core_charge;
        let rx = batch.x[a] - com[0];
        let ry = batch.y[a] - com[1];
        let rz = batch.z[a] - com[2];

        // Nuclear coupling: - Z_A * (E . r_A)
        e_nuc_field -= core_charge * (efield[0] * rx + efield[1] * ry + efield[2] * rz);

        let orb_start = batch.orbital_offsets[a];
        let norbs = batch.basis_types[a].num_orbitals();

        // Diagonal shift: + (E . r_A)
        let diag_shift = efield[0] * rx + efield[1] * ry + efield[2] * rz;
        for o in 0..norbs {
            let idx = orb_start + o;
            let cur = h_core.get(idx, idx);
            h_core.set(idx, idx, cur + diag_shift);
        }

        // On-atom hybridization shift: s-p transition dipole elements
        if norbs >= 4 {
            let d1 = crate::integrals::multipoles::DerivedMultipoleParams::from_element(&elem).dd;
            let d1_a = d1 * crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS;

            let s = orb_start;
            let px = orb_start + 1;
            let py = orb_start + 2;
            let pz = orb_start + 3;

            let hx = efield[0] * d1_a;
            let hy = efield[1] * d1_a;
            let hz = efield[2] * d1_a;

            h_core.set(s, px, h_core.get(s, px) + hx);
            h_core.set(px, s, h_core.get(px, s) + hx);

            h_core.set(s, py, h_core.get(s, py) + hy);
            h_core.set(py, s, h_core.get(py, s) + hy);

            h_core.set(s, pz, h_core.get(s, pz) + hz);
            h_core.set(pz, s, h_core.get(pz, s) + hz);
        }
    }

    e_nuc_field
}
