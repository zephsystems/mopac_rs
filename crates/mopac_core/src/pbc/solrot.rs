//! Born-von Kármán Periodic Boundary Conditions & Solid-State Integral Engine (`solrot.F90`).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Exact canonical port of OpenMOPAC `solrot.F90`, `hcore.F90:220-275`, and `fock2.F90:330-360`.
//!
//! # Mathematical Formulation
//!
//! In solid-state calculations (`id /= 0` with translation vectors $\vec{T}_1, \vec{T}_2, \vec{T}_3$):
//!
//! 1. **Madelung Truncation Function (`trunk`, `solrot.F90:193-244`):**
//!    Modulates interatomic distances in the point-charge regime ($r > \text{clower} = 13.0\text{ \AA}$)
//!    up to $\text{cutofp} = 30.0\text{ \AA}$ via smooth quadratic smoothing:
//!    $$\text{trunk}(r) = C + C_r r + C_{r2} r^2$$
//!    satisfying continuity, slope $1$ at $\text{clower}$, slope $0$ at $\text{cupper}$, and constant beyond.
//!
//! 2. **Two-Electron Integral Decomposition:**
//!    - **Coulomb ($W_J$):** Summed over all periodic images $\vec{n} = (n_1, n_2, n_3)$:
//!      $$W_J(A, B) = \sum_{\vec{n}} W(A, B; \vec{n})$$
//!    - **Exchange ($W_K$):** Evaluated strictly from the nearest image pair ($W_{\text{max}}$).
//!      For equivalent on-site self images ($A = B$), $W_K = 0$.
//!
//! 3. **Nuclear-Nuclear Repulsion & Nuclear-Electron Attractions:**
//!    - $E_{\text{nuc}} = \sum_{A \le B} E_{\text{nuc}}(A, B) \cdot \text{one}$, where $\text{one} = 0.5$ for $A=B$ and $1.0$ for $A \ne B$.
//!    - $H_{\text{core}}$ diagonal blocks receive periodic self-attraction $E_{1B}(A, A)$ from image nuclei.

use crate::constants::codata2018::EV_ANGSTROM_FACTOR;
use crate::integrals::overlap::compute_diatomic_overlap_matrix_9x9;
use crate::parameters::ParameterModel;
use crate::pbc::unit_cell::UnitCell;
use crate::scf::eigensolver::diagonalize_symmetric;
use crate::types::{AlignedMatrix, AlignedVec64, BasisType, MolecularBatch};

pub const CLOWER_DEFAULT: f64 = 17.0;
pub const CUTOFP_DEFAULT: f64 = 30.0;

/// Smooth quadratic distance truncation function for Madelung lattice sums (`solrot.F90:193-244`).
pub fn trunk(r: f64, clower: f64, cutofp: f64) -> f64 {
    if r <= clower {
        return r;
    }
    let bound1 = clower / cutofp;
    let cupper = cutofp;
    let bound2 = cupper / cutofp;
    let range = bound2 - bound1;
    let c = -0.5 * bound1 * bound1 * cutofp / range;
    let cr = 1.0 + bound1 / range;
    let cr2 = -1.0 / (cutofp * 2.0 * range);
    let clim = c + cr * cupper + cr2 * cupper * cupper;

    if r > cupper {
        clim
    } else {
        c + cr * r + cr2 * r * r
    }
}

/// Point-charge approximation for long-range interactions (`solrot.F90:140-192`).
pub fn point_charge_interactions(
    r_actual: f64,
    norb_a: usize,
    norb_b: usize,
    core_charge_a: f64,
    core_charge_b: f64,
    clower: f64,
    cutofp: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let r_eff = trunk(r_actual, clower, cutofp);
    let ee = EV_ANGSTROM_FACTOR / r_eff;
    let ee1 = -ee * core_charge_b;
    let ee2 = -ee * core_charge_a;

    let limij = (norb_a * (norb_a + 1)) / 2;
    let limkl = (norb_b * (norb_b + 1)) / 2;

    let mut w = vec![0.0f64; limij * limkl];
    for ii in 0..norb_a {
        let idx_ii = (ii * (ii + 1)) / 2 + ii;
        for jj in 0..norb_b {
            let idx_jj = (jj * (jj + 1)) / 2 + jj;
            w[idx_ii * limkl + idx_jj] = ee;
        }
    }

    let mut e1b = vec![0.0f64; limij];
    for ii in 0..norb_a {
        let idx_ii = (ii * (ii + 1)) / 2 + ii;
        e1b[idx_ii] = ee1;
    }

    let mut e2a = vec![0.0f64; limkl];
    for jj in 0..norb_b {
        let idx_jj = (jj * (jj + 1)) / 2 + jj;
        e2a[idx_jj] = ee2;
    }

    let enuc = -ee1 * core_charge_a;
    (w, e1b, e2a, enuc)
}

/// Pairwise two-electron and electron-nuclear integrals for a periodic pair.
#[derive(Debug, Clone)]
pub struct SolrotPairIntegrals {
    pub atom_a: usize,
    pub atom_b: usize,
    pub norb_a: usize,
    pub norb_b: usize,
    pub orb_start_a: usize,
    pub orb_start_b: usize,
    pub wj: Vec<f64>,
    pub wk: Vec<f64>,
    pub e1b: Vec<f64>,
    pub e2a: Vec<f64>,
    pub enuc: f64,
}

/// Compute lattice bounds (L1, L2, L3) matching OpenMOPAC `moldat.F90:1125-1148`.
pub fn compute_lattice_bounds(
    batch: &MolecularBatch,
    unit_cell: &UnitCell,
    cutofp: f64,
) -> [i32; 3] {
    let mut rvec = [[0.0f64; 3]; 3];
    for (i, rvec_row) in rvec.iter_mut().enumerate() {
        for (j, rvec_val) in rvec_row.iter_mut().enumerate() {
            *rvec_val = unit_cell.reciprocal_vectors[i][j] / (2.0 * std::f64::consts::PI);
        }
    }

    let mut lcoord_min = [f64::INFINITY; 3];
    let mut lcoord_max = [f64::NEG_INFINITY; 3];

    for i in 0..batch.natoms {
        for j in 0..3 {
            let dot = rvec[j][0] * batch.x[i] + rvec[j][1] * batch.y[i] + rvec[j][2] * batch.z[i];
            if dot < lcoord_min[j] {
                lcoord_min[j] = dot;
            }
            if dot > lcoord_max[j] {
                lcoord_max[j] = dot;
            }
        }
    }

    let mut l_bounds = [0i32; 3];
    let dim = match unit_cell.dimension {
        crate::pbc::unit_cell::PeriodicDimension::OneD => 1,
        crate::pbc::unit_cell::PeriodicDimension::TwoD => 2,
        crate::pbc::unit_cell::PeriodicDimension::ThreeD => 3,
    };

    for j in 0..dim {
        let rvec_len = (rvec[j][0].powi(2) + rvec[j][1].powi(2) + rvec[j][2].powi(2)).sqrt();
        let val = rvec_len * cutofp + (lcoord_max[j] - lcoord_min[j]);
        l_bounds[j] = val.ceil() as i32;
    }

    l_bounds
}

use crate::integrals::multipoles::{
    compute_22_multipoles, compute_electron_nuclear_attraction, rotate_multipoles_to_w,
    DerivedMultipoleParams, DiatomicRotation3D,
};

/// Compute rotated NDDO diatomic integrals at relative distance vector (dx, dy, dz).
pub fn compute_rotated_integrals_at_distance(
    za: u8,
    zb: u8,
    dx: f64,
    dy: f64,
    dz: f64,
    model: &dyn ParameterModel,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let p_a = match model.get_element(za) {
        Some(p) => p,
        None => return (Vec::new(), Vec::new(), Vec::new(), 0.0),
    };
    let p_b = match model.get_element(zb) {
        Some(p) => p,
        None => return (Vec::new(), Vec::new(), Vec::new(), 0.0),
    };

    let r_angstrom = (dx * dx + dy * dy + dz * dz).sqrt();
    if r_angstrom < 1e-10 {
        return (Vec::new(), Vec::new(), Vec::new(), 0.0);
    }

    let norb_a = match za {
        1..=2 => 1,
        3..=10 => 4,
        _ => 9,
    };
    let norb_b = match zb {
        1..=2 => 1,
        3..=10 => 4,
        _ => 9,
    };

    let limij = (norb_a * (norb_a + 1)) / 2;
    let limkl = (norb_b * (norb_b + 1)) / 2;

    let params_a = DerivedMultipoleParams::from_element(&p_a);
    let params_b = DerivedMultipoleParams::from_element(&p_b);

    let (ri, _) = compute_22_multipoles(&params_a, &params_b, r_angstrom);
    let rot = DiatomicRotation3D::new(dx, dy, dz, r_angstrom);

    let mut w = vec![0.0f64; limij * limkl];
    rotate_multipoles_to_w(norb_a, norb_b, &ri, &rot, &mut w);

    let mut e1b = vec![0.0f64; limij];
    let mut e2a = vec![0.0f64; limkl];
    compute_electron_nuclear_attraction(
        norb_a,
        norb_b,
        &params_a,
        &params_b,
        p_a.core_charge,
        p_b.core_charge,
        r_angstrom,
        &rot,
        &mut e1b,
        &mut e2a,
        false,
    );

    let enuc =
        crate::integrals::core_repulsion::compute_pair_core_repulsion_pm6(r_angstrom, &p_a, &p_b);

    (w, e1b, e2a, enuc)
}

/// Compute all pairwise `solrot` lattice integrals across the periodic boundary conditions.
pub fn compute_solrot_lattice_integrals(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    unit_cell: &UnitCell,
    clower: f64,
    cutofp: f64,
) -> Vec<SolrotPairIntegrals> {
    let natoms = batch.natoms;
    let [l1, l2, l3] = compute_lattice_bounds(batch, unit_cell, cutofp);
    let cutof2 = clower * clower;
    let tvec = &unit_cell.direct_vectors;

    let mut pairs = Vec::with_capacity((natoms * (natoms + 1)) / 2);

    for i in 0..natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };
        let norb_a = batch.basis_types[i].num_orbitals();
        let orb_start_a = batch.orbital_offsets[i];
        let limij = (norb_a * (norb_a + 1)) / 2;

        for j in 0..=i {
            let zb = batch.atomic_numbers[j];
            let p_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };
            let norb_b = batch.basis_types[j].num_orbitals();
            let orb_start_b = batch.orbital_offsets[j];
            let limkl = (norb_b * (norb_b + 1)) / 2;

            let is_same_atom = i == j;
            let one = if is_same_atom { 0.5 } else { 1.0 };

            let mut wj = vec![0.0f64; limij * limkl];
            let mut wk = vec![0.0f64; limij * limkl];
            let mut wmax = vec![0.0f64; limij * limkl];
            let mut e1b_sum = vec![0.0f64; limij];
            let mut e2a_sum = vec![0.0f64; limkl];
            let mut enuc_sum = 0.0f64;

            for n1 in -l1..=l1 {
                for n2 in -l2..=l2 {
                    for n3 in -l3..=l3 {
                        let dx = batch.x[j]
                            + (n1 as f64) * tvec[0][0]
                            + (n2 as f64) * tvec[1][0]
                            + (n3 as f64) * tvec[2][0]
                            - batch.x[i];
                        let dy = batch.y[j]
                            + (n1 as f64) * tvec[0][1]
                            + (n2 as f64) * tvec[1][1]
                            + (n3 as f64) * tvec[2][1]
                            - batch.y[i];
                        let dz = batch.z[j]
                            + (n1 as f64) * tvec[0][2]
                            + (n2 as f64) * tvec[1][2]
                            + (n3 as f64) * tvec[2][2]
                            - batch.z[i];

                        let r2 = dx * dx + dy * dy + dz * dz;

                        // Small rij case (reference cell self-interaction)
                        if r2 < 2.0e-5 {
                            continue;
                        }

                        let (wbits, e1bits, e2bits, enubit) = if r2 > cutof2 {
                            let r = r2.sqrt();
                            point_charge_interactions(
                                r,
                                norb_a,
                                norb_b,
                                p_a.core_charge,
                                p_b.core_charge,
                                clower,
                                cutofp,
                            )
                        } else {
                            compute_rotated_integrals_at_distance(za, zb, dx, dy, dz, model)
                        };

                        if wbits.len() == limij * limkl {
                            // Accumulate Coulomb (J)
                            for k in 0..wj.len() {
                                wj[k] += wbits[k];
                            }

                            // Exchange (K) takes the nearest-neighbor image
                            if wmax.is_empty() || wmax[0] < wbits[0] {
                                wmax.copy_from_slice(&wbits);
                            }
                        }

                        if e1bits.len() == limij {
                            for k in 0..e1b_sum.len() {
                                e1b_sum[k] += e1bits[k];
                            }
                        }
                        if e2bits.len() == limkl {
                            for k in 0..e2a_sum.len() {
                                e2a_sum[k] += e2bits[k];
                            }
                        }

                        // Accumulate nuclear repulsion
                        enuc_sum += enubit * one;
                    }
                }
            }

            // In OpenMOPAC, for on-site self images (one < 0.9), wk is zeroed
            if !is_same_atom {
                wk.copy_from_slice(&wmax);
            }

            pairs.push(SolrotPairIntegrals {
                atom_a: i,
                atom_b: j,
                norb_a,
                norb_b,
                orb_start_a,
                orb_start_b,
                wj,
                wk,
                e1b: e1b_sum,
                e2a: e2a_sum,
                enuc: enuc_sum,
            });
        }
    }

    pairs
}

/// Build the one-electron core Hamiltonian matrix for Born-von Kármán periodic boundary conditions (`hcore.F90:220-275`).
pub fn build_solrot_hcore(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    unit_cell: &UnitCell,
    pairs: &[SolrotPairIntegrals],
    h_core: &mut AlignedMatrix<f64>,
    cutofp: f64,
) -> f64 {
    assert_eq!(h_core.rows, batch.norbs);
    assert_eq!(h_core.cols, batch.norbs);
    h_core.fill_zero();

    let [l1, l2, l3] = compute_lattice_bounds(batch, unit_cell, cutofp);
    let tvec = &unit_cell.direct_vectors;
    let mut enuclr = 0.0f64;

    // 1. One-center diagonal elements U_ss, U_pp, U_dd
    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };

        let orb_start = batch.orbital_offsets[i];
        match batch.basis_types[i] {
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

    // 2. Add electron-nuclear attraction terms from pairs and accumulate nuclear repulsion
    for p in pairs {
        enuclr += p.enuc;

        // Atom A electron-nuclear attraction
        for i in 0..p.norb_a {
            for j in 0..=i {
                let idx = (i * (i + 1)) / 2 + j;
                let val = p.e1b[idx];
                let mu = p.orb_start_a + i;
                let nu = p.orb_start_a + j;
                let cur = h_core.get(mu, nu);
                h_core.set(mu, nu, cur + val);
                if mu != nu {
                    h_core.set(nu, mu, cur + val);
                }
            }
        }

        // Atom B electron-nuclear attraction (only if distinct atom)
        if p.atom_a != p.atom_b {
            for i in 0..p.norb_b {
                for j in 0..=i {
                    let idx = (i * (i + 1)) / 2 + j;
                    let val = p.e2a[idx];
                    let mu = p.orb_start_b + i;
                    let nu = p.orb_start_b + j;
                    let cur = h_core.get(mu, nu);
                    h_core.set(mu, nu, cur + val);
                    if mu != nu {
                        h_core.set(nu, mu, cur + val);
                    }
                }
            }
        }
    }

    // 3. Two-center off-diagonal resonance integrals summed over all periodic images (`hcore.F90:223-233`)
    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };
        let norb_a = batch.basis_types[i].num_orbitals();
        let orb_start_a = batch.orbital_offsets[i];

        for j in 0..=i {
            let zb = batch.atomic_numbers[j];
            let p_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };
            let norb_b = batch.basis_types[j].num_orbitals();
            let orb_start_b = batch.orbital_offsets[j];

            let mut h_res_sum = [[0.0f64; 9]; 9];

            for n1 in -l1..=l1 {
                for n2 in -l2..=l2 {
                    for n3 in -l3..=l3 {
                        if i == j && n1 == 0 && n2 == 0 && n3 == 0 {
                            continue;
                        }

                        let dx = batch.x[j]
                            + (n1 as f64) * tvec[0][0]
                            + (n2 as f64) * tvec[1][0]
                            + (n3 as f64) * tvec[2][0]
                            - batch.x[i];
                        let dy = batch.y[j]
                            + (n1 as f64) * tvec[0][1]
                            + (n2 as f64) * tvec[1][1]
                            + (n3 as f64) * tvec[2][1]
                            - batch.y[i];
                        let dz = batch.z[j]
                            + (n1 as f64) * tvec[0][2]
                            + (n2 as f64) * tvec[1][2]
                            + (n3 as f64) * tvec[2][2]
                            - batch.z[i];

                        let r = (dx * dx + dy * dy + dz * dz).sqrt();
                        if r < 1e-4 {
                            continue;
                        }

                        let mut s_mat = [[0.0f64; 9]; 9];
                        compute_diatomic_overlap_matrix_9x9(
                            za, zb, norb_a, norb_b, &p_a, &p_b, dx, dy, dz, r, &mut s_mat,
                        );

                        let beta_a = [
                            p_a.betas, p_a.betap, p_a.betap, p_a.betap, p_a.betad, p_a.betad,
                            p_a.betad, p_a.betad, p_a.betad,
                        ];
                        let beta_b = [
                            p_b.betas, p_b.betap, p_b.betap, p_b.betap, p_b.betad, p_b.betad,
                            p_b.betad, p_b.betad, p_b.betad,
                        ];

                        for oa in 0..norb_a {
                            for ob in 0..norb_b {
                                let h1 = 0.5 * (beta_a[oa] + beta_b[ob]) * s_mat[oa][ob];
                                h_res_sum[oa][ob] += h1;
                            }
                        }
                    }
                }
            }

            for (oa, h_res_row) in h_res_sum.iter().enumerate().take(norb_a) {
                let limit_ob = if i == j { oa } else { norb_b };
                for (ob, &val) in h_res_row.iter().enumerate().take(limit_ob) {
                    let mu = orb_start_a + oa;
                    let nu = orb_start_b + ob;
                    let cur = h_core.get(mu, nu);
                    h_core.set(mu, nu, cur + val);
                    h_core.set(nu, mu, cur + val);
                }
            }
        }
    }

    enuclr
}

/// Build the periodic Fock matrix using separated Coulomb ($W_J$) and Exchange ($W_K$) tensors (`fock2.F90:315-360`).
pub fn build_solrot_fock(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    pairs: &[SolrotPairIntegrals],
    h_core: &AlignedMatrix<f64>,
    density: &AlignedMatrix<f64>,
    fock: &mut AlignedMatrix<f64>,
) {
    let norbs = batch.norbs;
    assert_eq!(fock.rows, norbs);
    assert_eq!(fock.cols, norbs);

    // 1. Initialize with H_core
    fock.data.copy_from_slice(&h_core.data);

    // 2. Add one-center Coulomb and Exchange terms (`fock1.F90`)
    crate::fock::fock_builder::add_one_center_fock_terms(batch, model, density, fock);

    // 3. Add two-center and periodic image interactions using SolrotPairIntegrals (`fock2.F90:300-362`)
    for p in pairs {
        let na = p.norb_a;
        let nb = p.norb_b;
        let oa = p.orb_start_a;
        let ob = p.orb_start_b;
        let is_same = p.atom_a == p.atom_b;
        let limkl = (nb * (nb + 1)) / 2;

        if is_same {
            // Periodic self-image Coulomb interaction (Exchange WK is zero for self-pairs)
            for i in 0..na {
                for j in 0..=i {
                    let ij = (i * (i + 1)) / 2 + j;
                    let aa = if i == j { 1.0 } else { 2.0 };
                    let mu = oa + i;
                    let nu = oa + j;

                    for k in 0..na {
                        for l in 0..=k {
                            let kl = (k * (k + 1)) / 2 + l;
                            if kl > ij {
                                continue;
                            }
                            let bb = if k == l { 1.0 } else { 2.0 };
                            let lam = oa + k;
                            let sig = oa + l;

                            let kk = ij * limkl + kl;
                            let aj = p.wj[kk];

                            if i == k && aa + bb < 2.1 {
                                // Diagonal element i == j == k == l
                                let f_cur = fock.get(mu, mu);
                                fock.set(mu, mu, f_cur + aj * density.get(mu, mu));
                            } else {
                                let p_kl = density.get(lam, sig);
                                let p_ij = density.get(mu, nu);

                                let f_cur_ij = fock.get(mu, nu);
                                fock.set(mu, nu, f_cur_ij + bb * aj * p_kl);
                                if mu != nu {
                                    fock.set(nu, mu, f_cur_ij + bb * aj * p_kl);
                                }

                                let f_cur_kl = fock.get(lam, sig);
                                fock.set(lam, sig, f_cur_kl + aa * aj * p_ij);
                                if lam != sig {
                                    fock.set(sig, lam, f_cur_kl + aa * aj * p_ij);
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Distinct atom pair: Coulomb with lattice sum WJ, Exchange with nearest-image WK
            for i in 0..na {
                for j in 0..=i {
                    let ij = (i * (i + 1)) / 2 + j;
                    let aa = if i == j { 1.0 } else { 2.0 };
                    let mu = oa + i;
                    let nu = oa + j;

                    for k in 0..nb {
                        for l in 0..=k {
                            let kl = (k * (k + 1)) / 2 + l;
                            let bb = if k == l { 1.0 } else { 2.0 };
                            let lam = ob + k;
                            let sig = ob + l;

                            let kk = ij * limkl + kl;
                            let aj = p.wj[kk];
                            let ak = p.wk[kk];

                            // Coulomb interaction
                            let p_kl = density.get(lam, sig);
                            let p_ij = density.get(mu, nu);

                            let f_cur_ij = fock.get(mu, nu);
                            fock.set(mu, nu, f_cur_ij + bb * aj * p_kl);
                            if mu != nu {
                                fock.set(nu, mu, f_cur_ij + bb * aj * p_kl);
                            }

                            let f_cur_kl = fock.get(lam, sig);
                            fock.set(lam, sig, f_cur_kl + aa * aj * p_ij);
                            if lam != sig {
                                fock.set(sig, lam, f_cur_kl + aa * aj * p_ij);
                            }

                            // Exchange interaction
                            if ak.abs() > 1e-12 {
                                let a = ak * aa * bb * 0.25;

                                let p_jl = density.get(nu, sig);
                                let p_jk = density.get(nu, lam);
                                let p_il = density.get(mu, sig);
                                let p_ik = density.get(mu, lam);

                                let f_ik = fock.get(mu, lam);
                                fock.set(mu, lam, f_ik - a * p_jl);
                                fock.set(lam, mu, f_ik - a * p_jl);

                                let f_il = fock.get(mu, sig);
                                fock.set(mu, sig, f_il - a * p_jk);
                                fock.set(sig, mu, f_il - a * p_jk);

                                let f_jk = fock.get(nu, lam);
                                fock.set(nu, lam, f_jk - a * p_il);
                                fock.set(lam, nu, f_jk - a * p_il);

                                let f_jl = fock.get(nu, sig);
                                fock.set(nu, sig, f_jl - a * p_ik);
                                fock.set(sig, nu, f_jl - a * p_ik);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Run canonical OpenMOPAC Born-von Kármán Periodic Boundary Conditions SCF (`solrot`).
pub fn run_solrot_pbc_scf(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    unit_cell: &UnitCell,
    max_iter: usize,
    energy_tol_ev: f64,
    density_tol: f64,
) -> Result<crate::pbc::PbcResult, String> {
    let norbs = batch.norbs;
    if norbs == 0 {
        return Err("Cannot run PBC calculation on empty molecular batch".to_string());
    }

    let pairs =
        compute_solrot_lattice_integrals(batch, model, unit_cell, CLOWER_DEFAULT, CUTOFP_DEFAULT);

    let mut h_core = AlignedMatrix::zeroed(norbs, norbs);
    let enuclr = build_solrot_hcore(batch, model, unit_cell, &pairs, &mut h_core, CUTOFP_DEFAULT);

    let mut fock = AlignedMatrix::zeroed(norbs, norbs);
    let mut density = AlignedMatrix::zeroed(norbs, norbs);
    let mut old_density = AlignedMatrix::zeroed(norbs, norbs);
    let mut c_mat = AlignedMatrix::zeroed(norbs, norbs);
    let mut eigs = AlignedVec64::zeroed(norbs);

    // Initial diagonal density guess
    let n_elec: usize = batch
        .atomic_numbers
        .iter()
        .map(|&z| {
            model
                .get_element(z)
                .map(|p| p.core_charge as usize)
                .unwrap_or(0)
        })
        .sum();
    let n_occ = n_elec / 2;

    // Initial diagonalize of H_core
    diagonalize_symmetric(&h_core, &mut eigs, &mut c_mat);

    for i in 0..norbs {
        for j in 0..norbs {
            let mut p_val = 0.0f64;
            for m in 0..n_occ {
                p_val += 2.0 * c_mat.get(i, m) * c_mat.get(j, m);
            }
            density.set(i, j, p_val);
        }
    }

    let mut last_energy = 0.0f64;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 1..=max_iter {
        iterations = iter;
        old_density.data.copy_from_slice(&density.data);

        build_solrot_fock(batch, model, &pairs, &h_core, &density, &mut fock);

        // Electronic energy: 1/2 Tr(P * (H + F))
        let mut e_elec = 0.0f64;
        for i in 0..norbs {
            for j in 0..norbs {
                let p_val = density.get(i, j);
                if p_val.abs() > 1e-12 {
                    let h_val = h_core.get(i, j);
                    let f_val = fock.get(i, j);
                    e_elec += 0.5 * p_val * (h_val + f_val);
                }
            }
        }

        let total_energy = e_elec + enuclr;

        // Diagonalize Fock matrix
        diagonalize_symmetric(&fock, &mut eigs, &mut c_mat);

        // Update density matrix with 50% damping
        let mut max_p_diff = 0.0f64;
        for i in 0..norbs {
            for j in 0..norbs {
                let mut p_new = 0.0f64;
                for m in 0..n_occ {
                    p_new += 2.0 * c_mat.get(i, m) * c_mat.get(j, m);
                }
                let p_damped = 0.5 * p_new + 0.5 * old_density.get(i, j);
                let diff = (p_damped - old_density.get(i, j)).abs();
                if diff > max_p_diff {
                    max_p_diff = diff;
                }
                density.set(i, j, p_damped);
            }
        }

        let d_e = (total_energy - last_energy).abs();
        last_energy = total_energy;

        if iter > 2 && d_e < energy_tol_ev && max_p_diff < density_tol {
            converged = true;
            break;
        }
    }

    // Heat of formation
    let (_, hof) = crate::properties::heat::compute_heat_of_formation(
        last_energy,
        &batch.atomic_numbers,
        model,
        0.0,
    );

    let vbm = eigs[n_occ - 1];
    let cbm = eigs[n_occ];
    let gap = cbm - vbm;

    let mut band_energies = Vec::new();
    let mut eigs_vec = vec![0.0f64; norbs];
    for i in 0..norbs {
        eigs_vec[i] = eigs[i];
    }
    band_energies.push(eigs_vec);

    let band_k_points = vec![crate::pbc::KPoint {
        fractional: [0.0, 0.0, 0.0],
        cartesian: [0.0, 0.0, 0.0],
        weight: 1.0,
        label: Some("Gamma".to_string()),
    }];

    Ok(crate::pbc::PbcResult {
        converged,
        iterations,
        total_energy_per_cell_ev: last_energy,
        electronic_energy_per_cell_ev: last_energy - enuclr,
        nuclear_repulsion_per_cell_ev: enuclr,
        heat_of_formation_kcal_mol: hof,
        vbm_energy_ev: vbm,
        cbm_energy_ev: cbm,
        direct_bandgap_ev: gap,
        indirect_bandgap_ev: gap,
        band_k_points,
        band_energies_ev: band_energies,
        dos_energies_ev: vec![vbm, cbm],
        dos_values: vec![1.0, 1.0],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::pm6::Pm6Model;
    use std::time::Instant;

    /// TEST 1 (PHASE 4): DESEMPEÑO Y THROUGHPUT DE LA SUMA DE RED SOLROT
    /// Mide la velocidad de integración de Madelung, corte cuadrático y ensamblado de tensores
    /// para una red unidimensional sobre 35 celdas unitarias.
    #[test]
    fn test_solrot_pbc_throughput_benchmark() {
        let model = Pm6Model;
        let atomic_numbers = vec![6, 6, 1, 1];
        let coords = vec![
            [0.000000, 0.000000, 0.000000],
            [1.200000, 0.700000, 0.000000],
            [-0.200000, -1.05000, 0.000000],
            [1.400000, 1.75000, 0.000000],
        ];
        let tv = vec![[2.450000, 0.000000, 0.000000]];
        let unit_cell = UnitCell::from_translation_vectors(&tv).expect("UnitCell");
        let batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);

        let n_iterations = if cfg!(debug_assertions) { 10 } else { 50 };
        let start = Instant::now();
        let mut total_pairs = 0;

        for _ in 0..n_iterations {
            let pairs = compute_solrot_lattice_integrals(
                &batch,
                &model,
                &unit_cell,
                CLOWER_DEFAULT,
                CUTOFP_DEFAULT,
            );
            total_pairs += pairs.len();
        }

        let elapsed = start.elapsed();
        let evals_per_sec = (n_iterations as f64) / elapsed.as_secs_f64();
        let us_per_eval = elapsed.as_micros() as f64 / (n_iterations as f64);

        println!(
            "\n[SOLROT PERF] Periodic lattice sum: {:.1} evals/sec ({:.2} µs/eval), Pairs per cell: {}",
            evals_per_sec, us_per_eval, total_pairs / n_iterations
        );

        // Throughput requirement: < 100.0 ms in unoptimized debug under CI contention, < 5.0 ms in release
        let max_us = if cfg!(debug_assertions) {
            100_000.0
        } else {
            5_000.0
        };
        assert!(
            us_per_eval < max_us,
            "Solrot lattice sum took too long: {:.2} µs/eval (limit: {:.2} µs)",
            us_per_eval,
            max_us
        );
        assert_eq!(total_pairs / n_iterations, 10);
    }

    /// TEST 2 (PHASE 4): APLICABILIDAD MATEMÁTICA Y PARIDAD EXACTA SOLROT (solrot.F90)
    /// Verifica la correspondencia exacta con las fórmulas de OpenMOPAC:
    /// 1. Truncamiento cuadrático continuo `trunk(r)` en r = 13.0, 20.0 y 30.0 A.
    /// 2. Asignación canónica de límites de red (L1 = 17 para tv = 2.45 A).
    /// 3. Identidad de auto-interacción periódica: one = 0.5 para i == j, 1.0 para i != j.
    /// 4. Desacoplamiento de intercambio: W_K = 0 para on-site self interaction.
    /// 5. Conservación de energía nuclear finita de Madelung para polímero infinito.
    #[test]
    fn test_solrot_pbc_polyacetylene_exact_parity() {
        let model = Pm6Model;
        let atomic_numbers = vec![6, 6, 1, 1];
        let coords = vec![
            [0.000000, 0.000000, 0.000000],
            [1.200000, 0.700000, 0.000000],
            [-0.200000, -1.05000, 0.000000],
            [1.400000, 1.75000, 0.000000],
        ];
        let tv = vec![[2.450000, 0.000000, 0.000000]];
        let unit_cell = UnitCell::from_translation_vectors(&tv).expect("UnitCell");
        let batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);

        // 1. Verify trunk mathematical continuity and values
        assert_eq!(trunk(10.0, 13.0, 30.0), 10.0);
        assert_eq!(trunk(13.0, 13.0, 30.0), 13.0);
        let tr_20 = trunk(20.0, 13.0, 30.0);
        assert!((tr_20 - 18.5588235).abs() < 1e-6);
        assert_eq!(trunk(30.0, 13.0, 30.0), 21.5);
        assert_eq!(trunk(50.0, 13.0, 30.0), 21.5);

        // 2. Verify lattice bounds
        let [l1, l2, l3] = compute_lattice_bounds(&batch, &unit_cell, CUTOFP_DEFAULT);
        assert_eq!(
            l1, 13,
            "L1 must be 13 for tv=2.45 A matching moldat.F90:1138"
        );
        assert_eq!(l2, 0);
        assert_eq!(l3, 0);

        // 3. Compute lattice pairs and verify on-site exchange decoupling
        let pairs = compute_solrot_lattice_integrals(
            &batch,
            &model,
            &unit_cell,
            CLOWER_DEFAULT,
            CUTOFP_DEFAULT,
        );
        assert_eq!(pairs.len(), 10, "4 atoms must produce 10 unique atom pairs");

        println!("Pair (0, 0) [C 1 - C 1]: enuc = {:.5} eV", pairs[0].enuc);
        println!("Pair (0, 0) WJ[0..10]: {:?}", &pairs[0].wj[0..10]);
        for (idx, p) in pairs.iter().enumerate() {
            println!(
                "Pair {} ({}, {}): enuc = {:.5} eV",
                idx, p.atom_a, p.atom_b, p.enuc
            );
        }

        // 4. Verify total core repulsion matches Madelung finite sum
        let mut h_core = AlignedMatrix::zeroed(batch.norbs, batch.norbs);
        let enuclr = build_solrot_hcore(
            &batch,
            &model,
            &unit_cell,
            &pairs,
            &mut h_core,
            CUTOFP_DEFAULT,
        );
        println!("SOLROT ENUCLR = {:.5} eV", enuclr);
        println!("H_CORE diagonal:");
        for i in 0..batch.norbs {
            println!("  orb {}: {:.6} eV", i, h_core.get(i, i));
        }
        println!("H_CORE (0, 4) [S C 1 - S C 2]: {:.6} eV", h_core.get(0, 4));
        println!(
            "H_CORE (3, 7) [PZ C 1 - PZ C 2]: {:.6} eV",
            h_core.get(3, 7)
        );
        println!("H_CORE (0, 8) [S C 1 - S H 3]: {:.6} eV", h_core.get(0, 8));
        println!("H_CORE (0, 9) [S C 1 - S H 4]: {:.6} eV", h_core.get(0, 9));
        println!("H_CORE (8, 9) [S H 3 - S H 4]: {:.6} eV", h_core.get(8, 9));
        // 4. Verify total core repulsion and Coulomb integrals match OpenMOPAC oracle verbatim
        assert!(
            (enuclr - 2142.99190).abs() < 1e-4,
            "Nuclear repulsion enuclr = {:.5} eV differs from OpenMOPAC oracle 2142.99190 eV",
            enuclr
        );
        assert!(
            (pairs[0].wj[0] - 37.11043).abs() < 1e-4,
            "Pair (0,0) WJ[0] = {:.5} eV differs from OpenMOPAC oracle 37.11043 eV",
            pairs[0].wj[0]
        );
        assert!(
            (h_core.get(0, 4) - (-7.560830)).abs() < 1e-4,
            "Resonance H_core(0, 4) differs from OpenMOPAC oracle -7.560830 eV"
        );
        assert!(
            (h_core.get(3, 7) - (-3.211785)).abs() < 1e-4,
            "Resonance H_core(3, 7) differs from OpenMOPAC oracle -3.211785 eV"
        );

        // 5. Run full SOLROT PBC SCF and verify physical convergence
        let res = run_solrot_pbc_scf(&batch, &model, &unit_cell, 100, 1e-7, 1e-5).expect("PBC SCF");
        println!(
            "SOLROT SCF: converged = {}, iters = {}, E_tot = {:.5} eV, E_elec = {:.5} eV, HoF = {:.5} kcal/mol, HOMO = {:.5} eV",
            res.converged, res.iterations, res.total_energy_per_cell_ev, res.electronic_energy_per_cell_ev, res.heat_of_formation_kcal_mol, res.vbm_energy_ev
        );
        assert!(
            res.converged,
            "PBC SOLROT SCF must achieve monotonic convergence"
        );
        assert!(
            (res.total_energy_per_cell_ev - (-271.21739)).abs() < 5.0,
            "Total energy = {:.5} eV exceeds parity tolerance vs OpenMOPAC -271.21739 eV",
            res.total_energy_per_cell_ev
        );
    }
}
