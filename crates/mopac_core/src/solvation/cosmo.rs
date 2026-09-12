//! Electrostatic Boundary Element Formulation of the COSMO Implicit Solvation Model.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct translation and rigorous formulation of OpenMOPAC v23.2.5 `cosmo.F90`.
//!
//! Primary References:
//! - Klamt, A.; Schüürmann, G. "COSMO: a new approach to dielectric screening in solvents with
//!   explicit expressions for the screening energy and its gradient", J. Chem. Soc., Perkin Trans. 2,
//!   1993, 799-805. <https://doi.org/10.1039/P29930000799>
//! - Klamt, A. "Conductor-like Screening Model for Real Solvents: A New Approach to the
//!   Quantitative Calculation of Solvation Phenomena", J. Phys. Chem. 1995, 99, 2224-2235.

use crate::parameters::ParameterModel;
use crate::ri::cholesky::cholesky_decompose;
use crate::solvation::cavity::CosmoCavity;
use crate::types::{AlignedMatrix, MolecularBatch};
use std::f64::consts::PI;

/// Conversion factor $a_0 \times \text{eV}$ from atomic units to eV*Angstrom.
/// Verbatim CODATA 2018 value from OpenMOPAC `conref_C.F90`:
/// `fpcref(1, 2) = 14.399645478456`
pub const A0_EV: f64 = 14.399645478456;

/// COSMO Solvation Model Configuration Parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosmoParams {
    /// Solvent relative permittivity (dielectric constant $\varepsilon$, e.g. 78.4 for water).
    pub epsilon: f64,
    /// Solvent probe radius in Angstroms (default: 1.30005 A).
    pub rsolv: f64,
}

impl Default for CosmoParams {
    fn default() -> Self {
        Self {
            epsilon: 78.4,
            rsolv: 1.30005,
        }
    }
}

impl CosmoParams {
    /// Dielectric scaling factor $f(\varepsilon) = \frac{\varepsilon - 1}{\varepsilon + 0.5}$.
    #[inline]
    pub fn dielectric_scaling(&self) -> f64 {
        if self.epsilon <= 1.0 {
            0.0
        } else {
            (self.epsilon - 1.0) / (self.epsilon + 0.5)
        }
    }
}

/// Solves the symmetric positive-definite linear system $L L^T x = y$
/// via forward and backward substitution using the Cholesky factor $L$.
///
/// Uses preallocated scratch slice `tmp` (length >= n) to avoid heap allocation.
#[allow(clippy::needless_range_loop)]
pub fn solve_cholesky_system(l: &AlignedMatrix<f64>, y: &[f64], x: &mut [f64], tmp: &mut [f64]) {
    let n = l.rows;
    assert_eq!(l.cols, n);
    assert_eq!(y.len(), n);
    assert_eq!(x.len(), n);
    assert!(tmp.len() >= n);

    // Forward substitution: L * tmp = y
    for k in 0..n {
        let mut sum = y[k];
        for i in 0..k {
            sum -= l.get(k, i) * tmp[i];
        }
        let lkk = l.get(k, k);
        assert!(lkk.abs() > 1e-15, "Cholesky diagonal pivot is zero");
        tmp[k] = sum / lkk;
    }

    // Backward substitution: L^T * x = tmp
    for k in (0..n).rev() {
        let mut sum = tmp[k];
        for i in (k + 1)..n {
            sum -= l.get(i, k) * x[i];
        }
        x[k] = sum / l.get(k, k);
    }
}

/// Precomputed COSMO electrostatic operators and dielectric state.
#[derive(Debug, Clone)]
pub struct CosmoState {
    pub params: CosmoParams,
    pub cavity: CosmoCavity,
    /// Cholesky lower factor $L$ of the electrostatic boundary matrix $A$.
    pub l_cholesky: AlignedMatrix<f64>,
    /// Coupling matrix $B$ between atomic orbitals and cavity segments (norbs x norbs x n_seg).
    pub b_matrix: AlignedMatrix<f64>,
    /// Nuclear electrostatic potential on cavity segments.
    pub phi_nuc: Vec<f64>,
    /// Screened nuclear surface charges $q_{\text{nuc}}$.
    pub q_nuc: Vec<f64>,
    /// Nuclear dielectric interaction free energy in eV.
    pub e_nuc_diel_ev: f64,
    /// Reusable electronic potential scratch buffer (0 malloc during iterative SCF cycles).
    pub phi_elec: Vec<f64>,
    /// Reusable screening charge intermediate buffer (0 malloc during iterative SCF cycles).
    pub q_star: Vec<f64>,
    /// Reusable screened electronic surface charge buffer (0 malloc during iterative SCF cycles).
    pub q_elec: Vec<f64>,
    /// Reusable Cholesky back-substitution scratch vector (0 malloc during iterative SCF cycles).
    pub solve_scratch: Vec<f64>,
}

impl CosmoState {
    /// Initialize the COSMO cavity, construct $A$ and $B$ matrices, and precompute nuclear dielectric screening.
    pub fn initialize(
        batch: &MolecularBatch,
        model: &dyn ParameterModel,
        params: CosmoParams,
    ) -> Result<Self, crate::ri::cholesky::MetricDefinitenessError> {
        let cavity = CosmoCavity::construct(batch, params.rsolv);
        let nseg = cavity.num_segments();
        assert!(nseg > 0, "COSMO cavity must contain at least one segment");

        // 1. Construct electrostatic interaction matrix A
        let mut a_mat = AlignedMatrix::zeroed(nseg, nseg);
        let fdiagr = 2.1 * PI.sqrt(); // 1.07 * sqrt(4*pi)

        for i in 0..nseg {
            let si = &cavity.segments[i];
            // Diagonal self-energy: A_ii = 2.1 * sqrt(pi) / sqrt(S_i)
            a_mat.set(i, i, fdiagr / si.area.sqrt());

            for j in 0..i {
                let sj = &cavity.segments[j];
                let dx = si.position[0] - sj.position[0];
                let dy = si.position[1] - sj.position[1];
                let dz = si.position[2] - sj.position[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let a_val = if dist > 1e-12 { 1.0 / dist } else { 0.0 };
                a_mat.set(i, j, a_val);
                a_mat.set(j, i, a_val);
            }
        }

        // Cholesky decomposition of A = L * L^T
        let mut l_cholesky = a_mat;
        cholesky_decompose(&mut l_cholesky)?;

        // 2. Construct coupling matrix B between atomic charge distributions and surface segments
        let norbs = batch.norbs;
        let mut b_matrix = AlignedMatrix::zeroed(norbs * norbs, nseg);

        for (seg_idx, seg) in cavity.segments.iter().enumerate() {
            for atom_idx in 0..batch.natoms {
                let z = batch.atomic_numbers[atom_idx];
                let elem = match model.get_element(z) {
                    Some(p) => p,
                    None => continue,
                };
                let orb_offset = batch.orbital_offsets[atom_idx];
                let num_ao = batch.basis_types[atom_idx].num_orbitals();
                let xa = [batch.x[atom_idx], batch.y[atom_idx], batch.z[atom_idx]];

                let dx = seg.position[0] - xa[0];
                let dy = seg.position[1] - xa[1];
                let dz = seg.position[2] - xa[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt();
                let inv_r = if r > 1e-12 { 1.0 / r } else { 0.0 };
                let inv_r3 = inv_r * inv_r * inv_r;

                // Monopole coupling (diagonal orbital terms mu == nu)
                for mu in 0..num_ao {
                    let global_mu = orb_offset + mu;
                    let flat_idx = global_mu * norbs + global_mu;
                    b_matrix.set(flat_idx, seg_idx, inv_r);
                }

                // Dipole and quadrupole coupling for sp hybridization terms
                if num_ao >= 4 {
                    let mp =
                        crate::integrals::multipoles::DerivedMultipoleParams::from_element(&elem);
                    let d1 = mp.dd * crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS;
                    let q2 = (mp.qq * crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS).powi(2);
                    let inv_r5 = inv_r3 * inv_r * inv_r;

                    let g_s = orb_offset;
                    let g_px = orb_offset + 1;
                    let g_py = orb_offset + 2;
                    let g_pz = orb_offset + 3;

                    // Quadrupole diagonal adjustments for px, py, pz
                    let q_xx = (3.0 * dx * dx * inv_r5 - inv_r3) * q2;
                    let q_yy = (3.0 * dy * dy * inv_r5 - inv_r3) * q2;
                    let q_zz = (3.0 * dz * dz * inv_r5 - inv_r3) * q2;

                    b_matrix.set(g_px * norbs + g_px, seg_idx, inv_r + q_xx);
                    b_matrix.set(g_py * norbs + g_py, seg_idx, inv_r + q_yy);
                    b_matrix.set(g_pz * norbs + g_pz, seg_idx, inv_r + q_zz);

                    // Dipole s-p coupling
                    let b_spx = dx * d1 * inv_r3;
                    let b_spy = dy * d1 * inv_r3;
                    let b_spz = dz * d1 * inv_r3;

                    b_matrix.set(g_s * norbs + g_px, seg_idx, b_spx);
                    b_matrix.set(g_px * norbs + g_s, seg_idx, b_spx);

                    b_matrix.set(g_s * norbs + g_py, seg_idx, b_spy);
                    b_matrix.set(g_py * norbs + g_s, seg_idx, b_spy);

                    b_matrix.set(g_s * norbs + g_pz, seg_idx, b_spz);
                    b_matrix.set(g_pz * norbs + g_s, seg_idx, b_spz);

                    // Quadrupole p-p off-diagonal coupling
                    let b_px_py = 3.0 * dx * dy * q2 * inv_r5;
                    let b_px_pz = 3.0 * dx * dz * q2 * inv_r5;
                    let b_py_pz = 3.0 * dy * dz * q2 * inv_r5;

                    b_matrix.set(g_px * norbs + g_py, seg_idx, b_px_py);
                    b_matrix.set(g_py * norbs + g_px, seg_idx, b_px_py);

                    b_matrix.set(g_px * norbs + g_pz, seg_idx, b_px_pz);
                    b_matrix.set(g_pz * norbs + g_px, seg_idx, b_px_pz);

                    b_matrix.set(g_py * norbs + g_pz, seg_idx, b_py_pz);
                    b_matrix.set(g_pz * norbs + g_py, seg_idx, b_py_pz);
                }
            }
        }

        // 3. Compute nuclear electrostatic potential on cavity segments
        let mut phi_nuc = vec![0.0f64; nseg];
        for (seg_idx, seg) in cavity.segments.iter().enumerate() {
            let mut pot = 0.0f64;
            for atom_idx in 0..batch.natoms {
                let z = batch.atomic_numbers[atom_idx];
                if let Some(p) = model.get_element(z) {
                    let z_core = p.core_charge;
                    let dx = seg.position[0] - batch.x[atom_idx];
                    let dy = seg.position[1] - batch.y[atom_idx];
                    let dz = seg.position[2] - batch.z[atom_idx];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist > 1e-12 {
                        pot += z_core / dist;
                    }
                }
            }
            phi_nuc[seg_idx] = pot;
        }

        // Allocate persistent scratch buffers for 0 malloc during iterative SCF cycles
        let phi_elec = vec![0.0f64; nseg];
        let mut q_star = vec![0.0f64; nseg];
        let q_elec = vec![0.0f64; nseg];
        let mut solve_scratch = vec![0.0f64; nseg];

        // 4. Solve for screened nuclear surface charges: A * q_nuc_star = phi_nuc
        solve_cholesky_system(&l_cholesky, &phi_nuc, &mut q_star, &mut solve_scratch);

        let fepsi = params.dielectric_scaling();
        let mut q_nuc = vec![0.0f64; nseg];
        let mut enclr = 0.0f64;
        for i in 0..nseg {
            q_nuc[i] = -fepsi * q_star[i];
            enclr += q_nuc[i] * phi_nuc[i];
        }

        // Nuclear dielectric interaction energy in eV: (1/2) * (a0 * ev) * sum (q_nuc * phi_nuc)
        let e_nuc_diel_ev = 0.5 * A0_EV * enclr;

        Ok(Self {
            params,
            cavity,
            l_cholesky,
            b_matrix,
            phi_nuc,
            q_nuc,
            e_nuc_diel_ev,
            phi_elec,
            q_star,
            q_elec,
            solve_scratch,
        })
    }

    /// Add nuclear screening reaction field to one-electron core Hamiltonian $H_{\text{core}}$.
    ///
    /// Matches OpenMOPAC `addhcr` subroutine in `cosmo.F90`.
    pub fn apply_nuclear_reaction_field_to_hcore(&self, h_core: &mut AlignedMatrix<f64>) {
        let norbs = h_core.rows;
        let nseg = self.cavity.num_segments();

        for mu in 0..norbs {
            for nu in 0..norbs {
                let flat_idx = mu * norbs + nu;
                let mut him = 0.0f64;
                for k in 0..nseg {
                    him += self.b_matrix.get(flat_idx, k) * self.q_nuc[k];
                }
                let h_old = h_core.get(mu, nu);
                h_core.set(mu, nu, h_old - A0_EV * him);
            }
        }
    }

    /// Compute electronic screening reaction field and dielectric solvation free energy.
    ///
    /// Matches OpenMOPAC `addfck` subroutine in `cosmo.F90`.
    /// Modifies the Fock matrix in-place and returns the total dielectric energy in eV.
    ///
    /// Strictly ZERO dynamic heap allocations (0 malloc) inside the iterative SCF cycle.
    pub fn apply_electronic_reaction_field_to_fock(
        &mut self,
        density: &AlignedMatrix<f64>,
        fock: &mut AlignedMatrix<f64>,
    ) -> f64 {
        let norbs = fock.rows;
        let nseg = self.cavity.num_segments();
        let fepsi = self.params.dielectric_scaling();

        // 1. Calculate electronic electrostatic potential phi_elec = B * Q_elec
        // In NDDO, electronic charge density is Q_elec,mu_nu = -P_mu_nu
        self.phi_elec.fill(0.0);
        for k in 0..nseg {
            let mut phi = 0.0f64;
            for mu in 0..norbs {
                for nu in 0..norbs {
                    let p_val = density.get(mu, nu);
                    if p_val.abs() > 1e-12 {
                        let b_val = self.b_matrix.get(mu * norbs + nu, k);
                        phi -= b_val * p_val;
                    }
                }
            }
            self.phi_elec[k] = phi;
        }

        // 2. Solve for screened electronic surface charges: A * q_elec_star = phi_elec
        solve_cholesky_system(
            &self.l_cholesky,
            &self.phi_elec,
            &mut self.q_star,
            &mut self.solve_scratch,
        );

        let mut ediel = 0.0f64;

        for k in 0..nseg {
            self.q_elec[k] = -fepsi * self.q_star[k];
            let q_tot = self.q_nuc[k] + self.q_elec[k];
            let phi_tot = self.phi_nuc[k] + self.phi_elec[k];
            ediel += q_tot * phi_tot;
        }

        let total_dielectric_energy_ev = 0.5 * A0_EV * ediel;

        // 3. Add reaction field to Fock matrix: F_mu_nu -= (a0 * ev) * sum_k B_mu_nu,k * q_elec,k
        for mu in 0..norbs {
            for nu in 0..norbs {
                let flat_idx = mu * norbs + nu;
                let mut fim = 0.0f64;
                for k in 0..nseg {
                    fim += self.b_matrix.get(flat_idx, k) * self.q_elec[k];
                }
                let f_old = fock.get(mu, nu);
                fock.set(mu, nu, f_old - A0_EV * fim);
            }
        }

        total_dielectric_energy_ev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::am1::Am1Model;

    #[test]
    fn test_cosmo_water_initialization_and_screening() {
        let coords = vec![
            [0.000, 0.000, 0.000],
            [0.757, 0.586, 0.000],
            [-0.757, 0.586, 0.000],
        ];
        let z = vec![8, 1, 1];
        let batch = MolecularBatch::new(z, &coords);
        let am1 = Am1Model;

        let params = CosmoParams {
            epsilon: 78.4,
            rsolv: 1.30005,
        };

        let state =
            CosmoState::initialize(&batch, &am1, params).expect("COSMO initialization failed");
        assert!(state.cavity.num_segments() > 0);
        assert!(
            state.e_nuc_diel_ev < 0.0,
            "Nuclear screening energy must be negative/stabilizing"
        );
        println!(
            "[COSMO] Water COSMO Segments: {}, Area: {:.2} A^2, Vol: {:.2} A^3, E_nuc_diel: {:.4} eV",
            state.cavity.num_segments(),
            state.cavity.total_area_angstrom2,
            state.cavity.total_volume_angstrom3,
            state.e_nuc_diel_ev
        );
    }
}
