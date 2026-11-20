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
use rayon::prelude::*;
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
    /// Compact intra-atom coupling matrix $B$ (`total_intra_pairs x n_seg`).
    pub b_matrix: AlignedMatrix<f64>,
    /// Starting row offset in `b_matrix` for each atom's intra-atom orbital pairs.
    pub atom_pair_offsets: Vec<usize>,
    /// Orbital offset and count `(orb_offset, num_ao)` for each atom in `b_matrix`.
    atom_orbital_ranges: Vec<(usize, usize)>,
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

        // 1. Construct electrostatic interaction matrix A with canonical fine element integration
        let mut a_mat = AlignedMatrix::zeroed(nseg, nseg);
        let fdiagr = 2.1 * PI.sqrt(); // 1.07 * sqrt(4*pi)
        let disex2 = 4.0 * (1.7 * 4.0f64).powi(2) / 42.0; // 4.4038 A^2
        let segments = &cavity.segments;

        let fill_a_row = |i: usize, row: &mut [f64]| {
            let si = &segments[i];
            let nfl_i = si.fine_elements.len();

            // Diagonal self-energy with fine element pairwise integration (mfinel / coscav)
            let mut aa_diag = 0.0f64;
            for k in 0..nfl_i {
                let sk = si.fine_elements[k].area;
                aa_diag += fdiagr * sk.powf(1.5);
                let xk = si.fine_elements[k].position;
                for l in 0..k {
                    let sl = si.fine_elements[l].area;
                    let xl = si.fine_elements[l].position;
                    let dist = ((xk[0] - xl[0]).powi(2)
                        + (xk[1] - xl[1]).powi(2)
                        + (xk[2] - xl[2]).powi(2))
                    .sqrt();
                    if dist > 1e-12 {
                        aa_diag += 2.0 * sk * sl / dist;
                    }
                }
            }
            row[i] = if si.area > 1e-12 && aa_diag > 0.0 {
                aa_diag / (si.area * si.area)
            } else {
                fdiagr / si.area.max(1e-6).sqrt()
            };

            for j in 0..i {
                let sj = &segments[j];
                let dx = si.position[0] - sj.position[0];
                let dy = si.position[1] - sj.position[1];
                let dz = si.position[2] - sj.position[2];
                let d2 = dx * dx + dy * dy + dz * dz;

                let a_val = if d2 > disex2 {
                    1.0 / d2.sqrt()
                } else {
                    let nfl_j = sj.fine_elements.len();
                    if nfl_i > 0 && nfl_j > 0 {
                        let mut aa_near = 0.0f64;
                        for fk in &si.fine_elements {
                            for fl in &sj.fine_elements {
                                let dist = ((fk.position[0] - fl.position[0]).powi(2)
                                    + (fk.position[1] - fl.position[1]).powi(2)
                                    + (fk.position[2] - fl.position[2]).powi(2))
                                .sqrt();
                                if dist > 1e-12 {
                                    aa_near += fk.area * fl.area / dist;
                                }
                            }
                        }
                        if si.area > 1e-12 && sj.area > 1e-12 {
                            aa_near / (si.area * sj.area)
                        } else {
                            1.0 / d2.max(1e-6).sqrt()
                        }
                    } else {
                        1.0 / d2.max(1e-6).sqrt()
                    }
                };
                row[j] = a_val;
            }
        };

        if nseg >= 64 && rayon::current_num_threads() > 1 {
            a_mat
                .data
                .par_chunks_mut(nseg)
                .enumerate()
                .for_each(|(i, row)| fill_a_row(i, row));
        } else {
            a_mat
                .data
                .chunks_mut(nseg)
                .enumerate()
                .for_each(|(i, row)| fill_a_row(i, row));
        }

        // Mirror lower triangle to upper triangle
        for i in 1..nseg {
            let row_i_off = i * nseg;
            for j in 0..i {
                let val = a_mat.data[row_i_off + j];
                a_mat.data[j * nseg + i] = val;
            }
        }

        // Cholesky decomposition of A = L * L^T
        let mut l_cholesky = a_mat;
        cholesky_decompose(&mut l_cholesky)?;

        // 2. Construct compact intra-atom coupling matrix B between atomic charge distributions and segments
        //    Atom-outer layout: each atom owns contiguous rows [pair_base .. pair_base + num_ao^2],
        //    hoisting element & multipole parameter lookup once per atom instead of nseg * natoms.
        type AtomCosmoMeta = (usize, [f64; 3], Option<(f64, f64)>, f64);
        let mut atom_pair_offsets = Vec::with_capacity(batch.natoms);
        let mut atom_orbital_ranges = Vec::with_capacity(batch.natoms);
        let mut atom_meta: Vec<Option<AtomCosmoMeta>> = Vec::with_capacity(batch.natoms);
        let mut total_intra_pairs = 0usize;
        for atom_idx in 0..batch.natoms {
            let orb_offset = batch.orbital_offsets[atom_idx];
            let num_ao = batch.basis_types[atom_idx].num_orbitals();
            atom_pair_offsets.push(total_intra_pairs);
            atom_orbital_ranges.push((orb_offset, num_ao));
            total_intra_pairs += num_ao * num_ao;

            let z = batch.atomic_numbers[atom_idx];
            let meta = model.get_element(z).map(|elem| {
                let xa = [batch.x[atom_idx], batch.y[atom_idx], batch.z[atom_idx]];
                let sp_multipole = if num_ao >= 4 {
                    let mp =
                        crate::integrals::multipoles::DerivedMultipoleParams::from_element(&elem);
                    let d1 = mp.dd * crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS;
                    let q2 = (mp.qq * crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS).powi(2);
                    Some((d1, q2))
                } else {
                    None
                };
                (num_ao, xa, sp_multipole, elem.core_charge)
            });
            atom_meta.push(meta);
        }
        let mut b_matrix = AlignedMatrix::zeroed(total_intra_pairs, nseg);

        // Partition b_matrix.data into disjoint per-atom row bands for lock-free parallel writes
        let mut atom_bands: Vec<(usize, &mut [f64])> = Vec::with_capacity(batch.natoms);
        let mut rest: &mut [f64] = &mut b_matrix.data[..];
        for (atom_idx, &(_, num_ao)) in atom_orbital_ranges.iter().enumerate() {
            let band_len = num_ao * num_ao * nseg;
            let (head, tail) = rest.split_at_mut(band_len);
            atom_bands.push((atom_idx, head));
            rest = tail;
        }

        let fill_atom_b = |(atom_idx, band): &mut (usize, &mut [f64])| {
            let Some((num_ao, xa, sp_multipole, _)) = atom_meta[*atom_idx] else {
                return;
            };
            for (seg_idx, seg) in segments.iter().enumerate() {
                let dx = seg.position[0] - xa[0];
                let dy = seg.position[1] - xa[1];
                let dz = seg.position[2] - xa[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt();
                let inv_r = if r > 1e-12 { 1.0 / r } else { 0.0 };
                let inv_r3 = inv_r * inv_r * inv_r;

                // Monopole coupling (diagonal orbital terms mu == nu)
                for mu in 0..num_ao {
                    band[(mu * num_ao + mu) * nseg + seg_idx] = inv_r;
                }

                // Dipole and quadrupole coupling for sp hybridization terms
                if let Some((d1, q2)) = sp_multipole {
                    let inv_r5 = inv_r3 * inv_r * inv_r;

                    let s = 0usize;
                    let px = 1usize;
                    let py = 2usize;
                    let pz = 3usize;

                    // Quadrupole diagonal adjustments for px, py, pz
                    let q_xx = (3.0 * dx * dx * inv_r5 - inv_r3) * q2;
                    let q_yy = (3.0 * dy * dy * inv_r5 - inv_r3) * q2;
                    let q_zz = (3.0 * dz * dz * inv_r5 - inv_r3) * q2;

                    band[(px * num_ao + px) * nseg + seg_idx] = inv_r + q_xx;
                    band[(py * num_ao + py) * nseg + seg_idx] = inv_r + q_yy;
                    band[(pz * num_ao + pz) * nseg + seg_idx] = inv_r + q_zz;

                    // Dipole s-p coupling
                    let b_spx = dx * d1 * inv_r3;
                    let b_spy = dy * d1 * inv_r3;
                    let b_spz = dz * d1 * inv_r3;

                    band[(s * num_ao + px) * nseg + seg_idx] = b_spx;
                    band[(px * num_ao + s) * nseg + seg_idx] = b_spx;

                    band[(s * num_ao + py) * nseg + seg_idx] = b_spy;
                    band[(py * num_ao + s) * nseg + seg_idx] = b_spy;

                    band[(s * num_ao + pz) * nseg + seg_idx] = b_spz;
                    band[(pz * num_ao + s) * nseg + seg_idx] = b_spz;

                    // Quadrupole p-p off-diagonal coupling
                    let b_px_py = 3.0 * dx * dy * q2 * inv_r5;
                    let b_px_pz = 3.0 * dx * dz * q2 * inv_r5;
                    let b_py_pz = 3.0 * dy * dz * q2 * inv_r5;

                    band[(px * num_ao + py) * nseg + seg_idx] = b_px_py;
                    band[(py * num_ao + px) * nseg + seg_idx] = b_px_py;

                    band[(px * num_ao + pz) * nseg + seg_idx] = b_px_pz;
                    band[(pz * num_ao + px) * nseg + seg_idx] = b_px_pz;

                    band[(py * num_ao + pz) * nseg + seg_idx] = b_py_pz;
                    band[(pz * num_ao + py) * nseg + seg_idx] = b_py_pz;
                }
            }
        };

        if batch.natoms >= 8 && nseg >= 64 && rayon::current_num_threads() > 1 {
            atom_bands.par_iter_mut().for_each(fill_atom_b);
        } else {
            atom_bands.iter_mut().for_each(fill_atom_b);
        }

        // 3. Compute nuclear electrostatic potential on cavity segments
        let mut phi_nuc = vec![0.0f64; nseg];
        for (seg_idx, seg) in segments.iter().enumerate() {
            let mut pot = 0.0f64;
            for (_, xa, _, z_core) in atom_meta.iter().flatten() {
                let dx = seg.position[0] - xa[0];
                let dy = seg.position[1] - xa[1];
                let dz = seg.position[2] - xa[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist > 1e-12 {
                    pot += z_core / dist;
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
            atom_pair_offsets,
            atom_orbital_ranges,
            phi_nuc,
            q_nuc,
            e_nuc_diel_ev,
            phi_elec,
            q_star,
            q_elec,
            solve_scratch,
        })
    }

    /// Accumulate electronic electrostatic potential `phi_elec = B * Q_elec` over intra-atom blocks.
    fn compute_phi_elec_blocks(
        atom_pair_offsets: &[usize],
        atom_orbital_ranges: &[(usize, usize)],
        b_matrix: &AlignedMatrix<f64>,
        density: &AlignedMatrix<f64>,
        phi_out: &mut [f64],
    ) {
        let nseg = b_matrix.cols;
        phi_out[..nseg].fill(0.0);
        for (&pair_base, &(orb_offset, num_ao)) in
            atom_pair_offsets.iter().zip(atom_orbital_ranges.iter())
        {
            for i in 0..num_ao {
                let mu = orb_offset + i;
                let row_atom = pair_base + i * num_ao;
                for j in 0..num_ao {
                    let nu = orb_offset + j;
                    let p_val = density.get(mu, nu);
                    if p_val.abs() > 1e-12 {
                        let row_start = (row_atom + j) * nseg;
                        let b_row = &b_matrix.data[row_start..row_start + nseg];
                        for (phi_k, &b_val) in phi_out[..nseg].iter_mut().zip(b_row.iter()) {
                            *phi_k -= b_val * p_val;
                        }
                    }
                }
            }
        }
    }

    /// Apply reaction field `- A0_EV * B * q` over compact intra-atom blocks into `target`.
    fn apply_reaction_field_blocks(&self, q_vec: &[f64], target: &mut AlignedMatrix<f64>) {
        let nseg = q_vec.len();
        for (&pair_base, &(orb_offset, num_ao)) in self
            .atom_pair_offsets
            .iter()
            .zip(self.atom_orbital_ranges.iter())
        {
            for i in 0..num_ao {
                let mu = orb_offset + i;
                let row_atom = pair_base + i * num_ao;
                for j in 0..num_ao {
                    let nu = orb_offset + j;
                    let row_start = (row_atom + j) * nseg;
                    let b_row = &self.b_matrix.data[row_start..row_start + nseg];
                    let mut sum = 0.0f64;
                    for (&b_val, &q_val) in b_row.iter().zip(q_vec.iter()) {
                        sum += b_val * q_val;
                    }
                    let old_val = target.get(mu, nu);
                    target.set(mu, nu, old_val - A0_EV * sum);
                }
            }
        }
    }

    /// Add nuclear screening reaction field to one-electron core Hamiltonian $H_{\text{core}}$.
    ///
    /// Matches OpenMOPAC `addhcr` subroutine in `cosmo.F90`.
    pub fn apply_nuclear_reaction_field_to_hcore(&self, h_core: &mut AlignedMatrix<f64>) {
        self.apply_reaction_field_blocks(&self.q_nuc, h_core);
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
        let nseg = self.cavity.num_segments();
        let fepsi = self.params.dielectric_scaling();

        // 1. Calculate electronic electrostatic potential phi_elec = B * Q_elec
        // In NDDO, electronic charge density is Q_elec,mu_nu = -P_mu_nu
        Self::compute_phi_elec_blocks(
            &self.atom_pair_offsets,
            &self.atom_orbital_ranges,
            &self.b_matrix,
            density,
            &mut self.phi_elec,
        );

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
        self.apply_reaction_field_blocks(&self.q_elec, fock);

        total_dielectric_energy_ev
    }

    /// Compute analytical Cartesian nuclear gradients $\nabla E_{\text{diel}}$ (in eV / Å)
    /// of the dielectric reaction field matching OpenMOPAC `diegrd` (`cosmo.F90`).
    ///
    /// Evaluates:
    /// 1. Inter-segment electrostatic forces between conductor screening charges on different atoms.
    /// 2. Segment-atom electrostatic forces between screening charges and solute atomic net charges.
    pub fn compute_dielectric_gradients(
        &self,
        batch: &MolecularBatch,
        model: &dyn ParameterModel,
        density: &AlignedMatrix<f64>,
        gradients: &mut [[f64; 3]],
    ) {
        let fepsi = self.params.dielectric_scaling();
        if fepsi <= 1e-12 {
            return;
        }

        let nseg = self.cavity.num_segments();
        let natoms = batch.natoms;
        assert_eq!(gradients.len(), natoms);

        // 0. Compute actual electronic screening charges corresponding to the provided density matrix
        let mut phi_elec = vec![0.0f64; nseg];
        Self::compute_phi_elec_blocks(
            &self.atom_pair_offsets,
            &self.atom_orbital_ranges,
            &self.b_matrix,
            density,
            &mut phi_elec,
        );
        let mut q_star = vec![0.0f64; nseg];
        let mut scratch = vec![0.0f64; nseg];
        solve_cholesky_system(&self.l_cholesky, &phi_elec, &mut q_star, &mut scratch);
        let mut q_elec = vec![0.0f64; nseg];
        for k in 0..nseg {
            q_elec[k] = -fepsi * q_star[k];
        }

        // Precompute solute net atomic charges: Q_A = Z_core_A - sum_{mu in A} P_{mu, mu}
        let mut net_charges = vec![0.0f64; natoms];
        // Precompute solute atomic dipoles from sp hybridization
        let mut atomic_dipoles = vec![[0.0f64; 3]; natoms];
        for (a, net_charge) in net_charges.iter_mut().enumerate() {
            let z = batch.atomic_numbers[a];
            let z_core = model.get_element(z).map(|p| p.core_charge).unwrap_or(0.0);
            let orb_offset = batch.orbital_offsets[a];
            let num_ao = batch.basis_types[a].num_orbitals();
            let mut p_pop = 0.0f64;
            for mu in 0..num_ao {
                p_pop += density.get(orb_offset + mu, orb_offset + mu);
            }
            *net_charge = z_core - p_pop;

            if num_ao >= 4
                && let Some(elem) = model.get_element(z)
            {
                let mp = crate::integrals::multipoles::DerivedMultipoleParams::from_element(&elem);
                let d1 = mp.dd * crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS;
                let mux = 2.0 * density.get(orb_offset, orb_offset + 1) * d1;
                let muy = 2.0 * density.get(orb_offset, orb_offset + 2) * d1;
                let muz = 2.0 * density.get(orb_offset, orb_offset + 3) * d1;
                atomic_dipoles[a] = [mux, muy, muz];
            }
        }

        // 1. Conductor screening charge segment-segment forces
        for (k, &q_elec_k) in q_elec.iter().enumerate().take(nseg) {
            let iak = self.cavity.segments[k].atom_index;
            let xk = self.cavity.segments[k].position;
            let qsk = self.q_nuc[k] + q_elec_k;

            for (l, &q_elec_l) in q_elec.iter().enumerate().take(k) {
                let ial = self.cavity.segments[l].atom_index;
                if ial != iak {
                    let xl = self.cavity.segments[l].position;
                    let qsl = self.q_nuc[l] + q_elec_l;

                    let dx = xl[0] - xk[0];
                    let dy = xl[1] - xk[1];
                    let dz = xl[2] - xk[2];
                    let dist2 = dx * dx + dy * dy + dz * dz;

                    if dist2 > 1e-10 {
                        let dist3 = dist2 * dist2.sqrt();
                        let ff = qsk * qsl * (-A0_EV) / (dist3 * fepsi);

                        gradients[iak][0] -= dx * ff;
                        gradients[iak][1] -= dy * ff;
                        gradients[iak][2] -= dz * ff;

                        gradients[ial][0] += dx * ff;
                        gradients[ial][1] += dy * ff;
                        gradients[ial][2] += dz * ff;
                    }
                }
            }
        }

        // 2. Segment-atom electrostatic forces (monopole + multipolar dB derivatives)
        for (k, &q_elec_k) in q_elec.iter().enumerate().take(nseg) {
            let iak = self.cavity.segments[k].atom_index;
            let xk = self.cavity.segments[k].position;
            let qsk = self.q_nuc[k] + q_elec_k;

            for j in 0..natoms {
                if j != iak {
                    let xj = [batch.x[j], batch.y[j], batch.z[j]];
                    let dx = xk[0] - xj[0];
                    let dy = xk[1] - xj[1];
                    let dz = xk[2] - xj[2];
                    let dist2 = dx * dx + dy * dy + dz * dz;

                    if dist2 > 1e-10 {
                        let dist3 = dist2 * dist2.sqrt();
                        let dist5 = dist3 * dist2;
                        let ff0 = -qsk * (-A0_EV) / dist3;
                        let ff = -ff0 * net_charges[j];

                        // Monopole force
                        gradients[iak][0] += dx * ff;
                        gradients[iak][1] += dy * ff;
                        gradients[iak][2] += dz * ff;

                        gradients[j][0] -= dx * ff;
                        gradients[j][1] -= dy * ff;
                        gradients[j][2] -= dz * ff;

                        // Multipolar dipole force (dB derivative)
                        let mu = atomic_dipoles[j];
                        let mu_dot_r = mu[0] * dx + mu[1] * dy + mu[2] * dz;
                        let factor = qsk * A0_EV;
                        let f_dip_x = factor * (mu[0] / dist3 - 3.0 * mu_dot_r * dx / dist5);
                        let f_dip_y = factor * (mu[1] / dist3 - 3.0 * mu_dot_r * dy / dist5);
                        let f_dip_z = factor * (mu[2] / dist3 - 3.0 * mu_dot_r * dz / dist5);

                        gradients[iak][0] += f_dip_x;
                        gradients[iak][1] += f_dip_y;
                        gradients[iak][2] += f_dip_z;

                        gradients[j][0] -= f_dip_x;
                        gradients[j][1] -= f_dip_y;
                        gradients[j][2] -= f_dip_z;
                    }
                }
            }
        }
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

    #[test]
    fn test_cosmo_parallel_initialization_exact_parity() {
        let mut z = Vec::new();
        let mut coords = Vec::new();
        for c in 0..6 {
            z.push(6u8);
            coords.push([c as f64 * 1.53, (c % 2) as f64 * 0.5, 0.0]);
            let n_h = if c == 0 || c == 5 { 3 } else { 2 };
            for h in 0..n_h {
                z.push(1u8);
                coords.push([
                    c as f64 * 1.53 + 0.3 * h as f64,
                    (c % 2) as f64 * 0.5 + if h == 0 { 1.09 } else { -0.8 },
                    if h == 2 { 0.85 } else { 0.2 },
                ]);
            }
        }
        let batch = MolecularBatch::new(z, &coords);
        let am1 = Am1Model;
        let params = CosmoParams::default();

        let state_1 = crate::threading::run_with_thread_pool(Some(1), || {
            CosmoState::initialize(&batch, &am1, params).expect("1-thread COSMO init")
        });
        let state_4 = crate::threading::run_with_thread_pool(Some(4), || {
            CosmoState::initialize(&batch, &am1, params).expect("4-thread COSMO init")
        });

        assert_eq!(state_1.cavity.num_segments(), state_4.cavity.num_segments());
        assert!(state_1.cavity.num_segments() >= 64);

        let max_l_diff = state_1
            .l_cholesky
            .data
            .iter()
            .zip(state_4.l_cholesky.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let max_b_diff = state_1
            .b_matrix
            .data
            .iter()
            .zip(state_4.b_matrix.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let max_phi_diff = state_1
            .phi_nuc
            .iter()
            .zip(state_4.phi_nuc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_l_diff < 1e-13,
            "Parallel COSMO L_cholesky diverged: {:e}",
            max_l_diff
        );
        assert!(
            max_b_diff < 1e-13,
            "Parallel COSMO B matrix diverged: {:e}",
            max_b_diff
        );
        assert!(
            max_phi_diff < 1e-13,
            "Parallel COSMO phi_nuc diverged: {:e}",
            max_phi_diff
        );
        assert!(
            (state_1.e_nuc_diel_ev - state_4.e_nuc_diel_ev).abs() < 1e-13,
            "Parallel COSMO E_nuc_diel diverged"
        );
    }
}
