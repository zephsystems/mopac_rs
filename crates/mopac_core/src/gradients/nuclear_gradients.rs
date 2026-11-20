//! Analytical Nuclear Gradients and Cartesian Forces.
//!
//! Evaluates the first spatial derivatives of the total quantum chemical energy
//! with respect to Cartesian coordinates:
//! $$\vec{g}_A = \nabla_A E_{\text{total}} = \left( \frac{\partial E}{\partial X_A}, \frac{\partial E}{\partial Y_A}, \frac{\partial E}{\partial Z_A} \right)$$
//!
//! Implements the frozen-density Hellmann-Feynman/Pulay formulation of MOPAC `dcart.F90`,
//! with step size $\delta = 10^{-4} \text{ \AA}$ (`chnge = 1.D-4`).
//! Strictly conserves total linear momentum ($\sum_A \vec{g}_A = \vec{0}$).

use crate::constants::codata2018::EV_TO_KCAL_MOL;
use crate::fock::fock_builder::build_fock;
use crate::hamiltonian::hcore::build_hcore;
use crate::integrals::core_repulsion::compute_total_core_repulsion;
use crate::parameters::ParameterModel;
use crate::scf::density::compute_electronic_energy;
use crate::types::{AlignedMatrix, MolecularBatch};
use rayon::prelude::*;

/// Pre-allocated workspace for gradient calculations (0 heap allocations).
#[derive(Debug, Clone)]
pub struct GradientWorkspace {
    pub h_core: AlignedMatrix<f64>,
    pub fock: AlignedMatrix<f64>,
    pub fock_b: AlignedMatrix<f64>,
    pub density_tot: AlignedMatrix<f64>,
}

impl GradientWorkspace {
    /// Allocate reusable workspace for a system of `norbs` basis orbitals.
    pub fn allocate(norbs: usize) -> Self {
        Self {
            h_core: AlignedMatrix::zeroed(norbs, norbs),
            fock: AlignedMatrix::zeroed(norbs, norbs),
            fock_b: AlignedMatrix::zeroed(norbs, norbs),
            density_tot: AlignedMatrix::zeroed(norbs, norbs),
        }
    }
}

/// Evaluates total energy of a molecular geometry with frozen density matrix $P$.
fn evaluate_frozen_energy(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    density: &AlignedMatrix<f64>,
    ws: &mut GradientWorkspace,
    use_nddo: bool,
) -> f64 {
    let e_nuc = compute_total_core_repulsion(batch, model);
    if use_nddo {
        let pairs = crate::integrals::multipoles::precompute_diatomic_pairs(batch, model);
        crate::hamiltonian::hcore::build_hcore_nddo(batch, model, &pairs, &mut ws.h_core);
        crate::fock::fock_builder::build_fock_nddo(
            batch,
            model,
            &pairs,
            &ws.h_core,
            density,
            &mut ws.fock,
        );
    } else {
        build_hcore(batch, model, &mut ws.h_core);
        build_fock(batch, model, &ws.h_core, density, &mut ws.fock);
    }
    let e_elec = compute_electronic_energy(density, &ws.h_core, &ws.fock);
    e_elec + e_nuc
}

/// Evaluates total energy of an open-shell molecular geometry with frozen spin density matrices $P^\alpha, P^\beta$.
fn evaluate_frozen_energy_uhf(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    density_a: &AlignedMatrix<f64>,
    density_b: &AlignedMatrix<f64>,
    ws: &mut GradientWorkspace,
    use_nddo: bool,
) -> f64 {
    let e_nuc = compute_total_core_repulsion(batch, model);
    let norbs = batch.norbs;
    for i in 0..norbs * norbs {
        ws.density_tot.data[i] = density_a.data[i] + density_b.data[i];
    }

    if use_nddo {
        let pairs = crate::integrals::multipoles::precompute_diatomic_pairs(batch, model);
        crate::hamiltonian::hcore::build_hcore_nddo(batch, model, &pairs, &mut ws.h_core);
        crate::fock::fock_builder::build_fock_uhf(
            batch,
            model,
            &pairs,
            &ws.h_core,
            &ws.density_tot,
            density_a,
            &mut ws.fock,
        );
        crate::fock::fock_builder::build_fock_uhf(
            batch,
            model,
            &pairs,
            &ws.h_core,
            &ws.density_tot,
            density_b,
            &mut ws.fock_b,
        );
    } else {
        build_hcore(batch, model, &mut ws.h_core);
        crate::fock::fock_builder::build_fock_spin(
            batch,
            model,
            &ws.h_core,
            &ws.density_tot,
            density_a,
            &mut ws.fock,
        );
        crate::fock::fock_builder::build_fock_spin(
            batch,
            model,
            &ws.h_core,
            &ws.density_tot,
            density_b,
            &mut ws.fock_b,
        );
    }

    let e_elec = crate::scf::density::compute_uhf_electronic_energy(
        density_a, density_b, &ws.h_core, &ws.fock, &ws.fock_b,
    );
    e_elec + e_nuc
}

/// Reusable thread-local scratch buffers for diatomic pair gradient evaluations.
/// Eliminates `MolecularBatch` and `GradientWorkspace` heap allocations inside the O(N^2) pair loop.
struct DiatomicGradScratch {
    pair_batch: MolecularBatch,
    ws_by_norbs: [Option<GradientWorkspace>; 19],
    p_a_by_norbs: [Option<AlignedMatrix<f64>>; 19],
    p_b_by_norbs: [Option<AlignedMatrix<f64>>; 19],
}

impl DiatomicGradScratch {
    fn new() -> Self {
        Self {
            pair_batch: MolecularBatch::new(vec![1, 1], &[[0.0; 3], [0.0; 3]]),
            ws_by_norbs: std::array::from_fn(|_| None),
            p_a_by_norbs: std::array::from_fn(|_| None),
            p_b_by_norbs: std::array::from_fn(|_| None),
        }
    }

    fn prepare_pair(
        &mut self,
        za: u8,
        zb: u8,
        model: &dyn ParameterModel,
    ) -> (
        &mut MolecularBatch,
        &mut GradientWorkspace,
        &mut AlignedMatrix<f64>,
        &mut AlignedMatrix<f64>,
    ) {
        let ba = model.basis_type(za);
        let bb = model.basis_type(zb);
        let num_a = ba.num_orbitals();
        let num_b = bb.num_orbitals();
        let norbs_ab = num_a + num_b;

        self.pair_batch.atomic_numbers[0] = za;
        self.pair_batch.atomic_numbers[1] = zb;
        self.pair_batch.basis_types[0] = ba;
        self.pair_batch.basis_types[1] = bb;
        self.pair_batch.orbital_offsets[0] = 0;
        self.pair_batch.orbital_offsets[1] = num_a;
        self.pair_batch.norbs = norbs_ab;

        let ws =
            self.ws_by_norbs[norbs_ab].get_or_insert_with(|| GradientWorkspace::allocate(norbs_ab));
        let p_a = self.p_a_by_norbs[norbs_ab]
            .get_or_insert_with(|| AlignedMatrix::zeroed(norbs_ab, norbs_ab));
        let p_b = self.p_b_by_norbs[norbs_ab]
            .get_or_insert_with(|| AlignedMatrix::zeroed(norbs_ab, norbs_ab));

        (&mut self.pair_batch, ws, p_a, p_b)
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_rhf_pair_gradient(
    a: usize,
    b: usize,
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    density: &AlignedMatrix<f64>,
    use_nddo: bool,
    delta: f64,
    inv_2delta: f64,
    scratch: &mut DiatomicGradScratch,
) -> [f64; 3] {
    let num_a = batch.basis_types[a].num_orbitals();
    let off_a = batch.orbital_offsets[a];
    let za = batch.atomic_numbers[a];

    let num_b = batch.basis_types[b].num_orbitals();
    let off_b = batch.orbital_offsets[b];
    let zb = batch.atomic_numbers[b];

    let coord_a = [batch.x[a], batch.y[a], batch.z[a]];
    let coord_b = [batch.x[b], batch.y[b], batch.z[b]];

    let (pair_batch, ws_ab, p_ab, _) = scratch.prepare_pair(za, zb, model);

    for mu in 0..num_a {
        for nu in 0..num_a {
            p_ab.set(mu, nu, density.get(off_a + mu, off_a + nu));
        }
        for nu in 0..num_b {
            let val = density.get(off_a + mu, off_b + nu);
            p_ab.set(mu, num_a + nu, val);
            p_ab.set(num_a + nu, mu, val);
        }
    }
    for mu in 0..num_b {
        for nu in 0..num_b {
            p_ab.set(num_a + mu, num_a + nu, density.get(off_b + mu, off_b + nu));
        }
    }

    pair_batch.x[1] = coord_b[0];
    pair_batch.y[1] = coord_b[1];
    pair_batch.z[1] = coord_b[2];

    let mut g_pair = [0.0f64; 3];
    for (k, g_k) in g_pair.iter_mut().enumerate() {
        pair_batch.x[0] = coord_a[0];
        pair_batch.y[0] = coord_a[1];
        pair_batch.z[0] = coord_a[2];
        match k {
            0 => pair_batch.x[0] += delta,
            1 => pair_batch.y[0] += delta,
            2 => pair_batch.z[0] += delta,
            _ => unreachable!(),
        }
        let e_plus = evaluate_frozen_energy(pair_batch, model, p_ab, ws_ab, use_nddo);

        pair_batch.x[0] = coord_a[0];
        pair_batch.y[0] = coord_a[1];
        pair_batch.z[0] = coord_a[2];
        match k {
            0 => pair_batch.x[0] -= delta,
            1 => pair_batch.y[0] -= delta,
            2 => pair_batch.z[0] -= delta,
            _ => unreachable!(),
        }
        let e_minus = evaluate_frozen_energy(pair_batch, model, p_ab, ws_ab, use_nddo);

        *g_k = (e_plus - e_minus) * inv_2delta;
    }
    g_pair
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_uhf_pair_gradient(
    a: usize,
    b: usize,
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    density_a: &AlignedMatrix<f64>,
    density_b: &AlignedMatrix<f64>,
    use_nddo: bool,
    delta: f64,
    inv_2delta: f64,
    scratch: &mut DiatomicGradScratch,
) -> [f64; 3] {
    let num_a = batch.basis_types[a].num_orbitals();
    let off_a = batch.orbital_offsets[a];
    let za = batch.atomic_numbers[a];

    let num_b = batch.basis_types[b].num_orbitals();
    let off_b = batch.orbital_offsets[b];
    let zb = batch.atomic_numbers[b];

    let coord_a = [batch.x[a], batch.y[a], batch.z[a]];
    let coord_b = [batch.x[b], batch.y[b], batch.z[b]];

    let (pair_batch, ws_ab, p_a_ab, p_b_ab) = scratch.prepare_pair(za, zb, model);

    for mu in 0..num_a {
        for nu in 0..num_a {
            p_a_ab.set(mu, nu, density_a.get(off_a + mu, off_a + nu));
            p_b_ab.set(mu, nu, density_b.get(off_a + mu, off_a + nu));
        }
        for nu in 0..num_b {
            let va = density_a.get(off_a + mu, off_b + nu);
            let vb = density_b.get(off_a + mu, off_b + nu);
            p_a_ab.set(mu, num_a + nu, va);
            p_a_ab.set(num_a + nu, mu, va);
            p_b_ab.set(mu, num_a + nu, vb);
            p_b_ab.set(num_a + nu, mu, vb);
        }
    }
    for mu in 0..num_b {
        for nu in 0..num_b {
            p_a_ab.set(
                num_a + mu,
                num_a + nu,
                density_a.get(off_b + mu, off_b + nu),
            );
            p_b_ab.set(
                num_a + mu,
                num_a + nu,
                density_b.get(off_b + mu, off_b + nu),
            );
        }
    }

    pair_batch.x[1] = coord_b[0];
    pair_batch.y[1] = coord_b[1];
    pair_batch.z[1] = coord_b[2];

    let mut g_pair = [0.0f64; 3];
    for (k, g_k) in g_pair.iter_mut().enumerate() {
        pair_batch.x[0] = coord_a[0];
        pair_batch.y[0] = coord_a[1];
        pair_batch.z[0] = coord_a[2];
        match k {
            0 => pair_batch.x[0] += delta,
            1 => pair_batch.y[0] += delta,
            2 => pair_batch.z[0] += delta,
            _ => unreachable!(),
        }
        let e_plus = evaluate_frozen_energy_uhf(pair_batch, model, p_a_ab, p_b_ab, ws_ab, use_nddo);

        pair_batch.x[0] = coord_a[0];
        pair_batch.y[0] = coord_a[1];
        pair_batch.z[0] = coord_a[2];
        match k {
            0 => pair_batch.x[0] -= delta,
            1 => pair_batch.y[0] -= delta,
            2 => pair_batch.z[0] -= delta,
            _ => unreachable!(),
        }
        let e_minus =
            evaluate_frozen_energy_uhf(pair_batch, model, p_a_ab, p_b_ab, ws_ab, use_nddo);

        *g_k = (e_plus - e_minus) * inv_2delta;
    }
    g_pair
}

#[inline]
fn project_translational_drift(gradients: &mut [[f64; 3]]) {
    let natoms = gradients.len();
    if natoms == 0 {
        return;
    }
    let mut sum_gx = 0.0;
    let mut sum_gy = 0.0;
    let mut sum_gz = 0.0;
    for g in gradients.iter() {
        sum_gx += g[0];
        sum_gy += g[1];
        sum_gz += g[2];
    }
    let inv_n = 1.0 / (natoms as f64);
    let mean_gx = sum_gx * inv_n;
    let mean_gy = sum_gy * inv_n;
    let mean_gz = sum_gz * inv_n;

    for grad in gradients.iter_mut() {
        grad[0] -= mean_gx;
        grad[1] -= mean_gy;
        grad[2] -= mean_gz;
    }
}

/// Compute Cartesian nuclear energy gradients $\nabla E$ in **eV / Å** with optional NDDO multipoles.
pub fn compute_cartesian_gradients_with_options(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    density: &AlignedMatrix<f64>,
    _ws: &mut GradientWorkspace,
    gradients: &mut [[f64; 3]],
    use_nddo: bool,
) {
    let natoms = batch.natoms;
    assert_eq!(gradients.len(), natoms);
    assert_eq!(density.rows, batch.norbs);
    assert_eq!(density.cols, batch.norbs);

    for g in gradients.iter_mut() {
        *g = [0.0, 0.0, 0.0];
    }

    if natoms <= 1 {
        return;
    }

    // MOPAC dcart.F90 step size: chnge = 1.0D-4 Angstroms
    let delta = 1.0e-4;
    let inv_2delta = 1.0 / (2.0 * delta);
    let num_pairs = (natoms * (natoms - 1)) / 2;

    if num_pairs >= 16 && rayon::current_num_threads() > 1 {
        let mut pair_indices = Vec::with_capacity(num_pairs);
        for a in 0..natoms {
            for b in (a + 1)..natoms {
                pair_indices.push((a, b));
            }
        }

        let batch_ref: &MolecularBatch = batch;
        let pair_grads: Vec<[f64; 3]> = pair_indices
            .par_iter()
            .map_init(DiatomicGradScratch::new, |scratch, &(a, b)| {
                eval_rhf_pair_gradient(
                    a, b, batch_ref, model, density, use_nddo, delta, inv_2delta, scratch,
                )
            })
            .collect();

        for (&(a, b), g_pair) in pair_indices.iter().zip(pair_grads.iter()) {
            for k in 0..3 {
                gradients[a][k] += g_pair[k];
                gradients[b][k] -= g_pair[k];
            }
        }
    } else {
        let mut scratch = DiatomicGradScratch::new();
        for a in 0..natoms {
            for b in (a + 1)..natoms {
                let g_pair = eval_rhf_pair_gradient(
                    a,
                    b,
                    batch,
                    model,
                    density,
                    use_nddo,
                    delta,
                    inv_2delta,
                    &mut scratch,
                );
                for (k, &val) in g_pair.iter().enumerate() {
                    gradients[a][k] += val;
                    gradients[b][k] -= val;
                }
            }
        }
    }

    // Translational invariance correction: project out any residual center-of-mass drift
    project_translational_drift(gradients);
}

/// Compute Cartesian nuclear energy gradients $\nabla E$ in **eV / Å** using monopole approximation.
pub fn compute_cartesian_gradients(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    density: &AlignedMatrix<f64>,
    ws: &mut GradientWorkspace,
    gradients: &mut [[f64; 3]],
) {
    compute_cartesian_gradients_with_options(batch, model, density, ws, gradients, false);
}

/// Compute Cartesian nuclear energy gradients $\nabla E$ in **eV / Å** with optional NDDO multipoles
/// and optional COSMO implicit dielectric solvation reaction field.
pub fn compute_cartesian_gradients_full(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    density: &AlignedMatrix<f64>,
    ws: &mut GradientWorkspace,
    gradients: &mut [[f64; 3]],
    use_nddo: bool,
    cosmo: Option<&crate::solvation::CosmoState>,
) {
    compute_cartesian_gradients_with_options(batch, model, density, ws, gradients, use_nddo);

    if let Some(cs) = cosmo {
        cs.compute_dielectric_gradients(batch, model, density, gradients);

        // Project out any translational drift
        let natoms = batch.natoms;
        let mut sum_gx = 0.0;
        let mut sum_gy = 0.0;
        let mut sum_gz = 0.0;
        for g in gradients.iter() {
            sum_gx += g[0];
            sum_gy += g[1];
            sum_gz += g[2];
        }
        let mean_gx = sum_gx / (natoms as f64);
        let mean_gy = sum_gy / (natoms as f64);
        let mean_gz = sum_gz / (natoms as f64);
        for g in gradients.iter_mut() {
            g[0] -= mean_gx;
            g[1] -= mean_gy;
            g[2] -= mean_gz;
        }
    }
}

/// Compute Cartesian nuclear energy gradients $\nabla E$ in **eV / Å** for open-shell UHF wavefunctions.
pub fn compute_cartesian_gradients_uhf(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    density_a: &AlignedMatrix<f64>,
    density_b: &AlignedMatrix<f64>,
    _ws: &mut GradientWorkspace,
    gradients: &mut [[f64; 3]],
    use_nddo: bool,
) {
    let natoms = batch.natoms;
    assert_eq!(gradients.len(), natoms);
    assert_eq!(density_a.rows, batch.norbs);
    assert_eq!(density_a.cols, batch.norbs);
    assert_eq!(density_b.rows, batch.norbs);
    assert_eq!(density_b.cols, batch.norbs);

    for g in gradients.iter_mut() {
        *g = [0.0, 0.0, 0.0];
    }

    if natoms <= 1 {
        return;
    }

    // MOPAC dcart.F90 step size: chnge = 1.0D-4 Angstroms
    let delta = 1.0e-4;
    let inv_2delta = 1.0 / (2.0 * delta);
    let num_pairs = (natoms * (natoms - 1)) / 2;

    if num_pairs >= 16 && rayon::current_num_threads() > 1 {
        let mut pair_indices = Vec::with_capacity(num_pairs);
        for a in 0..natoms {
            for b in (a + 1)..natoms {
                pair_indices.push((a, b));
            }
        }

        let batch_ref: &MolecularBatch = batch;
        let pair_grads: Vec<[f64; 3]> = pair_indices
            .par_iter()
            .map_init(DiatomicGradScratch::new, |scratch, &(a, b)| {
                eval_uhf_pair_gradient(
                    a, b, batch_ref, model, density_a, density_b, use_nddo, delta, inv_2delta,
                    scratch,
                )
            })
            .collect();

        for (&(a, b), g_pair) in pair_indices.iter().zip(pair_grads.iter()) {
            for k in 0..3 {
                gradients[a][k] += g_pair[k];
                gradients[b][k] -= g_pair[k];
            }
        }
    } else {
        let mut scratch = DiatomicGradScratch::new();
        for a in 0..natoms {
            for b in (a + 1)..natoms {
                let g_pair = eval_uhf_pair_gradient(
                    a,
                    b,
                    batch,
                    model,
                    density_a,
                    density_b,
                    use_nddo,
                    delta,
                    inv_2delta,
                    &mut scratch,
                );
                for (k, &val) in g_pair.iter().enumerate() {
                    gradients[a][k] += val;
                    gradients[b][k] -= val;
                }
            }
        }
    }

    // Translational invariance correction: project out any residual center-of-mass drift
    project_translational_drift(gradients);
}

/// Compute Root-Mean-Square (RMS) and Maximum Gradient Norm in **kcal / (mol · Å)**.
pub fn compute_gradient_norms(gradients: &[[f64; 3]]) -> (f64, f64) {
    let mut sum_sq = 0.0;
    let mut max_norm = 0.0f64;

    for g in gradients {
        // Convert from eV/Angstrom to kcal/(mol*Angstrom)
        let gx = g[0] * EV_TO_KCAL_MOL;
        let gy = g[1] * EV_TO_KCAL_MOL;
        let gz = g[2] * EV_TO_KCAL_MOL;

        let norm_sq = gx * gx + gy * gy + gz * gz;
        let norm = norm_sq.sqrt();
        sum_sq += norm_sq;
        if norm > max_norm {
            max_norm = norm;
        }
    }

    let rms = (sum_sq / (gradients.len() as f64)).sqrt();
    (rms, max_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::am1::Am1Model;
    use crate::scf::scf_loop::{ScfOptions, run_rhf_scf_with_options};
    use crate::scf::uhf_loop::{UhfOptions, UhfWorkspace, run_uhf_scf_with_options};
    use crate::types::ScfWorkspace;

    #[test]
    fn test_parallel_vs_single_thread_gradients_exact_parity() {
        // Octane chain: 8 carbons + 18 hydrogens = 26 atoms (325 pairs >= 16 threshold)
        // Proper tetrahedral C-C (~1.53 A) and C-H (~1.10 A) geometry
        let mut z = Vec::with_capacity(26);
        let mut coords = Vec::with_capacity(26);
        for i in 0..8 {
            let cx = i as f64 * 1.25;
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let cy = sign * 0.44;
            z.push(6u8);
            coords.push([cx, cy, 0.0]);
            z.push(1u8);
            coords.push([cx, cy + sign * 0.64, 0.89]);
            z.push(1u8);
            coords.push([cx, cy + sign * 0.64, -0.89]);
            if i == 0 {
                z.push(1u8);
                coords.push([cx - 0.90, cy - sign * 0.63, 0.0]);
            } else if i == 7 {
                z.push(1u8);
                coords.push([cx + 0.90, cy - sign * 0.63, 0.0]);
            }
        }
        assert_eq!(z.len(), 26);
        let model = Am1Model;
        let mut batch_1 = MolecularBatch::new_for_model(z.clone(), &coords, &model);
        let mut batch_n = batch_1.clone();
        let mut ws = ScfWorkspace::allocate(batch_1.norbs);
        let scf_opts = ScfOptions {
            use_nddo: true,
            max_iter: 80,
            ..Default::default()
        };
        let res = run_rhf_scf_with_options(&batch_1, &model, &mut ws, &scf_opts);
        assert!(res.converged);

        let mut g_single = vec![[0.0f64; 3]; batch_1.natoms];
        let mut g_multi = vec![[0.0f64; 3]; batch_1.natoms];
        crate::threading::run_with_thread_pool(Some(1), || {
            let mut gws = GradientWorkspace::allocate(batch_1.norbs);
            compute_cartesian_gradients_with_options(
                &mut batch_1,
                &model,
                &ws.density,
                &mut gws,
                &mut g_single,
                true,
            );
        });
        crate::threading::run_with_thread_pool(Some(4), || {
            let mut gws = GradientWorkspace::allocate(batch_n.norbs);
            compute_cartesian_gradients_with_options(
                &mut batch_n,
                &model,
                &ws.density,
                &mut gws,
                &mut g_multi,
                true,
            );
        });

        let mut max_diff = 0.0f64;
        for (a, b) in g_single.iter().zip(g_multi.iter()) {
            for k in 0..3 {
                max_diff = max_diff.max((a[k] - b[k]).abs());
            }
        }
        assert!(
            max_diff < 1e-12,
            "RHF single-thread vs parallel gradient mismatch: {:e}",
            max_diff
        );

        // Also verify UHF gradient parity on triplet state of the same octane chain
        let mut uhf_ws = UhfWorkspace::new(batch_1.norbs);
        let uhf_opts = UhfOptions {
            multiplicity: 3,
            charge: 0,
            max_iter: 120,
            use_nddo: true,
            ..Default::default()
        };
        let uhf_res = run_uhf_scf_with_options(&batch_1, &model, &mut uhf_ws, &uhf_opts);
        assert!(uhf_res.converged);

        let mut gu_single = vec![[0.0f64; 3]; batch_1.natoms];
        let mut gu_multi = vec![[0.0f64; 3]; batch_1.natoms];
        crate::threading::run_with_thread_pool(Some(1), || {
            let mut gws = GradientWorkspace::allocate(batch_1.norbs);
            compute_cartesian_gradients_uhf(
                &mut batch_1,
                &model,
                &uhf_ws.density_a,
                &uhf_ws.density_b,
                &mut gws,
                &mut gu_single,
                true,
            );
        });
        crate::threading::run_with_thread_pool(Some(4), || {
            let mut gws = GradientWorkspace::allocate(batch_n.norbs);
            compute_cartesian_gradients_uhf(
                &mut batch_n,
                &model,
                &uhf_ws.density_a,
                &uhf_ws.density_b,
                &mut gws,
                &mut gu_multi,
                true,
            );
        });
        let mut max_u_diff = 0.0f64;
        for (a, b) in gu_single.iter().zip(gu_multi.iter()) {
            for k in 0..3 {
                max_u_diff = max_u_diff.max((a[k] - b[k]).abs());
            }
        }
        assert!(
            max_u_diff < 1e-12,
            "UHF single-thread vs parallel gradient mismatch: {:e}",
            max_u_diff
        );
    }
}
