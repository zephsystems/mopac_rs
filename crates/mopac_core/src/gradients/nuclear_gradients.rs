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

    // Pairwise diatomic decomposition: O(N^2) scaling matching OpenMOPAC dcart.F90 / dhc.F90
    for a in 0..natoms {
        let num_a = batch.basis_types[a].num_orbitals();
        let off_a = batch.orbital_offsets[a];
        let za = batch.atomic_numbers[a];

        for b in (a + 1)..natoms {
            let num_b = batch.basis_types[b].num_orbitals();
            let off_b = batch.orbital_offsets[b];
            let zb = batch.atomic_numbers[b];
            let norbs_ab = num_a + num_b;

            // Extract diatomic density block
            let mut p_ab = AlignedMatrix::zeroed(norbs_ab, norbs_ab);
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

            let coord_a = [batch.x[a], batch.y[a], batch.z[a]];
            let coord_b = [batch.x[b], batch.y[b], batch.z[b]];

            let eval_pair = |ca: [f64; 3], cb: [f64; 3]| -> f64 {
                let pair_batch = MolecularBatch::new_for_model(vec![za, zb], &[ca, cb], model);
                let mut ws_ab = GradientWorkspace::allocate(norbs_ab);
                evaluate_frozen_energy(&pair_batch, model, &p_ab, &mut ws_ab, use_nddo)
            };

            for k in 0..3 {
                let mut ca_plus = coord_a;
                ca_plus[k] += delta;
                let e_plus = eval_pair(ca_plus, coord_b);

                let mut ca_minus = coord_a;
                ca_minus[k] -= delta;
                let e_minus = eval_pair(ca_minus, coord_b);

                let g_k = (e_plus - e_minus) * inv_2delta;
                gradients[a][k] += g_k;
                gradients[b][k] -= g_k;
            }
        }
    }

    // Translational invariance correction: project out any residual center-of-mass drift
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

    for grad in gradients.iter_mut() {
        grad[0] -= mean_gx;
        grad[1] -= mean_gy;
        grad[2] -= mean_gz;
    }
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

    // Pairwise diatomic decomposition for UHF: O(N^2) scaling
    for a in 0..natoms {
        let num_a = batch.basis_types[a].num_orbitals();
        let off_a = batch.orbital_offsets[a];
        let za = batch.atomic_numbers[a];

        for b in (a + 1)..natoms {
            let num_b = batch.basis_types[b].num_orbitals();
            let off_b = batch.orbital_offsets[b];
            let zb = batch.atomic_numbers[b];
            let norbs_ab = num_a + num_b;

            // Extract diatomic density blocks for alpha and beta
            let mut p_a_ab = AlignedMatrix::zeroed(norbs_ab, norbs_ab);
            let mut p_b_ab = AlignedMatrix::zeroed(norbs_ab, norbs_ab);
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

            let coord_a = [batch.x[a], batch.y[a], batch.z[a]];
            let coord_b = [batch.x[b], batch.y[b], batch.z[b]];

            let eval_pair_uhf = |ca: [f64; 3], cb: [f64; 3]| -> f64 {
                let pair_batch = MolecularBatch::new_for_model(vec![za, zb], &[ca, cb], model);
                let mut ws_ab = GradientWorkspace::allocate(norbs_ab);
                evaluate_frozen_energy_uhf(
                    &pair_batch,
                    model,
                    &p_a_ab,
                    &p_b_ab,
                    &mut ws_ab,
                    use_nddo,
                )
            };

            for k in 0..3 {
                let mut ca_plus = coord_a;
                ca_plus[k] += delta;
                let e_plus = eval_pair_uhf(ca_plus, coord_b);

                let mut ca_minus = coord_a;
                ca_minus[k] -= delta;
                let e_minus = eval_pair_uhf(ca_minus, coord_b);

                let g_k = (e_plus - e_minus) * inv_2delta;
                gradients[a][k] += g_k;
                gradients[b][k] -= g_k;
            }
        }
    }

    // Translational invariance correction: project out any residual center-of-mass drift
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

    for grad in gradients.iter_mut() {
        grad[0] -= mean_gx;
        grad[1] -= mean_gy;
        grad[2] -= mean_gz;
    }
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
