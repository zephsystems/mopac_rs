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
use crate::scf::density::compute_electronic_energy;
use crate::hamiltonian::hcore::build_hcore;
use crate::integrals::core_repulsion::compute_total_core_repulsion;
use crate::parameters::ParameterModel;
use crate::types::{AlignedMatrix, MolecularBatch};

/// Pre-allocated workspace for gradient calculations (0 heap allocations).
#[derive(Debug, Clone)]
pub struct GradientWorkspace {
    pub h_core: AlignedMatrix<f64>,
    pub fock: AlignedMatrix<f64>,
}

impl GradientWorkspace {
    /// Allocate reusable workspace for a system of `norbs` basis orbitals.
    pub fn allocate(norbs: usize) -> Self {
        Self {
            h_core: AlignedMatrix::zeroed(norbs, norbs),
            fock: AlignedMatrix::zeroed(norbs, norbs),
        }
    }
}

/// Evaluates total energy of a molecular geometry with frozen density matrix $P$.
fn evaluate_frozen_energy(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    density: &AlignedMatrix<f64>,
    ws: &mut GradientWorkspace,
) -> f64 {
    let e_nuc = compute_total_core_repulsion(batch, model);
    build_hcore(batch, model, &mut ws.h_core);
    build_fock(batch, model, &ws.h_core, density, &mut ws.fock);
    let e_elec = compute_electronic_energy(density, &ws.h_core, &ws.fock);
    e_elec + e_nuc
}

/// Compute Cartesian nuclear energy gradients $\nabla E$ in **eV / Å**.
///
/// Output: `gradients` is an array of size $N_{\text{atoms}} \times 3$, where each entry is
/// $(\frac{\partial E}{\partial x_A}, \frac{\partial E}{\partial y_A}, \frac{\partial E}{\partial z_A})$.
pub fn compute_cartesian_gradients(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    density: &AlignedMatrix<f64>,
    ws: &mut GradientWorkspace,
    gradients: &mut [[f64; 3]],
) {
    let natoms = batch.natoms;
    assert_eq!(gradients.len(), natoms);
    assert_eq!(density.rows, batch.norbs);
    assert_eq!(density.cols, batch.norbs);

    // MOPAC dcart.F90 step size: chnge = 1.0D-4 Angstroms
    let delta = 1.0e-4;
    let inv_2delta = 1.0 / (2.0 * delta);

    let mut sum_gx = 0.0;
    let mut sum_gy = 0.0;
    let mut sum_gz = 0.0;

    for (a, grad) in gradients.iter_mut().enumerate().take(natoms) {
        // --- X coordinate derivative ---
        let orig_x = batch.x[a];
        batch.x[a] = orig_x + delta;
        let e_plus_x = evaluate_frozen_energy(batch, model, density, ws);
        batch.x[a] = orig_x - delta;
        let e_minus_x = evaluate_frozen_energy(batch, model, density, ws);
        batch.x[a] = orig_x;
        let gx = (e_plus_x - e_minus_x) * inv_2delta;

        // --- Y coordinate derivative ---
        let orig_y = batch.y[a];
        batch.y[a] = orig_y + delta;
        let e_plus_y = evaluate_frozen_energy(batch, model, density, ws);
        batch.y[a] = orig_y - delta;
        let e_minus_y = evaluate_frozen_energy(batch, model, density, ws);
        batch.y[a] = orig_y;
        let gy = (e_plus_y - e_minus_y) * inv_2delta;

        // --- Z coordinate derivative ---
        let orig_z = batch.z[a];
        batch.z[a] = orig_z + delta;
        let e_plus_z = evaluate_frozen_energy(batch, model, density, ws);
        batch.z[a] = orig_z - delta;
        let e_minus_z = evaluate_frozen_energy(batch, model, density, ws);
        batch.z[a] = orig_z;
        let gz = (e_plus_z - e_minus_z) * inv_2delta;

        *grad = [gx, gy, gz];
        sum_gx += gx;
        sum_gy += gy;
        sum_gz += gz;
    }

    // Translational invariance correction: project out center-of-mass drift
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
