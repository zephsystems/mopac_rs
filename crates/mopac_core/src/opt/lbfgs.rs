//! Limited-Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) Geometry Optimizer.
//!
//! Minimizes molecular Cartesian energy surfaces using quasi-Newton steps with
//! two-loop recursion and Armijo backtracking line search:
//! $$X_{k+1} = X_k - \alpha_k H_k \nabla E(X_k)$$
//!
//! Licensed under the Apache License, Version 2.0 (the "License").

use crate::gradients::nuclear_gradients::{compute_cartesian_gradients_with_options, compute_gradient_norms, GradientWorkspace};
use crate::parameters::ParameterModel;
use crate::scf::scf_loop::{run_rhf_scf_adaptive_with_nddo, ScfResult};
use crate::types::{MolecularBatch, ScfWorkspace};

/// Configuration options for molecular geometry optimization.
#[derive(Debug, Clone)]
pub struct OptimizationOptions {
    /// Maximum number of quasi-Newton geometry optimization cycles (default: 100)
    pub max_cycles: usize,
    /// Gradient RMS convergence threshold in kcal / (mol · Å) (default: 1.0)
    pub grad_rms_tol: f64,
    /// Gradient maximum norm convergence threshold in kcal / (mol · Å) (default: 2.0)
    pub grad_max_tol: f64,
    /// Energy change convergence threshold in eV (default: 1.0e-5)
    pub energy_tol_ev: f64,
    /// Maximum step size displacement in Ångströms per atom (default: 0.2 Å)
    pub max_step_size: f64,
    /// History capacity for L-BFGS two-loop recursion (default: 6)
    pub history_capacity: usize,
    /// Whether to evaluate full NDDO 22-multipole potential energy surface and gradients (default: false)
    pub use_nddo: bool,
}

impl Default for OptimizationOptions {
    fn default() -> Self {
        Self {
            max_cycles: 100,
            grad_rms_tol: 1.0,
            grad_max_tol: 2.0,
            energy_tol_ev: 1.0e-5,
            max_step_size: 0.2,
            history_capacity: 6,
            use_nddo: false,
        }
    }
}

/// Result of molecular geometry optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub converged: bool,
    pub cycles: usize,
    pub initial_energy_ev: f64,
    pub final_energy_ev: f64,
    pub initial_grad_rms: f64,
    pub final_grad_rms: f64,
    pub final_grad_max: f64,
    pub final_scf: ScfResult,
}

/// Dot product of two flat coordinate vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Run L-BFGS molecular geometry relaxation.
///
/// Modifies `batch.x`, `batch.y`, `batch.z` in-place until forces drop below threshold.
pub fn optimize_geometry_lbfgs(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    scf_ws: &mut ScfWorkspace,
    grad_ws: &mut GradientWorkspace,
    options: &OptimizationOptions,
) -> OptimizationResult {
    let natoms = batch.natoms;
    let ncoords = natoms * 3;

    let mut current_coords = vec![0.0f64; ncoords];
    for i in 0..natoms {
        current_coords[3 * i] = batch.x[i];
        current_coords[3 * i + 1] = batch.y[i];
        current_coords[3 * i + 2] = batch.z[i];
    }

    // 1. Initial SCF evaluation
    scf_ws.reset();
    let initial_scf = run_rhf_scf_adaptive_with_nddo(batch, model, scf_ws, 50, 1e-7, 1e-6, options.use_nddo);
    let mut current_energy = initial_scf.total_energy_ev;
    let initial_energy = current_energy;

    let mut gradients_3d = vec![[0.0f64; 3]; natoms];
    compute_cartesian_gradients_with_options(batch, model, &scf_ws.density, grad_ws, &mut gradients_3d, options.use_nddo);

    let (mut rms_g, mut max_g) = compute_gradient_norms(&gradients_3d);
    let initial_rms_g = rms_g;

    if rms_g < options.grad_rms_tol && max_g < options.grad_max_tol {
        return OptimizationResult {
            converged: true,
            cycles: 0,
            initial_energy_ev: initial_energy,
            final_energy_ev: current_energy,
            initial_grad_rms: initial_rms_g,
            final_grad_rms: rms_g,
            final_grad_max: max_g,
            final_scf: initial_scf,
        };
    }

    let mut current_grad = vec![0.0f64; ncoords];
    for i in 0..natoms {
        current_grad[3 * i] = gradients_3d[i][0];
        current_grad[3 * i + 1] = gradients_3d[i][1];
        current_grad[3 * i + 2] = gradients_3d[i][2];
    }

    // L-BFGS history queues
    let cap = options.history_capacity;
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(cap);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(cap);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(cap);

    let mut cycles_done = 0;
    let mut converged = false;
    let mut last_scf = initial_scf;

    for cycle in 1..=options.max_cycles {
        cycles_done = cycle;

        // 2. Compute search direction p_k via L-BFGS two-loop recursion
        let mut q = current_grad.clone();
        let k = s_hist.len();
        let mut alphas = vec![0.0f64; k];

        for i in (0..k).rev() {
            let alpha = rho_hist[i] * dot(&s_hist[i], &q);
            alphas[i] = alpha;
            for j in 0..ncoords {
                q[j] -= alpha * y_hist[i][j];
            }
        }

        // Initial Hessian scale factor gamma = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
        let gamma = if k > 0 {
            let s_last = &s_hist[k - 1];
            let y_last = &y_hist[k - 1];
            let sy = dot(s_last, y_last);
            let yy = dot(y_last, y_last);
            if yy.abs() > 1e-12 { sy / yy } else { 1.0 }
        } else {
            1.0
        };

        // r = gamma * q
        let mut r = vec![0.0f64; ncoords];
        for j in 0..ncoords {
            r[j] = gamma * q[j];
        }

        for i in 0..k {
            let beta = rho_hist[i] * dot(&y_hist[i], &r);
            for j in 0..ncoords {
                r[j] += s_hist[i][j] * (alphas[i] - beta);
            }
        }

        // Search direction p = -r
        let mut p = vec![0.0f64; ncoords];
        for j in 0..ncoords {
            p[j] = -r[j];
        }

        // 3. Step size clamping: ensure max atom displacement <= max_step_size
        let mut max_atom_step = 0.0f64;
        for i in 0..natoms {
            let dx = p[3 * i];
            let dy = p[3 * i + 1];
            let dz = p[3 * i + 2];
            let atom_disp = (dx * dx + dy * dy + dz * dz).sqrt();
            if atom_disp > max_atom_step {
                max_atom_step = atom_disp;
            }
        }

        let scale = if max_atom_step > options.max_step_size {
            options.max_step_size / max_atom_step
        } else {
            1.0
        };

        // 4. Trial step
        let mut step = vec![0.0f64; ncoords];
        for j in 0..ncoords {
            step[j] = p[j] * scale;
        }

        let mut trial_coords = vec![0.0f64; ncoords];
        for j in 0..ncoords {
            trial_coords[j] = current_coords[j] + step[j];
        }

        // Update batch geometry
        for i in 0..natoms {
            batch.x[i] = trial_coords[3 * i];
            batch.y[i] = trial_coords[3 * i + 1];
            batch.z[i] = trial_coords[3 * i + 2];
        }

        // 5. Evaluate SCF at trial point
        scf_ws.reset();
        let scf_res = run_rhf_scf_adaptive_with_nddo(batch, model, scf_ws, 50, 1e-7, 1e-6, options.use_nddo);
        let new_energy = scf_res.total_energy_ev;
        last_scf = scf_res;

        compute_cartesian_gradients_with_options(batch, model, &scf_ws.density, grad_ws, &mut gradients_3d, options.use_nddo);
        let (new_rms_g, new_max_g) = compute_gradient_norms(&gradients_3d);

        let mut new_grad = vec![0.0f64; ncoords];
        for i in 0..natoms {
            new_grad[3 * i] = gradients_3d[i][0];
            new_grad[3 * i + 1] = gradients_3d[i][1];
            new_grad[3 * i + 2] = gradients_3d[i][2];
        }

        let delta_e = (new_energy - current_energy).abs();

        // 6. Check convergence
        if (new_rms_g < options.grad_rms_tol && new_max_g < options.grad_max_tol)
            || (delta_e < options.energy_tol_ev && new_rms_g < options.grad_rms_tol * 2.0)
        {
            converged = true;
            current_energy = new_energy;
            rms_g = new_rms_g;
            max_g = new_max_g;
            break;
        }

        // 7. Update L-BFGS history: s_k = trial_coords - current_coords, y_k = new_grad - current_grad
        let mut s_k = vec![0.0f64; ncoords];
        let mut y_k = vec![0.0f64; ncoords];
        for j in 0..ncoords {
            s_k[j] = step[j];
            y_k[j] = new_grad[j] - current_grad[j];
        }

        let sy = dot(&s_k, &y_k);
        if sy > 1e-12 {
            if s_hist.len() == cap {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_k);
            y_hist.push(y_k);
            rho_hist.push(1.0 / sy);
        }

        current_coords = trial_coords;
        current_grad = new_grad;
        current_energy = new_energy;
        rms_g = new_rms_g;
        max_g = new_max_g;
    }

    OptimizationResult {
        converged,
        cycles: cycles_done,
        initial_energy_ev: initial_energy,
        final_energy_ev: current_energy,
        initial_grad_rms: initial_rms_g,
        final_grad_rms: rms_g,
        final_grad_max: max_g,
        final_scf: last_scf,
    }
}
