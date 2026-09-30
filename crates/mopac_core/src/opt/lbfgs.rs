//! Limited-Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) Geometry Optimizer.
//!
//! Minimizes molecular Cartesian energy surfaces using quasi-Newton steps with
//! two-loop recursion and Armijo backtracking line search:
//! $$X_{k+1} = X_k - \alpha_k H_k \nabla E(X_k)$$
//!
//! Licensed under the Apache License, Version 2.0 (the "License").

use crate::constants::codata2018::EV_TO_KCAL_MOL;
use crate::gradients::nuclear_gradients::{
    compute_cartesian_gradients_full, compute_cartesian_gradients_uhf, compute_gradient_norms,
    GradientWorkspace,
};
use crate::parameters::ParameterModel;
use crate::scf::scf_loop::{run_rhf_scf_adaptive_with_nddo_and_cosmo, ScfResult};
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
    /// Optional coordinate optimization mask (length: 3 * natoms).
    /// `true` = active degree of freedom, `false` = frozen/pinned coordinate.
    pub opt_mask: Option<Vec<bool>>,
    /// Optional COSMO implicit dielectric solvation parameters.
    pub cosmo: Option<crate::solvation::CosmoParams>,
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
            opt_mask: None,
            cosmo: None,
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

/// Compute Root-Mean-Square (RMS) and Maximum Gradient Norm considering optional coordinate mask.
fn compute_effective_gradient_norms(
    gradients_3d: &[[f64; 3]],
    mask: Option<&[bool]>,
) -> (f64, f64) {
    match mask {
        Some(m) => {
            let mut sum_sq = 0.0;
            let mut max_norm = 0.0f64;
            let mut n_active_coords = 0;
            for (a, g) in gradients_3d.iter().enumerate() {
                for c in 0..3 {
                    if m[3 * a + c] {
                        n_active_coords += 1;
                        let val = g[c] * EV_TO_KCAL_MOL;
                        let sq = val * val;
                        sum_sq += sq;
                        let abs_val = val.abs();
                        if abs_val > max_norm {
                            max_norm = abs_val;
                        }
                    }
                }
            }
            let rms = if n_active_coords > 0 {
                (sum_sq / (n_active_coords as f64)).sqrt()
            } else {
                0.0
            };
            (rms, max_norm)
        }
        None => compute_gradient_norms(gradients_3d),
    }
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
    let initial_scf = run_rhf_scf_adaptive_with_nddo_and_cosmo(
        batch,
        model,
        scf_ws,
        50,
        1e-7,
        1e-6,
        options.use_nddo,
        options.cosmo,
    );
    let mut current_energy = initial_scf.total_energy_ev;
    let initial_energy = current_energy;

    let mut gradients_3d = vec![[0.0f64; 3]; natoms];
    compute_cartesian_gradients_full(
        batch,
        model,
        &scf_ws.density,
        grad_ws,
        &mut gradients_3d,
        options.use_nddo,
        None,
    );

    // Apply optimization mask: zero out gradients on frozen degrees of freedom
    if let Some(ref m) = options.opt_mask {
        for a in 0..natoms {
            for c in 0..3 {
                if !m[3 * a + c] {
                    gradients_3d[a][c] = 0.0;
                }
            }
        }
    }

    let (mut rms_g, mut max_g) =
        compute_effective_gradient_norms(&gradients_3d, options.opt_mask.as_deref());
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
            if yy.abs() > 1e-12 {
                sy / yy
            } else {
                1.0
            }
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
        if let Some(ref m) = options.opt_mask {
            for j in 0..ncoords {
                if !m[j] {
                    step[j] = 0.0;
                }
            }
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
        let scf_res = run_rhf_scf_adaptive_with_nddo_and_cosmo(
            batch,
            model,
            scf_ws,
            50,
            1e-7,
            1e-6,
            options.use_nddo,
            options.cosmo,
        );
        let new_energy = scf_res.total_energy_ev;
        last_scf = scf_res;

        compute_cartesian_gradients_full(
            batch,
            model,
            &scf_ws.density,
            grad_ws,
            &mut gradients_3d,
            options.use_nddo,
            None,
        );

        // Apply optimization mask to trial gradients
        if let Some(ref m) = options.opt_mask {
            for a in 0..natoms {
                for c in 0..3 {
                    if !m[3 * a + c] {
                        gradients_3d[a][c] = 0.0;
                    }
                }
            }
        }

        let (new_rms_g, new_max_g) =
            compute_effective_gradient_norms(&gradients_3d, options.opt_mask.as_deref());

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

/// Result of open-shell UHF molecular geometry optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationUhfResult {
    pub converged: bool,
    pub cycles: usize,
    pub initial_energy_ev: f64,
    pub final_energy_ev: f64,
    pub initial_grad_rms: f64,
    pub final_grad_rms: f64,
    pub final_grad_max: f64,
    pub final_uhf: crate::scf::uhf_loop::UhfResult,
}

/// Run L-BFGS molecular geometry relaxation for open-shell UHF wavefunctions.
///
/// Modifies `batch.x`, `batch.y`, `batch.z` in-place until forces drop below threshold.
pub fn optimize_geometry_lbfgs_uhf(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    uhf_ws: &mut crate::scf::uhf_loop::UhfWorkspace,
    grad_ws: &mut GradientWorkspace,
    uhf_opts: &crate::scf::uhf_loop::UhfOptions,
    opt_opts: &OptimizationOptions,
) -> OptimizationUhfResult {
    let natoms = batch.natoms;
    let ncoords = natoms * 3;

    let mut current_coords = vec![0.0f64; ncoords];
    for i in 0..natoms {
        current_coords[3 * i] = batch.x[i];
        current_coords[3 * i + 1] = batch.y[i];
        current_coords[3 * i + 2] = batch.z[i];
    }

    // 1. Initial UHF evaluation
    let initial_uhf =
        crate::scf::uhf_loop::run_uhf_scf_with_options(batch, model, uhf_ws, uhf_opts);
    let mut current_energy = initial_uhf.total_energy_ev;
    let initial_energy = current_energy;

    let mut gradients_3d = vec![[0.0f64; 3]; natoms];
    compute_cartesian_gradients_uhf(
        batch,
        model,
        &uhf_ws.density_a,
        &uhf_ws.density_b,
        grad_ws,
        &mut gradients_3d,
        opt_opts.use_nddo,
    );

    // Apply optimization mask: zero out gradients on frozen degrees of freedom
    if let Some(ref m) = opt_opts.opt_mask {
        for a in 0..natoms {
            for c in 0..3 {
                if !m[3 * a + c] {
                    gradients_3d[a][c] = 0.0;
                }
            }
        }
    }

    let (mut rms_g, mut max_g) =
        compute_effective_gradient_norms(&gradients_3d, opt_opts.opt_mask.as_deref());
    let initial_rms_g = rms_g;

    if rms_g < opt_opts.grad_rms_tol && max_g < opt_opts.grad_max_tol {
        return OptimizationUhfResult {
            converged: true,
            cycles: 0,
            initial_energy_ev: initial_energy,
            final_energy_ev: current_energy,
            initial_grad_rms: initial_rms_g,
            final_grad_rms: rms_g,
            final_grad_max: max_g,
            final_uhf: initial_uhf,
        };
    }

    let mut current_grad = vec![0.0f64; ncoords];
    for i in 0..natoms {
        current_grad[3 * i] = gradients_3d[i][0];
        current_grad[3 * i + 1] = gradients_3d[i][1];
        current_grad[3 * i + 2] = gradients_3d[i][2];
    }

    // L-BFGS history queues
    let cap = opt_opts.history_capacity;
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(cap);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(cap);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(cap);

    let mut cycles_done = 0;
    let mut converged = false;
    let mut last_uhf = initial_uhf;

    for cycle in 1..=opt_opts.max_cycles {
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

        // Initial Hessian scale factor gamma
        let gamma = if k > 0 {
            let s_last = &s_hist[k - 1];
            let y_last = &y_hist[k - 1];
            let sy = dot(s_last, y_last);
            let yy = dot(y_last, y_last);
            if yy.abs() > 1e-12 {
                sy / yy
            } else {
                1.0
            }
        } else {
            1.0
        };

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

        let scale = if max_atom_step > opt_opts.max_step_size {
            opt_opts.max_step_size / max_atom_step
        } else {
            1.0
        };

        // 4. Trial step
        let mut step = vec![0.0f64; ncoords];
        for j in 0..ncoords {
            step[j] = p[j] * scale;
        }
        if let Some(ref m) = opt_opts.opt_mask {
            for j in 0..ncoords {
                if !m[j] {
                    step[j] = 0.0;
                }
            }
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

        // 5. Evaluate UHF at trial point
        let uhf_res =
            crate::scf::uhf_loop::run_uhf_scf_with_options(batch, model, uhf_ws, uhf_opts);
        let new_energy = uhf_res.total_energy_ev;
        last_uhf = uhf_res;

        compute_cartesian_gradients_uhf(
            batch,
            model,
            &uhf_ws.density_a,
            &uhf_ws.density_b,
            grad_ws,
            &mut gradients_3d,
            opt_opts.use_nddo,
        );

        // Apply optimization mask to trial gradients
        if let Some(ref m) = opt_opts.opt_mask {
            for a in 0..natoms {
                for c in 0..3 {
                    if !m[3 * a + c] {
                        gradients_3d[a][c] = 0.0;
                    }
                }
            }
        }

        let (new_rms_g, new_max_g) =
            compute_effective_gradient_norms(&gradients_3d, opt_opts.opt_mask.as_deref());

        let mut new_grad = vec![0.0f64; ncoords];
        for i in 0..natoms {
            new_grad[3 * i] = gradients_3d[i][0];
            new_grad[3 * i + 1] = gradients_3d[i][1];
            new_grad[3 * i + 2] = gradients_3d[i][2];
        }

        let delta_e = (new_energy - current_energy).abs();

        // 6. Check convergence
        if (new_rms_g < opt_opts.grad_rms_tol && new_max_g < opt_opts.grad_max_tol)
            || (delta_e < opt_opts.energy_tol_ev && new_rms_g < opt_opts.grad_rms_tol * 2.0)
        {
            converged = true;
            current_energy = new_energy;
            rms_g = new_rms_g;
            max_g = new_max_g;
            break;
        }

        // 7. Update L-BFGS history
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

    OptimizationUhfResult {
        converged,
        cycles: cycles_done,
        initial_energy_ev: initial_energy,
        final_energy_ev: current_energy,
        initial_grad_rms: initial_rms_g,
        final_grad_rms: rms_g,
        final_grad_max: max_g,
        final_uhf: last_uhf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::am1::Am1Model;
    use crate::solvation::CosmoParams;

    #[test]
    fn test_lbfgs_cosmo_water_optimization() {
        let z = vec![8, 1, 1];
        // Start slightly distorted from minimum
        let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.85, 0.50], [0.0, -0.85, 0.50]];
        let mut batch = MolecularBatch::new(z, &coords);
        let model = Am1Model;
        let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
        let mut grad_ws = GradientWorkspace::allocate(batch.norbs);

        let options = OptimizationOptions {
            max_cycles: 20,
            grad_rms_tol: 2.0,
            grad_max_tol: 3.0,
            energy_tol_ev: 1e-4,
            max_step_size: 0.1,
            history_capacity: 5,
            use_nddo: false,
            opt_mask: None,
            cosmo: Some(CosmoParams {
                epsilon: 78.4,
                rsolv: 1.30005,
            }),
        };

        let res = optimize_geometry_lbfgs(&mut batch, &model, &mut scf_ws, &mut grad_ws, &options);
        println!("[L-BFGS COSMO] cycles = {}, init E = {:.6} eV, final E = {:.6} eV, init RMS g = {:.3}, final RMS g = {:.3}",
            res.cycles, res.initial_energy_ev, res.final_energy_ev, res.initial_grad_rms, res.final_grad_rms);
        assert!(res.cycles > 0);
        assert!(res.final_grad_rms <= res.initial_grad_rms + 1e-2);
    }

    #[test]
    fn test_lbfgs_uhf_methyl_radical_optimization() {
        let z = vec![6, 1, 1, 1];
        // Pyramidal distorted initial geometry for methyl radical
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.08, 0.0, 0.3],
            [-0.54, 0.935, 0.3],
            [-0.54, -0.935, 0.3],
        ];
        let mut batch = MolecularBatch::new(z, &coords);
        let model = Am1Model;
        let mut uhf_ws = crate::scf::uhf_loop::UhfWorkspace::new(batch.norbs);
        let mut grad_ws = GradientWorkspace::allocate(batch.norbs);

        let uhf_opts = crate::scf::uhf_loop::UhfOptions {
            multiplicity: 2,
            charge: 0,
            max_iter: 60,
            energy_tol_ev: 1e-7,
            density_tol: 1e-6,
            damping: 0.5,
            use_nddo: false,
            cosmo: None,
        };

        let opt_opts = OptimizationOptions {
            max_cycles: 25,
            grad_rms_tol: 1.0,
            grad_max_tol: 2.0,
            energy_tol_ev: 1e-5,
            max_step_size: 0.1,
            history_capacity: 5,
            use_nddo: false,
            opt_mask: None,
            cosmo: None,
        };

        let res = optimize_geometry_lbfgs_uhf(
            &mut batch,
            &model,
            &mut uhf_ws,
            &mut grad_ws,
            &uhf_opts,
            &opt_opts,
        );

        println!("[L-BFGS UHF] Methyl radical: cycles = {}, init E = {:.6} eV, final E = {:.6} eV, final RMS g = {:.3}",
            res.cycles, res.initial_energy_ev, res.final_energy_ev, res.final_grad_rms);
        assert!(
            res.converged,
            "UHF geometry optimization of methyl radical must converge"
        );
        assert!(res.final_energy_ev < res.initial_energy_ev);
        // Planar methyl radical: carbon z coordinate close to hydrogen average z
        let avg_hz = (batch.z[1] + batch.z[2] + batch.z[3]) / 3.0;
        let out_of_plane = (batch.z[0] - avg_hz).abs();
        assert!(
            out_of_plane < 0.05,
            "Methyl radical must optimize to planar geometry in AM1, out-of-plane={:.4} A",
            out_of_plane
        );
    }
}
