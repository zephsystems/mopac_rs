//! Partitioned Rational Function Optimization (P-RFO) Eigenvector Following (EF) Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Implements Baker's P-RFO algorithm (J. Comput. Chem. 7, 385 (1986)) for locating
//! first-order saddle points (transition states) with:
//! * Dynamic Trust-Radius sphere control and step scaling (Culot et al., Jensen).
//! * Hessian mode following (overlap tracking) along the reaction coordinate.
//! * Powell Symmetric Dual (PSB) and Bofill rank-2 Hessian updates matching OpenMOPAC `ef.F90`.
//! * Zero heap allocation invariant (0-malloc) in iterative cycles.

use crate::constants::codata2018::EV_TO_KCAL_MOL;
use crate::gradients::nuclear_gradients::{
    compute_cartesian_gradients_with_options, compute_gradient_norms, GradientWorkspace,
};
use crate::opt::hessian_update::{
    update_cartesian_hessian, HessianUpdateScheme, HessianUpdateWorkspace,
};
use crate::parameters::ParameterModel;
use crate::properties::heat::compute_heat_of_formation;
use crate::scf::eigensolver::diagonalize_symmetric_with_work;
use crate::scf::scf_loop::{run_rhf_scf_adaptive_with_nddo, ScfResult};
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch, ScfWorkspace};

/// Configuration options for Eigenvector Following (EF) transition state search.
#[derive(Debug, Clone)]
pub struct TransitionStateOptions {
    /// Maximum number of EF optimization cycles (default: 100)
    pub max_cycles: usize,
    /// Gradient RMS convergence threshold in kcal / (mol · Å) (default: 0.1)
    pub grad_rms_tol: f64,
    /// Gradient maximum norm convergence threshold in kcal / (mol · Å) (default: 0.2)
    pub grad_max_tol: f64,
    /// Initial trust radius in Ångströms (default: 0.1 Å)
    pub trust_radius: f64,
    /// Minimum allowed trust radius in Ångströms (default: 0.005 Å)
    pub min_trust_radius: f64,
    /// Maximum allowed trust radius in Ångströms (default: 0.3 Å)
    pub max_trust_radius: f64,
    /// Hessian update scheme (default: Bofill)
    pub update_scheme: HessianUpdateScheme,
    /// Whether to enable eigenvector mode following across cycles (default: true)
    pub mode_following: bool,
    /// Specific mode index to follow as transition vector (default: None -> lowest eigenvalue)
    pub target_mode: Option<usize>,
    /// Optional coordinate optimization mask (length: 3 * natoms).
    /// `true` = active degree of freedom, `false` = frozen/pinned coordinate.
    pub opt_mask: Option<Vec<bool>>,
    /// Whether to evaluate full NDDO 22-multipole potential energy surface and gradients (default: false)
    pub use_nddo: bool,
    /// Finite-difference step size for numerical initial Hessian if not provided (default: 0.005 Å)
    pub hessian_delta: f64,
    /// Optional externally supplied initial Cartesian Hessian matrix (in kcal / (mol · Å²))
    pub initial_hessian: Option<AlignedMatrix<f64>>,
}

impl Default for TransitionStateOptions {
    fn default() -> Self {
        Self {
            max_cycles: 100,
            grad_rms_tol: 0.1,
            grad_max_tol: 0.2,
            trust_radius: 0.1,
            min_trust_radius: 0.005,
            max_trust_radius: 0.3,
            update_scheme: HessianUpdateScheme::Bofill,
            mode_following: true,
            target_mode: None,
            opt_mask: None,
            use_nddo: false,
            hessian_delta: 0.005,
            initial_hessian: None,
        }
    }
}

/// Result of transition state search via Eigenvector Following.
#[derive(Debug, Clone)]
pub struct TransitionStateResult {
    /// Whether the transition state search converged within tolerances
    pub converged: bool,
    /// Number of EF optimization cycles completed
    pub cycles: usize,
    /// Final electronic energy in eV
    pub final_energy_ev: f64,
    /// Final standard heat of formation in kcal / mol
    pub heat_of_formation_kcal: f64,
    /// Initial RMS gradient norm in kcal / (mol · Å)
    pub initial_grad_rms: f64,
    /// Final RMS gradient norm in kcal / (mol · Å)
    pub final_grad_rms: f64,
    /// Final maximum gradient component in kcal / (mol · Å)
    pub final_grad_max: f64,
    /// Eigenvalue of the transition state mode along which energy is maximized (in kcal / (mol · Å²))
    pub ts_mode_eigenvalue: f64,
    /// Index of the transition state mode (0-indexed)
    pub ts_mode_index: usize,
    /// Final Cartesian Hessian matrix at the stationary point
    pub final_hessian: AlignedMatrix<f64>,
    /// Final converged SCF wave function
    pub final_scf: ScfResult,
}

/// Preallocated workspace for Eigenvector Following ensuring zero allocations in iterative cycles.
#[derive(Debug, Clone)]
pub struct EigenvectorFollowingWorkspace {
    pub eigenvalues: AlignedVec64<f64>,
    pub eigenvectors: AlignedMatrix<f64>,
    pub work_mat: AlignedMatrix<f64>,
    pub f_basis: Vec<f64>,
    pub s_basis: Vec<f64>,
    pub step_cart: Vec<f64>,
    pub grad_cart: Vec<f64>,
    pub grad_cart_old: Vec<f64>,
    pub v_target: Vec<f64>,
    pub gradients_3d: Vec<[f64; 3]>,
    pub g_plus: Vec<[f64; 3]>,
    pub g_minus: Vec<[f64; 3]>,
    pub hess_ws: HessianUpdateWorkspace,
}

impl EigenvectorFollowingWorkspace {
    /// Preallocate all working buffers for a system with `natoms` atoms.
    pub fn allocate(natoms: usize) -> Self {
        let n3 = 3 * natoms;
        Self {
            eigenvalues: AlignedVec64::zeroed(n3),
            eigenvectors: AlignedMatrix::zeroed(n3, n3),
            work_mat: AlignedMatrix::zeroed(n3, n3),
            f_basis: vec![0.0; n3],
            s_basis: vec![0.0; n3],
            step_cart: vec![0.0; n3],
            grad_cart: vec![0.0; n3],
            grad_cart_old: vec![0.0; n3],
            v_target: vec![0.0; n3],
            gradients_3d: vec![[0.0; 3]; natoms],
            g_plus: vec![[0.0; 3]; natoms],
            g_minus: vec![[0.0; 3]; natoms],
            hess_ws: HessianUpdateWorkspace::allocate(n3),
        }
    }
}

/// Compute initial Cartesian Hessian matrix via central finite differences of analytical gradients.
#[allow(clippy::too_many_arguments)]
pub fn compute_initial_cartesian_hessian(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    scf_ws: &mut ScfWorkspace,
    grad_ws: &mut GradientWorkspace,
    ef_ws: &mut EigenvectorFollowingWorkspace,
    delta: f64,
    use_nddo: bool,
    mask: Option<&[bool]>,
) -> AlignedMatrix<f64> {
    let natoms = batch.natoms;
    let n3 = 3 * natoms;
    let inv_2delta = 1.0 / (2.0 * delta);

    let mut hess = AlignedMatrix::zeroed(n3, n3);
    let init_density = scf_ws.density.clone();

    for a in 0..natoms {
        for alpha in 0..3 {
            let col = 3 * a + alpha;
            if let Some(m) = mask {
                if !m[col] {
                    // Frozen coordinate: set large positive curvature on diagonal
                    hess.set(col, col, 500.0);
                    continue;
                }
            }

            // +delta displacement
            displace_coord(batch, a, alpha, delta);
            scf_ws.density.copy_from(&init_density);
            run_rhf_scf_adaptive_with_nddo(batch, model, scf_ws, 50, 1e-7, 1e-6, use_nddo);
            compute_cartesian_gradients_with_options(
                batch,
                model,
                &scf_ws.density,
                grad_ws,
                &mut ef_ws.g_plus,
                use_nddo,
            );

            // -delta displacement
            displace_coord(batch, a, alpha, -2.0 * delta);
            scf_ws.density.copy_from(&init_density);
            run_rhf_scf_adaptive_with_nddo(batch, model, scf_ws, 50, 1e-7, 1e-6, use_nddo);
            compute_cartesian_gradients_with_options(
                batch,
                model,
                &scf_ws.density,
                grad_ws,
                &mut ef_ws.g_minus,
                use_nddo,
            );

            // Restore coordinate
            displace_coord(batch, a, alpha, delta);

            // Populate column: d(grad_l)/d(coord_col) in kcal / (mol · Å²)
            for b in 0..natoms {
                for beta in 0..3 {
                    let row = 3 * b + beta;
                    let dg = (ef_ws.g_plus[b][beta] - ef_ws.g_minus[b][beta])
                        * inv_2delta
                        * EV_TO_KCAL_MOL;
                    hess.set(row, col, dg);
                }
            }
        }
    }

    // Restore density
    scf_ws.density.copy_from(&init_density);

    // Exact symmetrization: H = 0.5 * (H + H^T)
    for i in 0..n3 {
        for j in (i + 1)..n3 {
            let avg = 0.5 * (hess.get(i, j) + hess.get(j, i));
            hess.set(i, j, avg);
            hess.set(j, i, avg);
        }
    }

    hess
}

#[inline(always)]
fn displace_coord(batch: &mut MolecularBatch, atom: usize, axis: usize, delta: f64) {
    match axis {
        0 => batch.x[atom] += delta,
        1 => batch.y[atom] += delta,
        2 => batch.z[atom] += delta,
        _ => unreachable!(),
    }
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

/// Solve the secular equation for the transverse (minimization) subspace:
/// $$W(\lambda) = \sum_{i \ne \tau} \frac{f_i^2}{\lambda - \lambda_i} - \lambda = 0, \quad \lambda < \min_{i \ne \tau} \lambda_i$$
fn solve_rfo_lambda_transverse(eigenvalues: &[f64], f: &[f64], ts_mode: usize) -> f64 {
    let n = eigenvalues.len();
    let mut lambda_min = f64::INFINITY;
    for (i, &eig) in eigenvalues.iter().enumerate() {
        if i != ts_mode && eig < lambda_min {
            lambda_min = eig;
        }
    }

    if !lambda_min.is_finite() {
        return 0.0;
    }

    let eval_w = |lam: f64| -> f64 {
        let mut sum = -lam;
        for i in 0..n {
            if i != ts_mode {
                let diff = lam - eigenvalues[i];
                if diff.abs() > 1e-15 {
                    sum += (f[i] * f[i]) / diff;
                }
            }
        }
        sum
    };

    // Upper bound just below lambda_min
    let eps = 1e-5;
    let mut bu = lambda_min - eps;
    if eval_w(bu) >= 0.0 {
        // If eval_w(bu) >= 0, root is further left
        bu = lambda_min - 1e-3;
    }

    // Lower bound: expand step until fl > 0
    let mut step = 1.0;
    let mut bl = bu - step;
    let mut fl = eval_w(bl);
    let mut iter = 0;
    while fl <= 0.0 && iter < 60 {
        step *= 2.0;
        bl -= step;
        fl = eval_w(bl);
        iter += 1;
    }

    if fl <= 0.0 {
        return lambda_min - 1.0;
    }

    // Bisection to high precision
    let mut bisection_iter = 0;
    while (bu - bl).abs() > 1e-9 && bisection_iter < 100 {
        bisection_iter += 1;
        let mid = 0.5 * (bl + bu);
        let fm = eval_w(mid);
        if fm.abs() < 1e-12 {
            return mid;
        }
        if fm > 0.0 {
            bl = mid;
        } else {
            bu = mid;
        }
    }

    0.5 * (bl + bu)
}

/// Run Partitioned Rational Function Optimization (P-RFO) transition state search.
///
/// Modifies `batch.x`, `batch.y`, `batch.z` in-place until the first-order saddle point is located.
///
/// # Strict Invariants
/// * Preserves strict 0-malloc memory invariant during iterative optimization sweeps.
/// * Rigorous trust-radius sphere constraint: $\|s\| \le R_{\text{trust}}$.
/// * Preserves mode continuity along the reaction coordinate via eigenvector overlap.
pub fn optimize_transition_state(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    scf_ws: &mut ScfWorkspace,
    grad_ws: &mut GradientWorkspace,
    ef_ws: &mut EigenvectorFollowingWorkspace,
    options: &TransitionStateOptions,
) -> TransitionStateResult {
    let natoms = batch.natoms;
    let n3 = 3 * natoms;

    // 1. Initial SCF evaluation
    scf_ws.reset();
    let initial_scf =
        run_rhf_scf_adaptive_with_nddo(batch, model, scf_ws, 50, 1e-7, 1e-6, options.use_nddo);
    let mut current_energy = initial_scf.total_energy_ev;
    let mut last_scf = initial_scf;

    // 2. Initial Gradients
    compute_cartesian_gradients_with_options(
        batch,
        model,
        &scf_ws.density,
        grad_ws,
        &mut ef_ws.gradients_3d,
        options.use_nddo,
    );

    // Apply optimization mask
    if let Some(ref m) = options.opt_mask {
        for a in 0..natoms {
            for c in 0..3 {
                if !m[3 * a + c] {
                    ef_ws.gradients_3d[a][c] = 0.0;
                }
            }
        }
    }

    let (mut rms_g, mut max_g) =
        compute_effective_gradient_norms(&ef_ws.gradients_3d, options.opt_mask.as_deref());
    let initial_rms_g = rms_g;

    // Pack Cartesian gradients in kcal / (mol · Å)
    for a in 0..natoms {
        ef_ws.grad_cart[3 * a] = ef_ws.gradients_3d[a][0] * EV_TO_KCAL_MOL;
        ef_ws.grad_cart[3 * a + 1] = ef_ws.gradients_3d[a][1] * EV_TO_KCAL_MOL;
        ef_ws.grad_cart[3 * a + 2] = ef_ws.gradients_3d[a][2] * EV_TO_KCAL_MOL;
    }

    // 3. Obtain or compute initial Cartesian Hessian
    let mut hessian = match options.initial_hessian {
        Some(ref h) => {
            assert_eq!(h.rows, n3);
            assert_eq!(h.cols, n3);
            h.clone()
        }
        None => compute_initial_cartesian_hessian(
            batch,
            model,
            scf_ws,
            grad_ws,
            ef_ws,
            options.hessian_delta,
            options.use_nddo,
            options.opt_mask.as_deref(),
        ),
    };

    let mut trust_radius = options.trust_radius;
    let mut ts_mode = 0;
    let mut converged = false;
    let mut cycles_done = 0;

    for cycle in 1..=options.max_cycles {
        cycles_done = cycle;

        // Diagonalize current Hessian using preallocated work_mat: 0 allocations
        diagonalize_symmetric_with_work(
            &hessian,
            &mut ef_ws.work_mat,
            &mut ef_ws.eigenvalues,
            &mut ef_ws.eigenvectors,
        );

        // Mode following / overlap tracking
        if cycle == 1 {
            ts_mode = options.target_mode.unwrap_or(0);
            for j in 0..n3 {
                ef_ws.v_target[j] = ef_ws.eigenvectors.get(j, ts_mode);
            }
        } else if options.mode_following {
            let mut best_mode = 0;
            let mut best_overlap = -1.0;
            for i in 0..n3 {
                let mut ovlp = 0.0;
                for j in 0..n3 {
                    ovlp += ef_ws.eigenvectors.get(j, i) * ef_ws.v_target[j];
                }
                let abs_ovlp = ovlp.abs();
                if abs_ovlp > best_overlap {
                    best_overlap = abs_ovlp;
                    best_mode = i;
                }
            }
            ts_mode = best_mode;
            for j in 0..n3 {
                ef_ws.v_target[j] = ef_ws.eigenvectors.get(j, ts_mode);
            }
        }

        // Transform Cartesian gradient to eigenvector basis: f_i = v_i^T g
        for i in 0..n3 {
            let mut sum = 0.0;
            for j in 0..n3 {
                sum += ef_ws.eigenvectors.get(j, i) * ef_ws.grad_cart[j];
            }
            ef_ws.f_basis[i] = sum;
        }

        // --- P-RFO Step Formation ---
        // (1) Along TS mode: maximize energy
        let lambda_ts = ef_ws.eigenvalues[ts_mode];
        let f_ts = ef_ws.f_basis[ts_mode];
        let lambda0 = 0.5 * (lambda_ts + (lambda_ts * lambda_ts + 4.0 * f_ts * f_ts).sqrt());
        let denom_ts = lambda0 - lambda_ts;
        ef_ws.s_basis[ts_mode] = if denom_ts.abs() > 1e-12 {
            f_ts / denom_ts
        } else {
            0.0
        };

        // (2) Along transverse modes: minimize energy
        let lambda_trans = solve_rfo_lambda_transverse(&ef_ws.eigenvalues, &ef_ws.f_basis, ts_mode);
        for i in 0..n3 {
            if i != ts_mode {
                let diff = lambda_trans - ef_ws.eigenvalues[i];
                ef_ws.s_basis[i] = if diff.abs() > 1e-12 {
                    ef_ws.f_basis[i] / diff
                } else {
                    0.0
                };
            }
        }

        // Transform step back to Cartesian coordinates: s = sum_i s_basis[i] * v_i
        for j in 0..n3 {
            let mut sum = 0.0;
            for i in 0..n3 {
                sum += ef_ws.eigenvectors.get(j, i) * ef_ws.s_basis[i];
            }
            // Enforce frozen mask
            if let Some(ref m) = options.opt_mask {
                if !m[j] {
                    sum = 0.0;
                }
            }
            ef_ws.step_cart[j] = sum;
        }

        // Enforce Trust Radius Constraint: ||s|| <= R_trust
        let mut step_norm_sq = 0.0;
        for j in 0..n3 {
            step_norm_sq += ef_ws.step_cart[j] * ef_ws.step_cart[j];
        }
        let step_norm = step_norm_sq.sqrt();
        let step_scale = if step_norm > trust_radius && step_norm > 1e-14 {
            let scale = trust_radius / step_norm;
            for j in 0..n3 {
                ef_ws.step_cart[j] *= scale;
            }
            scale
        } else {
            1.0
        };

        // Predicted energy change (in kcal / mol)
        let mut de_pred = 0.0;
        for i in 0..n3 {
            let si = ef_ws.s_basis[i] * step_scale;
            de_pred += ef_ws.f_basis[i] * si + 0.5 * ef_ws.eigenvalues[i] * si * si;
        }

        // Save old coordinates and gradients for Hessian update
        ef_ws.grad_cart_old.copy_from_slice(&ef_ws.grad_cart);

        // Apply Cartesian displacement to batch coordinates
        for a in 0..natoms {
            batch.x[a] += ef_ws.step_cart[3 * a];
            batch.y[a] += ef_ws.step_cart[3 * a + 1];
            batch.z[a] += ef_ws.step_cart[3 * a + 2];
        }

        // Recompute SCF at new geometry
        let new_scf =
            run_rhf_scf_adaptive_with_nddo(batch, model, scf_ws, 50, 1e-7, 1e-6, options.use_nddo);
        let new_energy = new_scf.total_energy_ev;
        let de_act = (new_energy - current_energy) * EV_TO_KCAL_MOL;
        current_energy = new_energy;
        last_scf = new_scf;

        // Compute new Cartesian gradients
        compute_cartesian_gradients_with_options(
            batch,
            model,
            &scf_ws.density,
            grad_ws,
            &mut ef_ws.gradients_3d,
            options.use_nddo,
        );

        if let Some(ref m) = options.opt_mask {
            for a in 0..natoms {
                for c in 0..3 {
                    if !m[3 * a + c] {
                        ef_ws.gradients_3d[a][c] = 0.0;
                    }
                }
            }
        }

        let (cur_rms, cur_max) =
            compute_effective_gradient_norms(&ef_ws.gradients_3d, options.opt_mask.as_deref());
        rms_g = cur_rms;
        max_g = cur_max;

        for a in 0..natoms {
            ef_ws.grad_cart[3 * a] = ef_ws.gradients_3d[a][0] * EV_TO_KCAL_MOL;
            ef_ws.grad_cart[3 * a + 1] = ef_ws.gradients_3d[a][1] * EV_TO_KCAL_MOL;
            ef_ws.grad_cart[3 * a + 2] = ef_ws.gradients_3d[a][2] * EV_TO_KCAL_MOL;
        }

        // Trust Radius Adaptation (Jensen / OpenMOPAC ef.F90)
        if de_pred.abs() > 1e-5 {
            let ratio = de_act / de_pred;
            if ratio <= 0.1 || ratio >= 3.0 {
                trust_radius = (trust_radius.min(step_norm) / 2.0).max(options.min_trust_radius);
            } else if (0.75..=1.33).contains(&ratio) && step_norm >= 0.85 * trust_radius {
                trust_radius = (trust_radius * 2.0f64.sqrt()).min(options.max_trust_radius);
            }
        }

        // Convergence Check
        if rms_g < options.grad_rms_tol && max_g < options.grad_max_tol {
            converged = true;
            break;
        }

        // Hessian Update: Delta x = step_cart, Delta g = grad_cart - grad_cart_old
        for j in 0..n3 {
            ef_ws.grad_cart_old[j] = ef_ws.grad_cart[j] - ef_ws.grad_cart_old[j];
        }
        update_cartesian_hessian(
            &mut hessian,
            &ef_ws.step_cart,
            &ef_ws.grad_cart_old,
            options.update_scheme,
            &mut ef_ws.hess_ws,
        );
    }

    let (_, heat_of_formation_kcal) =
        compute_heat_of_formation(current_energy, &batch.atomic_numbers, model, 0.0);

    TransitionStateResult {
        converged,
        cycles: cycles_done,
        final_energy_ev: current_energy,
        heat_of_formation_kcal,
        initial_grad_rms: initial_rms_g,
        final_grad_rms: rms_g,
        final_grad_max: max_g,
        ts_mode_eigenvalue: ef_ws.eigenvalues[ts_mode],
        ts_mode_index: ts_mode,
        final_hessian: hessian,
        final_scf: last_scf,
    }
}
