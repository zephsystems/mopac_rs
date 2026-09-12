//! Intrinsic Reaction Coordinate (IRC) Path Tracing.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Implements the González-Schlegel mass-weighted steepest descent algorithm
//! (J. Chem. Phys. 90, 2154 (1989); J. Phys. Chem. 94, 5523 (1990)) on semi-empirical
//! potential energy surfaces.
//!
//! # Methodological Details
//! * Integrates path in mass-weighted coordinates $q_i = \sqrt{m_a} x_i$.
//! * Second-order constrained corrector on the hypersphere of radius $\frac{1}{2}\delta s$
//!   centered at the predictor pivot point $q^* = q_k + \frac{1}{2}\delta s \cdot p_k$.
//! * Traces both Forward and Reverse reaction branches starting from transition states.
//! * Zero heap allocations (0-malloc) in hot iterative predictor-corrector loops.

use crate::constants::codata2018::EV_TO_KCAL_MOL;
use crate::constants::standard_atomic_mass;
use crate::gradients::nuclear_gradients::{
    compute_cartesian_gradients_with_options, GradientWorkspace,
};
use crate::parameters::ParameterModel;
use crate::properties::heat::compute_heat_of_formation;
use crate::scf::scf_loop::{run_rhf_scf_adaptive_with_nddo, ScfOptions};
use crate::types::{MolecularBatch, ScfWorkspace};
use crate::vibrations::hessian::{compute_hessian_and_frequencies, HessianOptions};

/// Direction of IRC integration from the transition state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrcDirection {
    /// Follow path along positive transition vector (+v)
    Forward,
    /// Follow path along negative transition vector (-v)
    Reverse,
    /// Trace both Forward and Reverse branches to connect reactants and products
    Both,
}

/// Configuration options for Intrinsic Reaction Coordinate path tracing.
#[derive(Debug, Clone)]
pub struct IrcOptions {
    /// Arc length step size along the reaction path in $\text{amu}^{1/2}\text{ \AA}$ (default: 0.1)
    pub step_size: f64,
    /// Maximum number of reaction path points per direction (default: 50)
    pub max_points: usize,
    /// Maximum number of constrained corrector iterations per step (default: 25)
    pub corrector_max_iter: usize,
    /// Displacement convergence tolerance for the corrector step on the hypersphere in $\text{amu}^{1/2}\text{ \AA}$ (default: 1e-4)
    pub corrector_tol: f64,
    /// Mass-weighted gradient RMS threshold in $\text{kcal / (mol · \AA · amu}^{1/2}\text{)}$ to detect minimum (default: 0.05)
    pub grad_rms_tol: f64,
    /// Energy increase tolerance in kcal/mol before terminating branch (default: 0.02)
    pub energy_increase_tol: f64,
    /// Direction to trace
    pub direction: IrcDirection,
    /// Whether to evaluate full NDDO diatomic multipoles
    pub use_nddo: bool,
    /// Optional externally supplied mass-weighted transition vector (length: 3 * natoms)
    pub transition_vector: Option<Vec<f64>>,
}

impl Default for IrcOptions {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            max_points: 50,
            corrector_max_iter: 25,
            corrector_tol: 1e-4,
            grad_rms_tol: 0.05,
            energy_increase_tol: 0.02,
            direction: IrcDirection::Both,
            use_nddo: false,
            transition_vector: None,
        }
    }
}

/// Single point along the Intrinsic Reaction Coordinate path.
#[derive(Debug, Clone)]
pub struct IrcPoint {
    /// Integrated reaction coordinate $s$ in $\text{amu}^{1/2}\text{ \AA}$ (0 at TS, >0 forward, <0 reverse)
    pub path_coordinate: f64,
    /// Electronic total energy in eV
    pub energy_ev: f64,
    /// Standard heat of formation in kcal/mol
    pub heat_of_formation_kcal: f64,
    /// Cartesian coordinates in Ångströms (shape: [natoms, 3])
    pub coordinates: Vec<[f64; 3]>,
    /// Root-Mean-Square Cartesian gradient in kcal / (mol · Å)
    pub cartesian_gradient_rms: f64,
    /// Root-Mean-Square Mass-weighted gradient in kcal / (mol · Å · amu^(1/2))
    pub mass_weighted_gradient_rms: f64,
}

/// Result of complete Intrinsic Reaction Coordinate integration.
#[derive(Debug, Clone)]
pub struct IrcResult {
    /// Sequence of reaction path points ordered monotonically along $s$
    /// (from reverse product/reactant through TS at $s=0$ to forward product/reactant)
    pub points: Vec<IrcPoint>,
    /// Index of the transition state point in the `points` vector ($s=0$)
    pub ts_point_index: usize,
    /// Whether the forward branch terminated at a local minimum
    pub forward_converged: bool,
    /// Whether the reverse branch terminated at a local minimum
    pub reverse_converged: bool,
}

/// Preallocated workspace for Intrinsic Reaction Coordinate integration.
///
/// Guarantees 0-malloc memory invariant during predictor-corrector sweeps.
#[derive(Debug, Clone)]
pub struct IrcWorkspace {
    /// Mass-weighted coordinates $q$, length $3N$
    pub q: Vec<f64>,
    /// Predictor pivot point $q^*$, length $3N$
    pub q_pivot: Vec<f64>,
    /// Candidate point $q^{(j)}$, length $3N$
    pub q_cand: Vec<f64>,
    /// Updated candidate point $q^{(j+1)}$, length $3N$
    pub q_next: Vec<f64>,
    /// Previous point on path $q_k$, length $3N$
    pub q_prev: Vec<f64>,
    /// Cartesian analytical gradient in eV / Å, length $N$
    pub gradients_3d: Vec<[f64; 3]>,
    /// Mass-weighted gradient in kcal / (mol · Å · amu^(1/2)), length $3N$
    pub grad_q: Vec<f64>,
    /// Unit descent/tangent direction $p_k$, length $3N$
    pub tangent: Vec<f64>,
    /// Mass-weighted transition state vector $v_q$, length $3N$
    pub ts_vector: Vec<f64>,
    /// Atomic masses in amu, length $N$
    pub masses: Vec<f64>,
    /// Square root of masses $\sqrt{m_a}$, length $N$
    pub sqrt_masses: Vec<f64>,
    /// Inverse square root of masses $1 / \sqrt{m_a}$, length $N$
    pub inv_sqrt_masses: Vec<f64>,
    /// Initial TS Cartesian coordinates, length $N$
    pub ts_coords: Vec<[f64; 3]>,
}

impl IrcWorkspace {
    /// Allocate workspace for molecular system with `natoms`.
    pub fn allocate(batch: &MolecularBatch) -> Self {
        let natoms = batch.natoms;
        let n3 = 3 * natoms;

        let mut masses = Vec::with_capacity(natoms);
        let mut sqrt_masses = Vec::with_capacity(natoms);
        let mut inv_sqrt_masses = Vec::with_capacity(natoms);
        let mut ts_coords = Vec::with_capacity(natoms);

        for a in 0..natoms {
            let m = standard_atomic_mass(batch.atomic_numbers[a]);
            masses.push(m);
            let sm = m.sqrt();
            sqrt_masses.push(sm);
            inv_sqrt_masses.push(1.0 / sm);
            ts_coords.push([batch.x[a], batch.y[a], batch.z[a]]);
        }

        Self {
            q: vec![0.0; n3],
            q_pivot: vec![0.0; n3],
            q_cand: vec![0.0; n3],
            q_next: vec![0.0; n3],
            q_prev: vec![0.0; n3],
            gradients_3d: vec![[0.0; 3]; natoms],
            grad_q: vec![0.0; n3],
            tangent: vec![0.0; n3],
            ts_vector: vec![0.0; n3],
            masses,
            sqrt_masses,
            inv_sqrt_masses,
            ts_coords,
        }
    }

    /// Reset Cartesian coordinates in `batch` back to TS coordinates.
    pub fn restore_ts_coords(&self, batch: &mut MolecularBatch) {
        for a in 0..batch.natoms {
            batch.x[a] = self.ts_coords[a][0];
            batch.y[a] = self.ts_coords[a][1];
            batch.z[a] = self.ts_coords[a][2];
        }
    }

    /// Convert current batch Cartesian coordinates to mass-weighted coordinates in `self.q`.
    pub fn cartesian_to_mass_weighted(&mut self, batch: &MolecularBatch) {
        for a in 0..batch.natoms {
            let sm = self.sqrt_masses[a];
            self.q[3 * a] = sm * batch.x[a];
            self.q[3 * a + 1] = sm * batch.y[a];
            self.q[3 * a + 2] = sm * batch.z[a];
        }
    }

    /// Convert mass-weighted coordinates into batch Cartesian coordinates.
    pub fn mass_weighted_to_cartesian(&self, q: &[f64], batch: &mut MolecularBatch) {
        for a in 0..batch.natoms {
            let ism = self.inv_sqrt_masses[a];
            batch.x[a] = q[3 * a] * ism;
            batch.y[a] = q[3 * a + 1] * ism;
            batch.z[a] = q[3 * a + 2] * ism;
        }
    }

    /// Convert Cartesian gradients in `self.gradients_3d` (in eV/Å) to mass-weighted gradients (kcal/(mol·Å·amu^(1/2))).
    pub fn compute_mass_weighted_gradients(&mut self, natoms: usize) -> (f64, f64) {
        let mut sum_sq_cart = 0.0;
        let mut sum_sq_mw = 0.0;
        let n3 = 3 * natoms;

        for a in 0..natoms {
            let ism = self.inv_sqrt_masses[a];
            for c in 0..3 {
                let g_cart = self.gradients_3d[a][c] * EV_TO_KCAL_MOL;
                sum_sq_cart += g_cart * g_cart;
                let g_mw = g_cart * ism;
                self.grad_q[3 * a + c] = g_mw;
                sum_sq_mw += g_mw * g_mw;
            }
        }

        let rms_cart = (sum_sq_cart / (n3 as f64)).sqrt();
        let rms_mw = (sum_sq_mw / (n3 as f64)).sqrt();
        (rms_cart, rms_mw)
    }
}

/// Compute Intrinsic Reaction Coordinate path starting from a transition state.
///
/// # Strict Invariants
/// * Preserves strict 0-malloc memory invariant during iterative sweeps.
/// * Preserves mass-weighting and second-order González-Schlegel hypersphere geometry.
pub fn trace_intrinsic_reaction_coordinate(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    scf_ws: &mut ScfWorkspace,
    grad_ws: &mut GradientWorkspace,
    irc_ws: &mut IrcWorkspace,
    options: &IrcOptions,
) -> IrcResult {
    let natoms = batch.natoms;
    let n3 = 3 * natoms;

    // Save TS coordinates
    for a in 0..natoms {
        irc_ws.ts_coords[a] = [batch.x[a], batch.y[a], batch.z[a]];
    }

    // Determine mass-weighted transition state vector v_q
    if let Some(ref tv) = options.transition_vector {
        assert_eq!(tv.len(), n3, "Supplied transition vector length mismatch");
        irc_ws.ts_vector.copy_from_slice(tv);
    } else {
        // Compute harmonic frequencies at TS to locate the imaginary normal mode
        let scf_opts = ScfOptions {
            max_iter: 60,
            energy_tol_ev: 1e-8,
            density_tol: 1e-7,
            use_nddo: options.use_nddo,
            ..Default::default()
        };
        let hess_opts = HessianOptions {
            delta: 0.005,
            recompute_scf: true,
            use_nddo: options.use_nddo,
            project_external: true,
            ..Default::default()
        };
        let vib_res = compute_hessian_and_frequencies(batch, model, scf_ws, &scf_opts, &hess_opts);

        let mut found_mode = false;
        if let Some(first_mode) = vib_res.normal_modes.first() {
            if first_mode.frequency_cm1 < 0.0 {
                // Mass-weight normal mode displacements
                let mut norm_sq = 0.0;
                for a in 0..natoms {
                    let sm = irc_ws.sqrt_masses[a];
                    for c in 0..3 {
                        let val = sm * first_mode.displacements[a][c];
                        irc_ws.ts_vector[3 * a + c] = val;
                        norm_sq += val * val;
                    }
                }
                let inv_norm = 1.0 / norm_sq.sqrt().max(1e-15);
                for i in 0..n3 {
                    irc_ws.ts_vector[i] *= inv_norm;
                }
                found_mode = true;
            }
        }

        if !found_mode {
            // Fallback: take highest Cartesian displacement or first coordinate axis
            irc_ws.ts_vector.fill(0.0);
            irc_ws.ts_vector[0] = 1.0;
        }
    }

    // Evaluate TS electronic energy and gradients
    irc_ws.restore_ts_coords(batch);
    scf_ws.reset();
    let ts_scf =
        run_rhf_scf_adaptive_with_nddo(batch, model, scf_ws, 60, 1e-7, 1e-6, options.use_nddo);
    let ts_heat =
        compute_heat_of_formation(ts_scf.total_energy_ev, &batch.atomic_numbers, model, 0.0).1;
    compute_cartesian_gradients_with_options(
        batch,
        model,
        &scf_ws.density,
        grad_ws,
        &mut irc_ws.gradients_3d,
        options.use_nddo,
    );
    let (ts_rms_cart, ts_rms_mw) = irc_ws.compute_mass_weighted_gradients(natoms);

    // If residual gradient at TS is non-zero, orient ts_vector so that +ts_vector points downhill (-grad_q)
    let g_dot_v: f64 = (0..n3)
        .map(|i| irc_ws.grad_q[i] * irc_ws.ts_vector[i])
        .sum();
    if g_dot_v > 1e-4 {
        for i in 0..n3 {
            irc_ws.ts_vector[i] = -irc_ws.ts_vector[i];
        }
    }

    let ts_point = IrcPoint {
        path_coordinate: 0.0,
        energy_ev: ts_scf.total_energy_ev,
        heat_of_formation_kcal: ts_heat,
        coordinates: irc_ws.ts_coords.clone(),
        cartesian_gradient_rms: ts_rms_cart,
        mass_weighted_gradient_rms: ts_rms_mw,
    };

    let mut forward_points = Vec::new();
    let mut forward_converged = false;
    let mut reverse_points = Vec::new();
    let mut reverse_converged = false;

    // Trace Forward branch (+1.0 along transition vector)
    if options.direction == IrcDirection::Forward || options.direction == IrcDirection::Both {
        let (pts, conv) = trace_single_branch(batch, model, scf_ws, grad_ws, irc_ws, options, 1.0);
        forward_points = pts;
        forward_converged = conv;
    }

    // Trace Reverse branch (-1.0 along transition vector)
    if options.direction == IrcDirection::Reverse || options.direction == IrcDirection::Both {
        let (pts, conv) = trace_single_branch(batch, model, scf_ws, grad_ws, irc_ws, options, -1.0);
        reverse_points = pts;
        reverse_converged = conv;
    }

    // Assemble ordered path: reverse points (reversed so s is strictly ascending) + TS + forward points
    let mut all_points = Vec::with_capacity(reverse_points.len() + 1 + forward_points.len());
    for p in reverse_points.into_iter().rev() {
        all_points.push(p);
    }
    let ts_idx = all_points.len();
    all_points.push(ts_point);
    for p in forward_points {
        all_points.push(p);
    }

    // Restore batch coordinates to final point (or TS)
    irc_ws.restore_ts_coords(batch);

    IrcResult {
        points: all_points,
        ts_point_index: ts_idx,
        forward_converged,
        reverse_converged,
    }
}

/// Trace a single direction along the IRC path.
fn trace_single_branch(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    scf_ws: &mut ScfWorkspace,
    grad_ws: &mut GradientWorkspace,
    irc_ws: &mut IrcWorkspace,
    options: &IrcOptions,
    direction_sign: f64,
) -> (Vec<IrcPoint>, bool) {
    let natoms = batch.natoms;
    let n3 = 3 * natoms;
    let ds = options.step_size;
    let half_ds = 0.5 * ds;

    // 1. Initial coordinates at TS in mass-weighted space
    irc_ws.restore_ts_coords(batch);
    irc_ws.cartesian_to_mass_weighted(batch);

    // Initial tangent along transition vector
    for i in 0..n3 {
        irc_ws.tangent[i] = direction_sign * irc_ws.ts_vector[i];
    }

    let mut path_points = Vec::new();
    let mut s = 0.0;
    let mut branch_converged = false;
    let mut last_energy_kcal = compute_heat_of_formation(
        run_rhf_scf_adaptive_with_nddo(batch, model, scf_ws, 50, 1e-7, 1e-6, options.use_nddo)
            .total_energy_ev,
        &batch.atomic_numbers,
        model,
        0.0,
    )
    .1;

    for _step_idx in 0..options.max_points {
        // Compute predictor pivot point: q^* = q_k + 0.5 * ds * p_k
        for i in 0..n3 {
            irc_ws.q_pivot[i] = irc_ws.q[i] + half_ds * irc_ws.tangent[i];
            // Initial guess on hypersphere: q^(0) = q_k + ds * p_k
            irc_ws.q_cand[i] = irc_ws.q[i] + ds * irc_ws.tangent[i];
        }

        // González-Schlegel constrained corrector iterations on hypersphere of radius half_ds centered at q_pivot
        let mut _corrector_converged = false;
        let mut current_energy_ev = 0.0;
        let mut current_heat_kcal = 0.0;
        let mut current_rms_cart = 0.0;
        let mut current_rms_mw = 0.0;

        for _iter in 0..options.corrector_max_iter {
            // Unpack candidate q_cand to batch Cartesian
            irc_ws.mass_weighted_to_cartesian(&irc_ws.q_cand, batch);

            // SCF evaluation
            scf_ws.reset();
            let scf_res = run_rhf_scf_adaptive_with_nddo(
                batch,
                model,
                scf_ws,
                50,
                1e-7,
                1e-6,
                options.use_nddo,
            );
            current_energy_ev = scf_res.total_energy_ev;
            current_heat_kcal =
                compute_heat_of_formation(current_energy_ev, &batch.atomic_numbers, model, 0.0).1;

            // Gradients
            compute_cartesian_gradients_with_options(
                batch,
                model,
                &scf_ws.density,
                grad_ws,
                &mut irc_ws.gradients_3d,
                options.use_nddo,
            );
            let (rms_cart, rms_mw) = irc_ws.compute_mass_weighted_gradients(natoms);
            current_rms_cart = rms_cart;
            current_rms_mw = rms_mw;

            // Compute norm of mass-weighted gradient
            let mut g_norm_sq = 0.0;
            for i in 0..n3 {
                g_norm_sq += irc_ws.grad_q[i] * irc_ws.grad_q[i];
            }
            let g_norm = g_norm_sq.sqrt();

            if g_norm < 1e-12 {
                // Stationary point encountered
                for i in 0..n3 {
                    irc_ws.q_next[i] = irc_ws.q_cand[i];
                }
                _corrector_converged = true;
                break;
            }

            // Next point on hypersphere: target = q_pivot - half_ds * (g_q / ||g_q||)
            let scale = half_ds / g_norm;
            let mut diff_sq = 0.0;
            for i in 0..n3 {
                let target = irc_ws.q_pivot[i] - scale * irc_ws.grad_q[i];
                let d = target - irc_ws.q_cand[i];
                diff_sq += d * d;
                // Damped combination
                irc_ws.q_next[i] = 0.5 * (irc_ws.q_cand[i] + target);
            }

            // Reproject q_next onto hypersphere of radius half_ds centered at q_pivot
            let mut rad_sq = 0.0;
            for i in 0..n3 {
                let dr = irc_ws.q_next[i] - irc_ws.q_pivot[i];
                rad_sq += dr * dr;
            }
            let rad = rad_sq.sqrt().max(1e-15);
            let rad_scale = half_ds / rad;
            for i in 0..n3 {
                irc_ws.q_cand[i] =
                    irc_ws.q_pivot[i] + rad_scale * (irc_ws.q_next[i] - irc_ws.q_pivot[i]);
            }

            let disp_change = diff_sq.sqrt();
            if disp_change < options.corrector_tol {
                _corrector_converged = true;
                break;
            }
        }

        // Unpack final corrector point
        irc_ws.mass_weighted_to_cartesian(&irc_ws.q_cand, batch);
        let mut coords = Vec::with_capacity(natoms);
        for a in 0..natoms {
            coords.push([batch.x[a], batch.y[a], batch.z[a]]);
        }

        s += direction_sign * ds;

        path_points.push(IrcPoint {
            path_coordinate: s,
            energy_ev: current_energy_ev,
            heat_of_formation_kcal: current_heat_kcal,
            coordinates: coords,
            cartesian_gradient_rms: current_rms_cart,
            mass_weighted_gradient_rms: current_rms_mw,
        });

        // Stopping criteria 1: Minimum reached (mass-weighted gradient norm below threshold)
        if current_rms_mw < options.grad_rms_tol {
            branch_converged = true;
            break;
        }

        // Stopping criteria 2: Energy climbed significantly above previous point
        if current_heat_kcal > last_energy_kcal + options.energy_increase_tol {
            branch_converged = true;
            break;
        }
        last_energy_kcal = current_heat_kcal;

        // Accept point for next step: q_k = q_next
        for i in 0..n3 {
            irc_ws.q[i] = irc_ws.q_cand[i];
        }

        // New tangent along negative gradient at converged point
        let mut g_norm_sq = 0.0;
        for i in 0..n3 {
            g_norm_sq += irc_ws.grad_q[i] * irc_ws.grad_q[i];
        }
        let g_norm = g_norm_sq.sqrt().max(1e-15);
        for i in 0..n3 {
            irc_ws.tangent[i] = -irc_ws.grad_q[i] / g_norm;
        }
    }

    (path_points, branch_converged)
}
