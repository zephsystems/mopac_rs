//! Two-Ended Transition State Search (SADDLE / LOCATE-TS Engine).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Direct mathematical port and modernization of canonical OpenMOPAC `react1.F90` and `gmetry.F90`.
//!
//! # Methodological Details
//! * Takes reactant and product Cartesian geometries $X_R$ and $X_P$.
//! * Rigorously aligns $X_P$ to $X_R$ via Horn's quaternion method (closed-form optimal Eckart orientation).
//! * Measures Euclidean reaction coordinate distance $D = \|X_A - X_B\|_2$.
//! * Progressively steps the lower energy endpoint along the reaction vector and minimizes the
//!   energy on the hypersphere of radius $r_k$ perpendicular to the reaction path.
//! * Alternates/swaps moving endpoints upon uphill ascent or consecutive steps.
//! * Interpolates the saddle ridge apex when $D \le D_{\text{threshold}}$.
//! * Optionally refines the candidate structure to the exact stationary point via
//!   eigenvector-following (`optimize_transition_state`).

use crate::constants::codata2018::EV_TO_KCAL_MOL;
use crate::gradients::nuclear_gradients::{
    compute_cartesian_gradients_with_options, GradientWorkspace,
};
use crate::opt::eigenvector_following::{
    optimize_transition_state, EigenvectorFollowingWorkspace, TransitionStateOptions,
    TransitionStateResult,
};
use crate::parameters::ParameterModel;
use crate::properties::heat::compute_heat_of_formation;
use crate::scf::eigensolver::diagonalize_symmetric;
use crate::scf::scf_loop::run_rhf_scf_adaptive_with_nddo;
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch, ScfWorkspace};

/// Optimal rigid-body alignment of `mobile` coordinates onto `target` coordinates.
///
/// Translates centroids to the origin and evaluates the optimal rotation matrix $R \in \mathrm{SO}(3)$
/// via Horn's quaternion eigensystem of the cross-covariance matrix $H = X_{\text{target}}^T X_{\text{mobile}}$.
/// Returns `(aligned_mobile_coordinates, euclidean_distance)`.
pub fn dock(target: &[[f64; 3]], mobile: &[[f64; 3]]) -> (Vec<[f64; 3]>, f64) {
    let natoms = target.len();
    assert_eq!(
        mobile.len(),
        natoms,
        "Target and mobile coordinate sets must have equal atom count"
    );

    if natoms == 0 {
        return (Vec::new(), 0.0);
    }

    // 1. Centroids
    let mut c_target = [0.0f64; 3];
    let mut c_mobile = [0.0f64; 3];
    for i in 0..natoms {
        for c in 0..3 {
            c_target[c] += target[i][c];
            c_mobile[c] += mobile[i][c];
        }
    }
    let inv_n = 1.0 / (natoms as f64);
    for c in 0..3 {
        c_target[c] *= inv_n;
        c_mobile[c] *= inv_n;
    }

    // Centered coordinates
    let mut r_target = vec![[0.0f64; 3]; natoms];
    let mut r_mobile = vec![[0.0f64; 3]; natoms];
    for i in 0..natoms {
        for c in 0..3 {
            r_target[i][c] = target[i][c] - c_target[c];
            r_mobile[i][c] = mobile[i][c] - c_mobile[c];
        }
    }

    // 2. Cross-covariance matrix H = r_target^T * r_mobile
    let mut h = [[0.0f64; 3]; 3];
    for i in 0..natoms {
        for r in 0..3 {
            for c in 0..3 {
                h[r][c] += r_target[i][r] * r_mobile[i][c];
            }
        }
    }

    // 3. Horn's 4x4 symmetric quaternion matrix K
    let hxx = h[0][0];
    let hxy = h[0][1];
    let hxz = h[0][2];
    let hyx = h[1][0];
    let hyy = h[1][1];
    let hyz = h[1][2];
    let hzx = h[2][0];
    let hzy = h[2][1];
    let hzz = h[2][2];

    let mut k_mat = AlignedMatrix::zeroed(4, 4);
    // Row 0
    k_mat.set(0, 0, hxx + hyy + hzz);
    k_mat.set(0, 1, hzy - hyz);
    k_mat.set(0, 2, hxz - hzx);
    k_mat.set(0, 3, hyx - hxy);
    // Row 1
    k_mat.set(1, 0, hzy - hyz);
    k_mat.set(1, 1, hxx - hyy - hzz);
    k_mat.set(1, 2, hxy + hyx);
    k_mat.set(1, 3, hzx + hxz);
    // Row 2
    k_mat.set(2, 0, hxz - hzx);
    k_mat.set(2, 1, hxy + hyx);
    k_mat.set(2, 2, -hxx + hyy - hzz);
    k_mat.set(2, 3, hyz + hzy);
    // Row 3
    k_mat.set(3, 0, hyx - hxy);
    k_mat.set(3, 1, hzx + hxz);
    k_mat.set(3, 2, hyz + hzy);
    k_mat.set(3, 3, -hxx - hyy + hzz);

    let mut eig_vals = AlignedVec64::zeroed(4);
    let mut eig_vecs = AlignedMatrix::zeroed(4, 4);
    diagonalize_symmetric(&k_mat, &mut eig_vals, &mut eig_vecs);

    // Maximum eigenvalue corresponds to column 3 (sorted ascending)
    let q0 = eig_vecs.get(0, 3);
    let q1 = eig_vecs.get(1, 3);
    let q2 = eig_vecs.get(2, 3);
    let q3 = eig_vecs.get(3, 3);

    // 4. Rotation matrix from quaternion
    let rot = [
        [
            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
        ],
    ];

    // 5. Apply rotation and translate to target frame
    let mut aligned = vec![[0.0f64; 3]; natoms];
    let mut dist_sq = 0.0f64;
    for i in 0..natoms {
        for r in 0..3 {
            let mut val = 0.0;
            for c in 0..3 {
                val += rot[r][c] * r_mobile[i][c];
            }
            aligned[i][r] = val + c_target[r];
            let diff = target[i][r] - aligned[i][r];
            dist_sq += diff * diff;
        }
    }

    (aligned, dist_sq.sqrt())
}

/// Options controlling the two-ended SADDLE reaction transition state locator.
#[derive(Debug, Clone)]
pub struct SaddleOptions {
    /// Maximum number of macro SADDLE stepping cycles (default: 60)
    pub max_cycles: usize,
    /// Distance decrement step per cycle in Angstroms (default: 0.10 Å)
    pub step_size: f64,
    /// Termination distance between endpoint structures in Angstroms (default: 0.12 Å)
    pub convergence_distance: f64,
    /// Maximum inner relaxation steps per cycle (default: 12)
    pub inner_max_steps: usize,
    /// Gradient norm tolerance for transverse hyperplane relaxation in kcal/(mol * A) (default: 3.0)
    pub inner_grad_tol: f64,
    /// Whether full NDDO multipole integrals are evaluated (default: true)
    pub use_nddo: bool,
    /// Whether to refine the saddle estimate via eigenvector following (default: true)
    pub refine_with_ts: bool,
}

impl Default for SaddleOptions {
    fn default() -> Self {
        Self {
            max_cycles: 60,
            step_size: 0.10,
            convergence_distance: 0.12,
            inner_max_steps: 12,
            inner_grad_tol: 3.0,
            use_nddo: true,
            refine_with_ts: true,
        }
    }
}

/// A discrete point along the two-ended reaction barrier ascent.
#[derive(Debug, Clone)]
pub struct SaddlePoint {
    /// Macro cycle iteration index
    pub cycle: usize,
    /// Which structure was relaxed (1 = reactant side, 2 = product side)
    pub moving_structure: usize,
    /// Inter-structure distance in Angstroms
    pub distance: f64,
    /// Standard heat of formation in kcal/mol
    pub heat_of_formation_kcal: f64,
    /// Electronic total energy in eV
    pub total_energy_ev: f64,
    /// RMS gradient norm in kcal/(mol * A)
    pub grad_rms: f64,
    /// Cartesian coordinates
    pub coordinates: Vec<[f64; 3]>,
}

/// Comprehensive outcome of two-ended SADDLE transition state search.
#[derive(Debug, Clone)]
pub struct SaddleResult {
    /// Whether the two endpoints converged within the distance threshold
    pub converged: bool,
    /// Number of completed SADDLE macro cycles
    pub num_cycles: usize,
    /// Final distance between endpoint structures in Angstroms
    pub final_distance: f64,
    /// Estimated or refined transition state coordinates
    pub transition_state_coordinates: Vec<[f64; 3]>,
    /// Transition state total energy in eV
    pub transition_state_energy_ev: f64,
    /// Standard heat of formation at transition state in kcal/mol
    pub transition_state_heat_of_formation_kcal: f64,
    /// Forward reaction barrier height in kcal/mol (TS - Reactant)
    pub barrier_forward_kcal: f64,
    /// Reverse reaction barrier height in kcal/mol (TS - Product)
    pub barrier_reverse_kcal: f64,
    /// Trajectory of SADDLE relaxation points
    pub trajectory: Vec<SaddlePoint>,
    /// Optional eigenvector-following refinement result
    pub ts_refinement: Option<TransitionStateResult>,
}

/// Run two-ended transition state location between reactant and product geometries.
///
/// Port of canonical OpenMOPAC `react1.F90`.
#[allow(clippy::needless_range_loop)]
pub fn run_saddle(
    atomic_numbers: Vec<u8>,
    reactant_coords: &[[f64; 3]],
    product_coords: &[[f64; 3]],
    model: &dyn ParameterModel,
    options: &SaddleOptions,
) -> SaddleResult {
    let natoms = atomic_numbers.len();
    assert_eq!(reactant_coords.len(), natoms);
    assert_eq!(product_coords.len(), natoms);

    // 1. Initial Docking: align product onto reactant frame
    let (mut geo_b, mut dist) = dock(reactant_coords, product_coords);
    let mut geo_a = reactant_coords.to_vec();

    let mut batch_a = MolecularBatch::new_for_model(atomic_numbers.clone(), &geo_a, model);
    let mut batch_b = MolecularBatch::new_for_model(atomic_numbers.clone(), &geo_b, model);

    let mut scf_ws_a = ScfWorkspace::allocate(batch_a.norbs);
    let mut scf_ws_b = ScfWorkspace::allocate(batch_b.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch_a.norbs);

    // 2. Initial state evaluations
    let scf_a = run_rhf_scf_adaptive_with_nddo(
        &batch_a,
        model,
        &mut scf_ws_a,
        80,
        1e-9,
        1e-8,
        options.use_nddo,
    );
    let scf_b = run_rhf_scf_adaptive_with_nddo(
        &batch_b,
        model,
        &mut scf_ws_b,
        80,
        1e-9,
        1e-8,
        options.use_nddo,
    );

    let hof_a_init =
        compute_heat_of_formation(scf_a.total_energy_ev, &atomic_numbers, model, 0.0).1;
    let hof_b_init =
        compute_heat_of_formation(scf_b.total_energy_ev, &atomic_numbers, model, 0.0).1;

    let mut energy_a = scf_a.total_energy_ev;
    let mut energy_b = scf_b.total_energy_ev;

    // Moving structure: 1 = side A moves towards B, 2 = side B moves towards A
    let mut moving_flag = if hof_a_init <= hof_b_init { 1 } else { 2 };
    let mut consecutive_steps = 0;
    let mut prev_energy = if moving_flag == 1 { energy_a } else { energy_b };

    let mut trajectory = Vec::new();
    let mut converged = false;
    let mut last_grads_a = vec![[0.0; 3]; natoms];
    let mut last_grads_b = vec![[0.0; 3]; natoms];

    // Compute initial gradients
    compute_cartesian_gradients_with_options(
        &mut batch_a,
        model,
        &scf_ws_a.density,
        &mut grad_ws,
        &mut last_grads_a,
        options.use_nddo,
    );
    compute_cartesian_gradients_with_options(
        &mut batch_b,
        model,
        &scf_ws_b.density,
        &mut grad_ws,
        &mut last_grads_b,
        options.use_nddo,
    );

    for cycle in 1..=options.max_cycles {
        if dist <= options.convergence_distance {
            converged = true;
            break;
        }

        let target_dist = (dist - options.step_size).max(options.convergence_distance);

        // Perform stepping and hyperspherical relaxation on moving geometry
        let (cur_energy, cur_hof, cur_grad_rms) = if moving_flag == 1 {
            // Geometry A moves towards Geometry B
            let mut diff = vec![[0.0f64; 3]; natoms];
            for i in 0..natoms {
                for c in 0..3 {
                    diff[i][c] = geo_a[i][c] - geo_b[i][c];
                }
            }
            let cur_d = diff
                .iter()
                .flat_map(|v| v.iter())
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();

            if cur_d > 1e-8 {
                let scale = target_dist / cur_d;
                for i in 0..natoms {
                    for c in 0..3 {
                        geo_a[i][c] = geo_b[i][c] + diff[i][c] * scale;
                    }
                }
            }

            // Inner relaxation on hypersphere centered at geo_b
            let mut e_final = energy_a;
            let mut g_norm_final = 0.0;
            for _step in 0..options.inner_max_steps {
                for i in 0..natoms {
                    batch_a.x[i] = geo_a[i][0];
                    batch_a.y[i] = geo_a[i][1];
                    batch_a.z[i] = geo_a[i][2];
                }
                scf_ws_a.reset();
                let scf = run_rhf_scf_adaptive_with_nddo(
                    &batch_a,
                    model,
                    &mut scf_ws_a,
                    50,
                    1e-7,
                    1e-6,
                    options.use_nddo,
                );
                e_final = scf.total_energy_ev;
                compute_cartesian_gradients_with_options(
                    &mut batch_a,
                    model,
                    &scf_ws_a.density,
                    &mut grad_ws,
                    &mut last_grads_a,
                    options.use_nddo,
                );

                // Reaction vector unit vector u = (geo_a - geo_b) / target_dist
                let mut u = vec![[0.0f64; 3]; natoms];
                for i in 0..natoms {
                    for c in 0..3 {
                        u[i][c] = (geo_a[i][c] - geo_b[i][c]) / target_dist;
                    }
                }

                // Project gradient perpendicular to u: g_perp = g - (g . u) u
                let mut g_dot_u = 0.0f64;
                for i in 0..natoms {
                    for c in 0..3 {
                        g_dot_u += last_grads_a[i][c] * u[i][c];
                    }
                }

                let mut g_perp_sq = 0.0f64;
                let mut g_perp = vec![[0.0f64; 3]; natoms];
                for i in 0..natoms {
                    for c in 0..3 {
                        let val = (last_grads_a[i][c] - g_dot_u * u[i][c]) * EV_TO_KCAL_MOL;
                        g_perp[i][c] = val;
                        g_perp_sq += val * val;
                    }
                }
                let g_perp_norm = (g_perp_sq / (3.0 * natoms as f64)).sqrt();
                g_norm_final = g_perp_norm;

                if g_perp_norm < options.inner_grad_tol {
                    break;
                }

                // Step along -g_perp with adaptive step
                let alpha = (0.02 / (g_perp_norm + 1.0)).min(0.01);
                for i in 0..natoms {
                    for c in 0..3 {
                        geo_a[i][c] -= alpha * g_perp[i][c];
                    }
                }

                // Reproject onto sphere of radius target_dist
                let cur_len = geo_a
                    .iter()
                    .zip(geo_b.iter())
                    .flat_map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]])
                    .map(|x| x * x)
                    .sum::<f64>()
                    .sqrt();
                if cur_len > 1e-8 {
                    let fac = target_dist / cur_len;
                    for i in 0..natoms {
                        for c in 0..3 {
                            geo_a[i][c] = geo_b[i][c] + (geo_a[i][c] - geo_b[i][c]) * fac;
                        }
                    }
                }
            }

            let hof = compute_heat_of_formation(e_final, &atomic_numbers, model, 0.0).1;
            energy_a = e_final;
            (e_final, hof, g_norm_final)
        } else {
            // Geometry B moves towards Geometry A
            let mut diff = vec![[0.0f64; 3]; natoms];
            for i in 0..natoms {
                for c in 0..3 {
                    diff[i][c] = geo_b[i][c] - geo_a[i][c];
                }
            }
            let cur_d = diff
                .iter()
                .flat_map(|v| v.iter())
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();

            if cur_d > 1e-8 {
                let scale = target_dist / cur_d;
                for i in 0..natoms {
                    for c in 0..3 {
                        geo_b[i][c] = geo_a[i][c] + diff[i][c] * scale;
                    }
                }
            }

            // Inner relaxation on hypersphere centered at geo_a
            let mut e_final = energy_b;
            let mut g_norm_final = 0.0;
            for _step in 0..options.inner_max_steps {
                for i in 0..natoms {
                    batch_b.x[i] = geo_b[i][0];
                    batch_b.y[i] = geo_b[i][1];
                    batch_b.z[i] = geo_b[i][2];
                }
                scf_ws_b.reset();
                let scf = run_rhf_scf_adaptive_with_nddo(
                    &batch_b,
                    model,
                    &mut scf_ws_b,
                    50,
                    1e-7,
                    1e-6,
                    options.use_nddo,
                );
                e_final = scf.total_energy_ev;
                compute_cartesian_gradients_with_options(
                    &mut batch_b,
                    model,
                    &scf_ws_b.density,
                    &mut grad_ws,
                    &mut last_grads_b,
                    options.use_nddo,
                );

                // Reaction vector unit vector u = (geo_b - geo_a) / target_dist
                let mut u = vec![[0.0f64; 3]; natoms];
                for i in 0..natoms {
                    for c in 0..3 {
                        u[i][c] = (geo_b[i][c] - geo_a[i][c]) / target_dist;
                    }
                }

                // Project gradient perpendicular to u
                let mut g_dot_u = 0.0f64;
                for i in 0..natoms {
                    for c in 0..3 {
                        g_dot_u += last_grads_b[i][c] * u[i][c];
                    }
                }

                let mut g_perp_sq = 0.0f64;
                let mut g_perp = vec![[0.0f64; 3]; natoms];
                for i in 0..natoms {
                    for c in 0..3 {
                        let val = (last_grads_b[i][c] - g_dot_u * u[i][c]) * EV_TO_KCAL_MOL;
                        g_perp[i][c] = val;
                        g_perp_sq += val * val;
                    }
                }
                let g_perp_norm = (g_perp_sq / (3.0 * natoms as f64)).sqrt();
                g_norm_final = g_perp_norm;

                if g_perp_norm < options.inner_grad_tol {
                    break;
                }

                let alpha = (0.02 / (g_perp_norm + 1.0)).min(0.01);
                for i in 0..natoms {
                    for c in 0..3 {
                        geo_b[i][c] -= alpha * g_perp[i][c];
                    }
                }

                // Reproject onto sphere
                let cur_len = geo_b
                    .iter()
                    .zip(geo_a.iter())
                    .flat_map(|(b, a)| [b[0] - a[0], b[1] - a[1], b[2] - a[2]])
                    .map(|x| x * x)
                    .sum::<f64>()
                    .sqrt();
                if cur_len > 1e-8 {
                    let fac = target_dist / cur_len;
                    for i in 0..natoms {
                        for c in 0..3 {
                            geo_b[i][c] = geo_a[i][c] + (geo_b[i][c] - geo_a[i][c]) * fac;
                        }
                    }
                }
            }

            let hof = compute_heat_of_formation(e_final, &atomic_numbers, model, 0.0).1;
            energy_b = e_final;
            (e_final, hof, g_norm_final)
        };

        // Recalculate distance between A and B
        dist = geo_a
            .iter()
            .zip(geo_b.iter())
            .flat_map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]])
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        trajectory.push(SaddlePoint {
            cycle,
            moving_structure: moving_flag,
            distance: dist,
            heat_of_formation_kcal: cur_hof,
            total_energy_ev: cur_energy,
            grad_rms: cur_grad_rms,
            coordinates: if moving_flag == 1 {
                geo_a.clone()
            } else {
                geo_b.clone()
            },
        });

        // Swap condition (OpenMOPAC react1.F90):
        // If consecutive steps >= 3 or if energy increased on this side
        consecutive_steps += 1;
        if consecutive_steps >= 3 || cur_energy > prev_energy {
            moving_flag = 3 - moving_flag; // Toggle 1 <-> 2
            consecutive_steps = 0;
        }
        prev_energy = cur_energy;
    }

    // 3. Interpolate candidate transition state
    let norm_a = last_grads_a
        .iter()
        .flat_map(|v| v.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    let norm_b = last_grads_b
        .iter()
        .flat_map(|v| v.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    let total_norm = norm_a + norm_b;
    let c1 = if total_norm > 1e-6 {
        norm_b / total_norm
    } else {
        0.5
    };
    let c2 = 1.0 - c1;

    let mut ts_coords = vec![[0.0f64; 3]; natoms];
    for i in 0..natoms {
        for c in 0..3 {
            ts_coords[i][c] = c1 * geo_a[i][c] + c2 * geo_b[i][c];
        }
    }

    // Single point evaluation at interpolated TS
    let mut ts_batch = MolecularBatch::new_for_model(atomic_numbers.clone(), &ts_coords, model);
    let mut ts_scf_ws = ScfWorkspace::allocate(ts_batch.norbs);
    let ts_scf = run_rhf_scf_adaptive_with_nddo(
        &ts_batch,
        model,
        &mut ts_scf_ws,
        100,
        1e-9,
        1e-8,
        options.use_nddo,
    );
    let mut ts_energy = ts_scf.total_energy_ev;
    let mut ts_hof =
        compute_heat_of_formation(ts_scf.total_energy_ev, &atomic_numbers, model, 0.0).1;

    // 4. Optional EF Transition State Refinement
    let mut ts_refinement = None;
    if options.refine_with_ts {
        let ts_options = TransitionStateOptions {
            max_cycles: 60,
            grad_rms_tol: 0.5,
            grad_max_tol: 1.0,
            trust_radius: 0.08,
            use_nddo: options.use_nddo,
            ..Default::default()
        };

        let mut ef_ws = EigenvectorFollowingWorkspace::allocate(natoms);
        let ref_result = optimize_transition_state(
            &mut ts_batch,
            model,
            &mut ts_scf_ws,
            &mut grad_ws,
            &mut ef_ws,
            &ts_options,
        );

        if ref_result.converged {
            for i in 0..natoms {
                ts_coords[i][0] = ts_batch.x[i];
                ts_coords[i][1] = ts_batch.y[i];
                ts_coords[i][2] = ts_batch.z[i];
            }
            ts_energy = ref_result.final_energy_ev;
            ts_hof = ref_result.heat_of_formation_kcal;
        }
        ts_refinement = Some(ref_result);
    }

    let barrier_fwd = ts_hof - hof_a_init;
    let barrier_rev = ts_hof - hof_b_init;

    SaddleResult {
        converged,
        num_cycles: trajectory.len(),
        final_distance: dist,
        transition_state_coordinates: ts_coords,
        transition_state_energy_ev: ts_energy,
        transition_state_heat_of_formation_kcal: ts_hof,
        barrier_forward_kcal: barrier_fwd,
        barrier_reverse_kcal: barrier_rev,
        trajectory,
        ts_refinement,
    }
}
