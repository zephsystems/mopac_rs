//! Integration tests for two-ended SADDLE reaction transition state locator.
//!
//! Direct validation of Horn's quaternion docking and two-ended hyperspherical
//! saddle point searches against canonical transition state chemistry.

use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::reactions::saddle::{dock, run_saddle, SaddleOptions};

#[test]
fn test_horn_dock_rigid_alignment() {
    // Target geometry: water molecule in x-y plane
    let target = vec![[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]];

    // Apply an exact rigid 3D rotation: 90 deg around z axis, then translate
    // R_z(90 deg): x' = -y, y' = x, z' = z
    let shift = [10.0, 5.0, -2.0];
    let mobile = vec![
        [shift[0] - 0.0, shift[1] + 0.0, shift[2] + 0.0],
        [shift[0] - 0.586, shift[1] + 0.757, shift[2] + 0.0],
        [shift[0] - 0.586, shift[1] - 0.757, shift[2] + 0.0],
    ];

    let (aligned, dist) = dock(&target, &mobile);

    // Distance between congruent shapes after optimal docking must be 0
    assert!(
        dist < 1e-10,
        "Optimal dock distance should be < 1e-10 for congruent geometries, got {}",
        dist
    );

    for i in 0..target.len() {
        for c in 0..3 {
            let diff = (aligned[i][c] - target[i][c]).abs();
            assert!(
                diff < 1e-10,
                "Aligned coordinate mismatch atom {} axis {}: diff = {}",
                i,
                c,
                diff
            );
        }
    }
}

#[test]
fn test_saddle_ammonia_umbrella_inversion() {
    let model = Pm6Model;
    let atomic_numbers = vec![7, 1, 1, 1]; // N, H, H, H

    // Reactant: NH3 pyramidal (umbrella down)
    let reactant_coords = vec![
        [0.0, 0.0, 0.0],
        [0.93813, 0.0, -0.36130],
        [-0.46907, 0.81244, -0.36130],
        [-0.46907, -0.81244, -0.36130],
    ];

    // Product: NH3 pyramidal (umbrella up)
    let product_coords = vec![
        [0.0, 0.0, 0.0],
        [0.93813, 0.0, 0.36130],
        [-0.46907, 0.81244, 0.36130],
        [-0.46907, -0.81244, 0.36130],
    ];

    let options = SaddleOptions {
        max_cycles: 30,
        step_size: 0.08,
        convergence_distance: 0.15,
        inner_max_steps: 8,
        inner_grad_tol: 2.0,
        use_nddo: true,
        refine_with_ts: true,
    };

    let result = run_saddle(
        atomic_numbers,
        &reactant_coords,
        &product_coords,
        &model,
        &options,
    );

    println!(
        "[SADDLE RESULT] Ammonia: converged={}, cycles={}, final_dist={:.4} A, TS HoF={:.3} kcal/mol, barrier_fwd={:.3} kcal/mol",
        result.converged, result.num_cycles, result.final_distance, result.transition_state_heat_of_formation_kcal, result.barrier_forward_kcal
    );
    for (i, coord) in result.transition_state_coordinates.iter().enumerate() {
        println!(
            "  Atom {}: [{:.4}, {:.4}, {:.4}]",
            i, coord[0], coord[1], coord[2]
        );
    }

    assert!(
        result.converged,
        "SADDLE failed to converge within {} cycles, final distance: {:.4} A",
        options.max_cycles, result.final_distance
    );

    assert!(
        result.final_distance <= options.convergence_distance,
        "Final distance {} exceeded convergence threshold {}",
        result.final_distance,
        options.convergence_distance
    );

    // Barrier height for ammonia inversion in PM6 is approximately 5.0 to 6.5 kcal/mol
    assert!(
        result.barrier_forward_kcal > 2.0 && result.barrier_forward_kcal < 12.0,
        "Forward barrier {:.3} kcal/mol outside physical range [2, 12]",
        result.barrier_forward_kcal
    );

    // Ammonia inversion is symmetric: forward and reverse barriers must be equal within ~ 0.5 kcal/mol
    let barrier_diff = (result.barrier_forward_kcal - result.barrier_reverse_kcal).abs();
    assert!(
        barrier_diff < 0.8,
        "Ammonia inversion should be symmetric, barrier difference: {:.3} kcal/mol",
        barrier_diff
    );

    // Verify planar D3h geometry at TS: distance from N to H1-H2-H3 plane is virtually zero
    let r0 = result.transition_state_coordinates[0]; // N
    let r1 = result.transition_state_coordinates[1]; // H1
    let r2 = result.transition_state_coordinates[2]; // H2
    let r3 = result.transition_state_coordinates[3]; // H3

    let u = [r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]];
    let v = [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2]];
    let normal = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let norm_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    let n_hat = [
        normal[0] / norm_len,
        normal[1] / norm_len,
        normal[2] / norm_len,
    ];

    let dist_n_to_plane =
        ((r0[0] - r1[0]) * n_hat[0] + (r0[1] - r1[1]) * n_hat[1] + (r0[2] - r1[2]) * n_hat[2])
            .abs();

    println!(
        "[TS GEOMETRY] Distance of Nitrogen to H-plane: {:.6} A",
        dist_n_to_plane
    );
    assert!(
        dist_n_to_plane < 0.01,
        "Transition state is not planar: N distance to H plane = {:.6} A",
        dist_n_to_plane
    );
}

#[test]
fn test_saddle_hcn_hnc_isomerization() {
    let model = Pm6Model;
    let atomic_numbers = vec![6, 7, 1]; // C, N, H

    // Reactant: H-C=N (with small off-axis angle to break collinear C_inf_v singularity)
    let reactant_coords = vec![[0.0, 0.0, 0.0], [1.155, 0.0, 0.0], [-1.060, 0.05, 0.0]];

    // Product: C=N-H (bent precursor)
    let product_coords = vec![[0.0, 0.0, 0.0], [1.168, 0.0, 0.0], [1.800, 0.50, 0.0]];

    let options = SaddleOptions {
        max_cycles: 50,
        step_size: 0.10,
        convergence_distance: 0.15,
        inner_max_steps: 12,
        inner_grad_tol: 3.0,
        use_nddo: true,
        refine_with_ts: true,
    };

    let result = run_saddle(
        atomic_numbers,
        &reactant_coords,
        &product_coords,
        &model,
        &options,
    );

    println!(
        "[SADDLE RESULT] HCN-HNC: converged={}, cycles={}, final_dist={:.4} A, TS HoF={:.3} kcal/mol, barrier_fwd={:.3} kcal/mol, barrier_rev={:.3} kcal/mol",
        result.converged, result.num_cycles, result.final_distance, result.transition_state_heat_of_formation_kcal, result.barrier_forward_kcal, result.barrier_reverse_kcal
    );

    assert!(
        result.converged,
        "HCN-HNC SADDLE failed to converge within {} cycles, final distance: {:.4} A",
        options.max_cycles, result.final_distance
    );

    // The PM6 semi-empirical barrier for HCN -> HNC is ~ 86 kcal/mol
    // (exact TS HoF is 119.051 kcal/mol, matching single-ended EF TS)
    assert!(
        result.barrier_forward_kcal > 70.0 && result.barrier_forward_kcal < 95.0,
        "Forward barrier {:.3} kcal/mol outside expected PM6 range [70, 95]",
        result.barrier_forward_kcal
    );

    assert!(
        (result.transition_state_heat_of_formation_kcal - 119.051).abs() < 0.1,
        "SADDLE TS HoF {:.3} kcal/mol differs from canonical TS benchmark 119.051 kcal/mol",
        result.transition_state_heat_of_formation_kcal
    );

    // HNC is higher in energy than HCN by ~ 34 kcal/mol in PM6, so reverse barrier is lower
    assert!(
        result.barrier_forward_kcal > result.barrier_reverse_kcal,
        "HCN -> HNC barrier ({:.3}) should be higher than reverse HNC -> HCN barrier ({:.3})",
        result.barrier_forward_kcal,
        result.barrier_reverse_kcal
    );
}
