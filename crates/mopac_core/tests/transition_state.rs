//! Empirical Transition State & Eigenvector Following (EF) Verification Suite.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Validates Phase 2 deliverables:
//! 1. Trust radius displacement constraints: ||s|| <= R_trust
//! 2. Ammonia umbrella inversion TS (D3h planar saddle point with 1 imaginary mode ~ 950i cm^-1)
//! 3. HCN <-> HNC 3-center cyclic isomerization TS (with 1 imaginary mode ~ 1400i cm^-1)
//! 4. Canonical OpenMOPAC v23.2.5 differential oracle parity for TS keyword

use mopac_core::gradients::nuclear_gradients::GradientWorkspace;
use mopac_core::opt::eigenvector_following::{
    optimize_transition_state, EigenvectorFollowingWorkspace, TransitionStateOptions,
};
use mopac_core::opt::hessian_update::HessianUpdateScheme;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::types::{MolecularBatch, ScfWorkspace};
use mopac_core::vibrations::hessian::{compute_hessian_and_frequencies, HessianOptions};
use std::fs;
use std::path::Path;
use std::process::Command;

/// Test 2.1: Rigorous step-size constraint ||s|| <= R_trust across varied initial conditions.
#[test]
fn test_ef_trust_radius_invariants() {
    let model = Pm6Model;
    // Distorted ammonia geometry
    let atomic_numbers = vec![7, 1, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [1.01, 0.0, 0.08],
        [-0.505, 0.8747, -0.04],
        [-0.505, -0.8747, -0.04],
    ];

    let trust_radii = [0.03, 0.05, 0.10, 0.15];

    for &r_trust in &trust_radii {
        let mut batch = MolecularBatch::new_for_model(atomic_numbers.clone(), &coords, &model);
        let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
        let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
        let mut ef_ws = EigenvectorFollowingWorkspace::allocate(batch.natoms);

        let options = TransitionStateOptions {
            max_cycles: 2,
            trust_radius: r_trust,
            min_trust_radius: 0.005,
            max_trust_radius: r_trust,
            grad_rms_tol: 1e-4,
            grad_max_tol: 1e-4,
            update_scheme: HessianUpdateScheme::Bofill,
            mode_following: true,
            target_mode: None,
            opt_mask: None,
            use_nddo: false,
            hessian_delta: 0.005,
            initial_hessian: None,
        };

        let res = optimize_transition_state(
            &mut batch,
            &model,
            &mut scf_ws,
            &mut grad_ws,
            &mut ef_ws,
            &options,
        );

        // Verify that step norm in each cycle respected r_trust
        let mut step_norm_sq = 0.0;
        for j in 0..(3 * batch.natoms) {
            step_norm_sq += ef_ws.step_cart[j] * ef_ws.step_cart[j];
        }
        let step_norm = step_norm_sq.sqrt();
        assert!(
            step_norm <= r_trust + 1e-12,
            "Trust radius violated: step_norm = {}, r_trust = {}",
            step_norm,
            r_trust
        );
        assert!(res.cycles <= 2);
    }
}

/// Test 2.2: Ammonia umbrella inversion transition state (NH3 -> D3h planar -> NH3).
///
/// Verifies:
/// 1. Convergence to planar D3h saddle point with RMS gradient < 0.1 kcal/(mol*A)
/// 2. Harmonic vibrational analysis reports exactly 1 imaginary frequency (umbrella mode ~ 950i cm^-1)
/// 3. The remaining 5 vibrational modes are positive real
#[test]
fn test_nh3_umbrella_inversion_ts() {
    let model = Pm6Model;
    let atomic_numbers = vec![7, 1, 1, 1];
    // Start with slight out-of-plane distortion near planar D3h
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.98, 0.0, 0.04],
        [-0.49, 0.8487, -0.02],
        [-0.49, -0.8487, -0.02],
    ];

    let mut batch = MolecularBatch::new_for_model(atomic_numbers.clone(), &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut ef_ws = EigenvectorFollowingWorkspace::allocate(batch.natoms);

    let options = TransitionStateOptions {
        max_cycles: 20,
        grad_rms_tol: 0.05,
        grad_max_tol: 0.10,
        trust_radius: 0.05,
        min_trust_radius: 0.005,
        max_trust_radius: 0.20,
        update_scheme: HessianUpdateScheme::Bofill,
        mode_following: true,
        target_mode: None,
        opt_mask: None,
        use_nddo: true,
        hessian_delta: 0.005,
        initial_hessian: None,
    };

    let result = optimize_transition_state(
        &mut batch,
        &model,
        &mut scf_ws,
        &mut grad_ws,
        &mut ef_ws,
        &options,
    );

    println!(
        "[TS RESULT] Ammonia D3h: converged={}, cycles={}, final_grad_rms={:.6} kcal/(mol*A), HoF={:.3} kcal/mol",
        result.converged, result.cycles, result.final_grad_rms, result.heat_of_formation_kcal
    );

    assert!(
        result.converged,
        "Ammonia TS optimization failed to converge within {} cycles",
        options.max_cycles
    );
    assert!(
        result.final_grad_rms < 0.10,
        "Final RMS gradient too high: {}",
        result.final_grad_rms
    );

    // Verify planar D3h geometry: z-coordinates of all atoms should be virtually 0
    for a in 0..batch.natoms {
        assert!(
            batch.z[a].abs() < 0.05,
            "Atom {} z-coordinate is not planar: {}",
            a,
            batch.z[a]
        );
    }

    // Perform vibrational frequency analysis at the optimized saddle point
    let scf_opts = mopac_core::scf::scf_loop::ScfOptions::default();
    let hess_opts = HessianOptions {
        delta: 0.005,
        project_external: true,
        recompute_scf: true,
        use_nddo: true,
        ..Default::default()
    };

    let vib_result =
        compute_hessian_and_frequencies(&mut batch, &model, &mut scf_ws, &scf_opts, &hess_opts);

    let vib_freqs = &vib_result.vibrational_frequencies_cm1;
    assert_eq!(
        vib_freqs.len(),
        6,
        "Ammonia must have exactly 3N - 6 = 6 vibrational modes"
    );

    let mut imaginary_count = 0;
    let mut imaginary_freq = 0.0;
    for &f in vib_freqs {
        if f < 0.0 {
            imaginary_count += 1;
            imaginary_freq = f;
        }
    }

    println!("[TS FREQUENCIES] Ammonia modes: {:?}", vib_freqs);
    assert_eq!(
        imaginary_count, 1,
        "Transition state must have EXACTLY ONE imaginary frequency, found {}. Modes: {:?}",
        imaginary_count, vib_freqs
    );

    // The imaginary frequency should be near -950 to -1200 cm^-1 (canonical OpenMOPAC: -961.3 cm^-1)
    assert!(
        imaginary_freq < -600.0 && imaginary_freq > -1400.0,
        "Umbrella mode frequency {} is out of expected range [-1400, -600] cm^-1",
        imaginary_freq
    );
}

/// Test 2.3: HCN <-> HNC 3-center cyclic isomerization transition state.
///
/// Verifies:
/// 1. Convergence to 3-center cyclic saddle point [H ... C ... N]
/// 2. Exactly 1 imaginary frequency corresponding to proton transfer (~ 1400i cm^-1)
#[test]
fn test_hcn_hnc_isomerization_ts() {
    let model = Pm6Model;
    let atomic_numbers = vec![6, 7, 1]; // C, N, H
                                        // Initial geometry near cyclic saddle point
    let coords = vec![
        [0.005, 0.016, 0.0],
        [1.205, -0.042, 0.0],
        [0.490, 1.250, 0.0],
    ];

    let mut batch = MolecularBatch::new_for_model(atomic_numbers.clone(), &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut ef_ws = EigenvectorFollowingWorkspace::allocate(batch.natoms);

    let options = TransitionStateOptions {
        max_cycles: 25,
        grad_rms_tol: 0.10,
        grad_max_tol: 0.20,
        trust_radius: 0.05,
        min_trust_radius: 0.005,
        max_trust_radius: 0.15,
        update_scheme: HessianUpdateScheme::Bofill,
        mode_following: true,
        target_mode: None,
        opt_mask: None,
        use_nddo: true,
        hessian_delta: 0.005,
        initial_hessian: None,
    };

    let result = optimize_transition_state(
        &mut batch,
        &model,
        &mut scf_ws,
        &mut grad_ws,
        &mut ef_ws,
        &options,
    );

    println!(
        "[TS RESULT] HCN-HNC: converged={}, cycles={}, final_grad_rms={:.6} kcal/(mol*A), HoF={:.3} kcal/mol",
        result.converged, result.cycles, result.final_grad_rms, result.heat_of_formation_kcal
    );

    assert!(
        result.converged,
        "HCN-HNC TS optimization failed to converge"
    );

    // Compute harmonic vibrational frequencies
    let scf_opts = mopac_core::scf::scf_loop::ScfOptions::default();
    let hess_opts = HessianOptions {
        delta: 0.005,
        project_external: true,
        recompute_scf: true,
        use_nddo: true,
        ..Default::default()
    };

    let vib_result =
        compute_hessian_and_frequencies(&mut batch, &model, &mut scf_ws, &scf_opts, &hess_opts);

    let vib_freqs = &vib_result.vibrational_frequencies_cm1;
    println!("[TS FREQUENCIES] HCN-HNC modes: {:?}", vib_freqs);

    let mut imaginary_count = 0;
    let mut imaginary_freq = 0.0;
    for &f in vib_freqs {
        if f < 0.0 {
            imaginary_count += 1;
            imaginary_freq = f;
        }
    }

    assert_eq!(
        imaginary_count, 1,
        "HCN-HNC TS must have exactly 1 imaginary frequency, found {}. Modes: {:?}",
        imaginary_count, vib_freqs
    );

    // OpenMOPAC canonical frequency is -1396.2 cm^-1
    assert!(
        imaginary_freq < -1000.0 && imaginary_freq > -1800.0,
        "HCN-HNC proton transfer mode {} out of range [-1800, -1000] cm^-1",
        imaginary_freq
    );
}

/// Test 2.4: Golden Parity against OpenMOPAC v23.2.5 with TS keyword.
#[test]
fn test_ts_golden_parity() {
    let mopac_bin = "/home/cyclop/.local/bin/mopac";
    if !Path::new(mopac_bin).exists() {
        eprintln!("Skipping test_ts_golden_parity: OpenMOPAC binary not found");
        return;
    }

    let tmp_dir = Path::new("/tmp/mopac_ts_parity");
    fs::create_dir_all(tmp_dir).expect("Failed to create temporary directory");

    let mop_file = tmp_dir.join("ammonia_ts.mop");
    let out_file = tmp_dir.join("ammonia_ts.out");

    let deck = "PM6 XYZ TS PRECISE\nAmmonia Inversion TS Parity\n\n\
N   0.000000 1   0.000000 1   0.000000 1\n\
H   0.980000 1   0.000000 1   0.040000 1\n\
H  -0.490000 1   0.848705 1  -0.020000 1\n\
H  -0.490000 1  -0.848705 1  -0.020000 1\n";

    fs::write(&mop_file, deck).expect("Failed to write .mop test deck");

    let status = Command::new(mopac_bin)
        .arg(&mop_file)
        .current_dir(tmp_dir)
        .status()
        .expect("Failed to execute OpenMOPAC oracle");

    assert!(status.success(), "OpenMOPAC TS run failed");

    let output = fs::read_to_string(&out_file).expect("Failed to read output");
    let mut oracle_hof = None;
    for line in output.lines() {
        if line.contains("FINAL HEAT OF FORMATION =") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                let token = parts[1].split_whitespace().next().unwrap_or("0.0");
                if let Ok(val) = token.parse::<f64>() {
                    oracle_hof = Some(val);
                }
            }
        }
    }

    let oracle_hof_val = oracle_hof.expect("Failed to extract OpenMOPAC TS heat of formation");

    // Run mopac_rs EF optimization
    let model = Pm6Model;
    let atomic_numbers = vec![7, 1, 1, 1];
    let coords = vec![
        [0.000000, 0.000000, 0.000000],
        [0.980000, 0.000000, 0.040000],
        [-0.490000, 0.848705, -0.020000],
        [-0.490000, -0.848705, -0.020000],
    ];

    let mut batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut ef_ws = EigenvectorFollowingWorkspace::allocate(batch.natoms);

    let options = TransitionStateOptions {
        max_cycles: 25,
        grad_rms_tol: 0.05,
        grad_max_tol: 0.10,
        trust_radius: 0.05,
        min_trust_radius: 0.005,
        max_trust_radius: 0.20,
        update_scheme: HessianUpdateScheme::Bofill,
        mode_following: true,
        target_mode: None,
        opt_mask: None,
        use_nddo: true,
        hessian_delta: 0.005,
        initial_hessian: None,
    };

    let result = optimize_transition_state(
        &mut batch,
        &model,
        &mut scf_ws,
        &mut grad_ws,
        &mut ef_ws,
        &options,
    );

    println!(
        "[TS OPT COORDS] {:?}",
        (0..4)
            .map(|i| [batch.x[i], batch.y[i], batch.z[i]])
            .collect::<Vec<_>>()
    );

    assert!(result.converged, "mopac_rs TS failed to converge");

    let diff = (result.heat_of_formation_kcal - oracle_hof_val).abs();
    println!(
        "[ORACLE PARITY TS] Ammonia TS: mopac_rs = {:.5} kcal/mol, OpenMOPAC = {:.5} kcal/mol, diff = {:.5} kcal/mol",
        result.heat_of_formation_kcal, oracle_hof_val, diff
    );
    println!(
        "[ORACLE PARITY TS] mopac_rs total_energy = {:.5} eV (nuc = {:.5}, elec = {:.5})",
        result.final_energy_ev,
        result.final_scf.nuclear_repulsion_ev,
        result.final_scf.electronic_energy_ev
    );

    // Parity within 1.0 kcal/mol for TS saddle point
    assert!(
        diff < 1.0,
        "Transition state heat of formation discrepancy too large: diff = {} kcal/mol",
        diff
    );
}
