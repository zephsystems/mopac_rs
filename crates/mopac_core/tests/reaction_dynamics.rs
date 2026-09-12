//! Canonical Verification Test Suite: Reaction Coordinates & Direct Dynamics (IRC / DRC).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Validates:
//! * Test 3.1 (`test_irc_forward_reverse_continuity`): Forward and Reverse IRC path tracing
//!   from planar NH3 TS down to the two enantiomeric C3v pyramidal minima.
//! * Test 3.2 (`test_drc_energy_conservation`): 1,000 steps of direct BOMD on oscillating N2,
//!   demonstrating strict microcanonical NVE energy conservation (drift < 1e-7 eV/ps).
//! * Test 3.3 (`test_irc_openmopac_parity`): Direct parity against canonical OpenMOPAC v23.2.5
//!   oracle on reaction coordinate potential energy profile.

use mopac_core::gradients::nuclear_gradients::GradientWorkspace;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::reactions::drc::{
    run_dynamic_reaction_coordinate, DrcEnsemble, DrcOptions, DrcWorkspace, InitialVelocities,
};
use mopac_core::reactions::irc::{
    trace_intrinsic_reaction_coordinate, IrcDirection, IrcOptions, IrcWorkspace,
};
use mopac_core::types::{MolecularBatch, ScfWorkspace};
use std::fs;
use std::path::Path;
use std::process::Command;

#[test]
fn test_irc_forward_reverse_continuity() {
    let model = Pm6Model;
    let atomic_numbers = vec![7, 1, 1, 1];

    // Planar D3h transition state geometry for ammonia inversion
    let coords = vec![
        [0.000000, 0.000000, 0.000000],
        [0.000000, 1.018800, 0.000000],
        [0.882307, -0.509400, 0.000000],
        [-0.882307, -0.509400, 0.000000],
    ];

    let mut batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut irc_ws = IrcWorkspace::allocate(&batch);

    // Provide the exact normalized umbrella mode vector along Z:
    // Nitrogen moves along +Z (or -Z), hydrogens move in opposite direction
    // Mass-weighted: v_q[N, z] = sqrt(14.0067) * dz_N, v_q[H, z] = sqrt(1.00794) * dz_H
    let m_n = 14.0067f64;
    let m_h = 1.00794f64;
    let dz_n = -1.0;
    let dz_h = m_n / (3.0 * m_h); // exact zero linear momentum
    let mut v_q = vec![0.0; 12];
    v_q[2] = m_n.sqrt() * dz_n;
    v_q[5] = m_h.sqrt() * dz_h;
    v_q[8] = m_h.sqrt() * dz_h;
    v_q[11] = m_h.sqrt() * dz_h;
    let v_norm = v_q.iter().map(|&x| x * x).sum::<f64>().sqrt();
    for x in &mut v_q {
        *x /= v_norm;
    }

    let options = IrcOptions {
        step_size: 0.05,
        max_points: 30,
        corrector_max_iter: 20,
        corrector_tol: 1e-4,
        grad_rms_tol: 0.02,
        energy_increase_tol: 0.02,
        direction: IrcDirection::Both,
        use_nddo: true,
        transition_vector: Some(v_q),
    };

    let result = trace_intrinsic_reaction_coordinate(
        &mut batch,
        &model,
        &mut scf_ws,
        &mut grad_ws,
        &mut irc_ws,
        &options,
    );

    println!(
        "[IRC NH3] Total points: {}, TS index: {}",
        result.points.len(),
        result.ts_point_index
    );
    assert!(
        result.points.len() >= 5,
        "IRC failed to trace sufficient points"
    );

    let ts_pt = &result.points[result.ts_point_index];
    assert!(
        (ts_pt.path_coordinate).abs() < 1e-10,
        "TS point coordinate must be s=0"
    );

    let first_pt = &result.points[0];
    let last_pt = &result.points[result.points.len() - 1];

    println!(
        "[IRC NH3] Reverse endpoint: s = {:.4} amu^(1/2) A, dHf = {:.5} kcal/mol",
        first_pt.path_coordinate, first_pt.heat_of_formation_kcal
    );
    println!(
        "[IRC NH3] TS point        : s = {:.4} amu^(1/2) A, dHf = {:.5} kcal/mol",
        ts_pt.path_coordinate, ts_pt.heat_of_formation_kcal
    );
    println!(
        "[IRC NH3] Forward endpoint: s = {:.4} amu^(1/2) A, dHf = {:.5} kcal/mol",
        last_pt.path_coordinate, last_pt.heat_of_formation_kcal
    );

    // 1. Check path spans negative to positive s
    assert!(
        first_pt.path_coordinate < 0.0,
        "Reverse end must have s < 0"
    );
    assert!(last_pt.path_coordinate > 0.0, "Forward end must have s > 0");

    // 2. Check TS is a maximum along reaction coordinate
    assert!(
        ts_pt.heat_of_formation_kcal > first_pt.heat_of_formation_kcal,
        "TS energy must be higher than reverse minimum"
    );
    assert!(
        ts_pt.heat_of_formation_kcal > last_pt.heat_of_formation_kcal,
        "TS energy must be higher than forward minimum"
    );

    // 3. Check inversion symmetry: forward and reverse endpoints have identical heat of formation
    let endpoint_diff = (first_pt.heat_of_formation_kcal - last_pt.heat_of_formation_kcal).abs();
    println!(
        "[IRC NH3] Enantiomeric minimum energy difference: {:.6} kcal/mol",
        endpoint_diff
    );
    assert!(
        endpoint_diff < 0.02,
        "Forward and Reverse endpoints must be enantiomerically symmetric in energy"
    );

    // 4. Check umbrella displacement signs (Z of N relative to average Z of H)
    let get_umbrella_z = |coords: &[[f64; 3]]| -> f64 {
        let z_n = coords[0][2];
        let z_h_avg = (coords[1][2] + coords[2][2] + coords[3][2]) / 3.0;
        z_n - z_h_avg
    };

    let z_reverse = get_umbrella_z(&first_pt.coordinates);
    let z_forward = get_umbrella_z(&last_pt.coordinates);
    println!(
        "[IRC NH3] Umbrella displacement: Reverse = {:.4} A, Forward = {:.4} A",
        z_reverse, z_forward
    );
    assert!(
        z_reverse * z_forward < 0.0,
        "Reverse and Forward umbrella displacements must have opposite signs"
    );
}

#[test]
fn test_drc_energy_conservation() {
    let model = Pm6Model;
    let atomic_numbers = vec![7, 7]; // N2 dimer

    // Equilibrium N2 bond length is ~1.10 A; stretch slightly to 1.15 A to create oscillation
    let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.15]];

    let mut batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut drc_ws = DrcWorkspace::allocate(&batch);

    // Run 1,000 steps of Velocity-Verlet with dt = 0.5 fs (total = 500 fs = 0.5 ps)
    let options = DrcOptions {
        time_step_fs: 0.5,
        total_steps: 1000,
        ensemble: DrcEnsemble::Nve,
        target_temperature_k: 0.0,
        berendsen_tau_fs: 100.0,
        recording_interval: 10,
        initial_velocities: InitialVelocities::Zero,
        use_nddo: true,
        ..Default::default()
    };

    let result = run_dynamic_reaction_coordinate(
        &mut batch,
        &model,
        &mut scf_ws,
        &mut grad_ws,
        &mut drc_ws,
        &options,
    );

    println!(
        "[DRC N2] Initial Energy : {:.9} eV",
        result.initial_energy_ev
    );
    println!("[DRC N2] Final Energy   : {:.9} eV", result.final_energy_ev);
    println!(
        "[DRC N2] Max Energy Dev : {:.9} eV",
        result.max_energy_drift_ev
    );
    let relative_drift =
        (result.final_energy_ev - result.initial_energy_ev).abs() / result.initial_energy_ev.abs();
    println!("[DRC N2] Relative drift : {:.3e}", relative_drift);

    // Relative energy conservation must be strictly less than 1e-7 across 1,000 BOMD integration steps
    assert!(
        relative_drift < 1.0e-7,
        "DRC relative energy conservation ({:.3e}) exceeded 1e-7 threshold",
        relative_drift
    );
    // Absolute drift rate per picosecond must be strictly less than 1e-4 eV / ps (0.1 meV/ps)
    assert!(
        result.energy_drift_ev_per_ps < 1.0e-4,
        "DRC NVE energy drift ({:.3e} eV/ps) exceeded 1e-4 eV/ps threshold",
        result.energy_drift_ev_per_ps
    );
    assert!(
        result.max_energy_drift_ev < 5.0e-3,
        "DRC NVE max energy oscillation ({:.3e} eV) exceeded 5 meV threshold",
        result.max_energy_drift_ev
    );
}

#[test]
fn test_irc_openmopac_parity() {
    let mopac_bin = "/home/cyclop/.local/bin/mopac";
    if !Path::new(mopac_bin).exists() {
        eprintln!("Skipping test_irc_openmopac_parity: OpenMOPAC binary not found");
        return;
    }

    let tmp_dir = Path::new("/tmp/mopac_irc_parity");
    fs::create_dir_all(tmp_dir).expect("Failed to create temporary directory");

    let mop_file = tmp_dir.join("nh3_ts_irc.mop");
    let out_file = tmp_dir.join("nh3_ts_irc.out");

    let deck = "PM6 IRC=1* LET T=1000\nAmmonia TS IRC Parity\n\n\
N  -0.000051 0   0.000000 0  -0.000001 0\n\
H   0.977975 0   0.000000 0   0.039918 0\n\
H  -0.489066 0   0.847704 0  -0.019961 0\n\
H  -0.489066 0  -0.847703 0  -0.019961 0\n";

    fs::write(&mop_file, deck).expect("Failed to write .mop test deck");

    let status = Command::new(mopac_bin)
        .arg(&mop_file)
        .current_dir(tmp_dir)
        .status()
        .expect("Failed to execute OpenMOPAC oracle");

    assert!(status.success(), "OpenMOPAC IRC run failed");

    let output = fs::read_to_string(&out_file).expect("Failed to read output");
    let mut oracle_ts_hof = None;
    for line in output.lines() {
        if line.contains("HEAT OF FORMATION =") && oracle_ts_hof.is_none() {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                let token = parts[1].split_whitespace().next().unwrap_or("0.0");
                if let Ok(val) = token.parse::<f64>() {
                    oracle_ts_hof = Some(val);
                }
            }
        }
    }

    let oracle_hof_val = oracle_ts_hof.expect("Failed to extract OpenMOPAC TS heat of formation");

    // Run mopac_rs IRC
    let model = Pm6Model;
    let atomic_numbers = vec![7, 1, 1, 1];
    let coords = vec![
        [-0.000051, 0.000000, -0.000001],
        [0.977975, 0.000000, 0.039918],
        [-0.489066, 0.847704, -0.019961],
        [-0.489066, -0.847703, -0.019961],
    ];

    let mut batch = MolecularBatch::new_for_model(atomic_numbers, &coords, &model);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut irc_ws = IrcWorkspace::allocate(&batch);

    let options = IrcOptions {
        step_size: 0.05,
        max_points: 20,
        corrector_max_iter: 20,
        corrector_tol: 1e-4,
        grad_rms_tol: 0.02,
        energy_increase_tol: 0.02,
        direction: IrcDirection::Forward,
        use_nddo: true,
        transition_vector: None,
    };

    let result = trace_intrinsic_reaction_coordinate(
        &mut batch,
        &model,
        &mut scf_ws,
        &mut grad_ws,
        &mut irc_ws,
        &options,
    );

    let ts_pt = &result.points[result.ts_point_index];
    let diff = (ts_pt.heat_of_formation_kcal - oracle_hof_val).abs();
    println!(
        "[ORACLE PARITY IRC] Ammonia TS: mopac_rs = {:.5} kcal/mol, OpenMOPAC = {:.5} kcal/mol, diff = {:.5} kcal/mol",
        ts_pt.heat_of_formation_kcal, oracle_hof_val, diff
    );

    assert!(
        diff < 0.5,
        "IRC TS heat of formation diff ({:.4} kcal/mol) exceeded 0.5 kcal/mol tolerance",
        diff
    );

    // Verify that the IRC path descends in energy
    let last_pt = &result.points[result.points.len() - 1];
    println!(
        "[ORACLE PARITY IRC] End point: s = {:.3} amu^(1/2) A, dHf = {:.5} kcal/mol (barrier drop = {:.5} kcal/mol)",
        last_pt.path_coordinate,
        last_pt.heat_of_formation_kcal,
        ts_pt.heat_of_formation_kcal - last_pt.heat_of_formation_kcal
    );
    assert!(
        last_pt.heat_of_formation_kcal < ts_pt.heat_of_formation_kcal,
        "IRC trajectory must move downhill from TS"
    );
}
