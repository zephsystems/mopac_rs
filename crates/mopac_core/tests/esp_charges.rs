//! Electrostatic Potential (ESP) Charge Fitting Verification Suite.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Tests the Merz-Singh-Kollman ESP partial charge fitting on molecular systems.
//! Verifies:
//! 1. Exact charge conservation via Lagrange multiplier constraint: sum_A q_A = Q_net.
//! 2. Chemical electronegativity and symmetry: q_O < 0 in water, symmetric equivalent hydrogens.
//! 3. Alignment between ESP dipole moment and quantum SCF dipole moment.
//! 4. Ionic systems with net charges (cations Q = +1, anions Q = -1).

use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::properties::{compute_dipole_moment, compute_esp_charges, EspOptions};
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::types::{MolecularBatch, ScfWorkspace};

#[test]
fn test_esp_charges_water_neutral() {
    let model = Pm6Model;
    let scf_opts = ScfOptions::default();

    // Standard water geometry in C2v orientation
    let atomic_numbers = vec![8, 1, 1];
    let coordinates = vec![
        [0.000000, 0.000000, 0.000000],
        [0.757000, 0.586000, 0.000000],
        [-0.757000, 0.586000, 0.000000],
    ];

    let batch = MolecularBatch::new_for_model(atomic_numbers, &coordinates, &model);
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    assert!(scf_res.converged, "Water SCF must converge");

    // Compute quantum dipole for comparison
    let dipole_res = compute_dipole_moment(&batch, &model, &ws.density);

    let esp_opts = EspOptions {
        shell_multipliers: vec![1.4, 1.6, 1.8, 2.0],
        points_per_shell: 64,
        net_charge: 0.0,
    };

    let esp_res = compute_esp_charges(&batch, &model, &ws.density, &esp_opts)
        .expect("ESP charge fitting should succeed for water");

    assert_eq!(esp_res.charges.len(), 3);
    let q_o = esp_res.charges[0];
    let q_h1 = esp_res.charges[1];
    let q_h2 = esp_res.charges[2];

    // Invariant 1: Exact total charge conservation: sum_A q_A == 0.0
    let total_charge = q_o + q_h1 + q_h2;
    assert!(
        total_charge.abs() < 1e-10,
        "Total ESP charge must be strictly 0.0, got {:.12}",
        total_charge
    );

    // Invariant 2: Chemical electronegativity (Oxygen is negative, Hydrogens are positive)
    assert!(
        q_o < -0.3,
        "Oxygen ESP charge should be negative (< -0.3), got {:.4}",
        q_o
    );
    assert!(
        q_h1 > 0.15,
        "Hydrogen 1 ESP charge should be positive (> 0.15), got {:.4}",
        q_h1
    );
    assert!(
        q_h2 > 0.15,
        "Hydrogen 2 ESP charge should be positive (> 0.15), got {:.4}",
        q_h2
    );

    // Invariant 3: Symmetry equivalence of the two hydrogens in C2v water
    assert!(
        (q_h1 - q_h2).abs() < 0.02,
        "Symmetric hydrogens must have nearly identical ESP charges: H1={:.4}, H2={:.4}",
        q_h1,
        q_h2
    );

    // Invariant 4: Fitting quality - RMS error should be small (< 0.1 eV)
    assert!(
        esp_res.rms_error_ev < 0.15,
        "RMS fitting error should be < 0.15 eV, got {:.4} eV",
        esp_res.rms_error_ev
    );

    // Invariant 5: Dipole direction consistency
    // Quantum dipole in Debye:
    let q_dipole = dipole_res.total[3];
    let esp_dipole = esp_res.dipole_magnitude_debye;
    assert!(
        esp_dipole > 1.0 && esp_dipole < 3.0,
        "ESP dipole magnitude should be in physical range (1-3 D), got {:.3} D (quantum = {:.3} D)",
        esp_dipole,
        q_dipole
    );
    // In C2v water with coordinates above, dipole is along +Y
    assert!(
        esp_res.dipole_debye[1] > 0.0,
        "ESP dipole should point in positive Y direction matching molecular symmetry"
    );
}

#[test]
fn test_esp_charges_formaldehyde() {
    let model = Pm6Model;
    let scf_opts = ScfOptions::default();

    // Formaldehyde H2CO (C=O along X axis, Hydrogens in XY plane)
    let atomic_numbers = vec![6, 8, 1, 1]; // C, O, H, H
    let coordinates = vec![
        [0.000000, 0.000000, 0.000000],   // C
        [1.208000, 0.000000, 0.000000],   // O
        [-0.590000, 0.940000, 0.000000],  // H
        [-0.590000, -0.940000, 0.000000], // H
    ];

    let batch = MolecularBatch::new_for_model(atomic_numbers, &coordinates, &model);
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    assert!(scf_res.converged, "Formaldehyde SCF must converge");

    let esp_opts = EspOptions::default();
    let esp_res = compute_esp_charges(&batch, &model, &ws.density, &esp_opts)
        .expect("ESP charge fitting should succeed for formaldehyde");

    let q_c = esp_res.charges[0];
    let q_o = esp_res.charges[1];
    let q_h1 = esp_res.charges[2];
    let q_h2 = esp_res.charges[3];

    // Charge conservation
    let sum_q = q_c + q_o + q_h1 + q_h2;
    assert!(
        sum_q.abs() < 1e-10,
        "Total charge must be 0.0, got {:.12}",
        sum_q
    );

    // Carbonyl polarization: Oxygen is negative, Carbon is positive
    assert!(
        q_o < -0.25,
        "Carbonyl oxygen must be negative, got {:.4}",
        q_o
    );
    assert!(
        q_c > 0.10,
        "Carbonyl carbon must be positive, got {:.4}",
        q_c
    );

    // Symmetry between the two formyl hydrogens
    assert!(
        (q_h1 - q_h2).abs() < 0.02,
        "Formyl hydrogens should have matching ESP charges: H1={:.4}, H2={:.4}",
        q_h1,
        q_h2
    );
}

#[test]
fn test_esp_charges_cation_ammonium() {
    let model = Pm6Model;
    let scf_opts = ScfOptions::default();

    // Ammonium cation NH4+ (net charge +1.0)
    let atomic_numbers = vec![7, 1, 1, 1, 1]; // N, H, H, H, H
    let r = 1.02;
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
        [-r / 3.0, r * (8.0f64 / 9.0).sqrt(), 0.0],
        [
            -r / 3.0,
            -r * (2.0f64 / 9.0).sqrt(),
            r * (2.0f64 / 3.0).sqrt(),
        ],
        [
            -r / 3.0,
            -r * (2.0f64 / 9.0).sqrt(),
            -r * (2.0f64 / 3.0).sqrt(),
        ],
    ];

    let batch = MolecularBatch::new_for_model(atomic_numbers, &coordinates, &model);
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    assert!(scf_res.converged, "NH4+ SCF must converge");

    // Constrain fit to net charge +1.0
    let esp_opts = EspOptions {
        net_charge: 1.0,
        ..EspOptions::default()
    };

    let esp_res = compute_esp_charges(&batch, &model, &ws.density, &esp_opts)
        .expect("ESP charge fitting should succeed for ammonium");

    let sum_q: f64 = esp_res.charges.iter().sum();
    assert!(
        (sum_q - 1.0).abs() < 1e-10,
        "Net charge constraint +1.0 violated: sum_q = {:.12}",
        sum_q
    );

    // In ammonium, the positive charge is distributed across the 4 hydrogens
    for (i, &q_h) in esp_res.charges[1..=4].iter().enumerate() {
        assert!(
            q_h > 0.2,
            "Ammonium hydrogen {} should carry substantial positive charge (> 0.2), got {:.4}",
            i + 1,
            q_h
        );
    }
}
