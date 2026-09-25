//! Rigorous Quality and Parity Verification Suite for AM1-BCC Charge Model.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Validates:
//! 1. Exact charge conservation: \sum_i q_i^(BCC) == \sum_i q_i^(0) == Q_tot to 1.0e-14 e.
//! 2. Correct Jakalian 2002 BCC parameter transfers on aliphatic C-H, aromatic C-H, and carbonyl C=O.
//! 3. Full end-to-end SCF + Mulliken + AM1-BCC charge pipeline on realistic organic molecules.

use mopac_core::parameters::am1::Am1Model;
use mopac_core::properties::{compute_am1_bcc_charges, compute_mulliken_population};
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::types::{MolecularBatch, ScfWorkspace};

#[test]
fn test_methane_am1_bcc_charge_conservation_and_shifts() {
    // Methane CH4: tetrahedral geometry
    let atomic_numbers = vec![6, 1, 1, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.6276, 0.6276, 0.6276],
        [-0.6276, -0.6276, 0.6276],
        [0.0, 0.8875, -0.6276],
        [0.0, -0.8875, -0.6276],
    ];

    let batch = MolecularBatch::new(atomic_numbers, &coords);

    // Initial neutral Mulliken-like charges (sum = 0.0)
    let initial_charges = vec![-0.100, 0.025, 0.025, 0.025, 0.025];
    let result = compute_am1_bcc_charges(&batch, &initial_charges).expect("AM1-BCC failed");

    // 1. Verify exact charge conservation to 1.0e-14
    let sum_initial: f64 = initial_charges.iter().sum();
    let sum_final: f64 = result.bcc_charges.iter().sum();
    assert!(
        (sum_final - sum_initial).abs() < 1e-14,
        "Total charge not conserved: initial {}, final {}",
        sum_initial,
        sum_final
    );

    // 2. Verify C-H aliphatic BCC shifts: delta_CH = -0.0487
    // Carbon should receive 4 * (-0.0487) = -0.1948
    let c_correction = result.bond_charge_corrections[0];
    assert!(
        (c_correction - (-4.0 * 0.0487)).abs() < 1e-6,
        "Expected C correction -0.1948, got {}",
        c_correction
    );

    // Each hydrogen should receive -(-0.0487) = +0.0487
    for h_idx in 1..5 {
        let h_corr = result.bond_charge_corrections[h_idx];
        assert!(
            (h_corr - 0.0487).abs() < 1e-6,
            "Expected H correction +0.0487, got {}",
            h_corr
        );
    }
}

#[test]
fn test_formaldehyde_am1_bcc_carbonyl_shift() {
    // Formaldehyde H2C=O
    let atomic_numbers = vec![6, 8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.208],
        [0.94, 0.0, -0.58],
        [-0.94, 0.0, -0.58],
    ];

    let batch = MolecularBatch::new(atomic_numbers, &coords);

    let initial_charges = vec![0.15, -0.25, 0.05, 0.05];
    let result = compute_am1_bcc_charges(&batch, &initial_charges).expect("AM1-BCC failed");

    // Conservation
    let sum_initial: f64 = initial_charges.iter().sum();
    let sum_final: f64 = result.bcc_charges.iter().sum();
    assert!((sum_final - sum_initial).abs() < 1e-14);

    // Carbonyl C=O bond order is 2 -> delta = +0.1340 (C gets +0.1340, O gets -0.1340)
    let o_correction = result.bond_charge_corrections[1];
    assert!(
        (o_correction - (-0.1340)).abs() < 1e-6,
        "Expected O carbonyl correction -0.1340, got {}",
        o_correction
    );
}

#[test]
fn test_benzene_am1_bcc_aromatic_shifts() {
    // Benzene C6H6: planar hexagonal ring
    let r_cc = 1.397;
    let r_ch = 1.084;
    let mut atomic_numbers = Vec::new();
    let mut coords = Vec::new();

    for i in 0..6 {
        let angle = (i as f64) * std::f64::consts::PI / 3.0;
        atomic_numbers.push(6);
        coords.push([r_cc * angle.cos(), r_cc * angle.sin(), 0.0]);
    }
    for i in 0..6 {
        let angle = (i as f64) * std::f64::consts::PI / 3.0;
        let r = r_cc + r_ch;
        atomic_numbers.push(1);
        coords.push([r * angle.cos(), r * angle.sin(), 0.0]);
    }

    let batch = MolecularBatch::new(atomic_numbers, &coords);
    let initial_charges = vec![0.0; 12];
    let result = compute_am1_bcc_charges(&batch, &initial_charges).expect("AM1-BCC failed");

    // Exact conservation
    let sum_final: f64 = result.bcc_charges.iter().sum();
    assert!(sum_final.abs() < 1e-14);

    // Each aromatic C-H transfers -0.0407 to C (C gets -0.0407, H gets +0.0407)
    for c_idx in 0..6 {
        let c_corr = result.bond_charge_corrections[c_idx];
        assert!(
            (c_corr - (-0.0407)).abs() < 1e-6,
            "Expected aromatic C correction -0.0407, got {}",
            c_corr
        );
    }
    for h_idx in 6..12 {
        let h_corr = result.bond_charge_corrections[h_idx];
        assert!(
            (h_corr - 0.0407).abs() < 1e-6,
            "Expected aromatic H correction +0.0407, got {}",
            h_corr
        );
    }
}

#[test]
fn test_end_to_end_am1_scf_plus_am1_bcc() {
    // Water H2O with full AM1 SCF calculation
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.065],
        [0.0, 0.757, -0.521],
        [0.0, -0.757, -0.521],
    ];

    let batch = MolecularBatch::new(atomic_numbers, &coords);
    let model = Am1Model;
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        use_nddo: true,
        ..Default::default()
    };

    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    assert!(scf_res.converged);

    let mulliken = compute_mulliken_population(&batch, &model, &ws.eigenvectors, 4);
    let bcc_res =
        compute_am1_bcc_charges(&batch, &mulliken.net_charges).expect("AM1-BCC computation");

    // 1. Verify charge neutrality to machine precision
    assert!(
        bcc_res.total_charge.abs() < 1e-13,
        "Total charge not zero: {}",
        bcc_res.total_charge
    );

    // 2. Initial oxygen is negative, O-H bond transfers negative charge to O (-0.0620 per H)
    // O final charge is more negative than initial Mulliken charge
    assert!(
        bcc_res.bcc_charges[0] < mulliken.net_charges[0],
        "Oxygen charge should become more negative"
    );
    // Hydrogens should become more positive
    assert!(
        bcc_res.bcc_charges[1] > mulliken.net_charges[1],
        "Hydrogen charge should become more positive"
    );
}
