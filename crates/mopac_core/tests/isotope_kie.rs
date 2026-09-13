//! Kinetic Isotope Effect (KIE) and Custom Isotopic Masses Verification Suite.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Tests the harmonic frequency shifts and statistical thermodynamic properties
//! under isotopic substitutions (e.g. H2O vs D2O, 12CH4 vs 13CH4 vs CD4).
//! Verifies the Born-Oppenheimer invariant: the electronic force constants are
//! invariant under nuclear mass changes, while frequencies and reduced masses scale.

use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::scf::scf_loop::ScfOptions;
use mopac_core::types::{MolecularBatch, ScfWorkspace};
use mopac_core::vibrations::{compute_hessian_and_frequencies, HessianOptions};

#[test]
fn test_water_deuteration_isotope_shifts() {
    let model = Pm6Model;
    let scf_opts = ScfOptions::default();

    // Standard water geometry
    let atomic_numbers = vec![8, 1, 1];
    let coordinates = vec![
        [0.000000, 0.000000, 0.000000],
        [0.757000, 0.586000, 0.000000],
        [-0.757000, 0.586000, 0.000000],
    ];

    // 1. Standard H2O calculation (natural abundance masses: O ~ 15.9994, H ~ 1.008)
    let mut batch_h2o = MolecularBatch::new_for_model(atomic_numbers.clone(), &coordinates, &model);
    let mut ws_h2o = ScfWorkspace::allocate(batch_h2o.norbs);
    let hess_opts_h2o = HessianOptions::default();
    let res_h2o = compute_hessian_and_frequencies(
        &mut batch_h2o,
        &model,
        &mut ws_h2o,
        &scf_opts,
        &hess_opts_h2o,
    );

    assert_eq!(res_h2o.vibrational_frequencies_cm1.len(), 3);
    let f_h2o = &res_h2o.vibrational_frequencies_cm1;
    assert!(f_h2o[0] > 1000.0, "Bending mode should be > 1000 cm^-1");
    assert!(
        f_h2o[1] > 1800.0,
        "Symmetric stretch should be > 1800 cm^-1"
    );
    assert!(
        f_h2o[2] > 2000.0,
        "Asymmetric stretch should be > 2000 cm^-1"
    );

    // 2. Heavy water D2O calculation (O = 15.9994 amu, D = 2.0141018 amu)
    let d_mass = 2.0141018;
    let o_mass = 15.9994;
    let custom_masses_d2o = vec![o_mass, d_mass, d_mass];

    let mut batch_d2o = MolecularBatch::new_for_model(atomic_numbers.clone(), &coordinates, &model);
    let mut ws_d2o = ScfWorkspace::allocate(batch_d2o.norbs);
    let hess_opts_d2o = HessianOptions {
        custom_masses: Some(custom_masses_d2o),
        ..HessianOptions::default()
    };
    let res_d2o = compute_hessian_and_frequencies(
        &mut batch_d2o,
        &model,
        &mut ws_d2o,
        &scf_opts,
        &hess_opts_d2o,
    );

    assert_eq!(res_d2o.vibrational_frequencies_cm1.len(), 3);
    let f_d2o = &res_d2o.vibrational_frequencies_cm1;

    // Physical Invariant 1: In D2O, all 3 genuine vibrational frequencies must be lower than H2O
    for i in 0..3 {
        assert!(
            f_d2o[i] < f_h2o[i],
            "Frequency {} in D2O ({:.1} cm^-1) must be lower than H2O ({:.1} cm^-1)",
            i,
            f_d2o[i],
            f_h2o[i]
        );
    }

    // Physical Invariant 2: The isotope frequency shift ratio nu(H) / nu(D) for stretches
    // must be close to the theoretical sqrt(mu_D / mu_H) ratio (~1.34 - 1.38)
    let ratio_bend = f_h2o[0] / f_d2o[0];
    let ratio_symm = f_h2o[1] / f_d2o[1];
    let ratio_asym = f_h2o[2] / f_d2o[2];

    assert!(
        ratio_bend > 1.30 && ratio_bend < 1.42,
        "Bending isotope ratio expected ~1.36, got {:.3}",
        ratio_bend
    );
    assert!(
        ratio_symm > 1.30 && ratio_symm < 1.42,
        "Symmetric stretch isotope ratio expected ~1.36, got {:.3}",
        ratio_symm
    );
    assert!(
        ratio_asym > 1.30 && ratio_asym < 1.42,
        "Asymmetric stretch isotope ratio expected ~1.36, got {:.3}",
        ratio_asym
    );

    // Physical Invariant 3: Zero-Point Vibrational Energy (ZPVE) must be significantly lower in D2O
    assert!(
        res_d2o.zpve_kcal_mol < res_h2o.zpve_kcal_mol,
        "ZPVE(D2O) ({:.2} kcal/mol) must be lower than ZPVE(H2O) ({:.2} kcal/mol)",
        res_d2o.zpve_kcal_mol,
        res_h2o.zpve_kcal_mol
    );
    let zpve_ratio = res_h2o.zpve_kcal_mol / res_d2o.zpve_kcal_mol;
    assert!(
        zpve_ratio > 1.30 && zpve_ratio < 1.42,
        "ZPVE ratio expected ~1.36, got {:.3}",
        zpve_ratio
    );

    // Physical Invariant 4: Standard entropy S°(298.15 K) must be higher for D2O due to higher mass
    assert!(
        res_d2o.thermo.entropy_total_cal_k_mol > res_h2o.thermo.entropy_total_cal_k_mol,
        "S°(D2O) ({:.2}) must be greater than S°(H2O) ({:.2})",
        res_d2o.thermo.entropy_total_cal_k_mol,
        res_h2o.thermo.entropy_total_cal_k_mol
    );

    // Physical Invariant 5: Semi-heavy water HDO (asymmetric isotope substitution)
    let custom_masses_hdo = vec![o_mass, 1.00794, d_mass];
    let mut batch_hdo = MolecularBatch::new_for_model(atomic_numbers, &coordinates, &model);
    let mut ws_hdo = ScfWorkspace::allocate(batch_hdo.norbs);
    let hess_opts_hdo = HessianOptions {
        custom_masses: Some(custom_masses_hdo),
        ..HessianOptions::default()
    };
    let res_hdo = compute_hessian_and_frequencies(
        &mut batch_hdo,
        &model,
        &mut ws_hdo,
        &scf_opts,
        &hess_opts_hdo,
    );

    // ZPVE of HDO must lie strictly between D2O and H2O: ZPVE(D2O) < ZPVE(HDO) < ZPVE(H2O)
    assert!(
        res_d2o.zpve_kcal_mol < res_hdo.zpve_kcal_mol
            && res_hdo.zpve_kcal_mol < res_h2o.zpve_kcal_mol,
        "Monotonicity invariant: ZPVE(D2O) < ZPVE(HDO) < ZPVE(H2O) violated: {:.2} < {:.2} < {:.2}",
        res_d2o.zpve_kcal_mol,
        res_hdo.zpve_kcal_mol,
        res_h2o.zpve_kcal_mol
    );
}

#[test]
fn test_methane_carbon13_and_deuterium_shifts() {
    let model = Pm6Model;
    let scf_opts = ScfOptions::default();

    let atomic_numbers = vec![6, 1, 1, 1, 1];
    let r = 1.09;
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

    // Standard 12CH4
    let mut batch_12ch4 =
        MolecularBatch::new_for_model(atomic_numbers.clone(), &coordinates, &model);
    let mut ws_12ch4 = ScfWorkspace::allocate(batch_12ch4.norbs);
    let res_12ch4 = compute_hessian_and_frequencies(
        &mut batch_12ch4,
        &model,
        &mut ws_12ch4,
        &scf_opts,
        &HessianOptions::default(),
    );

    // 13CH4 (Carbon-13 substitution: 13.00335 amu)
    let masses_13c = vec![13.00335, 1.00794, 1.00794, 1.00794, 1.00794];
    let mut batch_13ch4 =
        MolecularBatch::new_for_model(atomic_numbers.clone(), &coordinates, &model);
    let mut ws_13ch4 = ScfWorkspace::allocate(batch_13ch4.norbs);
    let hess_opts_13c = HessianOptions {
        custom_masses: Some(masses_13c),
        ..HessianOptions::default()
    };
    let res_13ch4 = compute_hessian_and_frequencies(
        &mut batch_13ch4,
        &model,
        &mut ws_13ch4,
        &scf_opts,
        &hess_opts_13c,
    );

    // 13C isotope shift is subtle (~5-15 cm^-1 on C-H stretch/deformation), ZPVE must be slightly lower
    assert!(
        res_13ch4.zpve_kcal_mol < res_12ch4.zpve_kcal_mol,
        "13CH4 ZPVE ({:.3}) must be lower than 12CH4 ZPVE ({:.3})",
        res_13ch4.zpve_kcal_mol,
        res_12ch4.zpve_kcal_mol
    );
    let dzpve_13c = res_12ch4.zpve_kcal_mol - res_13ch4.zpve_kcal_mol;
    assert!(
        dzpve_13c > 0.01 && dzpve_13c < 0.20,
        "Expected 13C ZPVE shift between 0.01 and 0.20 kcal/mol, got {:.4}",
        dzpve_13c
    );

    // CD4 (Fully deuterated methane)
    let d_mass = 2.0141018;
    let masses_cd4 = vec![12.011, d_mass, d_mass, d_mass, d_mass];
    let mut batch_cd4 = MolecularBatch::new_for_model(atomic_numbers, &coordinates, &model);
    let mut ws_cd4 = ScfWorkspace::allocate(batch_cd4.norbs);
    let hess_opts_cd4 = HessianOptions {
        custom_masses: Some(masses_cd4),
        ..HessianOptions::default()
    };
    let res_cd4 = compute_hessian_and_frequencies(
        &mut batch_cd4,
        &model,
        &mut ws_cd4,
        &scf_opts,
        &hess_opts_cd4,
    );

    assert!(
        res_cd4.zpve_kcal_mol < res_13ch4.zpve_kcal_mol,
        "CD4 ZPVE ({:.2}) must be substantially lower than 13CH4 ZPVE ({:.2})",
        res_cd4.zpve_kcal_mol,
        res_13ch4.zpve_kcal_mol
    );
    let ratio_cd4 = res_12ch4.zpve_kcal_mol / res_cd4.zpve_kcal_mol;
    assert!(
        ratio_cd4 > 1.30 && ratio_cd4 < 1.40,
        "CD4 ZPVE ratio expected ~1.35, got {:.3}",
        ratio_cd4
    );
}
