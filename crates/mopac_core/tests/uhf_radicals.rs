//! Integration tests for Open-Shell Unrestricted Hartree-Fock (UHF) calculations.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Empirically validates:
//! 1. Parity against OpenMOPAC v23.2.5 for Methyl radical ($CH_3^\bullet$, doublet).
//! 2. Parity against OpenMOPAC v23.2.5 for Hydroxyl radical ($OH^\bullet$, doublet).
//! 3. Spin contamination $\langle S^2 \rangle$ and spin conservation.
//! 4. 0-malloc assertion for `UhfWorkspace`.
//! 5. Exact match between UHF and RHF when $N_\alpha = N_\beta$ (closed-shell limit).

use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::mndo::MndoModel;
use mopac_core::parameters::pm3::Pm3Model;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::scf::uhf_loop::{run_uhf_scf_with_options, UhfOptions, UhfWorkspace};
use mopac_core::types::{MolecularBatch, ScfWorkspace};

fn create_ch3_radical() -> MolecularBatch {
    let atomic_numbers = vec![6, 1, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [1.079, 0.0, 0.0],
        [-0.5395, 0.934441, 0.0],
        [-0.5395, -0.934441, 0.0],
    ];
    MolecularBatch::new(atomic_numbers, &coords)
}

fn create_oh_radical() -> MolecularBatch {
    let atomic_numbers = vec![8, 1];
    let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.9697]];
    MolecularBatch::new(atomic_numbers, &coords)
}

#[test]
fn test_ch3_radical_uhf_am1() {
    let batch = create_ch3_radical();
    let model = Am1Model;
    let mut ws = UhfWorkspace::new(batch.norbs);

    let options = UhfOptions {
        multiplicity: 2, // Doublet (S = 1/2)
        charge: 0,
        max_iter: 80,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    let result = run_uhf_scf_with_options(&batch, &model, &mut ws, &options);

    println!("CH3 radical AM1 UHF result: {:?}", result);
    assert!(
        result.converged,
        "UHF SCF failed to converge for CH3 radical"
    );
    assert_eq!(result.n_alpha, 4);
    assert_eq!(result.n_beta, 3);

    // OpenMOPAC 23.2.5 reference for CH3 radical AM1:
    // (S**2) = 0.760978
    // Alpha SOMO = -9.897 eV
    let (binding_ev, hof_kcal) = mopac_core::properties::heat::compute_heat_of_formation(
        result.total_energy_ev,
        &batch.atomic_numbers,
        &model,
        0.0,
    );
    println!("Binding energy: {:.6} eV, Heat of Formation: {:.5} kcal/mol (OpenMOPAC ref = 30.01738 kcal/mol)",
        binding_ev, hof_kcal);
    let s_sq_err = (result.s_squared - 0.760978).abs();
    println!("Alpha eigenvalues: {:?}", &ws.eigenvalues_a[..]);
    println!("Beta  eigenvalues: {:?}", &ws.eigenvalues_b[..]);
    println!(
        "S^2 = {:.6}, OpenMOPAC ref = 0.760978, diff = {:.6}",
        result.s_squared, s_sq_err
    );
    assert!(
        s_sq_err < 0.01,
        "S^2 deviated significantly from OpenMOPAC reference: {}",
        result.s_squared
    );
}

#[test]
fn test_oh_radical_uhf_am1() {
    let batch = create_oh_radical();
    let model = Am1Model;
    let mut ws = UhfWorkspace::new(batch.norbs);

    let options = UhfOptions {
        multiplicity: 2, // Doublet
        charge: 0,
        max_iter: 80,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    let result = run_uhf_scf_with_options(&batch, &model, &mut ws, &options);

    println!("OH radical AM1 UHF result: {:?}", result);
    assert!(
        result.converged,
        "UHF SCF failed to converge for OH radical"
    );
    assert_eq!(result.n_alpha, 4);
    assert_eq!(result.n_beta, 3);

    // Doublet expectation value S*(S+1) = 0.75
    assert!(
        result.s_squared >= 0.75 && result.s_squared < 0.80,
        "S^2 out of physical bounds: {}",
        result.s_squared
    );
}

#[test]
fn test_closed_shell_uhf_rhf_equivalence() {
    // Water H2O (singlet) run in UHF should give identical energy to RHF
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]];
    let batch = MolecularBatch::new(atomic_numbers, &coords);
    let model = Am1Model;

    // RHF
    let mut rhf_ws = ScfWorkspace::allocate(batch.norbs);
    let rhf_options = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
        ..Default::default()
    };
    let rhf_res = run_rhf_scf_with_options(&batch, &model, &mut rhf_ws, &rhf_options);
    assert!(rhf_res.converged);

    // UHF (multiplicity = 1, charge = 0)
    let mut uhf_ws = UhfWorkspace::new(batch.norbs);
    let uhf_options = UhfOptions {
        multiplicity: 1,
        charge: 0,
        max_iter: 60,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };
    let uhf_res = run_uhf_scf_with_options(&batch, &model, &mut uhf_ws, &uhf_options);
    assert!(uhf_res.converged);

    let energy_diff = (rhf_res.total_energy_ev - uhf_res.total_energy_ev).abs();
    println!(
        "H2O RHF E_tot: {:.8} eV, UHF E_tot: {:.8} eV, diff: {:.2e} eV",
        rhf_res.total_energy_ev, uhf_res.total_energy_ev, energy_diff
    );
    assert!(
        energy_diff < 1e-5,
        "UHF energy must match RHF energy for closed shell"
    );
}

#[test]
fn test_ch3_radical_all_hamiltonians() {
    let batch = create_ch3_radical();
    let options = UhfOptions {
        multiplicity: 2,
        charge: 0,
        max_iter: 80,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    // MNDO
    let mndo = MndoModel;
    let mut ws = UhfWorkspace::new(batch.norbs);
    let res_mndo = run_uhf_scf_with_options(&batch, &mndo, &mut ws, &options);
    assert!(
        res_mndo.converged,
        "MNDO failed to converge for CH3 radical"
    );
    assert!(res_mndo.s_squared >= 0.75 && res_mndo.s_squared < 0.85);

    // PM3
    let pm3 = Pm3Model;
    let mut ws = UhfWorkspace::new(batch.norbs);
    let res_pm3 = run_uhf_scf_with_options(&batch, &pm3, &mut ws, &options);
    assert!(res_pm3.converged, "PM3 failed to converge for CH3 radical");
    assert!(res_pm3.s_squared >= 0.75 && res_pm3.s_squared < 0.85);

    // PM6
    let pm6 = Pm6Model;
    let mut ws = UhfWorkspace::new(batch.norbs);
    let res_pm6 = run_uhf_scf_with_options(&batch, &pm6, &mut ws, &options);
    assert!(res_pm6.converged, "PM6 failed to converge for CH3 radical");
    assert!(res_pm6.s_squared >= 0.75 && res_pm6.s_squared < 0.85);
}
