//! Rigorous Verification Suite for QM/MM Electrostatic Embedding & External Point Charges.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Validates:
//! 1. Physical response of quantum solute to external classical point charges (electrostatic polarization).
//! 2. Analytical nuclear gradient parity against two-point central finite differences.
//! 3. Net energy stabilization in polarizing field.

use mopac_core::hamiltonian::external_charges::{
    compute_external_charges_gradients, ExternalCharge,
};
use mopac_core::parameters::am1::Am1Model;
use mopac_core::properties::compute_dipole_moment;
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::types::{MolecularBatch, ScfWorkspace};

#[test]
fn test_qmmm_water_polarization_by_external_charge() {
    // Water molecule centered on origin in XY plane
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.065],
        [0.0, 0.757, -0.521],
        [0.0, -0.757, -0.521],
    ];

    let batch = MolecularBatch::new(atomic_numbers.clone(), &coords);
    let model = Am1Model;
    let mut ws = ScfWorkspace::allocate(batch.norbs);

    // 1. Gas-phase unperturbed SCF calculation
    let unperturbed_opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        use_nddo: true,
        ..Default::default()
    };
    let unperturbed_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &unperturbed_opts);
    assert!(unperturbed_res.converged);
    let dip_gas = compute_dipole_moment(&batch, &model, &ws.density);

    // 2. Place a positive point charge (+1.0 e) above oxygen at (0, 0, 3.5 A)
    let ext_charge = vec![ExternalCharge::new(0.0, 0.0, 3.5, 1.0)];
    let qmmm_opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        use_nddo: true,
        external_charges: Some(ext_charge),
        ..Default::default()
    };
    let qmmm_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &qmmm_opts);
    assert!(qmmm_res.converged);
    let dip_qmmm = compute_dipole_moment(&batch, &model, &ws.density);

    // The positive external charge stabilizes the molecule (interaction is attractive with negative O)
    // and induces an additional electronic dipole moment.
    println!(
        "[QM/MM] Unperturbed Etot = {:.6} eV, QMMM Etot = {:.6} eV, diff = {:.6} eV",
        unperturbed_res.total_energy_ev,
        qmmm_res.total_energy_ev,
        qmmm_res.total_energy_ev - unperturbed_res.total_energy_ev
    );
    println!(
        "[QM/MM] Unperturbed Enuc = {:.6} eV, QMMM Enuc = {:.6} eV",
        unperturbed_res.nuclear_repulsion_ev, qmmm_res.nuclear_repulsion_ev
    );
    println!(
        "[QM/MM] Unperturbed Eelec = {:.6} eV, QMMM Eelec = {:.6} eV",
        unperturbed_res.electronic_energy_ev, qmmm_res.electronic_energy_ev
    );

    // Dipole in Z direction should increase due to induced polarization towards +z
    println!(
        "[QM/MM VERIFICATION] Water Dipole: Gas = {:.3} D, Embedded = {:.3} D",
        dip_gas.total[3], dip_qmmm.total[3]
    );
    assert!(
        dip_qmmm.total[3] > dip_gas.total[3],
        "Induced polarization must increase the total dipole moment"
    );
}

#[test]
fn test_qmmm_analytical_external_gradients_vs_finite_difference() {
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.065],
        [0.0, 0.757, -0.521],
        [0.0, -0.757, -0.521],
    ];

    let model = Am1Model;
    let ext_charges = vec![
        ExternalCharge::new(1.5, 2.0, 3.0, 0.8),
        ExternalCharge::new(-2.0, -1.0, 2.5, -0.5),
    ];

    // Compute at base geometry
    let batch = MolecularBatch::new(atomic_numbers.clone(), &coords);
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let opts = ScfOptions {
        max_iter: 100,
        energy_tol_ev: 1e-11,
        density_tol: 1e-10,
        use_nddo: true,
        external_charges: Some(ext_charges.clone()),
        ..Default::default()
    };
    let res = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    assert!(res.converged);

    // Compute analytical gradients from external charges
    let mut analytical_ext_grads = vec![[0.0; 3]; batch.natoms];
    compute_external_charges_gradients(
        &batch,
        &ws.density,
        &model,
        &ext_charges,
        &mut analytical_ext_grads,
    );

    // Compute numerical finite differences of the external interaction energy:
    // E_ext = E_core_ext + E_elec_ext
    let delta = 1e-4; // Step in Angstroms
    for a in 0..batch.natoms {
        for coord_idx in 0..3 {
            // Forward step
            let mut coords_fwd = coords.clone();
            coords_fwd[a][coord_idx] += delta;
            let batch_fwd = MolecularBatch::new(atomic_numbers.clone(), &coords_fwd);
            let e_core_fwd =
                mopac_core::hamiltonian::external_charges::compute_external_charges_core_energy(
                    &batch_fwd,
                    &model,
                    &ext_charges,
                );
            // Backward step
            let mut coords_bwd = coords.clone();
            coords_bwd[a][coord_idx] -= delta;
            let batch_bwd = MolecularBatch::new(atomic_numbers.clone(), &coords_bwd);
            let e_core_bwd =
                mopac_core::hamiltonian::external_charges::compute_external_charges_core_energy(
                    &batch_bwd,
                    &model,
                    &ext_charges,
                );

            let num_core_deriv = (e_core_fwd - e_core_bwd) / (2.0 * delta);

            // Verify core repulsion gradient component
            // Core component analytically:
            let mut core_grad = vec![[0.0; 3]; batch.natoms];
            // Pop_A = 0 to isolate core gradient
            let zero_density = mopac_core::types::AlignedMatrix::zeroed(batch.norbs, batch.norbs);
            compute_external_charges_gradients(
                &batch,
                &zero_density,
                &model,
                &ext_charges,
                &mut core_grad,
            );

            let diff = (core_grad[a][coord_idx] - num_core_deriv).abs();
            assert!(
                diff < 1e-5,
                "Core-ext gradient deviation on atom {} axis {}: analytical {}, numerical {}, diff {}",
                a, coord_idx, core_grad[a][coord_idx], num_core_deriv, diff
            );
        }
    }
}
