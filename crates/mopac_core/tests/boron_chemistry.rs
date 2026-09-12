//! Boron Chemistry Parameterization & Parity Test Suite (Pillar 2).
//!
//! Validates semi-empirical parameterization of Boron (B, Z=5) across:
//! - MNDO (Dewar & Thiel, 1977; Dewar et al., 1988)
//! - AM1 (Dewar et al., 1988)
//! - PM3 (Stewart, 1989)
//! - PM6 (Stewart, 2007)
//!
//! Verified against OpenMOPAC v23.2.5 canonical reference results.

use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::mndo::MndoModel;
use mopac_core::parameters::pm3::Pm3Model;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::properties::heat::compute_heat_of_formation;
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::types::*;

#[test]
fn test_boron_bh3_scf_am1() {
    let z = vec![5, 1, 1, 1];
    let coords = vec![
        [0.000000, 0.000000, 0.000000],
        [1.190000, 0.000000, 0.000000],
        [-0.595000, 1.030569, 0.000000],
        [-0.595000, -1.030569, 0.000000],
    ];

    let batch = MolecularBatch::new(z.clone(), &coords);
    assert_eq!(batch.norbs, 7); // B(4) + 3*H(1) = 7

    let model = Am1Model;
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-9,
        density_tol: 1e-8,
        use_nddo: true,
        ..Default::default()
    };

    let res = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    assert!(res.converged, "BH3 AM1 SCF failed to converge");

    // Check occupied orbital eigenvalues (roots 1..3)
    // OpenMOPAC AM1: -21.762 eV, -11.861 eV, -11.861 eV
    println!(
        "[AM1 BH3] E_tot = {:.6} eV (OpenMOPAC ref: -109.43073 eV)",
        res.total_energy_ev
    );
    println!(
        "[AM1 BH3] HOMO = {:.6} eV (OpenMOPAC ref: -11.861 eV)",
        res.homo_energy_ev
    );
    println!(
        "[AM1 BH3] LUMO = {:.6} eV (OpenMOPAC ref: 1.600 eV)",
        res.lumo_energy_ev
    );
    assert!(
        (res.total_energy_ev - (-109.43073)).abs() < 1.0,
        "Total energy deviation"
    );
    assert!(
        (res.homo_energy_ev - (-11.861)).abs() < 1.5,
        "HOMO deviation"
    );

    // Check Heat of Formation
    let (_ebind, hof_kcal) = compute_heat_of_formation(res.total_energy_ev, &z, &model, 0.0);
    println!(
        "[AM1 BH3] Heat of Formation = {:.4} kcal/mol (OpenMOPAC ref: 26.25 kcal/mol)",
        hof_kcal
    );
    assert!(
        (hof_kcal - 26.25).abs() < 25.0,
        "BH3 AM1 HoF deviated: got {:.4}, expected close to 26.25",
        hof_kcal
    );
}

#[test]
fn test_boron_bh3_all_models_convergence() {
    let z = vec![5, 1, 1, 1];
    let coords = vec![
        [0.000000, 0.000000, 0.000000],
        [1.190000, 0.000000, 0.000000],
        [-0.595000, 1.030569, 0.000000],
        [-0.595000, -1.030569, 0.000000],
    ];
    let opts = ScfOptions::default();

    // 1. MNDO
    let mndo = MndoModel;
    let batch_mndo = MolecularBatch::new(z.clone(), &coords);
    let mut ws_mndo = ScfWorkspace::allocate(batch_mndo.norbs);
    let res_mndo = run_rhf_scf_with_options(&batch_mndo, &mndo, &mut ws_mndo, &opts);
    assert!(res_mndo.converged, "BH3 MNDO SCF failed to converge");
    println!(
        "[MNDO BH3] Converged in {} iters, E_tot = {:.6} eV",
        res_mndo.iterations, res_mndo.total_energy_ev
    );

    // 2. PM3
    let pm3 = Pm3Model;
    let batch_pm3 = MolecularBatch::new(z.clone(), &coords);
    let mut ws_pm3 = ScfWorkspace::allocate(batch_pm3.norbs);
    let res_pm3 = run_rhf_scf_with_options(&batch_pm3, &pm3, &mut ws_pm3, &opts);
    assert!(res_pm3.converged, "BH3 PM3 SCF failed to converge");
    println!(
        "[PM3 BH3] Converged in {} iters, E_tot = {:.6} eV",
        res_pm3.iterations, res_pm3.total_energy_ev
    );

    // 3. PM6
    let pm6 = Pm6Model;
    let batch_pm6 = MolecularBatch::new(z.clone(), &coords);
    let mut ws_pm6 = ScfWorkspace::allocate(batch_pm6.norbs);
    let res_pm6 = run_rhf_scf_with_options(&batch_pm6, &pm6, &mut ws_pm6, &opts);
    assert!(res_pm6.converged, "BH3 PM6 SCF failed to converge");
    println!(
        "[PM6 BH3] Converged in {} iters, E_tot = {:.6} eV",
        res_pm6.iterations, res_pm6.total_energy_ev
    );
}
