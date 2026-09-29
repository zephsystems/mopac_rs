//! Empirical unit tests for Molstar export formats (SDF, Multi-Model XYZ, Gaussian Cube).

use mopac_core::export::*;
use mopac_core::parameters::Pm6Model;
use mopac_core::scf::scf_loop::run_rhf_scf_adaptive_with_nddo;
use mopac_core::types::{MolecularBatch, ScfWorkspace};

#[test]
fn test_sdf_v2000_export() {
    // Water molecule: O, H, H
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.469],
        [0.0, -0.757, -0.469],
    ];

    let props = vec![
        ("HEAT_OF_FORMATION_KCAL", "-57.80"),
        ("DIPOLE_DEBYE", "1.85"),
    ];

    let sdf = export_sdf_v2000(&atomic_numbers, &coords, "H2O Test", &props, None);
    assert!(sdf.contains("H2O Test"));
    assert!(sdf.contains("V2000"));
    assert!(sdf.contains("M  END"));
    assert!(sdf.contains("> <HEAT_OF_FORMATION_KCAL>"));
    assert!(sdf.contains("-57.80"));
    assert!(sdf.ends_with("$$$$\n"));
}

#[test]
fn test_trajectory_xyz_export() {
    let atomic_numbers = vec![8, 1, 1];
    let frame1 = vec![[0.0, 0.0, 0.0], [0.0, 0.8, -0.5], [0.0, -0.8, -0.5]];
    let frame2 = vec![[0.0, 0.0, 0.1], [0.0, 0.75, -0.47], [0.0, -0.75, -0.47]];

    let comments = vec![
        "Cycle 1 E = -380.00 eV".to_string(),
        "Cycle 2 E = -382.50 eV (Converged)".to_string(),
    ];

    let xyz = export_trajectory_xyz(&atomic_numbers, &[frame1, frame2], &comments);
    assert!(xyz.contains("Cycle 1 E = -380.00 eV"));
    assert!(xyz.contains("Cycle 2 E = -382.50 eV"));
    // Count occurrences of "3\n"
    assert_eq!(xyz.matches("3\n").count(), 2);
}

#[test]
fn test_gaussian_cube_generation() {
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.469],
        [0.0, -0.757, -0.469],
    ];

    let model = Pm6Model;
    let batch = MolecularBatch::new(atomic_numbers.clone(), &coords);
    let mut ws = ScfWorkspace::allocate(batch.norbs);

    let _scf = run_rhf_scf_adaptive_with_nddo(&batch, &model, &mut ws, 60, 1e-6, 1e-7, false);

    // Grid configuration: coarse grid for fast test
    let config = CubeGridConfig {
        padding_angstrom: 2.0,
        resolution_angstrom: 0.5,
        ..Default::default()
    };

    // 1. Generate HOMO Cube
    let homo_idx = batch.norbs / 2 - 1; // 4 occupied orbitals in water valence (8 valence electrons / 2 = 4)
    let homo_energy = ws.eigenvalues[homo_idx];
    let homo_coeffs = ws.eigenvectors.row(homo_idx);

    let mo_cube = generate_molecular_orbital_cube(
        &batch,
        &model,
        homo_coeffs,
        homo_idx + 1,
        homo_energy,
        &config,
    );
    assert!(mo_cube.contains("MOPAC_RS Molecular Orbital Cube File"));
    assert!(mo_cube.contains(&format!("Orbital {} Energy", homo_idx + 1)));
    assert!(mo_cube.lines().count() > 10);

    // 2. Generate Density Cube
    let density_cube = generate_density_cube(&batch, &model, &ws.density, &config);
    assert!(density_cube.contains("MOPAC_RS Total Electron Density Cube File"));
    assert!(density_cube.lines().count() > 10);
}
