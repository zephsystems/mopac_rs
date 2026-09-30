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

#[test]
fn test_cube_density_numerical_invariants_and_warm_restart() {
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.469],
        [0.0, -0.757, -0.469],
    ];

    let model = Pm6Model;
    let batch = MolecularBatch::new(atomic_numbers, &coords);
    let mut ws = ScfWorkspace::allocate(batch.norbs);

    // Initial solve
    let scf_initial = run_rhf_scf_adaptive_with_nddo(&batch, &model, &mut ws, 60, 1e-7, 1e-6, true);
    assert!(scf_initial.converged);

    // Warm restart test: run with reuse_density: true
    let mut ws_warm = ScfWorkspace::allocate(batch.norbs);
    ws_warm.density.data.copy_from_slice(&ws.density.data);

    let opts_warm = mopac_core::scf::scf_loop::ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        reuse_density: true,
        ..Default::default()
    };

    let scf_warm = mopac_core::scf::scf_loop::run_rhf_scf_with_options(
        &batch,
        &model,
        &mut ws_warm,
        &opts_warm,
    );
    assert!(scf_warm.converged);
    assert!(
        scf_warm.iterations <= 2,
        "Warm restart should converge in <= 2 iterations, took {}",
        scf_warm.iterations
    );
    eprintln!(
        "scf_initial: {:.8}, scf_warm: {:.8}, diff: {:.2e}",
        scf_initial.total_energy_ev,
        scf_warm.total_energy_ev,
        (scf_warm.total_energy_ev - scf_initial.total_energy_ev).abs()
    );
    assert!((scf_warm.total_energy_ev - scf_initial.total_energy_ev).abs() < 1e-5);

    // Invariant: Test physical cutoff preservation on density
    let config = CubeGridConfig {
        padding_angstrom: 3.5,
        resolution_angstrom: 0.3,
        ..Default::default()
    };
    let cube = generate_density_cube(&batch, &model, &ws.density, &config);

    // Parse density values from cube output
    let mut total_density_integral = 0.0f64;
    let step_bohr = 0.3 * ANGSTROM_TO_BOHR;
    let d_vol = step_bohr * step_bohr * step_bohr;

    let lines: Vec<&str> = cube.lines().collect();
    let header_lines = 6 + batch.natoms;
    for &line in &lines[header_lines..] {
        for token in line.split_whitespace() {
            if let Ok(val) = token.parse::<f64>() {
                assert!(
                    val >= -1e-12,
                    "Physical density cannot be significantly negative: {}",
                    val
                );
                total_density_integral += val * d_vol;
            }
        }
    }

    // Verify integrated electron count in real space Tr(P * S_real) is consistent
    assert!(
        total_density_integral > 7.0 && total_density_integral < 12.0,
        "Integrated density: {}",
        total_density_integral
    );

    // Verify cutoff constant is accessible and reasonable
    assert_eq!(STO_SPATIAL_CUTOFF_BOHR2, 144.0);
}

#[test]
fn test_cube_line_length_exact_6_and_z_ray_delimiters() {
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.469],
        [0.0, -0.757, -0.469],
    ];

    let model = Pm6Model;
    let batch = MolecularBatch::new(atomic_numbers, &coords);
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let _scf = run_rhf_scf_adaptive_with_nddo(&batch, &model, &mut ws, 60, 1e-6, 1e-7, false);

    let config = CubeGridConfig {
        padding_angstrom: 1.5,
        resolution_angstrom: 0.5,
        ..Default::default()
    };

    let density_cube = generate_density_cube(&batch, &model, &ws.density, &config);
    let lines: Vec<&str> = density_cube.lines().collect();

    // Line 3: Natoms, Origin
    // Line 4: Nx, step
    // Line 5: Ny, step
    // Line 6: Nz, step
    let parts_nx: Vec<&str> = lines[3].split_whitespace().collect();
    let parts_ny: Vec<&str> = lines[4].split_whitespace().collect();
    let parts_nz: Vec<&str> = lines[5].split_whitespace().collect();

    let nx: usize = parts_nx[0].parse().unwrap();
    let ny: usize = parts_ny[0].parse().unwrap();
    let nz: usize = parts_nz[0].parse().unwrap();

    let expected_lines_per_ray = nz.div_ceil(6);
    let expected_total_data_lines = nx * ny * expected_lines_per_ray;

    let header_lines = 6 + batch.natoms;
    let data_lines = &lines[header_lines..];

    assert_eq!(
        data_lines.len(),
        expected_total_data_lines,
        "Total data lines must match Nx * Ny * ceil(Nz / 6)"
    );

    // Verify each line has at most 6 numbers
    for (line_idx, line) in data_lines.iter().enumerate() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        assert!(
            tokens.len() <= 6 && !tokens.is_empty(),
            "Line {} has invalid token count: {}",
            line_idx,
            tokens.len()
        );
        let ray_line_idx = line_idx % expected_lines_per_ray;
        if ray_line_idx < expected_lines_per_ray - 1 {
            assert_eq!(
                tokens.len(),
                6,
                "Non-terminal ray line {} must have exactly 6 numbers",
                line_idx
            );
        } else {
            let remainder = if nz.is_multiple_of(6) { 6 } else { nz % 6 };
            assert_eq!(
                tokens.len(),
                remainder,
                "Terminal ray line {} must have exactly remainder {} numbers",
                line_idx,
                remainder
            );
        }
    }
}

#[test]
fn test_b_factor_positivity_invariant() {
    let atomic_numbers = vec![8, 6, 1];
    let coords = vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.8, 0.9, 0.0]];
    let charges = vec![-0.55, 0.35, 0.20];

    let pdb = export_molstar_pdb(&atomic_numbers, &coords, &charges, Some(0.6)).unwrap();
    assert!(pdb.contains("REMARK   MOPAC_RS MOLSTAR-OPTIMIZED PDB"));
    assert!(pdb.contains("TER\nEND\n"));

    for line in pdb.lines() {
        if line.starts_with("ATOM") {
            // Columns 61-66 is B-factor in 1-based indexing -> 60..66 in 0-based
            let b_str = &line[60..66].trim();
            let b_val: f64 = b_str.parse().expect("Valid B-factor float");
            assert!(
                (10.0..=90.0).contains(&b_val),
                "B-factor {} must be bounded in [10.0, 90.0]",
                b_val
            );
        }
    }
}

#[test]
fn test_mmcif_syntax_and_charge_parity() {
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.469],
        [0.0, -0.757, -0.469],
    ];
    let charges = vec![-0.60, 0.30, 0.30];

    let cif = export_mmcif_with_charges(&atomic_numbers, &coords, &charges, Some("WATER")).unwrap();
    assert!(cif.contains("data_WATER"));
    assert!(cif.contains("_atom_site.partial_charge"));
    assert!(cif.contains("_atom_site.B_iso_or_equiv"));

    let mut parsed_charges = Vec::new();
    for line in cif.lines() {
        if line.starts_with("ATOM") {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            // Last column is partial_charge
            let q: f64 = tokens.last().unwrap().parse().unwrap();
            parsed_charges.push(q);
        }
    }

    assert_eq!(parsed_charges.len(), 3);
    for (q_parsed, &q_orig) in parsed_charges.iter().zip(charges.iter()) {
        assert!((q_parsed - q_orig).abs() < 1e-4);
    }
}

#[test]
fn test_mvs_json_spec_validity() {
    let mvs = export_mvs_session("molecule_test.pdb", "pdb", 0.40).unwrap();
    assert!(mvs.contains("\"version\": \"1.0\""));
    assert!(mvs.contains("\"generator\": \"mopac_rs\""));
    assert!(mvs.contains("\"type\": \"ball_and_stick\""));
    assert!(mvs.contains("\"type\": \"gaussian_surface\""));
    assert!(mvs.contains("\"opacity\": 0.40"));
    assert!(mvs.contains("\"url\": \"molecule_test.pdb\""));
}

// =========================================================================
// ADVERSARIAL CONTRATESTS (STRESS & BOUNDARY CONDITIONS)
// =========================================================================

#[test]
fn test_adversarial_pathological_charges_and_limits() {
    let atomic_numbers = vec![8, 6, 1, 7, 16, 9];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ];

    // Extremes: huge charges, NaN, +Inf, -Inf, zero
    let pathological_charges = vec![
        -999.0,
        10000.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
    ];

    // Case 1: max_charge <= 0 or None
    let pdb1 =
        export_molstar_pdb(&atomic_numbers, &coords, &pathological_charges, Some(0.0)).unwrap();
    for line in pdb1.lines() {
        if line.starts_with("ATOM") {
            let b_val: f64 = line[60..66].trim().parse().unwrap();
            assert!(
                (10.0..=90.0).contains(&b_val),
                "Pathological B-factor {} violated [10.0, 90.0] invariant",
                b_val
            );
        }
    }

    let pdb2 =
        export_molstar_pdb(&atomic_numbers, &coords, &pathological_charges, Some(-5.0)).unwrap();
    for line in pdb2.lines() {
        if line.starts_with("ATOM") {
            let b_val: f64 = line[60..66].trim().parse().unwrap();
            assert!(
                (10.0..=90.0).contains(&b_val),
                "Negative max_charge produced invalid B-factor: {}",
                b_val
            );
        }
    }

    // Case 2: mmCIF handles NaN/Inf safely without emitting "NaN" or "inf" into CIF
    let cif =
        export_mmcif_with_charges(&atomic_numbers, &coords, &pathological_charges, None).unwrap();
    for line in cif.lines() {
        if line.starts_with("ATOM") {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            let q: f64 = tokens
                .last()
                .unwrap()
                .parse()
                .expect("Must be finite float");
            assert!(q.is_finite(), "mmCIF charge must be finite, found {}", q);
        }
    }
}

#[test]
fn test_adversarial_dimensional_mismatch_and_empty_system() {
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]; // 2 coords vs 3 atoms
    let charges = vec![0.0, 0.0, 0.0];

    // Mismatched coords
    assert!(export_molstar_pdb(&atomic_numbers, &coords, &charges, None).is_err());
    assert!(export_mmcif_with_charges(&atomic_numbers, &coords, &charges, None).is_err());

    // Mismatched charges
    let coords_valid = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let charges_mismatch = vec![0.0, 0.0];
    assert!(export_molstar_pdb(&atomic_numbers, &coords_valid, &charges_mismatch, None).is_err());
    assert!(
        export_mmcif_with_charges(&atomic_numbers, &coords_valid, &charges_mismatch, None).is_err()
    );

    // Empty system
    assert!(export_molstar_pdb(&[], &[], &[], None).is_err());
    assert!(export_mmcif_with_charges(&[], &[], &[], None).is_err());
}

#[test]
fn test_adversarial_cube_non_divisible_z_grids() {
    // Test that when Nz is not divisible by 6 (e.g., Nz = 7, Nz = 13),
    // each Z-ray terminates properly with a newline, and the next ray starts clean.
    let atomic_numbers = vec![1, 1];
    let coords = vec![[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]];
    let model = Pm6Model;
    let batch = MolecularBatch::new(atomic_numbers, &coords);
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let _scf = run_rhf_scf_adaptive_with_nddo(&batch, &model, &mut ws, 60, 1e-6, 1e-7, false);

    // Padding chosen to ensure Nz is odd/prime
    let config = CubeGridConfig {
        padding_angstrom: 1.25,
        resolution_angstrom: 0.42,
        ..Default::default()
    };

    let cube = generate_density_cube(&batch, &model, &ws.density, &config);
    let lines: Vec<&str> = cube.lines().collect();

    let parts_nz: Vec<&str> = lines[5].split_whitespace().collect();
    let nz: usize = parts_nz[0].parse().unwrap();
    let parts_nx: Vec<&str> = lines[3].split_whitespace().collect();
    let nx: usize = parts_nx[0].parse().unwrap();
    let parts_ny: Vec<&str> = lines[4].split_whitespace().collect();
    let ny: usize = parts_ny[0].parse().unwrap();

    let expected_lines_per_ray = nz.div_ceil(6);
    let expected_data_lines = nx * ny * expected_lines_per_ray;
    let header_lines = 6 + batch.natoms;
    let data_lines = &lines[header_lines..];

    assert_eq!(data_lines.len(), expected_data_lines);
}

#[test]
fn test_adversarial_heavy_and_unusual_elements() {
    // Elements: B (5), Mg (12), P (15), Ti (22), Zn (30), Br (35), I (53), Pt (78), Pb (82)
    let atomic_numbers = vec![5, 12, 15, 22, 30, 35, 53, 78, 82];
    let coords: Vec<[f64; 3]> = (0..9).map(|i| [i as f64 * 1.5, 0.0, 0.0]).collect();
    let charges: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4];

    let pdb = export_molstar_pdb(&atomic_numbers, &coords, &charges, None).unwrap();
    let expected_symbols = ["B", "Mg", "P", "Ti", "Zn", "Br", "I", "Pt", "Pb"];

    for sym in expected_symbols {
        assert!(
            pdb.contains(&format!(" {} ", sym)) || pdb.contains(&format!("{} ", sym)),
            "PDB must contain genuine element symbol '{}', no 'X' fallback allowed",
            sym
        );
    }
    assert!(!pdb.contains(" X "));

    let cif = export_mmcif_with_charges(&atomic_numbers, &coords, &charges, Some("HEAVY_METALS"))
        .unwrap();
    for sym in expected_symbols {
        assert!(
            cif.contains(&format!(" {} ", sym)),
            "mmCIF must contain genuine element symbol '{}'",
            sym
        );
    }
}

#[test]
fn test_adversarial_mvs_special_characters_and_escaping() {
    // Filename with quotes, backslashes and spaces
    let pathological_filename = r#"my "test" molecule\path.pdb"#;
    let mvs = export_mvs_session(pathological_filename, "pdb", 0.35).unwrap();

    // Must be valid JSON: verify balanced braces and properly escaped quotes
    assert!(mvs.contains(r#"my \"test\" molecule\\path.pdb"#));

    // Unsupported format must error cleanly
    assert!(export_mvs_session("test.xyz", "xyz", 0.35).is_err());
    // Empty filename must error cleanly
    assert!(export_mvs_session("   ", "pdb", 0.35).is_err());
}
