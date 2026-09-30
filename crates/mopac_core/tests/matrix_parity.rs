//! Automated Differential CI Suite for Core Hamiltonian and Overlap Matrices.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Systematically verifies the tensor invariants of:
//! 1. One-electron Core Hamiltonian Matrix $H_{\text{core}}$
//! 2. Overlap Matrix $S$ and Löwdin metric $S^{-1/2}$
//! 3. Diatomic Resonance Contractions and Two-Center Repulsions
//!
//! Evaluated across organic, halogenated, and coordination test cases.

use mopac_core::integrals::overlap::build_overlap_matrix;
use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::scf::eigensolver::diagonalize_symmetric;
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::types::{MolecularBatch, ScfWorkspace};

#[test]
fn test_core_hamiltonian_and_overlap_water_am1() {
    let z = vec![8, 1, 1];
    let coords = vec![
        [0.000, 0.000, 0.000],
        [0.000, 0.757, 0.586],
        [0.000, -0.757, 0.586],
    ];
    let batch = MolecularBatch::new(z, &coords);
    let model = Am1Model;
    let norbs = batch.norbs; // 4 (O) + 1 (H) + 1 (H) = 6 orbitals

    // 1. Overlap Matrix S Verification
    let s_mat = build_overlap_matrix(&batch, &model);
    assert_eq!(s_mat.rows, norbs);
    assert_eq!(s_mat.cols, norbs);

    // Diagonal elements of S must be exactly 1.0 (normalization)
    for i in 0..norbs {
        assert!(
            (s_mat.get(i, i) - 1.0).abs() < 1e-12,
            "Overlap matrix diagonal S_{{{},{}}} != 1.0: {}",
            i,
            i,
            s_mat.get(i, i)
        );
    }

    // Overlap matrix must be strictly symmetric: S_ij == S_ji
    for i in 0..norbs {
        for j in 0..norbs {
            let diff = (s_mat.get(i, j) - s_mat.get(j, i)).abs();
            assert!(
                diff < 1e-14,
                "Symmetry violation in overlap matrix S_{{{},{}}}: diff = {}",
                i,
                j,
                diff
            );
        }
    }

    // Overlap matrix eigenvalues must all be strictly positive (positive definite metric)
    let mut ws = ScfWorkspace::allocate(norbs);
    diagonalize_symmetric(&s_mat, &mut ws.eigenvalues, &mut ws.eigenvectors);
    for (k, &eval) in ws.eigenvalues.iter().enumerate() {
        assert!(
            eval > 1e-4,
            "Overlap eigenvalue {} is not strictly positive: {}",
            k,
            eval
        );
    }

    // 2. Core Hamiltonian Matrix H_core Verification
    mopac_core::hamiltonian::hcore::build_hcore(&batch, &model, &mut ws.h_core);

    // H_core must be strictly symmetric: H_ij == H_ji
    for i in 0..norbs {
        for j in 0..norbs {
            let diff = (ws.h_core.get(i, j) - ws.h_core.get(j, i)).abs();
            assert!(
                diff < 1e-14,
                "Symmetry violation in core Hamiltonian H_{{{},{}}}: diff = {}",
                i,
                j,
                diff
            );
        }
    }

    // Oxygen 2s atomic diagonal energy in AM1: U_ss ~ -98.2 eV
    let h_oo_2s = ws.h_core.get(0, 0);
    println!("[MATRIX AUDIT] H2O AM1 H_core(O 2s) = {:.4} eV", h_oo_2s);
    assert!(h_oo_2s < -80.0 && h_oo_2s > -120.0);

    // Hydrogen 1s molecular diagonal energy in AM1: U_ss (-13 eV) + V_nuc,O (-67 eV) ~ -80.6 eV
    let h_hh_1s = ws.h_core.get(4, 4);
    println!("[MATRIX AUDIT] H2O AM1 H_core(H1 1s) = {:.4} eV", h_hh_1s);
    assert!(h_hh_1s < -70.0 && h_hh_1s > -90.0);

    // 3. Full SCF convergence and density matrix trace
    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &ScfOptions::default());
    assert!(scf_res.converged);

    // Trace of density matrix P must equal total number of valence electrons: 6 + 1 + 1 = 8
    let mut tr_p = 0.0f64;
    for i in 0..norbs {
        tr_p += ws.density.get(i, i);
    }
    println!("[MATRIX AUDIT] Tr(P) = {:.12} (Exact expected = 8.0)", tr_p);
    assert!(
        (tr_p - 8.0).abs() < 1e-10,
        "Density matrix electron trace conservation violated: Tr(P) = {}",
        tr_p
    );
}

#[test]
fn test_core_hamiltonian_and_overlap_methane_pm6() {
    let z = vec![6, 1, 1, 1, 1];
    let r = 1.09;
    let coords = vec![
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
    let batch = MolecularBatch::new(z, &coords);
    let model = Pm6Model;
    let norbs = batch.norbs; // 4 (C) + 4 (H) = 8 orbitals

    let s_mat = build_overlap_matrix(&batch, &model);
    assert_eq!(s_mat.rows, norbs);

    // Diagonal elements of S must be 1.0
    for i in 0..norbs {
        assert!((s_mat.get(i, i) - 1.0).abs() < 1e-12);
    }

    let mut ws = ScfWorkspace::allocate(norbs);
    mopac_core::hamiltonian::hcore::build_hcore(&batch, &model, &mut ws.h_core);

    // H_core symmetry
    for i in 0..norbs {
        for j in 0..norbs {
            assert!((ws.h_core.get(i, j) - ws.h_core.get(j, i)).abs() < 1e-14);
        }
    }

    // SCF convergence and idempotency of P
    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &ScfOptions::default());
    assert!(scf_res.converged);

    // Verify P^2 = 2P in orthogonal basis
    for i in 0..norbs {
        for j in 0..norbs {
            let mut p2_ij = 0.0f64;
            for k in 0..norbs {
                p2_ij += ws.density.get(i, k) * ws.density.get(k, j);
            }
            let diff = (p2_ij - 2.0 * ws.density.get(i, j)).abs();
            assert!(
                diff < 1e-8,
                "Density matrix idempotency (P^2 = 2P) violated at ({}, {}): diff = {}",
                i,
                j,
                diff
            );
        }
    }
}

#[test]
fn test_core_hamiltonian_ch3cl_am1() {
    let z = vec![6, 17, 1, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.78],
        [1.03, 0.0, -0.36],
        [-0.51, 0.89, -0.36],
        [-0.51, -0.89, -0.36],
    ];
    let batch = MolecularBatch::new(z, &coords);
    let model = Am1Model;
    let norbs = batch.norbs;

    let mut ws = ScfWorkspace::allocate(norbs);
    ws.diatomic_pairs = mopac_core::integrals::multipoles::precompute_diatomic_pairs(&batch, &model);
    mopac_core::hamiltonian::hcore::build_hcore_nddo(&batch, &model, &ws.diatomic_pairs, &mut ws.h_core);

    println!("[CH3Cl AM1 H_CORE NDDO]");
    for i in 0..norbs {
        for j in 0..=i {
            print!("{:12.6} ", ws.h_core.get(i, j));
        }
        println!();
    }

    let mut opts = ScfOptions::default();
    opts.use_nddo = true;
    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    println!("[CH3Cl AM1 SCF] converged: {}, E_tot = {:.6} eV, E_nuc = {:.6} eV, E_elec = {:.6} eV",
        scf_res.converged, scf_res.total_energy_ev, scf_res.nuclear_repulsion_ev, scf_res.electronic_energy_ev);
    
    let (_, hof) = mopac_core::properties::heat::compute_heat_of_formation(
        scf_res.total_energy_ev,
        &batch.atomic_numbers,
        &model,
        0.0,
    );
    println!("[CH3Cl AM1 Hf] = {:.5} kcal/mol (OpenMOPAC = -17.64012 kcal/mol)", hof);
    assert!((hof - (-17.64012)).abs() < 0.05, "Hf difference too large: {} vs -17.64012", hof);
}

#[test]
fn test_core_hamiltonian_ch3cl_pm6() {
    let z = vec![6, 17, 1, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.78],
        [1.03, 0.0, -0.36],
        [-0.51, 0.89, -0.36],
        [-0.51, -0.89, -0.36],
    ];
    let model = Pm6Model;
    let batch = MolecularBatch::new_for_model(z, &coords, &model);
    let norbs = batch.norbs;

    let mut ws = ScfWorkspace::allocate(norbs);
    let mut opts = ScfOptions::default();
    opts.use_nddo = true;
    let scf_res = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    println!("[CH3Cl PM6 SCF] converged: {}, E_tot = {:.6} eV, E_nuc = {:.6} eV, E_elec = {:.6} eV",
        scf_res.converged, scf_res.total_energy_ev, scf_res.nuclear_repulsion_ev, scf_res.electronic_energy_ev);
    
    let (_, hof) = mopac_core::properties::heat::compute_heat_of_formation(
        scf_res.total_energy_ev,
        &batch.atomic_numbers,
        &model,
        0.0,
    );
    println!("[CH3Cl PM6 Hf] = {:.5} kcal/mol", hof);
}

#[test]
fn test_core_hamiltonian_cc61_pm3() {
    let z = vec![6, 1, 1, 35, 35];
    let coords = vec![
        [0.0, 0.0, 0.945128],
        [-0.902579, 0.0, 1.547601],
        [0.902579, 0.0, 1.547601],
        [0.0, 1.629635, -0.125228],
        [0.0, -1.629635, -0.125228],
    ];
    let model = mopac_core::parameters::pm3::Pm3Model;
    let batch = MolecularBatch::new_for_model(z, &coords, &model);
    let norbs = batch.norbs;

    let mut ws = ScfWorkspace::allocate(norbs);
    ws.diatomic_pairs = mopac_core::integrals::multipoles::precompute_diatomic_pairs(&batch, &model);
    mopac_core::hamiltonian::hcore::build_hcore_nddo(&batch, &model, &ws.diatomic_pairs, &mut ws.h_core);

    println!("[cc-61 PM3 H_CORE NDDO diag]");
    for i in 0..norbs {
        print!("{:12.6} ", ws.h_core.get(i, i));
    }
    println!();
    
    println!("Row 6 Col 0 (C 2s, Br1 4s): {} (OM: -1.160351)", ws.h_core.get(6, 0));
    println!("Row 6 Col 3 (C 2pz, Br1 4s): {} (OM: 0.624075)", ws.h_core.get(6, 3));
    println!("Row 6 Col 4 (H1 1s, Br1 4s): {} (OM: -0.444908)", ws.h_core.get(6, 4));

    let mut opts = ScfOptions::default();
    opts.use_nddo = true;
    opts.reuse_density = true;

    use mopac_core::parameters::ParameterModel;

    // Fill ws.density with atomic orbital populations pdiag
    ws.density.data.fill(0.0);
    for i in 0..batch.natoms {
        let z_i = batch.atomic_numbers[i];
        let p_i = model.get_element(z_i).unwrap();
        let off = batch.orbital_offsets[i];
        let norb = batch.basis_types[i].num_orbitals();
        if norb == 1 {
            ws.density.set(off, off, p_i.core_charge);
        } else if norb == 4 {
            let pop = p_i.core_charge * 0.25;
            for o in 0..4 {
                ws.density.set(off + o, off + o, pop);
            }
        } else if norb == 9 {
            let pop = p_i.core_charge * 0.25;
            for o in 0..4 {
                ws.density.set(off + o, off + o, pop);
            }
            // d orbitals remain 0.0
        }
    }

    let res = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    println!("SCF res with atomic guess: conv={}, iter={}, E_tot={}, E_elec={}",
        res.converged, res.iterations, res.total_energy_ev, res.electronic_energy_ev);
    let (_, hof) = mopac_core::properties::heat::compute_heat_of_formation(
        res.total_energy_ev,
        &batch.atomic_numbers,
        &model,
        0.0,
    );
    println!("[cc-61 PM3 Hf with atomic guess] = {:.5} kcal/mol (OpenMOPAC = 12.16934 kcal/mol)", hof);
}
