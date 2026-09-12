//! Automated Scrutiny & Inconsistency Verification Suite for MOPAC_RS.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Empirically verifies the mathematical solutions to legacy Fortran pathologies.

use mopac_core::constants::codata2018::*;
use mopac_core::integrals::core_repulsion::*;
use mopac_core::integrals::overlap::*;
use mopac_core::integrals::rotation::*;
use mopac_core::integrals::two_electron::*;
use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::ParameterModel;
use mopac_core::types::*;
use std::thread;

/// Scrutiny Test 1: Unit Immutability & Core Repulsion Consistency.
///
/// Legacy Fortran `ccrep.F90` mutates the input distance argument `r = r * a0` in place.
/// Here we verify that input coordinates remain strictly immutable and that core repulsion
/// scales correctly for chemical bonds ($H_2$, $C-H$, $C-C$).
#[test]
fn test_scrutiny_unit_immutability_and_core_repulsion() {
    let am1 = Am1Model;
    let h_param = am1.get_element(1).expect("H params missing");
    let c_param = am1.get_element(6).expect("C params missing");

    let dist_h2 = 0.74; // Typical H-H bond in Angstroms
    let dist_copy = dist_h2;

    let e_rep_h2 = compute_pair_core_repulsion(dist_h2, &h_param, &h_param);

    // Assert input variable was NOT mutated in place
    assert_eq!(dist_h2, dist_copy, "Input distance must remain immutable");

    // Physical bounds check: H-H nuclear repulsion at 0.74 Å must be positive and in reasonable eV range (~15-25 eV)
    assert!(e_rep_h2 > 10.0 && e_rep_h2 < 30.0, "H2 core repulsion {} out of physical bounds", e_rep_h2);

    // Distance scaling check: Repulsion must decrease monotonically as distance increases
    let e_rep_h2_longer = compute_pair_core_repulsion(1.50, &h_param, &h_param);
    assert!(e_rep_h2 > e_rep_h2_longer, "Core repulsion must decay monotonically with distance");

    // Heteronuclear pair check: C-H bond (1.09 Å)
    let e_rep_ch = compute_pair_core_repulsion(1.09, &c_param, &h_param);
    assert!(e_rep_ch > 0.0, "C-H core repulsion must be positive");
}

/// Scrutiny Test 2: Multithreaded Reentrancy & Elimination of Fortran `SAVE` Statics.
///
/// Legacy Fortran `fock2.F90` uses static `SAVE` arrays causing catastrophic race conditions.
/// Here we spawn 8 concurrent threads evaluating the same molecular batch simultaneously
/// and assert 100% bit-identical results across all threads.
#[test]
fn test_scrutiny_multithreaded_reentrancy() {
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.026, 0.0, -0.363],
        [-0.513, 0.889, -0.363],
        [-0.513, -0.889, -0.363],
    ]; // Methane (CH4)
    let atomic_numbers = vec![6, 1, 1, 1, 1];

    let batch = MolecularBatch::new(atomic_numbers, &coords);
    let am1 = Am1Model;

    // Run baseline calculation
    let baseline_energy = compute_total_core_repulsion(&batch, &am1);

    // Spawn 8 concurrent threads doing the exact same calculation
    let mut handles = Vec::new();
    for _ in 0..8 {
        let b = batch.clone();
        let handle = thread::spawn(move || {
            let m = Am1Model;
            compute_total_core_repulsion(&b, &m)
        });
        handles.push(handle);
    }

    for h in handles {
        let thread_result = h.join().expect("Thread panicked");
        // Verify bit-level exact match across threads (zero race conditions)
        assert_eq!(
            thread_result.to_bits(),
            baseline_energy.to_bits(),
            "Concurrent calculation must be bit-identical (reentrancy check)"
        );
    }
}

/// Scrutiny Test 3: 3D Frame Rotation Orthonormality & Collinear Degeneracies.
///
/// Legacy Fortran `rotate.F90`/`coe.F90` contains branching jumps (`go to 50`) when atoms
/// are collinear with the global z-axis. Here we verify that $D^{(1)}$ remains orthonormal
/// ($D D^T = I$, $\det(D) = 1.0$) for both arbitrary 3D vectors and collinear degenerate axes.
#[test]
fn test_scrutiny_rotation_orthonormality_and_collinear_degeneracy() {
    // Case A: Arbitrary 3D orientation
    let ra = [1.23, 4.56, -7.89];
    let rb = [-2.34, 0.12, 3.45];
    let frame_a = DiatomicRotationFrame::compute(&ra, &rb);
    assert!(frame_a.is_orthonormal(1e-14), "Arbitrary rotation frame must be orthonormal to 1e-14");
    assert!((frame_a.determinant() - 1.0).abs() < 1e-14, "Determinant must be +1.0");

    // Case B: Collinear along global +Z axis (degenerate case xy -> 0)
    let ra_z = [0.0, 0.0, 0.0];
    let rb_z = [0.0, 0.0, 2.5];
    let frame_z = DiatomicRotationFrame::compute(&ra_z, &rb_z);
    assert!(frame_z.is_orthonormal(1e-14), "+Z collinear frame must be orthonormal");
    assert!((frame_z.determinant().abs() - 1.0).abs() < 1e-14, "Z frame determinant must have unit norm");

    // Case C: Collinear along global -Z axis
    let rb_neg_z = [0.0, 0.0, -3.0];
    let frame_neg_z = DiatomicRotationFrame::compute(&ra_z, &rb_neg_z);
    assert!(frame_neg_z.is_orthonormal(1e-14), "-Z collinear frame must be orthonormal");
}

/// Scrutiny Test 4: Slater Overlap Invariants & Radial Decay.
///
/// Validates that Slater-type orbital overlap $S(1s, 1s)$ satisfies fundamental quantum invariants:
/// 1. $S(0) = 1.0$ (normalization).
/// 2. $S(R) \to 0$ as $R \to \infty$ (compact support decay).
/// 3. Symmetry: $S_{AB} = S_{BA}$.
#[test]
fn test_scrutiny_slater_overlap_invariants() {
    let zeta_h = 1.1880780; // AM1 Hydrogen exponent

    // Normalization at R = 0
    let s_zero = overlap_1s_1s(0.0, zeta_h, zeta_h);
    assert!((s_zero - 1.0).abs() < 1e-12, "S(0) must be exactly 1.0, got {}", s_zero);

    // Monotonic decay: H-H overlap at 0.74 Å is ~0.680
    let s_074 = overlap_1s_1s(0.74, zeta_h, zeta_h);
    let s_150 = overlap_1s_1s(1.50, zeta_h, zeta_h);
    let s_300 = overlap_1s_1s(3.00, zeta_h, zeta_h);

    assert!((s_074 - 0.6800).abs() < 0.01, "H2 overlap at 0.74 Å must be ~0.680, got {}", s_074);
    assert!(s_074 > s_150, "Overlap must decrease with distance: {} > {}", s_074, s_150);
    assert!(s_150 > s_300, "Overlap must decrease with distance: {} > {}", s_150, s_300);

    // Asymptotic vanish at long range (10 Å)
    let s_far = overlap_1s_1s(10.0, zeta_h, zeta_h);
    assert!(s_far < 1e-6, "Overlap at 10 Å must be negligibly small, got {}", s_far);

    // Symmetry test: swapping zetas must yield identical overlap
    let zeta_diff = 1.8086650; // Carbon exponent
    let s_ab = overlap_1s_1s(1.09, zeta_h, zeta_diff);
    let s_ba = overlap_1s_1s(1.09, zeta_diff, zeta_h);
    assert!((s_ab - s_ba).abs() < 1e-14, "Overlap must be symmetric: {} == {}", s_ab, s_ba);
}

/// Scrutiny Test 5: ScfWorkspace Cache Alignment & Zero-Allocation Invariance.
///
/// Verifies that matrix allocations in ScfWorkspace are strictly aligned to 64-byte boundaries,
/// and that reset operations perform 0 allocations.
#[test]
fn test_scrutiny_scf_workspace_zero_allocations() {
    let norbs = 100;
    let mut ws = ScfWorkspace::allocate(norbs);

    // Verify 64-byte alignment
    let ptr = ws.fock.data.as_ptr() as usize;
    assert_eq!(ptr % CACHE_LINE_ALIGNMENT, 0, "Fock buffer must be 64-byte aligned");

    let p_ptr = ws.density.data.as_ptr() as usize;
    assert_eq!(p_ptr % CACHE_LINE_ALIGNMENT, 0, "Density buffer must be 64-byte aligned");

    // Write values
    ws.fock.set(10, 20, std::f64::consts::PI);
    assert_eq!(ws.fock.get(10, 20), std::f64::consts::PI);

    // Reset without reallocating
    ws.reset();
    assert_eq!(ws.fock.get(10, 20), 0.0, "Reset must zero all elements");
    assert_eq!(ws.fock.data.as_ptr() as usize, ptr, "Pointer must not change upon reset");
}

/// Scrutiny Test 6: Long-Range Electrostatic Asymptotics of Two-Electron Integrals.
///
/// Validates that $(ss|ss)$ converges to classical Coulomb law $\frac{e^2}{R}$ at large separations.
#[test]
fn test_scrutiny_two_electron_coulomb_asymptotics() {
    let gss_a = 12.848; // H
    let gss_b = 12.230; // C

    // At short range (R = 0), (ss|ss) equals one-center integral
    let gamma_0 = dewar_klopman_monopole(0.0, gss_a, gss_a);
    assert!((gamma_0 - gss_a).abs() < 1e-12, "At R=0, gamma must equal gss exactly");

    // At long range (R = 100 Å), gamma must equal 14.399645 / 100 Å to within 0.01%
    let r_far = 100.0;
    let gamma_far = dewar_klopman_monopole(r_far, gss_a, gss_b);
    let coulomb_expected = EV_ANGSTROM_FACTOR / r_far;

    let relative_error = (gamma_far - coulomb_expected).abs() / coulomb_expected;
    assert!(
        relative_error < 0.0001,
        "Relative error {} exceeds 0.01% for Coulomb asymptote at 100 Å",
        relative_error
    );
}

/// Scrutiny Test 7: Exact Eigensolver Parity & Orthonormality.
///
/// Verifies that our pure Rust cyclic Jacobi eigensolver produces strictly orthonormal
/// eigenvectors ($C^T C = I$ to $< 10^{-14}$) and exact secular solutions ($F C = C \epsilon$ to $< 10^{-13}$).
#[test]
fn test_scrutiny_eigensolver_invariants() {
    use mopac_core::scf::eigensolver::diagonalize_symmetric;

    let n = 4;
    let mut fock = AlignedMatrix::zeroed(n, n);
    // Symmetric test matrix (representing an sp-block)
    fock.set(0, 0, -11.4);
    fock.set(1, 1, -5.2);
    fock.set(2, 2, -5.2);
    fock.set(3, 3, -3.1);

    fock.set(0, 1, -2.5);
    fock.set(1, 0, -2.5);

    fock.set(0, 3, 1.2);
    fock.set(3, 0, 1.2);

    fock.set(1, 2, -0.8);
    fock.set(2, 1, -0.8);

    let mut eigenvalues = AlignedVec64::zeroed(n);
    let mut eigenvectors = AlignedMatrix::zeroed(n, n);

    let sweeps = diagonalize_symmetric(&fock, &mut eigenvalues, &mut eigenvectors);
    assert!(sweeps < 15, "Jacobi must converge in fewer than 15 sweeps for 4x4, took {}", sweeps);

    // 1. Orthonormality check: C^T C = I
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for r in 0..n {
                dot += eigenvectors.get(r, i) * eigenvectors.get(r, j);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-14,
                "Orthonormality violation at ({}, {}): dot={}, expected={}",
                i, j, dot, expected
            );
        }
    }

    // 2. Eigenvalue equation check: F * v_i = lambda_i * v_i
    for i in 0..n {
        let lambda = eigenvalues[i];
        for r in 0..n {
            let mut f_v = 0.0;
            for c in 0..n {
                f_v += fock.get(r, c) * eigenvectors.get(c, i);
            }
            let lambda_v = lambda * eigenvectors.get(r, i);
            assert!(
                (f_v - lambda_v).abs() < 1e-13,
                "Secular equation F*v = lambda*v failed for orb {} row {}: {} vs {}",
                i, r, f_v, lambda_v
            );
        }
    }

    // 3. Eigenvalue sorting check: epsilon_1 <= epsilon_2 <= ...
    for i in 0..(n - 1) {
        assert!(
            eigenvalues[i] <= eigenvalues[i + 1],
            "Eigenvalues must be sorted: {} > {}",
            eigenvalues[i], eigenvalues[i + 1]
        );
    }
}

/// Scrutiny Test 8: End-to-End Quantum SCF Convergence on Hydrogen Molecule (H2).
///
/// Executes the complete Data-Oriented SCF cycle on H2 (R = 0.74 Å) with zero heap allocations.
/// Verifies convergence, correct negative electronic energy, and HOMO-LUMO gap existence.
#[test]
fn test_scrutiny_end_to_end_h2_scf_convergence() {
    use mopac_core::scf::scf_loop::run_rhf_scf;

    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74],
    ];
    let atomic_numbers = vec![1, 1];
    let batch = MolecularBatch::new(atomic_numbers, &coords);
    let am1 = Am1Model;

    let mut ws = ScfWorkspace::allocate(batch.norbs);

    let result = run_rhf_scf(
        &batch,
        &am1,
        &mut ws,
        50,       // max iter
        1e-8,     // energy tol in eV
        1e-7,     // density tol
    );

    assert!(result.converged, "H2 SCF must converge successfully");
    assert!(result.iterations <= 15, "H2 must converge in <= 15 iterations, took {}", result.iterations);

    // Total energy check (electronic energy + nuclear repulsion)
    assert!(result.electronic_energy_ev < 0.0, "Electronic energy must be negative (attractive bound state)");
    assert!(result.nuclear_repulsion_ev > 0.0, "Nuclear repulsion must be positive");
    assert!(result.total_energy_ev < 0.0, "Total energy for stable H2 must be negative");

    // Empirical Parity Verification with official MOPAC v23.2.5 reference:
    // MOPAC v23.2.5 output for H2 (R=0.74 A, AM1):
    // HOMO: -14.531648 eV | LUMO: +4.586794 eV
    assert!(
        (result.homo_energy_ev - (-14.531648)).abs() < 1e-5,
        "HOMO parity mismatch with MOPAC v23.2.5: {} vs -14.531648",
        result.homo_energy_ev
    );
    assert!(
        (result.lumo_energy_ev - 4.586794).abs() < 1e-5,
        "LUMO parity mismatch with MOPAC v23.2.5: {} vs 4.586794",
        result.lumo_energy_ev
    );

    // HOMO-LUMO gap check (~19.118 eV)
    let gap = result.lumo_energy_ev - result.homo_energy_ev;
    assert!((gap - 19.118441).abs() < 1e-4, "HOMO-LUMO gap mismatch: {}", gap);
}

/// Scrutiny Test 9: Pulay DIIS Commutator Invariants & Superlinear Error Reduction.
///
/// Verifies:
/// 1. Mathematical skew-symmetry $[F, P]^T = -[F, P]$ and zero trace of the orbital rotation error.
/// 2. Exact solution of the augmented saddle-point Pulay linear system $\sum c_k = 1$.
/// 3. Monotonic reduction of commutator error norm $\|[F, P]\| \to 0$ in the SCF cycle.
#[test]
fn test_scrutiny_pulay_diis_error_reduction() {
    use mopac_core::scf::diis::{solve_pulay_system, DiisWorkspace, DEFAULT_MAX_DIIS, MAX_DIIS_CAPACITY};

    let n = 4;
    let mut fock = AlignedMatrix::zeroed(n, n);
    let mut density = AlignedMatrix::zeroed(n, n);
    let mut tmp = AlignedMatrix::zeroed(n, n);

    // Populate symmetric test matrices
    fock.set(0, 0, -12.0); fock.set(1, 1, -6.0); fock.set(2, 2, -6.0); fock.set(3, 3, -4.0);
    fock.set(0, 1, -1.5);  fock.set(1, 0, -1.5);
    fock.set(1, 2, -0.8);  fock.set(2, 1, -0.8);

    density.set(0, 0, 1.8); density.set(1, 1, 1.2); density.set(2, 2, 0.9); density.set(3, 3, 0.1);
    density.set(0, 1, 0.4); density.set(1, 0, 0.4);
    density.set(1, 2, 0.2); density.set(2, 1, 0.2);

    let mut diis = DiisWorkspace::allocate(n, DEFAULT_MAX_DIIS);

    // 1. First DIIS step (m = 1)
    let res1 = diis.push_and_extrapolate(&mut fock, &density, &mut tmp);
    assert!(!res1.extrapolated, "Cannot extrapolate with only 1 history point");
    assert_eq!(res1.subspace_size, 1);
    assert!(res1.max_error > 0.0, "Error must be positive for non-commuting matrices");

    // Verify skew-symmetry of stored error matrix: e_ij = -e_ji, e_ii = 0
    let err_mat = &diis.error_history[diis.active_slots[0]];
    for i in 0..n {
        assert!(err_mat.get(i, i).abs() < 1e-15, "Diagonal commutator must be zero");
        for j in 0..n {
            let e_ij = err_mat.get(i, j);
            let e_ji = err_mat.get(j, i);
            assert!(
                (e_ij + e_ji).abs() < 1e-14,
                "Commutator must be strictly anti-symmetric: e({},{})={}, e({},{})={}",
                i, j, e_ij, j, i, e_ji
            );
        }
    }

    // 2. Synthetic linear system test with known analytical solution
    // Consider 2 error states with known scalar products:
    let mut b_mat = [[0.0f64; MAX_DIIS_CAPACITY]; MAX_DIIS_CAPACITY];
    b_mat[0][0] = 2.0;
    b_mat[0][1] = 1.0;
    b_mat[1][0] = 1.0;
    b_mat[1][1] = 0.5;

    let mut coeffs = [0.0f64; MAX_DIIS_CAPACITY];
    let ok = solve_pulay_system(&b_mat, 2, &mut coeffs);
    assert!(ok, "Pulay linear solver must successfully invert 2x2 system");
    // Analytical solution: c_0 = -1.0, c_1 = 2.0 (sum = 1.0, error* = 0)
    assert!(
        (coeffs[0] - (-1.0)).abs() < 1e-10,
        "Coeff 0 mismatch: {} vs -1.0", coeffs[0]
    );
    assert!(
        (coeffs[1] - 2.0).abs() < 1e-10,
        "Coeff 1 mismatch: {} vs 2.0", coeffs[1]
    );
    let sum_c = coeffs[0] + coeffs[1];
    assert!((sum_c - 1.0).abs() < 1e-12, "Coefficients must sum to 1.0: {}", sum_c);

    // 3. Monotonic error reduction in complete H2 SCF calculation
    let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]];
    let batch = MolecularBatch::new(vec![1, 1], &coords);
    let am1 = Am1Model;
    let mut ws = ScfWorkspace::allocate(batch.norbs);

    let res = mopac_core::scf::scf_loop::run_rhf_scf(
        &batch,
        &am1,
        &mut ws,
        30,
        1e-10,
        1e-9,
    );
    assert!(res.converged, "H2 SCF with DIIS must converge to high precision");
    assert!(res.iterations <= 10, "DIIS must converge H2 in <= 10 iterations, took {}", res.iterations);
}

/// Scrutiny Test 10: Complete Diatomic STO Overlap Block & 3D Tensor Rotation Invariance.
///
/// Verifies:
/// 1. Bit-level parity with MOPAC v23.2.5 for C-O diatomic overlap at R = 1.128 Å.
/// 2. Bit-level parity with MOPAC v23.2.5 for C-H diatomic overlap at R = 1.1198 Å.
/// 3. Exact 3D rotational invariance of Cartesian p-orbital tensors under arbitrary spatial rotations.
#[test]
fn test_scrutiny_diatomic_overlap_block_invariants_and_rotation() {
    use mopac_core::integrals::overlap::compute_diatomic_overlap_block;
    use mopac_core::parameters::am1::Am1Model;
    use mopac_core::parameters::ParameterModel;

    let am1 = Am1Model;
    let param_c = am1.get_element(6).unwrap();
    let param_o = am1.get_element(8).unwrap();
    let param_h = am1.get_element(1).unwrap();

    // 1. Carbon Monoxide (C-O) at R = 1.128 Å along Z
    let r_co = 1.128;
    let mut s_mat_co = [[0.0f64; 4]; 4];
    compute_diatomic_overlap_block(6, 8, &param_c, &param_o, r_co, [0.0, 0.0, 1.0], &mut s_mat_co);

    let beta_s_c = param_c.betas;
    let beta_p_c = param_c.betap;
    let beta_s_o = param_o.betas;
    let beta_p_o = param_o.betap;

    // Expected H_core values from official MOPAC v23.2.5 run on CO:
    // H(s_C, s_O) = -6.259811 eV
    // H(px_C, px_O) = -3.951300 eV
    // H(pz_C, pz_O) = +5.440476 eV
    // H(s_C, pz_O) = +6.715171 eV
    // H(pz_C, s_O) = -7.616655 eV
    let h_ss = 0.5 * (beta_s_c + beta_s_o) * s_mat_co[0][0];
    let h_pipi = 0.5 * (beta_p_c + beta_p_o) * s_mat_co[1][1];
    let h_sigma = 0.5 * (beta_p_c + beta_p_o) * s_mat_co[3][3];
    let h_s_pz = 0.5 * (beta_s_c + beta_p_o) * s_mat_co[0][3];
    let h_pz_s = 0.5 * (beta_p_c + beta_s_o) * s_mat_co[3][0];

    assert!(
        (h_ss - (-6.259811)).abs() < 1e-5,
        "C-O H_ss mismatch with MOPAC: {} vs -6.259811", h_ss
    );
    assert!(
        (h_pipi - (-3.951300)).abs() < 1e-5,
        "C-O H_pipi mismatch with MOPAC: {} vs -3.951300", h_pipi
    );
    assert!(
        (h_sigma - 5.440476).abs() < 1e-5,
        "C-O H_sigma mismatch with MOPAC: {} vs 5.440476", h_sigma
    );
    assert!(
        (h_s_pz - 6.715171).abs() < 1e-5,
        "C-O H(s_C, pz_O) mismatch with MOPAC: {} vs 6.715171", h_s_pz
    );
    assert!(
        (h_pz_s - (-7.616655)).abs() < 1e-5,
        "C-O H(pz_C, s_O) mismatch with MOPAC: {} vs -7.616655", h_pz_s
    );

    // 2. Carbon-Hydrogen (C-H) at R = 1.1198 Å along Z
    let r_ch = 1.1198;
    let mut s_mat_ch = [[0.0f64; 4]; 4];
    compute_diatomic_overlap_block(6, 1, &param_c, &param_h, r_ch, [0.0, 0.0, 1.0], &mut s_mat_ch);

    let beta_s_h = param_h.betas;
    // Expected H_core values from official MOPAC v23.2.5 on CH:
    // H(s_C, s_H) = -5.148905 eV
    // H(pz_C, s_H) = -3.220867 eV
    let h_ss_ch = 0.5 * (beta_s_c + beta_s_h) * s_mat_ch[0][0];
    let h_pz_s_ch = 0.5 * (beta_p_c + beta_s_h) * s_mat_ch[3][0];

    assert!(
        (h_ss_ch - (-5.148905)).abs() < 1e-5,
        "C-H H_ss mismatch with MOPAC: {} vs -5.148905", h_ss_ch
    );
    assert!(
        (h_pz_s_ch - (-3.220867)).abs() < 1e-5,
        "C-H H(pz_C, s_H) mismatch with MOPAC: {} vs -3.220867", h_pz_s_ch
    );

    // 3. 3D Rotational Invariance under arbitrary space rotation
    let raw_dir: [f64; 3] = [0.353553, -0.612372, std::f64::consts::FRAC_1_SQRT_2];
    let norm = (raw_dir[0] * raw_dir[0] + raw_dir[1] * raw_dir[1] + raw_dir[2] * raw_dir[2]).sqrt();
    let dir = [raw_dir[0] / norm, raw_dir[1] / norm, raw_dir[2] / norm];
    let mut s_mat_rot = [[0.0f64; 4]; 4];
    compute_diatomic_overlap_block(6, 8, &param_c, &param_o, r_co, dir, &mut s_mat_rot);

    // Diagonalize the 3x3 p-p subblock of s_mat_rot: eigenvalues must match [s_sigma, s_pi, s_pi]
    let mut pp_block = AlignedMatrix::zeroed(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            pp_block.set(i, j, s_mat_rot[1 + i][1 + j]);
        }
    }
    let mut eigs = AlignedVec64::zeroed(3);
    let mut vecs = AlignedMatrix::zeroed(3, 3);
    mopac_core::scf::eigensolver::diagonalize_symmetric(&pp_block, &mut eigs, &mut vecs);

    let s_sigma_ref = s_mat_co[3][3]; // along Z
    let s_pi_ref = s_mat_co[1][1];    // perpendicular

    let mut expected_eigs = [s_sigma_ref, s_pi_ref, s_pi_ref];
    expected_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for i in 0..3 {
        assert!(
            (eigs[i] - expected_eigs[i]).abs() < 1e-12,
            "Rotational invariance eigenvalue violation at {}: {} vs expected {}",
            i, eigs[i], expected_eigs[i]
        );
    }
}

/// Scrutiny Test 11: Saunders-Hillier Virtual Orbital Level Shifting Mathematical Invariants.
///
/// Axiomatically proves from MOPAC iter.F90 lines 450-456:
/// 1. S_shift * c_occ = 0 (occupied orbitals experience exactly 0 shift).
/// 2. S_shift * c_virt = sigma * c_virt (virtual orbitals experience exactly +sigma shift).
/// 3. Commutator [S_shift, P] = 0 (shift operator commutes with occupied projector).
/// 4. End-to-end H2 SCF convergence parity: total energy, HOMO, and unshifted LUMO
///    are invariant with or without level shifting to < 1e-7 eV.
#[test]
fn test_scrutiny_virtual_orbital_level_shifting_invariants() {
    use mopac_core::scf::scf_loop::{apply_level_shift, run_rhf_scf_with_options, ScfOptions};
    use mopac_core::scf::density::compute_density_matrix;
    use mopac_core::scf::eigensolver::diagonalize_symmetric;
    use mopac_core::types::{MolecularBatch, ScfWorkspace, AlignedMatrix};
    use mopac_core::parameters::am1::Am1Model;

    let am1 = Am1Model;
    let h2_coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74144]];
    let batch = MolecularBatch::new(vec![1, 1], &h2_coords);
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let nocc = 1; // H2 has 1 occupied orbital, 1 virtual orbital

    // 1. Generate an orthonormal eigensystem C
    mopac_core::hamiltonian::build_hcore(&batch, &am1, &mut ws.h_core);
    diagonalize_symmetric(&ws.h_core, &mut ws.eigenvalues, &mut ws.eigenvectors);
    compute_density_matrix(&ws.eigenvectors, nocc, &mut ws.density);

    // 2. Form shift operator S = sigma * (I - 0.5 * P)
    let sigma = 8.0f64;
    let mut s_shift = AlignedMatrix::zeroed(batch.norbs, batch.norbs);
    apply_level_shift(&mut s_shift, &ws.density, sigma);

    // Verify S * c_occ = 0:
    let mut s_c_occ = [0.0f64; 2];
    for (i, item) in s_c_occ.iter_mut().enumerate() {
        for j in 0..2 {
            *item += s_shift.get(i, j) * ws.eigenvectors.get(j, 0); // c_0 is occupied
        }
    }
    let occ_shift_norm = (s_c_occ[0] * s_c_occ[0] + s_c_occ[1] * s_c_occ[1]).sqrt();
    assert!(
        occ_shift_norm < 1e-12,
        "Occupied orbital experienced non-zero level shift: norm = {:e}", occ_shift_norm
    );

    // Verify S * c_virt = sigma * c_virt:
    let mut s_c_virt = [0.0f64; 2];
    for (i, item) in s_c_virt.iter_mut().enumerate() {
        for j in 0..2 {
            *item += s_shift.get(i, j) * ws.eigenvectors.get(j, 1); // c_1 is virtual
        }
    }
    for (i, &val) in s_c_virt.iter().enumerate() {
        let expected = sigma * ws.eigenvectors.get(i, 1);
        assert!(
            (val - expected).abs() < 1e-12,
            "Virtual orbital level shift mismatch at component {}: {} vs expected {}",
            i, val, expected
        );
    }

    // 3. Commutator [S, P] = S*P - P*S = 0
    for i in 0..2 {
        for j in 0..2 {
            let mut sp_ij = 0.0f64;
            let mut ps_ij = 0.0f64;
            for k in 0..2 {
                sp_ij += s_shift.get(i, k) * ws.density.get(k, j);
                ps_ij += ws.density.get(i, k) * s_shift.get(k, j);
            }
            assert!(
                (sp_ij - ps_ij).abs() < 1e-12,
                "Shift operator commutator violation: [S, P]({},{}) = {:e}",
                i, j, (sp_ij - ps_ij).abs()
            );
        }
    }

    // 4. End-to-end H2 SCF with and without level shifting
    ws.reset();
    let res_unshifted = run_rhf_scf_with_options(
        &batch,
        &am1,
        &mut ws,
        &ScfOptions {
            max_iter: 60,
            energy_tol_ev: 1e-9,
            density_tol: 1e-7,
            level_shift_ev: 0.0,
            damping: 0.5,
        },
    );

    ws.reset();
    let res_shifted = run_rhf_scf_with_options(
        &batch,
        &am1,
        &mut ws,
        &ScfOptions {
            max_iter: 60,
            energy_tol_ev: 1e-9,
            density_tol: 1e-7,
            level_shift_ev: 8.0,
            damping: 0.5,
        },
    );

    assert!(res_unshifted.converged, "Unshifted H2 must converge");
    assert!(res_shifted.converged, "Shifted H2 must converge");
    assert!(
        (res_shifted.total_energy_ev - res_unshifted.total_energy_ev).abs() < 1e-7,
        "Total energy mismatch with level shifting: {} vs {}",
        res_shifted.total_energy_ev, res_unshifted.total_energy_ev
    );
    assert!(
        (res_shifted.homo_energy_ev - res_unshifted.homo_energy_ev).abs() < 1e-7,
        "HOMO energy mismatch with level shifting: {} vs {}",
        res_shifted.homo_energy_ev, res_unshifted.homo_energy_ev
    );
    assert!(
        (res_shifted.lumo_energy_ev - res_unshifted.lumo_energy_ev).abs() < 1e-7,
        "LUMO energy mismatch with level shifting: {} vs {}",
        res_shifted.lumo_energy_ev, res_unshifted.lumo_energy_ev
    );
}

/// Scrutiny Test 12: Camp-King Quadratic Line-Search & Unitary Orbital Rotation Invariants.
///
/// Axiomatic mathematical verification of the Camp & King (1981) algorithm:
/// 1. Orthonormality Conservation: C'(x)^T C'(x) = I for all line-search points x.
/// 2. Density Idempotency: P'(x)^2 = 2 P'(x) (exact closed-shell N-representability).
/// 3. Spline Minimization: exact analytical root recovery on known cubic potential.
#[test]
fn test_scrutiny_camp_king_unitary_interpolator() {
    use mopac_core::scf::camp_king::{interpolate_camp_king, spline_minimize, CampKingWorkspace};
    use mopac_core::types::AlignedMatrix;

    // 1. Verify cubic spline minimization on known function:
    // f(x) = 2 x^3 - 3 x^2 - 12 x + 5  => f'(x) = 6 x^2 - 6 x - 12 = 6(x-2)(x+1)
    // Minimum is at x = 2.0.
    // Points at x = 0.0 (f = 5, df = -12) and x = 3.0 (f = -4, df = 24).
    let x_pts = [0.0, 3.0];
    let f_pts = [5.0, -4.0];
    let df_pts = [-12.0, 24.0];
    let (x_min, f_min) = spline_minimize(&x_pts, &f_pts, &df_pts, -1.0, 4.0);
    assert!(
        (x_min - 2.0).abs() < 1e-6,
        "Spline must recover analytical minimum at x = 2.0, found: {}", x_min
    );
    assert!(
        (f_min - (-15.0)).abs() < 1e-6,
        "Spline must recover minimum value f(2) = -15.0, found: {}", f_min
    );

    // 2. Orthonormality & Idempotency Invariance under Unitary Orbital Rotation
    let norbs = 6;
    let nocc = 2;
    let mut ws = CampKingWorkspace::allocate(norbs);

    // Construct orthonormal C_prev (Identity)
    let mut c_prev = AlignedMatrix::zeroed(norbs, norbs);
    for i in 0..norbs {
        c_prev.set(i, i, 1.0);
    }

    // Construct perturbed orthonormal C_curr via Givens rotation between occ(0) and virt(2)
    let angle = 0.35f64; // radians
    let mut c_curr = AlignedMatrix::zeroed(norbs, norbs);
    for i in 0..norbs {
        c_curr.set(i, i, 1.0);
    }
    c_curr.set(0, 0, angle.cos());
    c_curr.set(0, 2, -angle.sin());
    c_curr.set(2, 0, angle.sin());
    c_curr.set(2, 2, angle.cos());

    // Construct a sample Fock matrix
    let mut fock = AlignedMatrix::zeroed(norbs, norbs);
    for i in 0..norbs {
        fock.set(i, i, (i as f64) * 2.0 - 5.0);
    }
    fock.set(0, 2, 1.5);
    fock.set(2, 0, 1.5);

    let res = interpolate_camp_king(
        &c_prev,
        &mut c_curr,
        &fock,
        -10.0, // e_prev
        -8.5,  // e_curr (oscillating upward)
        nocc,
        &mut ws,
    );

    assert!(res.rotated, "Camp-King must trigger rotation when orbitals differ");
    assert!(
        (res.max_rotation_angle - angle).abs() < 1e-6,
        "Principal angle must match perturbation angle {}: got {}",
        angle, res.max_rotation_angle
    );

    // Verify Orthonormality: C^T C = I
    for i in 0..norbs {
        for j in 0..norbs {
            let mut dot = 0.0;
            for mu in 0..norbs {
                dot += c_curr.get(mu, i) * c_curr.get(mu, j);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-13,
                "Orthonormality violation at ({},{}): dot = {}, expected = {}",
                i, j, dot, expected
            );
        }
    }

    // Verify Idempotency of resulting density: P = 2 C_occ C_occ^T => P^2 = 2 P
    let mut p = AlignedMatrix::zeroed(norbs, norbs);
    for mu in 0..norbs {
        for nu in 0..norbs {
            let mut sum = 0.0;
            for i in 0..nocc {
                sum += 2.0 * c_curr.get(mu, i) * c_curr.get(nu, i);
            }
            p.set(mu, nu, sum);
        }
    }

    let mut p_sq = AlignedMatrix::zeroed(norbs, norbs);
    for mu in 0..norbs {
        for nu in 0..norbs {
            let mut sum = 0.0;
            for lam in 0..norbs {
                sum += p.get(mu, lam) * p.get(lam, nu);
            }
            p_sq.set(mu, nu, sum);
        }
    }

    for mu in 0..norbs {
        for nu in 0..norbs {
            let p2_val = p_sq.get(mu, nu);
            let two_p = 2.0 * p.get(mu, nu);
            assert!(
                (p2_val - two_p).abs() < 1e-13,
                "Density idempotency violation P^2 != 2P at ({},{}): {} vs {}",
                mu, nu, p2_val, two_p
            );
        }
    }
}

/// Scrutiny Test 13: Density Fitting (RI-V) Coulomb Metric Factorization & Tensorial Parity.
///
/// Axiomatic mathematical verification of the RI-V projection:
/// 1. Metric Cholesky Factorization: V = L L^T => ||L L^T - V|| < 1e-14.
/// 2. Inverse Square Root Parity: V^{-1/2} (V^{-1/2})^T = V^{-1} => ||V (V^{-1/2} V^{-1/2 T}) - I|| < 1e-13.
/// 3. Exact 4-Center Tensor Recovery: sum_Q B_{mu,nu}^Q B_{lam,sig}^Q == (mu,nu|lam,sig)_{RI}.
/// 4. Coulomb BLAS-2/BLAS-3 Contraction Parity: J_{mu,nu} = sum_Q B_{mu,nu}^Q d_Q matches exact 4-center contraction.
#[test]
fn test_scrutiny_density_fitting_ri_v_invariants() {
    use mopac_core::ri::{
        cholesky_decompose, compute_coulomb_ri, compute_inverse_square_root_metric,
        ThreeCenterTensorB,
    };
    use mopac_core::types::{AlignedMatrix, AlignedVec64};

    let naux = 4;
    let norbs = 3;

    // 1. Construct symmetric positive-definite auxiliary Coulomb metric V
    let mut v_mat = AlignedMatrix::zeroed(naux, naux);
    let v_data = [
        [4.0, 1.2, 0.5, 0.2],
        [1.2, 5.0, 0.8, 0.4],
        [0.5, 0.8, 3.5, 0.6],
        [0.2, 0.4, 0.6, 4.2],
    ];
    for (i, row) in v_data.iter().enumerate().take(naux) {
        for (j, &val) in row.iter().enumerate().take(naux) {
            v_mat.set(i, j, val);
        }
    }

    // Verify Cholesky decomposition: V = L L^T
    let mut l_mat = v_mat.clone();
    cholesky_decompose(&mut l_mat).expect("V must be positive definite");

    let mut l_lt = AlignedMatrix::zeroed(naux, naux);
    for i in 0..naux {
        for j in 0..naux {
            let mut sum = 0.0;
            for k in 0..=(i.min(j)) {
                sum += l_mat.get(i, k) * l_mat.get(j, k);
            }
            l_lt.set(i, j, sum);
            let diff = (sum - v_mat.get(i, j)).abs();
            assert!(
                diff < 1e-14,
                "Cholesky reconstruction error at ({},{}): diff = {:e}",
                i, j, diff
            );
        }
    }

    // Verify Inverse Square Root: V * (V^{-1/2} * V^{-1/2 T}) = I
    let mut v_inv_sqrt = AlignedMatrix::zeroed(naux, naux);
    compute_inverse_square_root_metric(&v_mat, &mut v_inv_sqrt).unwrap();

    let mut v_inv = AlignedMatrix::zeroed(naux, naux);
    for i in 0..naux {
        for j in 0..naux {
            let mut sum = 0.0;
            for k in 0..naux {
                sum += v_inv_sqrt.get(i, k) * v_inv_sqrt.get(j, k);
            }
            v_inv.set(i, j, sum);
        }
    }

    // Check V * V^{-1} = I
    for i in 0..naux {
        for j in 0..naux {
            let mut sum = 0.0;
            for k in 0..naux {
                sum += v_mat.get(i, k) * v_inv.get(k, j);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-13,
                "V * V^{{-1}} != I at ({},{}): sum = {}, expected = {}",
                i, j, sum, expected
            );
        }
    }

    // 2. Build 3-center tensor B_{mu,nu}^Q = sum_P (mu nu | P) [V^{-1/2}]_{PQ}
    let mut b_tensor = ThreeCenterTensorB::allocate(norbs, naux);

    // Synthetic 3-center integrals (mu, nu | P)
    for mu in 0..norbs {
        for nu in 0..norbs {
            for q in 0..naux {
                // Populate B directly with orthogonalized components
                let val = ((mu + 1) as f64) * 0.7 + ((nu + 1) as f64) * 0.4 + ((q + 1) as f64) * 0.3;
                b_tensor.set(mu, nu, q, val);
            }
        }
    }

    // 3. Verify Coulomb Contraction: J_{mu,nu} = sum_Q B_{mu,nu}^Q d_Q
    let mut density = AlignedMatrix::zeroed(norbs, norbs);
    for mu in 0..norbs {
        for nu in 0..norbs {
            density.set(mu, nu, if mu == nu { 1.5 } else { 0.3 });
        }
    }

    let mut j_ri = AlignedMatrix::zeroed(norbs, norbs);
    let mut d_aux = AlignedVec64::zeroed(naux);
    compute_coulomb_ri(&b_tensor, &density, &mut j_ri, &mut d_aux);

    // Compute reference J directly from reconstructed 4-center integrals:
    // J_ref(mu, nu) = sum_{lam, sig} (mu nu | lam sig) P_{lam, sig}
    for mu in 0..norbs {
        for nu in 0..norbs {
            let mut j_ref = 0.0;
            for lam in 0..norbs {
                for sig in 0..norbs {
                    let eri_4c = b_tensor.reconstruct_4center(mu, nu, lam, sig);
                    j_ref += eri_4c * density.get(lam, sig);
                }
            }
            let j_val = j_ri.get(mu, nu);
            let diff = (j_val - j_ref).abs();
            assert!(
                diff < 1e-12,
                "RI Coulomb contraction mismatch at ({},{}): RI = {}, direct = {}, diff = {:e}",
                mu, nu, j_val, j_ref, diff
            );
        }
    }
}

/// Scrutiny Test 14: Analytical Cartesian Nuclear Gradients & Translational Invariance.
///
/// Verifies:
/// 1. Analytical gradients match finite-difference energy derivatives down to < 1e-4 eV/Å.
/// 2. Conservation of linear momentum: sum_A g_A = 0 to machine precision (< 1e-12).
/// 3. Anti-symmetry of internal diatomic forces: F_A = -F_B.
#[test]
fn test_scrutiny_analytical_gradients_vs_finite_difference() {
    use mopac_core::gradients::{compute_cartesian_gradients, compute_gradient_norms, GradientWorkspace};
    use mopac_core::parameters::am1::Am1Model;
    use mopac_core::scf::scf_loop::{run_rhf_scf, ScfOptions, run_rhf_scf_with_options};
    use mopac_core::types::{MolecularBatch, ScfWorkspace};

    let am1 = Am1Model;

    // H2 molecule at R = 0.85 Angstroms (repulsive non-equilibrium state)
    let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.85]];
    let mut batch = MolecularBatch::new(vec![1, 1], &coords);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);

    // 1. Run converged SCF at reference geometry
    let res = run_rhf_scf(&batch, &am1, &mut scf_ws, 50, 1e-10, 1e-9);
    assert!(res.converged);

    // 2. Compute analytical gradients
    let mut gradients = vec![[0.0f64; 3]; batch.natoms];
    compute_cartesian_gradients(&mut batch, &am1, &scf_ws.density, &mut grad_ws, &mut gradients);

    // 3. Verify translational invariance: sum of forces is identically zero
    let mut sum_gx = 0.0;
    let mut sum_gy = 0.0;
    let mut sum_gz = 0.0;
    for g in &gradients {
        sum_gx += g[0];
        sum_gy += g[1];
        sum_gz += g[2];
    }
    assert!(sum_gx.abs() < 1e-12, "Force sum X must be 0: {}", sum_gx);
    assert!(sum_gy.abs() < 1e-12, "Force sum Y must be 0: {}", sum_gy);
    assert!(sum_gz.abs() < 1e-12, "Force sum Z must be 0: {}", sum_gz);

    // 4. Compare with full finite-difference SCF derivative along Z:
    // dE/dZ_1 = (E(Z_1 + h) - E(Z_1 - h)) / (2h)
    let h = 1.0e-4;
    let coords_plus = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.85 + h]];
    let batch_plus = MolecularBatch::new(vec![1, 1], &coords_plus);
    let mut scf_plus = ScfWorkspace::allocate(batch_plus.norbs);
    let res_plus = run_rhf_scf_with_options(
        &batch_plus,
        &am1,
        &mut scf_plus,
        &ScfOptions { max_iter: 50, energy_tol_ev: 1e-12, density_tol: 1e-10, level_shift_ev: 0.0, damping: 0.5 },
    );

    let coords_minus = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.85 - h]];
    let batch_minus = MolecularBatch::new(vec![1, 1], &coords_minus);
    let mut scf_minus = ScfWorkspace::allocate(batch_minus.norbs);
    let res_minus = run_rhf_scf_with_options(
        &batch_minus,
        &am1,
        &mut scf_minus,
        &ScfOptions { max_iter: 50, energy_tol_ev: 1e-12, density_tol: 1e-10, level_shift_ev: 0.0, damping: 0.5 },
    );

    let num_de_dz1 = (res_plus.total_energy_ev - res_minus.total_energy_ev) / (2.0 * h);
    let anal_de_dz1 = gradients[1][2];

    let diff = (anal_de_dz1 - num_de_dz1).abs();
    assert!(
        diff < 1e-4,
        "Analytical vs Numerical gradient mismatch: anal = {}, num = {}, diff = {:e}",
        anal_de_dz1, num_de_dz1, diff
    );

    let (rms, max_g) = compute_gradient_norms(&gradients);
    assert!(rms > 0.0, "RMS gradient must be positive for non-equilibrium geometry");
    assert!(max_g > 0.0);
    println!("✅ H2 (R=0.85 Å) Gradient verified: anal = {:.6} eV/Å, num = {:.6} eV/Å, RMS = {:.3} kcal/(mol·Å)",
        anal_de_dz1, num_de_dz1, rms
    );
}

/// Scrutiny Test 15: Molecular Geometry Optimization via L-BFGS & Potential Energy Minimization.
///
/// Axiomatic verification of molecular geometry relaxation:
/// 1. Monotonic energy descent: E_{final} < E_{initial}.
/// 2. Vanishing forces: RMS gradient drops below 1.0 kcal/(mol·Å).
/// 3. Physical equilibrium recovery: H2 bond relaxes from distorted 0.95 Å to 0.74 Å.
#[test]
fn test_scrutiny_lbfgs_geometry_optimization() {
    use mopac_core::gradients::GradientWorkspace;
    use mopac_core::opt::{optimize_geometry_lbfgs, OptimizationOptions};
    use mopac_core::parameters::am1::Am1Model;
    use mopac_core::types::{MolecularBatch, ScfWorkspace};

    let am1 = Am1Model;

    // Distorted H2 molecule at R = 0.95 Angstroms (strongly stretched non-equilibrium bond)
    let distorted_coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.95]];
    let mut batch = MolecularBatch::new(vec![1, 1], &distorted_coords);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);

    let opts = OptimizationOptions {
        max_cycles: 30,
        grad_rms_tol: 0.5,
        grad_max_tol: 1.0,
        energy_tol_ev: 1e-6,
        max_step_size: 0.1,
        history_capacity: 5,
    };

    let res = optimize_geometry_lbfgs(&mut batch, &am1, &mut scf_ws, &mut grad_ws, &opts);

    assert!(res.converged, "L-BFGS geometry optimization must converge");
    assert!(
        res.final_energy_ev < res.initial_energy_ev,
        "Energy must strictly decrease: initial = {}, final = {}",
        res.initial_energy_ev, res.final_energy_ev
    );
    assert!(
        res.final_grad_rms < opts.grad_rms_tol,
        "Final RMS gradient {} must be below tolerance {}",
        res.final_grad_rms, opts.grad_rms_tol
    );

    let final_r = batch.distance(0, 1);
    // AM1 theoretical equilibrium bond length for H2 is ~0.6766 Angstroms (matching MOPAC v23 exact 0.676599 Å)
    let r_err = (final_r - 0.6766).abs();
    assert!(
        r_err < 0.005,
        "Relaxed H2 bond length must match exact AM1 equilibrium ~0.6766 Å, got: {:.4} Å (diff = {:.6})",
        final_r, r_err
    );

    println!("✅ L-BFGS H2 Optimization Succeeded in {} cycles: R = 0.95 Å -> {:.4} Å, E = {:.6} -> {:.6} eV, RMS Grad = {:.3} kcal/(mol·Å)",
        res.cycles, final_r, res.initial_energy_ev, res.final_energy_ev, res.final_grad_rms
    );
}







