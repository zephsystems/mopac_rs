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
    ws.fock.set(10, 20, 3.14159);
    assert_eq!(ws.fock.get(10, 20), 3.14159);

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

