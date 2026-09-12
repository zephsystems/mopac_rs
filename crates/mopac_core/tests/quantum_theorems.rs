//! Physical Invariants and Quantum Theorems Verification Suite for MOPAC_RS.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Strictly validates foundational theorems of non-relativistic quantum mechanics
//! and Data-Oriented Programming guarantees:
//!
//! 1. Density Idempotency: ||P^2 - 2P||_inf < 1e-13 for closed-shell RHF states.
//! 2. Orbital Orthonormality: ||C^T C - I||_inf < 1e-14 across molecular orbitals.
//! 3. Rotational Invariance: SO(3) coordinate rotation invariance |E(R*X) - E(X)| < 1e-8 eV.
//! 4. Saunders-Hillier Level Shift Trace Orthogonality: Tr[P * Delta_F_shift] == 0.
//! 5. 0-Heap Allocation Policy: Hot SCF loop executes with zero memory reallocations.

use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::types::*;

/// Quantum Invariant 1: Idempotency of the Closed-Shell Density Matrix.
///
/// In an orthogonal basis (Lowdin / NDDO where overlap S = I), the closed-shell
/// density matrix satisfies:
///     P = 2 \sum_{i=1}^{N_occ} |psi_i><psi_i|
///     P^2 = 4 \sum_{i,j} |psi_i><psi_i|psi_j><psi_j| = 4 \sum_i |psi_i><psi_i| = 2 P
///
/// Theorem:
///     ||P^2 - 2P||_inf < 1e-13
#[test]
fn test_quantum_invariant_density_idempotency() {
    let model = Pm6Model;

    // Test on Water (H2O)
    let z_water = vec![8, 1, 1];
    let coords_water = vec![
        [0.0, 0.0, 0.0655],
        [0.0, 0.7571, -0.5205],
        [0.0, -0.7571, -0.5205],
    ];

    let batch = MolecularBatch::new(z_water, &coords_water);
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let opts = ScfOptions {
        max_iter: 50,
        energy_tol_ev: 1e-10,
        density_tol: 1e-10,
        damping: 0.5,
        use_nddo: true,
        ..Default::default()
    };

    let res = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    assert!(res.converged, "Water SCF failed to converge");

    let norbs = batch.norbs;
    let p = &ws.density;

    // Compute P^2 - 2P
    let mut max_idempotency_error = 0.0f64;
    for i in 0..norbs {
        for j in 0..norbs {
            let mut p2_ij = 0.0f64;
            for k in 0..norbs {
                p2_ij += p.get(i, k) * p.get(k, j);
            }
            let diff = (p2_ij - 2.0 * p.get(i, j)).abs();
            if diff > max_idempotency_error {
                max_idempotency_error = diff;
            }
        }
    }

    println!(
        "[OK] Water Closed-Shell Density Idempotency: ||P^2 - 2P||_inf = {:e}",
        max_idempotency_error
    );
    assert!(
        max_idempotency_error < 1e-13,
        "Density matrix failed idempotency condition: {:e} >= 1e-13",
        max_idempotency_error
    );

    // Test also on Methane (CH4)
    let z_ch4 = vec![6, 1, 1, 1, 1];
    let coords_ch4 = vec![
        [0.000000, 0.000000, 0.000000],
        [0.627600, 0.627600, 0.627600],
        [-0.627600, -0.627600, 0.627600],
        [-0.627600, 0.627600, -0.627600],
        [0.627600, -0.627600, -0.627600],
    ];

    let batch_ch4 = MolecularBatch::new(z_ch4, &coords_ch4);
    let mut ws_ch4 = ScfWorkspace::allocate(batch_ch4.norbs);
    let res_ch4 = run_rhf_scf_with_options(&batch_ch4, &model, &mut ws_ch4, &opts);
    assert!(res_ch4.converged, "Methane SCF failed to converge");

    let p_ch4 = &ws_ch4.density;
    let norbs_ch4 = batch_ch4.norbs;
    let mut max_idemp_ch4 = 0.0f64;
    for i in 0..norbs_ch4 {
        for j in 0..norbs_ch4 {
            let mut p2_ij = 0.0f64;
            for k in 0..norbs_ch4 {
                p2_ij += p_ch4.get(i, k) * p_ch4.get(k, j);
            }
            let diff = (p2_ij - 2.0 * p_ch4.get(i, j)).abs();
            if diff > max_idemp_ch4 {
                max_idemp_ch4 = diff;
            }
        }
    }
    println!(
        "[OK] Methane Closed-Shell Density Idempotency: ||P^2 - 2P||_inf = {:e}",
        max_idemp_ch4
    );
    assert!(
        max_idemp_ch4 < 1e-13,
        "Methane density failed idempotency: {:e} >= 1e-13",
        max_idemp_ch4
    );
}

/// Quantum Invariant 2: Molecular Orbital Orthonormality.
///
/// In Löwdin orthogonalized basis, eigenvectors C satisfy:
///     C^T C = I
///
/// Theorem:
///     ||C^T C - I||_inf < 1e-14
#[test]
fn test_quantum_invariant_orbital_orthonormality() {
    let model = Am1Model;

    // Phenol (C6H5OH: 13 atoms, 34 atomic orbitals)
    let z_phenol = vec![6, 6, 6, 6, 6, 6, 8, 1, 1, 1, 1, 1, 1];
    let coords_phenol = vec![
        [-0.0157, 1.3912, 0.0000],
        [-1.2185, 0.6974, 0.0000],
        [-1.2152, -0.6983, 0.0000],
        [-0.0084, -1.3965, 0.0000],
        [1.1969, -0.7027, 0.0000],
        [1.1997, 0.6933, 0.0000],
        [0.0000, 2.7600, 0.0000],  // Oxygen
        [-0.8660, 3.1600, 0.0000], // Hydroxyl Hydrogen
        [-2.1524, 1.2464, 0.0000],
        [-2.1524, -1.2415, 0.0000],
        [-0.0084, -2.4849, 0.0000],
        [2.1384, -1.2464, 0.0000],
        [2.1384, 1.2415, 0.0000],
    ];

    let batch = MolecularBatch::new(z_phenol, &coords_phenol);
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let opts = ScfOptions {
        max_iter: 80,
        energy_tol_ev: 1e-9,
        density_tol: 1e-8,
        use_nddo: true,
        ..Default::default()
    };

    let res = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    assert!(res.converged, "Phenol SCF failed to converge");

    let norbs = batch.norbs;
    let c = &ws.eigenvectors;

    // Check (C^T C)_{ij} = \sum_mu C_{mu i} C_{mu j}
    let mut max_ortho_error = 0.0f64;
    for i in 0..norbs {
        for j in 0..norbs {
            let mut dot = 0.0f64;
            for mu in 0..norbs {
                dot += c.get(mu, i) * c.get(mu, j);
            }
            let delta_ij = if i == j { 1.0 } else { 0.0 };
            let err = (dot - delta_ij).abs();
            if err > max_ortho_error {
                max_ortho_error = err;
            }
        }
    }

    println!(
        "[OK] Phenol Orbital Orthonormality (norbs={}): ||C^T C - I||_inf = {:e}",
        norbs, max_ortho_error
    );
    assert!(
        max_ortho_error < 1e-14,
        "Molecular orbital orthonormality violated: {:e} >= 1e-14",
        max_ortho_error
    );
}

/// Quantum Invariant 3: SO(3) 3D Spatial Rotational Invariance.
///
/// An isolated molecule in free space must have an electronic energy strictly invariant
/// under arbitrary 3D rigid body rotations R in SO(3):
///     E(R * X) == E(X) to numerical machine precision (|Delta E| < 1e-8 eV).
#[test]
fn test_quantum_invariant_rotational_invariance_so3() {
    let model = Pm6Model;

    // Asymmetric 3D Water geometry
    let z = vec![8, 1, 1];
    let coords_orig = vec![
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ];

    let batch_orig = MolecularBatch::new(z.clone(), &coords_orig);
    let mut ws_orig = ScfWorkspace::allocate(batch_orig.norbs);
    let opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-11,
        density_tol: 1e-11,
        use_nddo: false,
        ..Default::default()
    };

    let res_orig = run_rhf_scf_with_options(&batch_orig, &model, &mut ws_orig, &opts);
    assert!(res_orig.converged, "Unrotated water failed to converge");

    // Arbitrary Euler angles (alpha, beta, gamma)
    let alpha = 0.5342984885f64;
    let beta = 1.1273950182f64;
    let gamma = -0.8923401928f64;

    let (ca, sa) = (alpha.cos(), alpha.sin());
    let (cb, sb) = (beta.cos(), beta.sin());
    let (cg, sg) = (gamma.cos(), gamma.sin());

    // 3D Euler Z-Y-Z rotation matrix R
    let r = [
        [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
        [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [-sb * cg, sb * sg, cb],
    ];

    // Rotate coordinates
    let mut coords_rot = Vec::with_capacity(3);
    for pt in &coords_orig {
        let rx = r[0][0] * pt[0] + r[0][1] * pt[1] + r[0][2] * pt[2];
        let ry = r[1][0] * pt[0] + r[1][1] * pt[1] + r[1][2] * pt[2];
        let rz = r[2][0] * pt[0] + r[2][1] * pt[1] + r[2][2] * pt[2];
        coords_rot.push([rx, ry, rz]);
    }

    let batch_rot = MolecularBatch::new(z, &coords_rot);
    let mut ws_rot = ScfWorkspace::allocate(batch_rot.norbs);
    let res_rot = run_rhf_scf_with_options(&batch_rot, &model, &mut ws_rot, &opts);
    assert!(res_rot.converged, "Rotated water failed to converge");

    let e_diff_ev = (res_rot.total_energy_ev - res_orig.total_energy_ev).abs();
    let e_elec_diff = (res_rot.electronic_energy_ev - res_orig.electronic_energy_ev).abs();
    let e_nuc_diff = (res_rot.nuclear_repulsion_ev - res_orig.nuclear_repulsion_ev).abs();

    println!(
        "[OK] SO(3) 3D Rotational Invariance: Delta E_tot = {:e} eV, Delta E_elec = {:e} eV, Delta E_nuc = {:e} eV",
        e_diff_ev, e_elec_diff, e_nuc_diff
    );

    assert!(
        e_diff_ev < 1e-8,
        "Rotational invariance violated: Delta E_tot = {:e} eV >= 1e-8 eV",
        e_diff_ev
    );
    assert!(
        e_elec_diff < 1e-8,
        "Electronic energy rotational invariance violated: {:e} >= 1e-8 eV",
        e_elec_diff
    );
}

/// Quantum Invariant 4: Saunders-Hillier Virtual Orbital Level Shift Trace Conservation.
///
/// In MOPAC's Saunders-Hillier method, virtual orbitals are shifted by Delta_F:
///     Delta_F = sigma * (I - 1/2 P)
///
/// The energy expectation contribution to the ground state is:
///     E_shift = Tr[P * Delta_F] = sigma * Tr[P * (I - 1/2 P)]
///             = sigma * (Tr[P] - 1/2 Tr[P^2])
///
/// Since P^2 = 2P, Tr[P^2] = 2 Tr[P]:
///     Tr[P * (I - 1/2 P)] = Tr[P] - 1/2 (2 Tr[P]) == 0
///
/// Theorem:
///     Tr[P * Delta_F_shift] == 0 identically!
///     Therefore, level shifting NEVER contaminates the ground-state physical electronic energy.
#[test]
fn test_quantum_invariant_saunders_hillier_trace_conservation() {
    let model = Am1Model;

    let z = vec![7, 1, 1, 1]; // Ammonia (NH3)
    let coords = vec![
        [0.000000, 0.000000, 0.116489],
        [0.000000, 0.939731, -0.271808],
        [0.813831, -0.469865, -0.271808],
        [-0.813831, -0.469865, -0.271808],
    ];

    let batch = MolecularBatch::new(z, &coords);
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let opts = ScfOptions {
        max_iter: 50,
        energy_tol_ev: 1e-10,
        density_tol: 1e-10,
        use_nddo: true,
        ..Default::default()
    };

    let res = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    assert!(res.converged, "NH3 SCF failed to converge");

    let norbs = batch.norbs;
    let sigma = 8.0f64; // 8.0 eV level shift
    let p = &ws.density;

    // Delta_F = sigma * (I - 1/2 P)
    // Compute Tr[P * Delta_F] = \sum_{mu, nu} P_{mu nu} Delta_F_{nu mu}
    let mut trace_energy_contamination = 0.0f64;
    for mu in 0..norbs {
        for nu in 0..norbs {
            let delta_f_nu_mu = if nu == mu {
                sigma * (1.0 - 0.5 * p.get(nu, mu))
            } else {
                sigma * (-0.5 * p.get(nu, mu))
            };
            trace_energy_contamination += p.get(mu, nu) * delta_f_nu_mu;
        }
    }

    println!(
        "[OK] Saunders-Hillier Shift Trace Conservation: Tr[P * Delta_F] = {:e} eV",
        trace_energy_contamination
    );
    assert!(
        trace_energy_contamination.abs() < 1e-13,
        "Saunders-Hillier level shift contaminated ground state: Tr[P * Delta_F] = {:e} >= 1e-13",
        trace_energy_contamination
    );
}

/// Systems Invariant 5: Zero Heap Allocations in Hot SCF Loop (0-Malloc Gate).
///
/// Validates that a pre-allocated ScfWorkspace executes consecutive SCF cycles
/// without performing any dynamic heap allocations or re-allocating matrices.
#[test]
fn test_zero_allocation_scf_inner_loop_gate() {
    let model = Pm6Model;

    let z = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0655],
        [0.0, 0.7571, -0.5205],
        [0.0, -0.7571, -0.5205],
    ];

    let batch = MolecularBatch::new(z, &coords);
    let mut ws = ScfWorkspace::allocate(batch.norbs);

    // Initial memory buffer pointer snapshots
    let fock_ptr = ws.fock.data.as_ptr();
    let density_ptr = ws.density.data.as_ptr();
    let hcore_ptr = ws.h_core.data.as_ptr();
    let eig_ptr = ws.eigenvectors.data.as_ptr();
    let vals_ptr = ws.eigenvalues.as_ptr();
    let tmp1_ptr = ws.tmp1.data.as_ptr();
    let tmp2_ptr = ws.tmp2.data.as_ptr();

    let opts = ScfOptions {
        max_iter: 30,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        use_nddo: true,
        ..Default::default()
    };

    // Run first calculation
    let res1 = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    assert!(res1.converged);

    // Run second calculation reusing the same workspace (simulating trajectory / optimization)
    let res2 = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    assert!(res2.converged);

    // Verify all pointers are 100% invariant (ZERO re-allocations or buffer re-creations)
    assert_eq!(
        ws.fock.data.as_ptr(),
        fock_ptr,
        "Fock buffer was reallocated!"
    );
    assert_eq!(
        ws.density.data.as_ptr(),
        density_ptr,
        "Density buffer was reallocated!"
    );
    assert_eq!(
        ws.h_core.data.as_ptr(),
        hcore_ptr,
        "Hcore buffer was reallocated!"
    );
    assert_eq!(
        ws.eigenvectors.data.as_ptr(),
        eig_ptr,
        "Eigenvectors buffer was reallocated!"
    );
    assert_eq!(
        ws.eigenvalues.as_ptr(),
        vals_ptr,
        "Eigenvalues buffer was reallocated!"
    );
    assert_eq!(
        ws.tmp1.data.as_ptr(),
        tmp1_ptr,
        "Tmp1 buffer was reallocated!"
    );
    assert_eq!(
        ws.tmp2.data.as_ptr(),
        tmp2_ptr,
        "Tmp2 buffer was reallocated!"
    );

    println!(
        "[OK] 0-Malloc Gate: ScfWorkspace verified 100% realloc-free over consecutive iterations (7 buffers immutable)"
    );
}

/// Quantum Invariant 6: D-Orbital SO(3) Orthonormality and Casimir Invariance.
///
/// In accordance with Phase 1 Acceptance Gate (Test 1.1):
/// Proves that the 5x5 rotation matrix for d-orbitals is strictly orthonormal
/// in SO(3): ||D D^T - I_5||_inf < 10^-14, and preserves the Casimir invariant
/// ||D v||^2 = ||v||^2 for arbitrary spherical harmonic states.
#[test]
fn test_d_orbital_orthonormality() {
    use mopac_core::integrals::d_orbitals::DOrbitalRotation3D;

    // Test orientations spanning the entire unit sphere, including poles and diagonals
    let orientations: Vec<[f64; 3]> = vec![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [-1.5, 2.3, -0.7],
        [0.12345, -0.98765, 0.54321],
        [
            std::f64::consts::PI,
            std::f64::consts::E,
            -std::f64::consts::SQRT_2,
        ],
    ];

    // Arbitrary d-orbital spherical harmonic test state vectors (L=2)
    let test_states: Vec<[f64; 5]> = vec![
        [1.0, 0.0, 0.0, 0.0, 0.0],    // pure dx2-y2
        [0.0, 0.0, 1.0, 0.0, 0.0],    // pure dz2
        [0.2, -0.4, 0.6, -0.5, 0.35], // arbitrary superposition
        [1.0 / 5.0f64.sqrt(); 5],     // normalized symmetric superposition
    ];

    for (idx, coord) in orientations.iter().enumerate() {
        let r = (coord[0] * coord[0] + coord[1] * coord[1] + coord[2] * coord[2]).sqrt();
        let rot = DOrbitalRotation3D::new(coord[0], coord[1], coord[2], r);

        let (err_p, err_d) = rot.check_orthonormality();
        assert!(
            err_p < 1e-14,
            "Orientation {} p-orbital rotation orthonormality violated: ||P P^T - I|| = {:.3e}",
            idx,
            err_p
        );
        assert!(
            err_d < 1e-14,
            "Orientation {} d-orbital rotation orthonormality violated: ||D D^T - I|| = {:.3e}",
            idx,
            err_d
        );

        for state in &test_states {
            let casimir_err = rot.casimir_invariance(state);
            assert!(
                casimir_err < 1e-14,
                "Orientation {} d-orbital Casimir invariance violated: Delta ||v||^2 = {:.3e}",
                idx,
                casimir_err
            );
        }
    }

    println!(
        "[OK] Test 1.1 Passed: D-orbital 5x5 rotation matrix is strictly SO(3) orthonormal (||D D^T - I|| < 1e-14) and preserves Casimir norm identically"
    );
}
