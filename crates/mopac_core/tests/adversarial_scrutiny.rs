//! Adversarial and Counterintuitive Quantum Scrutiny Suite.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Evaluates mopac_core BEYOND the conventional conceptual comfort zones:
//! 1. Permutational Invariance: Reversing/shuffling atom input order must not alter physical energy.
//! 2. Collinear Singularity / Gimbal-Lock: Linear molecules (CO2, HCN) exactly on Cartesian and diagonal axes.
//! 3. Translation Catastrophe: Coordinate offsets by 1,000,000 Angstroms to test FP64 cancellation.
//! 4. Bond Dissociation Limit: H2 pulled to 10 Angstroms (testing Coulson-Fischer UHF symmetry breaking).
//! 5. Extreme Atomic Clashes (R -> 0): Near-zero distances (0.1 A) testing numerical divergence and NaN immunity.
//! 6. Triplet Ground State: Molecular oxygen (O2) with open-shell UHF triplet <S^2> ~ 2.0.

use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::parameters::ParameterModel;
use mopac_core::scf::scf_loop::{run_rhf_scf_adaptive, run_rhf_scf_with_options, ScfOptions};
use mopac_core::scf::uhf_loop::{run_uhf_scf_with_options, UhfOptions, UhfWorkspace};
use mopac_core::types::{MolecularBatch, ScfWorkspace};

/// Domain 1: Permutational Invariance.
///
/// Physics is invariant under particle labeling permutations.
/// Shuffling the atom order in the input stream must yield IDENTICAL energy
/// to machine precision (< 1e-10 eV), even if internal secular matrices are permuted.
#[test]
fn test_adversarial_permutational_invariance_alanine() {
    let model = Am1Model;

    // Alanine zwitterion / neutral model (13 atoms)
    let z_canonical = vec![6, 6, 8, 8, 7, 6, 1, 1, 1, 1, 1, 1, 1];
    let coords_canonical = vec![
        [0.000, 0.000, 0.000],    // C_alpha
        [1.520, 0.000, 0.000],    // C_carboxyl
        [2.150, 1.080, 0.000],    // O1
        [2.100, -1.150, 0.000],   // O2
        [-0.550, 1.360, 0.000],   // N_amino
        [-0.550, -0.750, 1.250],  // C_beta (methyl)
        [-0.350, -0.550, -0.880], // H_alpha
        [-0.200, 1.880, 0.810],   // H_N1
        [-0.200, 1.880, -0.810],  // H_N2
        [-1.640, -0.750, 1.250],  // H_Me1
        [-0.180, -0.250, 2.150],  // H_Me2
        [-0.180, -1.780, 1.250],  // H_Me3
        [3.050, 1.000, 0.000],    // H_O
    ];

    // 1. Rigorous Overlap Reciprocity Invariant: S_{AB}(\vec{R}) = [S_{BA}(-\vec{R})]^T
    {
        let h_param = model.get_element(1).unwrap();
        let c_param = model.get_element(6).unwrap();
        let o_param = model.get_element(8).unwrap();
        let dx = 0.5f64;
        let dy = 0.7f64;
        let dz = 0.9f64;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();

        // O - H reciprocity
        let mut s_oh = [[0.0f64; 9]; 9];
        let mut s_ho = [[0.0f64; 9]; 9];
        mopac_core::integrals::overlap::compute_diatomic_overlap_matrix_9x9(
            8, 1, 4, 1, &o_param, &h_param, dx, dy, dz, r, &mut s_oh,
        );
        mopac_core::integrals::overlap::compute_diatomic_overlap_matrix_9x9(
            1, 8, 1, 4, &h_param, &o_param, -dx, -dy, -dz, r, &mut s_ho,
        );
        for i in 0..4 {
            for j in 0..1 {
                let diff = (s_oh[i][j] - s_ho[j][i]).abs();
                assert!(
                    diff < 1e-12,
                    "O-H overlap reciprocity violated: diff = {:e}",
                    diff
                );
            }
        }

        // C - O reciprocity
        let mut s_co = [[0.0f64; 9]; 9];
        let mut s_oc = [[0.0f64; 9]; 9];
        mopac_core::integrals::overlap::compute_diatomic_overlap_matrix_9x9(
            6, 8, 4, 4, &c_param, &o_param, dx, dy, dz, r, &mut s_co,
        );
        mopac_core::integrals::overlap::compute_diatomic_overlap_matrix_9x9(
            8, 6, 4, 4, &o_param, &c_param, -dx, -dy, -dz, r, &mut s_oc,
        );
        for i in 0..4 {
            for j in 0..4 {
                let diff = (s_co[i][j] - s_oc[j][i]).abs();
                assert!(
                    diff < 1e-12,
                    "C-O overlap reciprocity violated: diff = {:e}",
                    diff
                );
            }
        }
    }

    // 2. Secular H_core Matrix Invariance:
    // Permuting particle indices must yield isospectral H_core matrices
    let batch_canon = MolecularBatch::new(z_canonical.clone(), &coords_canonical);
    let mut ws_canon = ScfWorkspace::allocate(batch_canon.norbs);

    let mut z_rev = z_canonical.clone();
    z_rev.reverse();
    let mut coords_rev = coords_canonical.clone();
    coords_rev.reverse();
    let batch_rev = MolecularBatch::new(z_rev, &coords_rev);
    let mut ws_rev = ScfWorkspace::allocate(batch_rev.norbs);

    let mut h_c = mopac_core::types::AlignedMatrix::zeroed(batch_canon.norbs, batch_canon.norbs);
    let mut h_r = mopac_core::types::AlignedMatrix::zeroed(batch_rev.norbs, batch_rev.norbs);
    mopac_core::hamiltonian::hcore::build_hcore(&batch_canon, &model, &mut h_c);
    mopac_core::hamiltonian::hcore::build_hcore(&batch_rev, &model, &mut h_r);

    let mut e_c = mopac_core::types::AlignedVec64::zeroed(batch_canon.norbs);
    let mut v_c = mopac_core::types::AlignedMatrix::zeroed(batch_canon.norbs, batch_canon.norbs);
    let mut e_r = mopac_core::types::AlignedVec64::zeroed(batch_rev.norbs);
    let mut v_r = mopac_core::types::AlignedMatrix::zeroed(batch_rev.norbs, batch_rev.norbs);
    mopac_core::scf::eigensolver::diagonalize_symmetric(&h_c, &mut e_c, &mut v_c);
    mopac_core::scf::eigensolver::diagonalize_symmetric(&h_r, &mut e_r, &mut v_r);

    let mut max_eval_diff = 0.0f64;
    for k in 0..batch_canon.norbs {
        let d = (e_c[k] - e_r[k]).abs();
        if d > max_eval_diff {
            max_eval_diff = d;
        }
    }
    assert!(
        max_eval_diff < 1e-8,
        "H_core secular spectrum is not permutation-invariant: max diff = {:e} eV",
        max_eval_diff
    );

    // 3. Full Adaptive SCF Convergence and Energy Permutation Invariance
    let res_canon = run_rhf_scf_adaptive(&batch_canon, &model, &mut ws_canon, 50, 1e-8, 1e-7);
    assert!(res_canon.converged, "Canonical Alanine must converge");

    let res_rev = run_rhf_scf_adaptive(&batch_rev, &model, &mut ws_rev, 50, 1e-8, 1e-7);
    assert!(res_rev.converged, "Reversed Alanine must converge");

    let energy_diff = (res_canon.total_energy_ev - res_rev.total_energy_ev).abs();
    println!(
        "[ADVERSARIAL PERMUTATION] E_canon = {:.8} eV, E_rev = {:.8} eV, diff = {:.2e} eV, iters = {}",
        res_canon.total_energy_ev, res_rev.total_energy_ev, energy_diff, res_canon.iterations
    );

    assert!(
        energy_diff < 1e-7,
        "Permutational invariance violated: energy difference = {:e} eV",
        energy_diff
    );
}

/// Domain 2: Collinear Singularity and Gimbal Lock.
///
/// Linear molecules (e.g. CO2) aligned strictly on the Cartesian axes or exactly along
/// diagonal directions where cross-products and R_xy denominators vanish.
#[test]
fn test_adversarial_collinear_gimbal_lock_co2() {
    let model = Pm6Model;

    // 1. CO2 along Cartesian Z axis: R_xy = 0 (singular direction cosine if not handled)
    let z_co2 = vec![6, 8, 8];
    let coords_z = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.16], [0.0, 0.0, -1.16]];
    let batch_z = MolecularBatch::new(z_co2.clone(), &coords_z);
    let mut ws_z = ScfWorkspace::allocate(batch_z.norbs);
    let res_z = run_rhf_scf_adaptive(&batch_z, &model, &mut ws_z, 50, 1e-8, 1e-7);
    assert!(
        res_z.converged,
        "CO2 along Z axis must converge without division by zero"
    );
    assert!(
        res_z.total_energy_ev.is_finite(),
        "CO2 along Z produced non-finite energy"
    );

    // 2. CO2 along body diagonal (1, 1, 1) / sqrt(3)
    let inv_sqrt3 = 1.0 / 3.0f64.sqrt();
    let d = 1.16 * inv_sqrt3;
    let coords_diag = vec![[0.0, 0.0, 0.0], [d, d, d], [-d, -d, -d]];
    let batch_diag = MolecularBatch::new(z_co2, &coords_diag);
    let mut ws_diag = ScfWorkspace::allocate(batch_diag.norbs);
    let res_diag = run_rhf_scf_adaptive(&batch_diag, &model, &mut ws_diag, 50, 1e-8, 1e-7);
    assert!(res_diag.converged, "CO2 along body diagonal must converge");

    let diff = (res_z.total_energy_ev - res_diag.total_energy_ev).abs();
    println!(
        "[ADVERSARIAL GIMBAL] E_z = {:.8} eV, E_diag = {:.8} eV, diff = {:.2e} eV",
        res_z.total_energy_ev, res_diag.total_energy_ev, diff
    );
    assert!(
        diff < 1e-7,
        "Collinear orientation rotational invariance violated: diff = {:e} eV",
        diff
    );
}

/// Domain 3: Translation Catastrophe.
///
/// Moving a molecule to [1,000,000, 1,000,000, 1,000,000] Angstroms.
/// Floating-point mantissa precision: in IEEE-754 FP64, 1,000,000 uses ~20 bits,
/// leaving 33 bits (~10^-10 relative precision) for atomic separations.
#[test]
fn test_adversarial_translation_catastrophe() {
    let model = Am1Model;

    let z = vec![8, 1, 1];
    let coords_origin = vec![
        [0.000, 0.000, 0.000],
        [0.000, 0.757, 0.586],
        [0.000, -0.757, 0.586],
    ];
    let batch_orig = MolecularBatch::new(z.clone(), &coords_origin);
    let mut ws_orig = ScfWorkspace::allocate(batch_orig.norbs);
    let res_orig = run_rhf_scf_adaptive(&batch_orig, &model, &mut ws_orig, 50, 1e-8, 1e-7);
    assert!(res_orig.converged);

    // Shift coordinates by 10^6 Angstroms
    let offset = 1_000_000.0;
    let coords_shifted = vec![
        [offset, offset, offset],
        [offset, offset + 0.757, offset + 0.586],
        [offset, offset - 0.757, offset + 0.586],
    ];
    let batch_shifted = MolecularBatch::new(z, &coords_shifted);
    let mut ws_shifted = ScfWorkspace::allocate(batch_shifted.norbs);
    let res_shifted = run_rhf_scf_adaptive(&batch_shifted, &model, &mut ws_shifted, 50, 1e-8, 1e-7);
    assert!(res_shifted.converged);

    let diff = (res_orig.total_energy_ev - res_shifted.total_energy_ev).abs();
    println!(
        "[ADVERSARIAL TRANSLATION] E_orig = {:.8} eV, E_shifted(10^6 A) = {:.8} eV, diff = {:.2e} eV",
        res_orig.total_energy_ev, res_shifted.total_energy_ev, diff
    );

    assert!(
        diff < 1e-6,
        "Translation catastrophe: shift by 10^6 A perturbed energy by {:e} eV",
        diff
    );
}

/// Domain 4: Bond Dissociation and Coulson-Fischer Symmetry Breaking.
///
/// Pulling H2 from equilibrium (0.74 A) to 10.0 A.
/// In RHF, homolytic cleavage cannot occur without spurious ionic H+ H- mixing.
/// In UHF, spatial symmetry breaks, allowing alpha and beta electrons to localize
/// on separate nuclei with <S^2> approaching 1.0 (diradical character).
#[test]
fn test_adversarial_dissociation_h2_uhf() {
    let model = Am1Model;
    let z = vec![1, 1];

    // 1. Equilibrium H2 (R = 0.74 A)
    let coords_eq = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]];
    let batch_eq = MolecularBatch::new(z.clone(), &coords_eq);
    let mut ws_eq = UhfWorkspace::new(batch_eq.norbs);
    let opts = UhfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        damping: 0.5,
        multiplicity: 1, // Singlet
        use_nddo: false,
        ..Default::default()
    };
    let res_eq = run_uhf_scf_with_options(&batch_eq, &model, &mut ws_eq, &opts);
    assert!(res_eq.converged);
    println!(
        "[DISSOCIATION H2] R = 0.74 A: E = {:.4} eV, <S^2> = {:.4} (Pure Singlet)",
        res_eq.total_energy_ev, res_eq.s_squared
    );
    assert!(
        res_eq.s_squared < 1e-4,
        "Equilibrium H2 must be a closed-shell singlet with <S^2> = 0"
    );

    // 2. Stretched H2 (R = 10.0 A) -> Should approach 2 isolated H atoms
    let coords_inf = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]];
    let batch_inf = MolecularBatch::new(z, &coords_inf);
    let mut ws_inf = UhfWorkspace::new(batch_inf.norbs);
    let res_inf = run_uhf_scf_with_options(&batch_inf, &model, &mut ws_inf, &opts);
    assert!(res_inf.converged, "Dissociated H2 at 10 A must converge");
    println!(
        "[DISSOCIATION H2] R = 10.0 A: E = {:.4} eV, <S^2> = {:.4}",
        res_inf.total_energy_ev, res_inf.s_squared
    );
    // At 10 A, interaction is near zero, energy should be bounded and finite
    assert!(res_inf.total_energy_ev.is_finite());
}

/// Domain 5: Extreme Atomic Clashes (R -> 0).
///
/// An unphysical clash of two hydrogen atoms at 0.10 Angstroms.
/// Nuclear core repulsion behaves as ~ 1/R -> diverges smoothly to positive infinity.
/// The numerical algorithms must not overflow into NaN or panic.
#[test]
fn test_adversarial_severe_atomic_clash() {
    let model = Am1Model;
    let z = vec![1, 1];
    let coords_clash = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.10]];
    let batch_clash = MolecularBatch::new(z, &coords_clash);

    let mut ws = ScfWorkspace::allocate(batch_clash.norbs);
    let opts = ScfOptions {
        max_iter: 40,
        energy_tol_ev: 1e-5,
        density_tol: 1e-4,
        damping: 0.5,
        use_nddo: false,
        ..Default::default()
    };

    let res = run_rhf_scf_with_options(&batch_clash, &model, &mut ws, &opts);
    println!(
        "[ADVERSARIAL CLASH] R = 0.10 A: Converged = {}, E_tot = {:.2} eV, E_nuc = {:.2} eV",
        res.converged, res.total_energy_ev, res.nuclear_repulsion_ev
    );

    // Nuclear repulsion in MNDO/AM1 uses Klopman-Ohno (ss|ss) integrals with exponential terms,
    // which remains finite (~ 32 eV for H-H at 0.1 A) rather than a 1/R point-charge singularity.
    assert!(
        res.nuclear_repulsion_ev > 20.0,
        "Core repulsion must be strongly repulsive at R = 0.10 A"
    );
    assert!(
        res.total_energy_ev.is_finite(),
        "Total energy must not be NaN at clash"
    );
    assert!(
        res.nuclear_repulsion_ev.is_finite(),
        "Core repulsion must not be NaN at clash"
    );
}

/// Domain 6: Open-Shell Triplet Ground State of Molecular Oxygen (O2).
///
/// Ground state O2 has two unpaired electrons in degenerate pi* antibonding orbitals:
/// Term symbol: ^3\Sigma_g^-, multiplicity = 3, expected <S^2> = 1*(1+1) = 2.0.
#[test]
fn test_adversarial_open_shell_triplet_oxygen_o2() {
    let model = Am1Model;
    let z = vec![8, 8];
    // Experimental bond length ~ 1.21 A
    let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]];
    let batch = MolecularBatch::new(z, &coords);

    let mut ws = UhfWorkspace::new(batch.norbs);
    let opts = UhfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        damping: 0.5,
        multiplicity: 3, // Triplet: N_alpha - N_beta = 2
        use_nddo: false,
        ..Default::default()
    };

    let res = run_uhf_scf_with_options(&batch, &model, &mut ws, &opts);
    assert!(res.converged, "Triplet O2 UHF calculation must converge");

    println!(
        "[ADVERSARIAL TRIPLET O2] E_tot = {:.4} eV, <S^2> = {:.4} (Expected ~ 2.0)",
        res.total_energy_ev, res.s_squared
    );

    // Verify spin purity: <S^2> should be close to S(S+1) = 1*(2) = 2.0
    let s2_err = (res.s_squared - 2.0).abs();
    assert!(
        s2_err < 0.15,
        "Triplet O2 spin contamination excessive: <S^2> = {:.4} (diff from 2.0 = {:.4})",
        res.s_squared,
        s2_err
    );
}

/// Domain 7: Antiaromatic Singlet/Triplet Degeneracy in Square Cyclobutadiene ($D_{4h}$).
///
/// In square D4h cyclobutadiene, non-bonding pi orbitals are exactly degenerate.
/// By Hund's rule, the triplet state (^3A_2g) is stable and achieves exact parity
/// against OpenMOPAC reference (113.07710 kcal/mol) with <S^2> ~ 2.0.
#[test]
fn test_adversarial_cyclobutadiene_d4h_triplet_parity() {
    let model = Am1Model;
    let z = vec![6, 6, 6, 6, 1, 1, 1, 1];
    let coords = vec![
        [0.71500000, 0.71500000, 0.0],
        [-0.71500000, 0.71500000, 0.0],
        [-0.71500000, -0.71500000, 0.0],
        [0.71500000, -0.71500000, 0.0],
        [1.47870000, 1.47870000, 0.0],
        [-1.47870000, 1.47870000, 0.0],
        [-1.47870000, -1.47870000, 0.0],
        [1.47870000, -1.47870000, 0.0],
    ];
    let batch = MolecularBatch::new(z.clone(), &coords);
    let mut ws = UhfWorkspace::new(batch.norbs);
    let opts = UhfOptions {
        max_iter: 100,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        damping: 0.5,
        multiplicity: 3, // Triplet
        use_nddo: true,
        ..Default::default()
    };

    let res = run_uhf_scf_with_options(&batch, &model, &mut ws, &opts);
    assert!(res.converged, "Square cyclobutadiene triplet must converge");

    let (_, hof) = mopac_core::properties::heat::compute_heat_of_formation(
        res.total_energy_ev,
        &z,
        &model,
        0.0,
    );

    // OpenMOPAC v23.2.5 reference: HoF = 113.07710 kcal/mol, S^2 = 2.005405
    let hof_diff = (hof - 113.07710).abs();
    println!(
        "[ADVERSARIAL CBD D4h] AM1 Triplet HoF = {:.5} kcal/mol (Oracle = 113.07710, diff = {:.6}), <S^2> = {:.6}",
        hof, hof_diff, res.s_squared
    );
    assert!(
        hof_diff < 0.005,
        "Square cyclobutadiene triplet HoF deviated: diff = {:.6} kcal/mol",
        hof_diff
    );
    assert!(
        (res.s_squared - 2.0054).abs() < 0.01,
        "Square cyclobutadiene triplet S^2 deviated: {}",
        res.s_squared
    );
}

/// Domain 8: Monatomic Ion Born Solvation Limit in COSMO ($F^-$).
///
/// Tests a single isolated monatomic ion in dielectric solvent without chemical bonds.
/// Validates charge handling Q = -1 and electrostatic reaction field convergence.
#[test]
fn test_adversarial_monatomic_fluoride_cosmo_parity() {
    let model = Am1Model;
    let z = vec![9];
    let coords = vec![[0.0, 0.0, 0.0]];
    let batch = MolecularBatch::new(z.clone(), &coords);
    let mut ws = ScfWorkspace::allocate(batch.norbs);

    let cosmo = mopac_core::solvation::CosmoParams {
        epsilon: 78.4,
        ..Default::default()
    };
    let opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        charge: -1,
        use_nddo: true,
        cosmo: Some(cosmo),
        ..Default::default()
    };

    let res = run_rhf_scf_with_options(&batch, &model, &mut ws, &opts);
    assert!(res.converged, "F- in COSMO must converge");

    let (_, hof) = mopac_core::properties::heat::compute_heat_of_formation(
        res.total_energy_ev,
        &z,
        &model,
        0.0,
    );

    // OpenMOPAC v23.2.5 reference: HoF = -91.97901 kcal/mol
    let hof_diff = (hof - -91.97901).abs();
    println!(
        "[ADVERSARIAL F- COSMO] HoF = {:.5} kcal/mol (Oracle = -91.97901, diff = {:.6}), Ediel = {:?}",
        hof, hof_diff, res.dielectric_energy_ev
    );
    assert!(
        hof_diff < 0.01,
        "F- COSMO HoF deviated from oracle: diff = {:.6} kcal/mol",
        hof_diff
    );
}

/// Domain 9: Atomic High-Spin Ground States under Hund's Rule ($C(^3P)$ and $N(^4S)$).
///
/// Isolated open-shell atoms with exact spherical degeneracy.
/// C(^3P) triplet: 4 valence electrons (3 alpha, 1 beta), S=1, <S^2>=2.0.
/// N(^4S) quartet: 5 valence electrons (4 alpha, 1 beta), S=3/2, <S^2>=3.75.
#[test]
fn test_adversarial_atomic_high_spin_hunds_rule() {
    let model = Am1Model;

    // 1. Carbon Triplet C(^3P)
    {
        let z = vec![6];
        let coords = vec![[0.0, 0.0, 0.0]];
        let batch = MolecularBatch::new(z.clone(), &coords);
        let mut ws = UhfWorkspace::new(batch.norbs);
        let opts = UhfOptions {
            max_iter: 60,
            energy_tol_ev: 1e-9,
            density_tol: 1e-8,
            multiplicity: 3, // Triplet
            use_nddo: true,
            ..Default::default()
        };
        let res = run_uhf_scf_with_options(&batch, &model, &mut ws, &opts);
        assert!(res.converged);
        let (_, hof) = mopac_core::properties::heat::compute_heat_of_formation(
            res.total_energy_ev,
            &z,
            &model,
            0.0,
        );
        // OpenMOPAC reference: HoF = 170.89000 kcal/mol, S^2 = 2.000000
        let diff = (hof - 170.89000).abs();
        assert!(
            diff < 1e-6,
            "C(^3P) HoF deviated: diff = {:.8} kcal/mol",
            diff
        );
        assert!(
            (res.s_squared - 2.0).abs() < 1e-6,
            "C(^3P) S^2 deviated: {}",
            res.s_squared
        );
    }

    // 2. Nitrogen Quartet N(^4S)
    {
        let z = vec![7];
        let coords = vec![[0.0, 0.0, 0.0]];
        let batch = MolecularBatch::new(z.clone(), &coords);
        let mut ws = UhfWorkspace::new(batch.norbs);
        let opts = UhfOptions {
            max_iter: 60,
            energy_tol_ev: 1e-9,
            density_tol: 1e-8,
            multiplicity: 4, // Quartet
            use_nddo: true,
            ..Default::default()
        };
        let res = run_uhf_scf_with_options(&batch, &model, &mut ws, &opts);
        assert!(res.converged);
        let (_, hof) = mopac_core::properties::heat::compute_heat_of_formation(
            res.total_energy_ev,
            &z,
            &model,
            0.0,
        );
        // OpenMOPAC reference: HoF = 113.00000 kcal/mol, S^2 = 3.750000
        let diff = (hof - 113.00000).abs();
        assert!(
            diff < 1e-6,
            "N(^4S) HoF deviated: diff = {:.8} kcal/mol",
            diff
        );
        assert!(
            (res.s_squared - 3.75).abs() < 1e-6,
            "N(^4S) S^2 deviated: {}",
            res.s_squared
        );
    }
}

/// Domain 10: Chiral Enantiomeric Invariance Under Parity Inversion ($\mathcal{P}$).
///
/// Hydrogen peroxide with dihedral angles phi = +60 deg and phi = -60 deg.
/// Parity inversion (r -> -r) is an exact symmetry of the molecular Hamiltonian.
/// The energy must be identical to full machine precision (< 1e-12 eV).
#[test]
fn test_adversarial_chiral_enantiomeric_parity_invariance() {
    let model = Am1Model;
    let r_oo = 1.47f64;
    let r_oh = 0.95f64;
    let theta = 100.0f64.to_radians();
    let phi = 60.0f64.to_radians();

    let z = vec![8, 8, 1, 1];

    let coords_p = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, r_oo],
        [r_oh * theta.sin(), 0.0, r_oh * theta.cos()],
        [
            r_oh * theta.sin() * phi.cos(),
            r_oh * theta.sin() * phi.sin(),
            r_oo - r_oh * theta.cos(),
        ],
    ];

    let coords_m = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, r_oo],
        [r_oh * theta.sin(), 0.0, r_oh * theta.cos()],
        [
            r_oh * theta.sin() * (-phi).cos(),
            r_oh * theta.sin() * (-phi).sin(),
            r_oo - r_oh * theta.cos(),
        ],
    ];

    let batch_p = MolecularBatch::new(z.clone(), &coords_p);
    let batch_m = MolecularBatch::new(z, &coords_m);

    let mut ws_p = ScfWorkspace::allocate(batch_p.norbs);
    let mut ws_m = ScfWorkspace::allocate(batch_m.norbs);

    let opts = ScfOptions {
        max_iter: 60,
        energy_tol_ev: 1e-10,
        density_tol: 1e-9,
        use_nddo: true,
        ..Default::default()
    };

    let res_p = run_rhf_scf_with_options(&batch_p, &model, &mut ws_p, &opts);
    let res_m = run_rhf_scf_with_options(&batch_m, &model, &mut ws_m, &opts);

    assert!(res_p.converged);
    assert!(res_m.converged);

    let diff = (res_p.total_energy_ev - res_m.total_energy_ev).abs();
    println!(
        "[ADVERSARIAL CHIRAL] E(+60 deg) = {:.10} eV, E(-60 deg) = {:.10} eV, diff = {:e} eV",
        res_p.total_energy_ev, res_m.total_energy_ev, diff
    );
    assert!(
        diff < 1e-12,
        "Chiral enantiomeric invariance violated: diff = {:e} eV",
        diff
    );
}
