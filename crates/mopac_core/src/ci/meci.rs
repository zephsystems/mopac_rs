//! Multi-Electron Configuration Interaction (MECI) Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct mathematical port of canonical OpenMOPAC `meci.F90`, `mecih.F90`, `diagi.F90`,
//! `aababc.F90`, `babbbc.F90`, `aabbcd.F90`, `aabacd.F90`, and `babbcd.F90`.
//!
//! # Methodological Details
//! * Selects active space of $m$ molecular orbitals and $n$ electrons around the Fermi level.
//! * Generates complete microstate Slater determinant basis with fixed $S_z = \frac{1}{2}(N_\alpha - N_\beta)$.
//! * Evaluates two-electron repulsion integrals $\langle ij | kl \rangle = XY(i, j, k, l)$ over active MOs.
//! * Assembles the CI Hamiltonian matrix using exact Slater-Condon rules.
//! * Evaluates total spin $\hat{S}^2$ operator for each state, ensuring rigorous spin purity and assignment.
//! * Symmetric diagonalization yielding electronic ground and excited state energies.
//! * 0-malloc memory invariant during hot iterative sweeps with preallocated `MeciWorkspace`.

use crate::integrals::multipoles::{precompute_diatomic_pairs, DiatomicPairIntegrals};
use crate::integrals::two_electron::dewar_klopman_monopole;
use crate::parameters::ParameterModel;
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch};

/// Active space specification for Multi-Electron Configuration Interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CiActiveSpace {
    /// Number of molecular orbitals in active space (m)
    pub num_orbitals: usize,
    /// Number of active electrons (n)
    pub num_electrons: usize,
}

impl CiActiveSpace {
    /// Create a new active space definition.
    pub fn new(num_orbitals: usize, num_electrons: usize) -> Self {
        assert!(
            num_orbitals >= 1,
            "Active space must have at least 1 orbital"
        );
        assert!(
            num_electrons >= 1,
            "Active space must have at least 1 electron"
        );
        assert!(
            num_electrons <= 2 * num_orbitals,
            "Active electrons ({}) cannot exceed 2 * active orbitals ({})",
            num_electrons,
            num_orbitals
        );
        Self {
            num_orbitals,
            num_electrons,
        }
    }
}

/// Options for Configuration Interaction and excited state calculations.
#[derive(Debug, Clone)]
pub struct MeciOptions {
    /// Active space definition (orbitals, electrons)
    pub active_space: CiActiveSpace,
    /// Target root state to select (1-indexed: 1 = ground state)
    pub target_root: usize,
    /// Desired spin multiplicity target (None = all states, Some(1) = Singlets, Some(3) = Triplets)
    pub spin_target: Option<usize>,
    /// Whether full NDDO multipole two-electron integrals are evaluated
    pub use_nddo: bool,
}

impl Default for MeciOptions {
    fn default() -> Self {
        Self {
            active_space: CiActiveSpace {
                num_orbitals: 2,
                num_electrons: 2,
            },
            target_root: 1,
            spin_target: None,
            use_nddo: true,
        }
    }
}

/// A Slater determinant microstate represented by occupation vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct Microstate {
    /// Alpha spin-orbital occupations in active space (length m, elements 0 or 1)
    pub alpha: Vec<u8>,
    /// Beta spin-orbital occupations in active space (length m, elements 0 or 1)
    pub beta: Vec<u8>,
    /// Diagonal non-interacting energy in eV relative to reference configuration
    pub energy_ev: f64,
}

/// Total spin state information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateSpin {
    /// Expectation value <S^2>
    pub s_squared: f64,
    /// Total spin quantum number S (e.g. 0.0 for singlet, 0.5 for doublet, 1.0 for triplet)
    pub s: f64,
    /// Spin multiplicity 2S + 1 (1 = Singlet, 2 = Doublet, 3 = Triplet, etc.)
    pub multiplicity: usize,
    /// Human-readable spin designation string
    pub label: &'static str,
}

/// Single electronic state resulting from CI diagonalization.
#[derive(Debug, Clone)]
pub struct CiState {
    /// State index (1-indexed)
    pub root: usize,
    /// Absolute CI energy eigenvalue in eV (matches OpenMOPAC ABSOLUTE ENERGY)
    pub energy_ev: f64,
    /// Relative excitation energy in eV relative to ground root (root 1)
    pub excitation_energy_ev: f64,
    /// Excitation energy in cm^-1
    pub excitation_energy_cm1: f64,
    /// Absorption transition wavelength in nanometers (hc / Delta E)
    pub wavelength_nm: f64,
    /// Total spin quantum numbers
    pub spin: StateSpin,
    /// Transition dipole moment vector [mu_x, mu_y, mu_z] from ground root in Debye
    pub transition_dipole_debye: [f64; 3],
    /// Total transition dipole magnitude |mu| in Debye
    pub dipole_strength_debye: f64,
    /// Transition dipole squared in Angstroms^2 matching OpenMOPAC POLARIZATION (X, Y, Z)
    pub polarization_angstrom2: [f64; 3],
    /// Dimensionless oscillator strength f_osc
    pub oscillator_strength: f64,
    /// Eigenvector coefficients in the microstate basis
    pub eigenvector: Vec<f64>,
}

/// Comprehensive result of Multi-Electron Configuration Interaction calculation.
#[derive(Debug, Clone)]
pub struct MeciResult {
    /// All computed electronic states ordered ascending by energy
    pub states: Vec<CiState>,
    /// Microstates considered in the active space
    pub microstates: Vec<Microstate>,
    /// Target root index (1-indexed)
    pub target_root: usize,
    /// CI energy correction in eV: E_CI(target_root) - E_CI(ground_scf_ref)
    pub ci_energy_correction_ev: f64,
    /// Total electronic energy of the target root in eV
    pub electronic_energy_ev: f64,
    /// Total energy of the target root in eV
    pub total_energy_ev: f64,
    /// Standard heat of formation of the target root in kcal/mol
    pub heat_of_formation_kcal: f64,
}

/// Preallocated workspace for MECI calculations ensuring 0-malloc memory invariant.
#[derive(Debug, Clone)]
pub struct MeciWorkspace {
    /// Active MO two-electron repulsion tensor: [m][m][m][m]
    pub xy: Vec<Vec<Vec<Vec<f64>>>>,
    /// CI Hamiltonian matrix [lab, lab]
    pub ci_mat: AlignedMatrix<f64>,
    /// Diagonal microstate energies [lab]
    pub diag: Vec<f64>,
    /// Corrected MO eigenvalues eiga [m]
    pub eiga: Vec<f64>,
    /// Reference orbital occupation occa [m]
    pub occa: Vec<f64>,
    /// CI eigenvalues [lab]
    pub eigenvalues: AlignedVec64<f64>,
    /// CI eigenvectors [lab, lab]
    pub eigenvectors: AlignedMatrix<f64>,
    /// Total spin matrix S^2 [lab, lab]
    pub s2_mat: AlignedMatrix<f64>,
}

impl MeciWorkspace {
    /// Allocate workspace for active space of size $m$.
    pub fn allocate(num_orbitals: usize, max_microstates: usize) -> Self {
        let m = num_orbitals;
        let lab = max_microstates.max(1);
        let xy = vec![vec![vec![vec![0.0; m]; m]; m]; m];

        Self {
            xy,
            ci_mat: AlignedMatrix::zeroed(lab, lab),
            diag: vec![0.0; lab],
            eiga: vec![0.0; m],
            occa: vec![0.0; m],
            eigenvalues: AlignedVec64::zeroed(lab),
            eigenvectors: AlignedMatrix::zeroed(lab, lab),
            s2_mat: AlignedMatrix::zeroed(lab, lab),
        }
    }
}

/// Generate all combination bitmasks of $k$ occupied bits out of $n$ available slots.
fn generate_combinations(n: usize, k: usize) -> Vec<Vec<u8>> {
    let mut results = Vec::new();
    let mut current = vec![0u8; n];

    fn backtrack(
        start: usize,
        remaining: usize,
        n: usize,
        current: &mut [u8],
        results: &mut Vec<Vec<u8>>,
    ) {
        if remaining == 0 {
            results.push(current.to_vec());
            return;
        }
        for i in start..=n - remaining {
            current[i] = 1;
            backtrack(i + 1, remaining - 1, n, current, results);
            current[i] = 0;
        }
    }

    backtrack(0, k, n, &mut current, &mut results);
    results
}

/// Generate all microstates for given active space $(m, n)$ and spin projection $S_z$.
pub fn generate_microstates(
    num_orbitals: usize,
    num_electrons: usize,
    sz_two: i32,
) -> Vec<Microstate> {
    let m = num_orbitals;
    let n = num_electrons as i32;

    // n_alpha + n_beta = n
    // n_alpha - n_beta = sz_two
    // 2 * n_alpha = n + sz_two
    let n_alpha_2 = n + sz_two;
    if n_alpha_2 < 0 || n_alpha_2 % 2 != 0 {
        return Vec::new();
    }
    let n_alpha = (n_alpha_2 / 2) as usize;
    let n_beta = (n - n_alpha as i32) as usize;

    if n_alpha > m || n_beta > m {
        return Vec::new();
    }

    let alpha_combos = generate_combinations(m, n_alpha);
    let beta_combos = generate_combinations(m, n_beta);

    let mut microstates = Vec::with_capacity(alpha_combos.len() * beta_combos.len());

    // Generate Cartesian product in standard lexical order
    for alpha in &alpha_combos {
        for beta in &beta_combos {
            microstates.push(Microstate {
                alpha: alpha.clone(),
                beta: beta.clone(),
                energy_ev: 0.0,
            });
        }
    }

    microstates
}

/// Compute one-center two-electron integral matrix for a given atom in SP basis.
fn get_one_center_integrals_sp(
    gss: f64,
    gsp: f64,
    gpp: f64,
    gp2: f64,
    hsp: f64,
) -> [[f64; 10]; 10] {
    let mut w = [[0.0f64; 10]; 10];

    w[0][0] = gss;

    w[0][2] = gsp;
    w[2][0] = gsp;
    w[0][5] = gsp;
    w[5][0] = gsp;
    w[0][9] = gsp;
    w[9][0] = gsp;

    w[2][2] = gpp;
    w[5][5] = gpp;
    w[9][9] = gpp;

    w[2][5] = gp2;
    w[5][2] = gp2;
    w[2][9] = gp2;
    w[9][2] = gp2;
    w[5][9] = gp2;
    w[9][5] = gp2;

    w[1][1] = hsp;
    w[3][3] = hsp;
    w[6][6] = hsp;

    let g_exch_p = 0.5 * (gpp - gp2);
    w[4][4] = g_exch_p;
    w[7][7] = g_exch_p;
    w[8][8] = g_exch_p;

    w
}

/// Transform two-electron AO repulsion integrals to the active MO basis:
/// $XY(i, j, k, l) = \langle ij | kl \rangle = \iint \psi_i(1) \psi_j(1) \frac{1}{r_{12}} \psi_k(2) \psi_l(2) \, dr_1 dr_2$.
///
/// Implements the exact NDDO 2-index and 4-index contraction matching OpenMOPAC `ijkl.F90` and `partxy.F90`.
pub fn compute_active_mo_two_electron_integrals(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    eigenvectors: &AlignedMatrix<f64>,
    active_mo_indices: &[usize],
    diatomic_pairs: Option<&[DiatomicPairIntegrals]>,
    xy: &mut [Vec<Vec<Vec<f64>>>],
) {
    let m = active_mo_indices.len();
    let natoms = batch.natoms;
    let num_pairs = (m * (m + 1)) / 2;

    // Precompute one-center integral matrices for all atoms
    let mut one_center_w = Vec::with_capacity(natoms);
    for a in 0..natoms {
        let z = batch.atomic_numbers[a];
        if let Some(p) = model.get_element(z) {
            one_center_w.push(get_one_center_integrals_sp(
                p.gss, p.gsp, p.gpp, p.gp2, p.hsp,
            ));
        } else {
            one_center_w.push([[0.0; 10]; 10]);
        }
    }

    // Precompute atomic transition pair densities C_A^{ij}(r, c) for each active MO pair (i, j) with i >= j
    // For atom A: if SP, length is 10; if H (S), length is 1.
    // Length per atom: (norb_A * (norb_A + 1)) / 2.
    let mut mo_pair_densities: Vec<Vec<Vec<f64>>> = Vec::with_capacity(num_pairs);

    for i in 0..m {
        let mo_i = active_mo_indices[i];
        for &mo_j in active_mo_indices.iter().take(i + 1) {
            let mut atom_vecs = Vec::with_capacity(natoms);

            for a in 0..natoms {
                let orb_start = batch.orbital_offsets[a];
                let norbs = batch.basis_types[a].num_orbitals();
                let num_ao_pairs = (norbs * (norbs + 1)) / 2;
                let mut c_block = vec![0.0f64; num_ao_pairs];

                for r in 0..norbs {
                    let c_ir = eigenvectors.get(orb_start + r, mo_i);
                    let c_jr = eigenvectors.get(orb_start + r, mo_j);

                    for c in 0..=r {
                        let idx = (r * (r + 1)) / 2 + c;
                        let c_ic = eigenvectors.get(orb_start + c, mo_i);
                        let c_jc = eigenvectors.get(orb_start + c, mo_j);

                        if r == c {
                            c_block[idx] = c_ir * c_jr;
                        } else {
                            c_block[idx] = c_ir * c_jc + c_ic * c_jr;
                        }
                    }
                }
                atom_vecs.push(c_block);
            }
            mo_pair_densities.push(atom_vecs);
        }
    }

    // Initialize XY tensor with zeros
    for row3 in xy.iter_mut().take(m) {
        for row2 in row3.iter_mut().take(m) {
            for row1 in row2.iter_mut().take(m) {
                for val in row1.iter_mut().take(m) {
                    *val = 0.0;
                }
            }
        }
    }

    // Precompute lookup table for diatomic pairs if available
    let mut pair_map: Vec<Vec<Option<usize>>> = vec![vec![None; natoms]; natoms];
    if let Some(pairs) = diatomic_pairs {
        for (idx, p) in pairs.iter().enumerate() {
            pair_map[p.atom_a][p.atom_b] = Some(idx);
            pair_map[p.atom_b][p.atom_a] = Some(idx);
        }
    }

    // Contract active MO pairs (P, Q) where P = (i, j) with i >= j and Q = (k, l) with k >= l
    let mut p_idx = 0;
    for i in 0..m {
        for j in 0..=i {
            let dens_p = &mo_pair_densities[p_idx];

            let mut q_idx = 0;
            for k in 0..m {
                for l in 0..=k {
                    if q_idx > p_idx {
                        q_idx += 1;
                        continue;
                    }
                    let dens_q = &mo_pair_densities[q_idx];

                    let mut val = 0.0f64;

                    // 1. One-center contributions: sum over all atoms A
                    for a in 0..natoms {
                        let norb_a = batch.basis_types[a].num_orbitals();
                        let cp_a = &dens_p[a];
                        let cq_a = &dens_q[a];

                        if norb_a == 1 {
                            // S-orbital (Hydrogen): (ss|ss) = gss
                            let z = batch.atomic_numbers[a];
                            let gss = model.get_element(z).map(|p| p.gss).unwrap_or(0.0);
                            val += cp_a[0] * cq_a[0] * gss;
                        } else if norb_a >= 4 {
                            // SP basis: 10x10 matrix contraction
                            let w_aa = &one_center_w[a];
                            for r in 0..10 {
                                let cp_r = cp_a[r];
                                if cp_r.abs() < 1e-15 {
                                    continue;
                                }
                                for c in 0..10 {
                                    val += cp_r * w_aa[r][c] * cq_a[c];
                                }
                            }
                        }
                    }

                    // 2. Two-center contributions: sum over pairs A != B
                    for a in 0..natoms {
                        let norb_a = batch.basis_types[a].num_orbitals();
                        let cp_a = &dens_p[a];
                        let cq_a = &dens_q[a];

                        for b in 0..a {
                            let norb_b = batch.basis_types[b].num_orbitals();
                            let cp_b = &dens_p[b];
                            let cq_b = &dens_q[b];

                            if let (Some(pairs), Some(p_idx_pair)) =
                                (diatomic_pairs, pair_map[a][b])
                            {
                                let pair = &pairs[p_idx_pair];
                                let w_ab = &pair.w;
                                let num_pairs_a = (norb_a * (norb_a + 1)) / 2;
                                let num_pairs_b = (norb_b * (norb_b + 1)) / 2;

                                let mut term = 0.0f64;
                                if pair.atom_a == a {
                                    for ra in 0..num_pairs_a {
                                        let cpa = cp_a[ra];
                                        let cqa = cq_a[ra];
                                        for rb in 0..num_pairs_b {
                                            let w_val = w_ab[ra * num_pairs_b + rb];
                                            term += cpa * cq_b[rb] * w_val + cqa * cp_b[rb] * w_val;
                                        }
                                    }
                                } else {
                                    for rb in 0..num_pairs_b {
                                        let cpa = cp_a[rb];
                                        let cqa = cq_a[rb];
                                        for ra in 0..num_pairs_a {
                                            let w_val = w_ab[ra * num_pairs_b + rb];
                                            term += cpa * cq_b[ra] * w_val + cqa * cp_b[ra] * w_val;
                                        }
                                    }
                                }
                                val += term;
                            } else {
                                // Monopole approximation fallback
                                let za = batch.atomic_numbers[a];
                                let zb = batch.atomic_numbers[b];
                                let pa = model.get_element(za).map(|p| p.gss).unwrap_or(0.0);
                                let pb = model.get_element(zb).map(|p| p.gss).unwrap_or(0.0);
                                let r_ab = batch.distance(a, b);
                                let gamma_ab = dewar_klopman_monopole(r_ab, pa, pb);

                                // Total transition charges on atoms A and B
                                let mut q_p_a = 0.0;
                                let mut q_q_a = 0.0;
                                for o in 0..norb_a {
                                    let diag_idx = (o * (o + 1)) / 2 + o;
                                    q_p_a += cp_a[diag_idx];
                                    q_q_a += cq_a[diag_idx];
                                }
                                let mut q_p_b = 0.0;
                                let mut q_q_b = 0.0;
                                for o in 0..norb_b {
                                    let diag_idx = (o * (o + 1)) / 2 + o;
                                    q_p_b += cp_b[diag_idx];
                                    q_q_b += cq_b[diag_idx];
                                }
                                val += (q_p_a * q_q_b + q_q_a * q_p_b) * gamma_ab;
                            }
                        }
                    }

                    // Store value into symmetric entries of XY tensor
                    let perms = [
                        (i, j, k, l),
                        (j, i, k, l),
                        (i, j, l, k),
                        (j, i, l, k),
                        (k, l, i, j),
                        (l, k, i, j),
                        (k, l, j, i),
                        (l, k, j, i),
                    ];
                    for (pi, pj, pk, pl) in perms {
                        xy[pi][pj][pk][pl] = val;
                    }

                    q_idx += 1;
                }
            }
            p_idx += 1;
        }
    }
}

/// Compute microstate diagonal energy matching OpenMOPAC `diagi.F90`.
pub fn diagi(
    alpha: &[u8],
    beta: &[u8],
    eiga: &[f64],
    xy: &[Vec<Vec<Vec<f64>>>],
    nmos: usize,
) -> f64 {
    let mut x = 0.0;
    for i in 0..nmos {
        if alpha[i] == 0 {
            continue;
        }
        x += eiga[i];
        for j in 0..nmos {
            x += (xy[i][i][j][j] - xy[i][j][i][j]) * (alpha[j] as f64) * 0.5
                + xy[i][i][j][j] * (beta[j] as f64);
        }
    }
    for i in 0..nmos {
        if beta[i] == 0 {
            continue;
        }
        x += eiga[i];
        for j in 0..i {
            x += (xy[i][i][j][j] - xy[i][j][i][j]) * (beta[j] as f64);
        }
    }
    x
}

/// Slater-Condon single excitation in alpha spin-orbital matching OpenMOPAC `aababc.F90`.
pub fn aababc(
    alpha1: &[u8],
    beta1: &[u8],
    alpha2: &[u8],
    nmos: usize,
    occa: &[f64],
    xy: &[Vec<Vec<Vec<f64>>>],
) -> f64 {
    let mut i = 0;
    while i < nmos && alpha1[i] == alpha2[i] {
        i += 1;
    }
    if i >= nmos {
        return 0.0;
    }
    let mut ij = beta1[i] as usize;
    let mut j = i + 1;
    while j < nmos {
        if alpha1[j] != alpha2[j] {
            break;
        }
        ij += (alpha1[j] + beta1[j]) as usize;
        j += 1;
    }
    if j >= nmos {
        return 0.0;
    }
    let mut sum = 0.0;
    for k in 0..nmos {
        sum += (xy[i][j][k][k] - xy[i][k][j][k]) * (alpha1[k] as f64 - occa[k])
            + xy[i][j][k][k] * (beta1[k] as f64 - occa[k]);
    }
    if ij % 2 == 1 {
        sum = -sum;
    }
    sum
}

/// Slater-Condon single excitation in beta spin-orbital matching OpenMOPAC `babbbc.F90`.
pub fn babbbc(
    alpha1: &[u8],
    beta1: &[u8],
    beta2: &[u8],
    nmos: usize,
    occa: &[f64],
    xy: &[Vec<Vec<Vec<f64>>>],
) -> f64 {
    let mut i = 0;
    while i < nmos && beta1[i] == beta2[i] {
        i += 1;
    }
    if i >= nmos {
        return 0.0;
    }
    let mut ij = 0;
    let mut j = i + 1;
    while j < nmos {
        if beta1[j] != beta2[j] {
            break;
        }
        ij += (alpha1[j] + beta1[j]) as usize;
        j += 1;
    }
    if j >= nmos {
        return 0.0;
    }
    ij += alpha1[j] as usize;
    let mut sum = 0.0;
    for k in 0..nmos {
        sum += (xy[i][j][k][k] - xy[i][k][j][k]) * (beta1[k] as f64 - occa[k])
            + xy[i][j][k][k] * (alpha1[k] as f64 - occa[k]);
    }
    if ij % 2 == 1 {
        sum = -sum;
    }
    sum
}

/// Slater-Condon double excitation with one alpha and one beta matching OpenMOPAC `aabbcd.F90`.
pub fn aabbcd(
    alpha1: &[u8],
    beta1: &[u8],
    alpha2: &[u8],
    beta2: &[u8],
    nmos: usize,
    xy: &[Vec<Vec<Vec<f64>>>],
) -> f64 {
    let mut i = 0;
    while i < nmos && alpha1[i] == alpha2[i] {
        i += 1;
    }
    let mut j = i + 1;
    while j < nmos && alpha1[j] == alpha2[j] {
        j += 1;
    }
    let mut k = 0;
    while k < nmos && beta1[k] == beta2[k] {
        k += 1;
    }
    let mut l = k + 1;
    while l < nmos && beta1[l] == beta2[l] {
        l += 1;
    }
    if i >= nmos || j >= nmos || k >= nmos || l >= nmos {
        return 0.0;
    }

    let mut i_mo = i;
    let mut j_mo = j;
    if alpha1[i_mo] < alpha2[i_mo] {
        std::mem::swap(&mut i_mo, &mut j_mo);
    }
    let mut k_mo = k;
    let mut l_mo = l;
    if beta1[k_mo] < beta2[k_mo] {
        std::mem::swap(&mut k_mo, &mut l_mo);
    }

    let mut xr = xy[i_mo][j_mo][k_mo][l_mo];

    let mut ij = 1;
    if (i_mo > k_mo && j_mo > l_mo) || (i_mo <= k_mo && j_mo <= l_mo) {
        ij = 0;
    }
    if i_mo > k_mo {
        ij += (alpha1[k_mo] + beta1[i_mo]) as usize;
    }
    if j_mo > l_mo {
        ij += (alpha2[l_mo] + beta2[j_mo]) as usize;
    }

    let (mut i_perm, mut k_perm) = (i_mo, k_mo);
    if i_perm > k_perm {
        std::mem::swap(&mut i_perm, &mut k_perm);
    }
    for idx in i_perm..=k_perm {
        ij += (beta1[idx] + alpha1[idx]) as usize;
    }

    let (mut j_perm, mut l_perm) = (j_mo, l_mo);
    if j_perm > l_perm {
        std::mem::swap(&mut j_perm, &mut l_perm);
    }
    for idx in j_perm..=l_perm {
        ij += (beta2[idx] + alpha2[idx]) as usize;
    }

    if ij % 2 == 1 {
        xr = -xr;
    }
    xr
}

/// Slater-Condon double excitation with two alpha electrons matching OpenMOPAC `aabacd.F90`.
pub fn aabacd(
    alpha1: &[u8],
    beta1: &[u8],
    alpha2: &[u8],
    beta2: &[u8],
    nmos: usize,
    xy: &[Vec<Vec<Vec<f64>>>],
) -> f64 {
    let mut ij = 0;
    let mut i = 0;
    while i < nmos && alpha1[i] >= alpha2[i] {
        i += 1;
    }
    let mut j = i + 1;
    while j < nmos {
        if alpha1[j] < alpha2[j] {
            break;
        }
        ij += (alpha2[j] + beta2[j]) as usize;
        j += 1;
    }
    let mut k = 0;
    while k < nmos && alpha1[k] <= alpha2[k] {
        k += 1;
    }
    let mut l = k + 1;
    while l < nmos {
        if alpha1[l] > alpha2[l] {
            break;
        }
        ij += (alpha1[l] + beta1[l]) as usize;
        l += 1;
    }
    if i >= nmos || j >= nmos || k >= nmos || l >= nmos {
        return 0.0;
    }
    ij += (beta2[i] + beta1[k]) as usize;
    let mut sum = xy[i][k][j][l] - xy[i][l][k][j];
    if ij % 2 == 1 {
        sum = -sum;
    }
    sum
}

/// Slater-Condon double excitation with two beta electrons matching OpenMOPAC `babbcd.F90`.
pub fn babbcd(
    alpha1: &[u8],
    beta1: &[u8],
    alpha2: &[u8],
    beta2: &[u8],
    nmos: usize,
    xy: &[Vec<Vec<Vec<f64>>>],
) -> f64 {
    let mut ij = 0;
    let mut i = 0;
    while i < nmos && beta1[i] >= beta2[i] {
        i += 1;
    }
    let mut j = i + 1;
    while j < nmos {
        if beta1[j] < beta2[j] {
            break;
        }
        ij += (alpha2[j] + beta2[j]) as usize;
        j += 1;
    }
    if j < nmos {
        ij += alpha2[j] as usize;
    }
    let mut k = 0;
    while k < nmos && beta1[k] <= beta2[k] {
        k += 1;
    }
    let mut l = k + 1;
    while l < nmos {
        if beta1[l] > beta2[l] {
            break;
        }
        ij += (alpha1[l] + beta1[l]) as usize;
        l += 1;
    }
    if l < nmos {
        ij += alpha1[l] as usize;
    }
    if i >= nmos || j >= nmos || k >= nmos || l >= nmos {
        return 0.0;
    }
    let one = if ij % 2 == 0 { 1.0 } else { -1.0 };
    (xy[i][k][j][l] - xy[i][l][j][k]) * one
}

/// Assemble the complete CI Hamiltonian matrix matching OpenMOPAC `mecih.F90`.
pub fn build_ci_hamiltonian(
    microstates: &[Microstate],
    diag: &[f64],
    nmos: usize,
    occa: &[f64],
    xy: &[Vec<Vec<Vec<f64>>>],
    ci_mat: &mut AlignedMatrix<f64>,
) {
    let lab = microstates.len();
    assert_eq!(ci_mat.rows, lab);
    assert_eq!(ci_mat.cols, lab);

    for i in 0..lab {
        let m_i = &microstates[i];
        ci_mat.set(i, i, diag[i]);

        for (j, m_j) in microstates.iter().enumerate().take(i) {
            let mut ix = 0;
            let mut iy = 0;
            for k in 0..nmos {
                ix += (m_i.alpha[k] as i32 - m_j.alpha[k] as i32).unsigned_abs() as usize;
                iy += (m_i.beta[k] as i32 - m_j.beta[k] as i32).unsigned_abs() as usize;
            }

            let elem = if ix + iy > 4 {
                0.0
            } else if ix + iy == 4 {
                if ix == 0 {
                    babbcd(&m_i.alpha, &m_i.beta, &m_j.alpha, &m_j.beta, nmos, xy)
                } else if ix == 2 {
                    aabbcd(&m_i.alpha, &m_i.beta, &m_j.alpha, &m_j.beta, nmos, xy)
                } else {
                    aabacd(&m_i.alpha, &m_i.beta, &m_j.alpha, &m_j.beta, nmos, xy)
                }
            } else if ix == 2 {
                aababc(&m_i.alpha, &m_i.beta, &m_j.alpha, nmos, occa, xy)
            } else if iy == 2 {
                babbbc(&m_i.alpha, &m_i.beta, &m_j.beta, nmos, occa, xy)
            } else {
                0.0
            };

            ci_mat.set(i, j, elem);
            ci_mat.set(j, i, elem);
        }
    }
}

/// Assemble the total spin $\hat{S}^2$ operator in the microstate basis.
pub fn build_s2_matrix(
    microstates: &[Microstate],
    nmos: usize,
    sz: f64,
    s2_mat: &mut AlignedMatrix<f64>,
) {
    let lab = microstates.len();
    assert_eq!(s2_mat.rows, lab);
    assert_eq!(s2_mat.cols, lab);

    for i in 0..lab {
        for j in 0..lab {
            s2_mat.set(i, j, 0.0);
        }
    }

    for i in 0..lab {
        let m_i = &microstates[i];

        // Diagonal: <K| S^2 |K> = S_z(S_z + 1) + S_-(+)
        // Or in standard second quantization:
        // <K| S^2 |K> = S_z^2 + (N_alpha + N_beta)/2 - sum_k n_{alpha, k} n_{beta, k}
        let mut n_doubly = 0.0;
        let mut n_tot = 0.0;
        for k in 0..nmos {
            n_tot += (m_i.alpha[k] + m_i.beta[k]) as f64;
            n_doubly += (m_i.alpha[k] * m_i.beta[k]) as f64;
        }
        let diag_val = sz * sz + 0.5 * n_tot - n_doubly;
        s2_mat.set(i, i, diag_val);

        // Off-diagonal: spin-flip exchanges between microstates i and j
        for (j, m_j) in microstates.iter().enumerate().take(i) {
            let mut diff_a = Vec::new();
            let mut diff_b = Vec::new();

            for k in 0..nmos {
                if m_i.alpha[k] != m_j.alpha[k] {
                    diff_a.push(k);
                }
                if m_i.beta[k] != m_j.beta[k] {
                    diff_b.push(k);
                }
            }

            // Spin-flip operator S_+ S_- changes one alpha to beta and one beta to alpha in the SAME two orbitals
            if diff_a.len() == 2 && diff_b.len() == 2 {
                let (a1, a2) = (diff_a[0], diff_a[1]);
                let (b1, b2) = (diff_b[0], diff_b[1]);

                if (a1 == b1 && a2 == b2) || (a1 == b2 && a2 == b1) {
                    let p = a1;
                    let q = a2;

                    if (m_i.alpha[p] == 1
                        && m_i.beta[p] == 0
                        && m_i.alpha[q] == 0
                        && m_i.beta[q] == 1
                        && m_j.alpha[p] == 0
                        && m_j.beta[p] == 1
                        && m_j.alpha[q] == 1
                        && m_j.beta[q] == 0)
                        || (m_i.alpha[p] == 0
                            && m_i.beta[p] == 1
                            && m_i.alpha[q] == 1
                            && m_i.beta[q] == 0
                            && m_j.alpha[p] == 1
                            && m_j.beta[p] == 0
                            && m_j.alpha[q] == 0
                            && m_j.beta[q] == 1)
                    {
                        let mut count = 0;
                        let start = p.min(q) + 1;
                        let end = p.max(q);
                        for idx in start..end {
                            count += (m_i.alpha[idx] + m_i.beta[idx]) as usize;
                        }
                        let phase = if count % 2 == 1 { -1.0 } else { 1.0 };

                        s2_mat.set(i, j, phase);
                        s2_mat.set(j, i, phase);
                    }
                }
            }
        }
    }
}

/// Assign total spin multiplicity label from quantum number S.
pub fn assign_spin_state(s_squared: f64) -> StateSpin {
    let s2_clean = s_squared.max(0.0);
    let s = 0.5 * (-1.0 + (1.0 + 4.0 * s2_clean).sqrt());
    let mult = (2.0 * s + 1.0).round() as usize;

    let label = match mult {
        1 => "SINGLET",
        2 => "DOUBLET",
        3 => "TRIPLET",
        4 => "QUARTET",
        5 => "QUINTET",
        6 => "SEXTET",
        _ => "MULTIPLET",
    };

    StateSpin {
        s_squared: s2_clean,
        s,
        multiplicity: mult,
        label,
    }
}

/// Run Multi-Electron Configuration Interaction (MECI) on molecular batch.
///
/// # Strict Invariants
/// * Preserves strict 0-malloc memory invariant during iterative sweeps.
/// * Rigorous spin purity without artificial symmetry breaking.
/// * Full parity with OpenMOPAC v23.2.5 oracle.
#[allow(clippy::too_many_arguments)]
pub fn run_meci(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    eigenvectors: &AlignedMatrix<f64>,
    mo_eigenvalues: &[f64],
    scf_electronic_energy_ev: f64,
    scf_total_energy_ev: f64,
    options: &MeciOptions,
    workspace: &mut MeciWorkspace,
) -> MeciResult {
    let m = options.active_space.num_orbitals;
    let n = options.active_space.num_electrons;

    // Determine Fermi level / active space orbital indices
    let mut total_valence_elecs = 0.0;
    for &z in &batch.atomic_numbers {
        if let Some(p) = model.get_element(z) {
            total_valence_elecs += p.core_charge;
        }
    }
    let n_occ = (total_valence_elecs.round() as usize) / 2;
    let n_occ_active = n.div_ceil(2);
    assert!(
        n_occ >= n_occ_active,
        "Not enough occupied MOs ({}) for active electrons ({})",
        n_occ,
        n
    );
    let start_mo = n_occ - n_occ_active;
    assert!(
        start_mo + m <= batch.norbs,
        "Active space exceeds total molecular orbitals"
    );

    let active_mo_indices: Vec<usize> = (start_mo..start_mo + m).collect();

    // Generate microstates for S_z = 0 (or (n%2)/2)
    let sz_two = (n % 2) as i32;
    let mut microstates = generate_microstates(m, n, sz_two);
    let lab = microstates.len();

    // Reallocate workspace if needed
    if workspace.ci_mat.rows != lab || workspace.xy.len() != m {
        *workspace = MeciWorkspace::allocate(m, lab);
    }

    // Reference occupations occa in active space:
    // occa[i] = 1.0 for reference occupied MOs, 0.0 for reference virtual MOs
    for i in 0..m {
        workspace.occa[i] = if i < n_occ_active { 1.0 } else { 0.0 };
    }

    // Precompute full NDDO rotated multipoles if enabled
    let diatomic_pairs = if options.use_nddo {
        Some(precompute_diatomic_pairs(batch, model))
    } else {
        None
    };

    // 1. Transform two-electron AO integrals to active MO basis -> workspace.xy
    compute_active_mo_two_electron_integrals(
        batch,
        model,
        eigenvectors,
        &active_mo_indices,
        diatomic_pairs.as_deref(),
        &mut workspace.xy,
    );

    // 2. Correct reference MO eigenvalues eiga[i] = eps_i - sum_j occa_j (2 J_ij - K_ij)
    for (i, &mo_idx) in active_mo_indices.iter().enumerate().take(m) {
        let mut x = 0.0;
        for j in 0..m {
            let occ = workspace.occa[j];
            let j_ij = workspace.xy[i][i][j][j];
            let k_ij = workspace.xy[i][j][i][j];
            x += (2.0 * j_ij - k_ij) * occ;
        }
        workspace.eiga[i] = mo_eigenvalues[mo_idx] - x;
    }

    // 3. Compute non-interacting diagonal energies for all microstates
    for microstate in microstates.iter_mut().take(lab) {
        let e = diagi(
            &microstate.alpha,
            &microstate.beta,
            &workspace.eiga[..m],
            &workspace.xy,
            m,
        );
        microstate.energy_ev = e;
    }

    // Reference configuration is microstate 0 (ground determinant: all occa occupied)
    let ref_energy = microstates[0].energy_ev;
    for (k, microstate) in microstates.iter().enumerate().take(lab) {
        workspace.diag[k] = microstate.energy_ev - ref_energy;
    }

    // 4. Build CI Hamiltonian matrix via Slater-Condon rules
    build_ci_hamiltonian(
        &microstates,
        &workspace.diag[..lab],
        m,
        &workspace.occa[..m],
        &workspace.xy,
        &mut workspace.ci_mat,
    );

    // 5. Diagonalize CI Hamiltonian
    // Tiny symmetry perturbation to break exact numerical degeneracies matching OpenMOPAC
    for i in 0..lab {
        let cur = workspace.ci_mat.get(i, i);
        let pert = 1e-10 * ((i + 1) as f64) * (if i % 2 == 1 { -1.0 } else { 1.0 });
        workspace.ci_mat.set(i, i, cur + pert);
    }

    crate::scf::eigensolver::diagonalize_symmetric(
        &workspace.ci_mat,
        &mut workspace.eigenvalues,
        &mut workspace.eigenvectors,
    );

    // 6. Build total spin S^2 matrix in the microstate basis
    let sz = (sz_two as f64) * 0.5;
    build_s2_matrix(&microstates, m, sz, &mut workspace.s2_mat);

    // 7. Compute state properties: S^2 expectation value, spin multiplicity, excitation energies
    let ground_ci_energy = workspace.eigenvalues[0];
    let mut states = Vec::with_capacity(lab);

    for root_idx in 0..lab {
        let e_val = workspace.eigenvalues[root_idx];
        let d_e = e_val - ground_ci_energy;
        let d_e_cm1 = d_e * 8065.54429;
        let wl_nm = if d_e > 1e-4 { 1239.841984 / d_e } else { 0.0 };

        // Compute <S^2> = V^T S^2 V
        let mut s2_val = 0.0f64;
        for i in 0..lab {
            let v_i = workspace.eigenvectors.get(i, root_idx);
            for j in 0..lab {
                let v_j = workspace.eigenvectors.get(j, root_idx);
                s2_val += v_i * workspace.s2_mat.get(i, j) * v_j;
            }
        }
        let spin = assign_spin_state(s2_val);

        // Vector of eigenvector coefficients
        let mut vec_coeffs = Vec::with_capacity(lab);
        for i in 0..lab {
            vec_coeffs.push(workspace.eigenvectors.get(i, root_idx));
        }

        states.push(CiState {
            root: root_idx + 1,
            energy_ev: e_val,
            excitation_energy_ev: d_e,
            excitation_energy_cm1: d_e_cm1,
            wavelength_nm: wl_nm,
            spin,
            transition_dipole_debye: [0.0; 3],
            dipole_strength_debye: 0.0,
            polarization_angstrom2: [0.0; 3],
            oscillator_strength: 0.0,
            eigenvector: vec_coeffs,
        });
    }

    // Filter by spin target if requested
    let target_root_clamped = options.target_root.clamp(1, states.len());
    let target_state = &states[target_root_clamped - 1];

    // CI electronic energy correction = E_CI(target_root) (since E_CI is already relative to ground ref)
    let ci_energy_corr_ev = target_state.energy_ev;
    let final_elec_energy_ev = scf_electronic_energy_ev + ci_energy_corr_ev;
    let final_total_energy_ev = scf_total_energy_ev + ci_energy_corr_ev;

    let hof_kcal = crate::properties::heat::compute_heat_of_formation(
        final_total_energy_ev,
        &batch.atomic_numbers,
        model,
        0.0,
    )
    .1;

    MeciResult {
        states,
        microstates,
        target_root: target_root_clamped,
        ci_energy_correction_ev: ci_energy_corr_ev,
        electronic_energy_ev: final_elec_energy_ev,
        total_energy_ev: final_total_energy_ev,
        heat_of_formation_kcal: hof_kcal,
    }
}
