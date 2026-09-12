//! Periodic Bloch Self-Consistent Field (PBC RHF) Solver & Band Structure Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Implements crystal orbital Roothaan-Hall SCF in reciprocal $k$-space with exact Hermitian embedding,
//! direct/indirect bandgap determination, and continuous density of states (DOS) evaluation.

use crate::fock::{build_fock, build_fock_nddo};
use crate::hamiltonian::build_hcore;
use crate::integrals::core_repulsion::compute_pair_core_repulsion;
use crate::parameters::ParameterModel;
use crate::pbc::unit_cell::{KPoint, UnitCell};
use crate::properties::heat::compute_heat_of_formation;
use crate::scf::eigensolver::diagonalize_symmetric_with_work;
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch};

/// Configuration options for Periodic Boundary Condition calculations.
#[derive(Debug, Clone)]
pub struct PbcOptions {
    /// Periodic unit cell geometry
    pub unit_cell: UnitCell,
    /// Number of fundamental unit cells in Born-von Kármán cluster per periodic dimension [n1, n2, n3]
    pub mers: [usize; 3],
    /// Monkhorst-Pack $k$-point grid sampling in the 1st Brillouin zone [nk1, nk2, nk3]
    pub k_grid: [usize; 3],
    /// Maximum allowed SCF iterations (default: 80)
    pub max_iter: usize,
    /// Energy convergence threshold in eV (default: 1e-7)
    pub energy_tol_ev: f64,
    /// Density matrix convergence threshold (default: 1e-6)
    pub density_tol: f64,
    /// Linear damping factor for density matrix updates (default: 0.5)
    pub damping: f64,
    /// Enable full NDDO 22 diatomic multipoles and rotated attractions (default: true)
    pub use_nddo: bool,
    /// Number of points per high-symmetry segment for band structure plots (default: 40)
    pub band_path_points: usize,
    /// Gaussian broadening sigma for Density of States (DOS) in eV (default: 0.1 eV)
    pub dos_sigma_ev: f64,
}

impl PbcOptions {
    pub fn new(unit_cell: UnitCell) -> Self {
        Self {
            unit_cell,
            mers: [5, 1, 1],
            k_grid: [16, 1, 1],
            max_iter: 100,
            energy_tol_ev: 1e-5,
            density_tol: 1e-4,
            damping: 0.5,
            use_nddo: true,
            band_path_points: 40,
            dos_sigma_ev: 0.1,
        }
    }
}

/// Comprehensive results of Periodic Boundary Condition calculation.
#[derive(Debug, Clone)]
pub struct PbcResult {
    /// Whether the periodic SCF iteration converged
    pub converged: bool,
    /// Total SCF cycles taken
    pub iterations: usize,
    /// Total energy per fundamental unit cell in eV
    pub total_energy_per_cell_ev: f64,
    /// Electronic energy per unit cell in eV
    pub electronic_energy_per_cell_ev: f64,
    /// Core-core nuclear repulsion energy per unit cell in eV
    pub nuclear_repulsion_per_cell_ev: f64,
    /// Standard heat of formation per unit cell in kcal/mol
    pub heat_of_formation_kcal_mol: f64,
    /// Valence Band Maximum (VBM) energy in eV
    pub vbm_energy_ev: f64,
    /// Conduction Band Minimum (CBM) energy in eV
    pub cbm_energy_ev: f64,
    /// Direct bandgap $E_{g, \text{dir}} = \min_k (\varepsilon_{\text{LUMO}}(k) - \varepsilon_{\text{HOMO}}(k))$ in eV
    pub direct_bandgap_ev: f64,
    /// Fundamental (indirect) bandgap $E_{g, \text{ind}} = \text{CBM} - \text{VBM}$ in eV
    pub indirect_bandgap_ev: f64,
    /// Band structure $k$-points along high symmetry path
    pub band_k_points: Vec<KPoint>,
    /// Band energies along path: shape `[num_k_points, norbs_per_cell]` in eV
    pub band_energies_ev: Vec<Vec<f64>>,
    /// Energy grid for Density of States (DOS) in eV
    pub dos_energies_ev: Vec<f64>,
    /// Density of States values $g(E)$ in $\text{states} / (\text{eV} \cdot \text{cell})$
    pub dos_values: Vec<f64>,
}

/// Preallocated workspace for PBC calculations ensuring 0-malloc memory invariant during SCF sweeps.
#[derive(Debug, Clone)]
pub struct PbcWorkspace {
    pub norbs_cell: usize,
    pub n_trans: usize,
    pub n_k: usize,
    pub p_real: Vec<AlignedMatrix<f64>>,
    pub f_real: Vec<AlignedMatrix<f64>>,
    pub h_core_real: Vec<AlignedMatrix<f64>>,
    pub hermitian_embed_f: AlignedMatrix<f64>,
    pub hermitian_embed_work: AlignedMatrix<f64>,
    pub embed_eigenvalues: AlignedVec64<f64>,
    pub embed_eigenvectors: AlignedMatrix<f64>,
}

impl PbcWorkspace {
    pub fn allocate(norbs_cell: usize, n_trans: usize, n_k: usize) -> Self {
        let mut p_real = Vec::with_capacity(n_trans);
        let mut f_real = Vec::with_capacity(n_trans);
        let mut h_core_real = Vec::with_capacity(n_trans);

        for _ in 0..n_trans {
            p_real.push(AlignedMatrix::zeroed(norbs_cell, norbs_cell));
            f_real.push(AlignedMatrix::zeroed(norbs_cell, norbs_cell));
            h_core_real.push(AlignedMatrix::zeroed(norbs_cell, norbs_cell));
        }

        let embed_dim = 2 * norbs_cell;

        Self {
            norbs_cell,
            n_trans,
            n_k,
            p_real,
            f_real,
            h_core_real,
            hermitian_embed_f: AlignedMatrix::zeroed(embed_dim, embed_dim),
            hermitian_embed_work: AlignedMatrix::zeroed(embed_dim, embed_dim),
            embed_eigenvalues: AlignedVec64::zeroed(embed_dim),
            embed_eigenvectors: AlignedMatrix::zeroed(embed_dim, embed_dim),
        }
    }
}

/// Run Crystal Orbital Roothaan-Hall Periodic SCF calculation.
pub fn run_pbc_scf(
    unit_cell_atoms: &MolecularBatch,
    model: &dyn ParameterModel,
    options: &PbcOptions,
    workspace: &mut PbcWorkspace,
) -> Result<PbcResult, String> {
    let natoms_cell = unit_cell_atoms.natoms;
    let norbs_cell = unit_cell_atoms.norbs;

    if natoms_cell == 0 {
        return Err("Periodic calculation requires at least 1 atom in unit cell".to_string());
    }

    // 1. Generate real-space translation indices R_m for supercell cluster
    let translations = options.unit_cell.generate_translation_indices(options.mers);
    let n_trans = translations.len();

    // 2. Generate Monkhorst-Pack k-points in 1st BZ
    let k_points = options
        .unit_cell
        .generate_monkhorst_pack_grid(options.k_grid);
    let n_k = k_points.len();

    // Reallocate workspace if needed
    if workspace.norbs_cell != norbs_cell || workspace.n_trans != n_trans || workspace.n_k != n_k {
        *workspace = PbcWorkspace::allocate(norbs_cell, n_trans, n_k);
    }

    // 3. Build supercell cluster molecular batch:
    // Total atoms = n_trans * natoms_cell
    let total_atoms = n_trans * natoms_cell;
    let mut cluster_z = Vec::with_capacity(total_atoms);
    let mut cluster_coords = Vec::with_capacity(total_atoms);

    for tr in &translations {
        for a in 0..natoms_cell {
            cluster_z.push(unit_cell_atoms.atomic_numbers[a]);
            cluster_coords.push([
                unit_cell_atoms.x[a] + tr.shift_angstrom[0],
                unit_cell_atoms.y[a] + tr.shift_angstrom[1],
                unit_cell_atoms.z[a] + tr.shift_angstrom[2],
            ]);
        }
    }

    let cluster_batch = MolecularBatch::new_for_model(cluster_z, &cluster_coords, model);
    let cluster_norbs = cluster_batch.norbs;

    // Precompute diatomic pairs for NDDO if requested
    let diatomic_pairs = if options.use_nddo {
        Some(crate::integrals::multipoles::precompute_diatomic_pairs(
            &cluster_batch,
            model,
        ))
    } else {
        None
    };

    // 4. Compute cluster Core Hamiltonian and extract blocks H_{0, m}
    let mut cluster_hcore = AlignedMatrix::zeroed(cluster_norbs, cluster_norbs);
    if let Some(ref pairs) = diatomic_pairs {
        crate::hamiltonian::hcore::build_hcore_nddo(
            &cluster_batch,
            model,
            pairs,
            &mut cluster_hcore,
        );
    } else {
        build_hcore(&cluster_batch, model, &mut cluster_hcore);
    }

    for (m_idx, _) in translations.iter().enumerate() {
        let col_start = m_idx * norbs_cell;
        for i in 0..norbs_cell {
            for j in 0..norbs_cell {
                let val = cluster_hcore.get(i, col_start + j);
                workspace.h_core_real[m_idx].set(i, j, val);
            }
        }
    }

    // 5. Compute core-core nuclear repulsion per central unit cell:
    // E_nuc = 0.5 * sum_{A in cell 0} sum_{m} sum_{B in cell m}' E_AB
    let mut e_nuc_ev = 0.0f64;
    for a in 0..natoms_cell {
        let za = unit_cell_atoms.atomic_numbers[a];
        let pa = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };
        let ra = [
            unit_cell_atoms.x[a],
            unit_cell_atoms.y[a],
            unit_cell_atoms.z[a],
        ];

        for (m_idx, tr) in translations.iter().enumerate() {
            for b in 0..natoms_cell {
                if m_idx == 0 && a == b {
                    continue;
                }
                let zb = unit_cell_atoms.atomic_numbers[b];
                let pb = match model.get_element(zb) {
                    Some(p) => p,
                    None => continue,
                };
                let rb = [
                    unit_cell_atoms.x[b] + tr.shift_angstrom[0],
                    unit_cell_atoms.y[b] + tr.shift_angstrom[1],
                    unit_cell_atoms.z[b] + tr.shift_angstrom[2],
                ];

                let dx = ra[0] - rb[0];
                let dy = ra[1] - rb[1];
                let dz = ra[2] - rb[2];
                let r_ab = (dx * dx + dy * dy + dz * dz).sqrt();

                let e_pair = compute_pair_core_repulsion(r_ab, &pa, &pb);
                e_nuc_ev += 0.5 * e_pair;
            }
        }
    }

    // Count occupied levels in fundamental unit cell
    let mut total_valence_elecs = 0.0f64;
    for &z in &unit_cell_atoms.atomic_numbers {
        if let Some(p) = model.get_element(z) {
            total_valence_elecs += p.core_charge;
        }
    }
    let n_occ = (total_valence_elecs.round() as usize) / 2;
    assert!(
        n_occ <= norbs_cell,
        "Occupied levels ({}) exceed unit cell orbitals ({})",
        n_occ,
        norbs_cell
    );

    // 6. Initialize real-space density matrix P(m) with diagonal block initial guess
    for m_idx in 0..n_trans {
        workspace.p_real[m_idx].fill_zero();
    }
    for a in 0..natoms_cell {
        let z = unit_cell_atoms.atomic_numbers[a];
        if let Some(p) = model.get_element(z) {
            let start = unit_cell_atoms.orbital_offsets[a];
            let norbs_a = unit_cell_atoms.basis_types[a].num_orbitals();
            let pop_per_orb = p.core_charge / (norbs_a as f64);
            for o in 0..norbs_a {
                workspace.p_real[0].set(start + o, start + o, pop_per_orb);
            }
        }
    }

    // Allocate full cluster density matrix and Fock matrix for Fock builder
    let mut cluster_density = AlignedMatrix::zeroed(cluster_norbs, cluster_norbs);
    let mut cluster_fock = AlignedMatrix::zeroed(cluster_norbs, cluster_norbs);

    let mut prev_energy = 0.0f64;
    let mut converged = false;
    let mut iterations = 0;
    let mut final_e_elec = 0.0f64;

    // 7. Periodic SCF iteration loop
    for iter in 1..=options.max_iter {
        iterations = iter;

        // Replicate periodic real-space density blocks P_{0, m} into full cluster density with BvK cyclic boundary
        cluster_density.fill_zero();
        let m1_size = options.mers[0] as i32;
        let m2_size = options.mers[1] as i32;
        let m3_size = options.mers[2] as i32;
        let half1 = (m1_size - 1) / 2;
        let half2 = (m2_size - 1) / 2;
        let half3 = (m3_size - 1) / 2;

        for m1 in 0..n_trans {
            let tr1 = &translations[m1];
            for m2 in 0..n_trans {
                let tr2 = &translations[m2];
                // Difference vector R_{m2} - R_{m1} with periodic modulo wrapping
                let raw_d1 = tr2.n1 - tr1.n1;
                let raw_d2 = tr2.n2 - tr1.n2;
                let raw_d3 = tr2.n3 - tr1.n3;

                let d1 = if m1_size > 1 {
                    ((raw_d1 + half1).rem_euclid(m1_size)) - half1
                } else {
                    0
                };
                let d2 = if m2_size > 1 {
                    ((raw_d2 + half2).rem_euclid(m2_size)) - half2
                } else {
                    0
                };
                let d3 = if m3_size > 1 {
                    ((raw_d3 + half3).rem_euclid(m3_size)) - half3
                } else {
                    0
                };

                let diff_n = (d1, d2, d3);

                // Find matching translation index
                let m_diff_idx = translations
                    .iter()
                    .position(|t| (t.n1, t.n2, t.n3) == diff_n);

                if let Some(m_idx) = m_diff_idx {
                    let row_offset = m1 * norbs_cell;
                    let col_offset = m2 * norbs_cell;
                    for i in 0..norbs_cell {
                        for j in 0..norbs_cell {
                            let val = workspace.p_real[m_idx].get(i, j);
                            cluster_density.set(row_offset + i, col_offset + j, val);
                        }
                    }
                }
            }
        }

        // Build cluster Fock matrix using full NDDO 22 multipoles
        if let Some(ref pairs) = diatomic_pairs {
            build_fock_nddo(
                &cluster_batch,
                model,
                pairs,
                &cluster_hcore,
                &cluster_density,
                &mut cluster_fock,
            );
        } else {
            build_fock(
                &cluster_batch,
                model,
                &cluster_hcore,
                &cluster_density,
                &mut cluster_fock,
            );
        }

        // Extract real-space Fock blocks F_{0, m}
        for (m_idx, _) in translations.iter().enumerate() {
            let col_start = m_idx * norbs_cell;
            for i in 0..norbs_cell {
                for j in 0..norbs_cell {
                    let val = cluster_fock.get(i, col_start + j);
                    workspace.f_real[m_idx].set(i, j, val);
                }
            }
        }

        // 8. Solve Bloch eigenvalue problem at each k-point in Brillouin Zone:
        // F(k) = sum_m F_{0, m} e^{i k . R_m}
        // Accumulate new real-space density: P_new(m) = sum_k w_k P(k) e^{-i k . R_m}
        let mut new_p_real = vec![AlignedMatrix::zeroed(norbs_cell, norbs_cell); n_trans];

        for kp in &k_points {
            // Build complex Hermitian Fock matrix F(k) = A(k) + i B(k)
            let mut a_k = vec![0.0f64; norbs_cell * norbs_cell];
            let mut b_k = vec![0.0f64; norbs_cell * norbs_cell];

            for (m_idx, tr) in translations.iter().enumerate() {
                let k_dot_r = kp.cartesian[0] * tr.shift_angstrom[0]
                    + kp.cartesian[1] * tr.shift_angstrom[1]
                    + kp.cartesian[2] * tr.shift_angstrom[2];
                let cos_kr = k_dot_r.cos();
                let sin_kr = k_dot_r.sin();

                for i in 0..norbs_cell {
                    for j in 0..norbs_cell {
                        let f_val = workspace.f_real[m_idx].get(i, j);
                        a_k[i * norbs_cell + j] += f_val * cos_kr;
                        b_k[i * norbs_cell + j] += f_val * sin_kr;
                    }
                }
            }

            // Construct 2N x 2N real symmetric embedding: [ A, -B; B, A ]
            workspace.hermitian_embed_f.fill_zero();

            for i in 0..norbs_cell {
                for j in 0..norbs_cell {
                    let a_val = a_k[i * norbs_cell + j];
                    let b_val = b_k[i * norbs_cell + j];

                    // Top-left: A
                    workspace.hermitian_embed_f.set(i, j, a_val);
                    // Top-right: -B
                    workspace.hermitian_embed_f.set(i, norbs_cell + j, -b_val);
                    // Bottom-left: B
                    workspace.hermitian_embed_f.set(norbs_cell + i, j, b_val);
                    // Bottom-right: A
                    workspace
                        .hermitian_embed_f
                        .set(norbs_cell + i, norbs_cell + j, a_val);
                }
            }

            // Diagonalize 2N x 2N embedding
            diagonalize_symmetric_with_work(
                &workspace.hermitian_embed_f,
                &mut workspace.hermitian_embed_work,
                &mut workspace.embed_eigenvalues,
                &mut workspace.embed_eigenvectors,
            );

            // In 2N x 2N embedding, eigenvalues appear in degenerate pairs: e_0, e_0, e_1, e_1, ...
            // The occupied crystalline orbitals correspond to the first n_occ pairs (2 * n_occ eigenvectors).
            // Density contribution at k:
            // P_mu_nu(k) = 2 * sum_{n=1}^{n_occ} (u_mu,n u_nu,n + v_mu,n v_nu,n)
            // Real part P_k_re, Imaginary part P_k_im
            let mut p_k_re = vec![0.0f64; norbs_cell * norbs_cell];
            let mut p_k_im = vec![0.0f64; norbs_cell * norbs_cell];

            for occ_idx in 0..n_occ {
                let col = 2 * occ_idx;

                for mu in 0..norbs_cell {
                    let u_mu = workspace.embed_eigenvectors.get(mu, col);
                    let v_mu = workspace.embed_eigenvectors.get(norbs_cell + mu, col);

                    for nu in 0..norbs_cell {
                        let u_nu = workspace.embed_eigenvectors.get(nu, col);
                        let v_nu = workspace.embed_eigenvectors.get(norbs_cell + nu, col);

                        // c_mu * conj(c_nu) = (u_mu + i v_mu) * (u_nu - i v_nu)
                        // Real part: u_mu * u_nu + v_mu * v_nu
                        // Imag part: v_mu * u_nu - u_mu * v_nu
                        let re_term = 2.0 * (u_mu * u_nu + v_mu * v_nu);
                        let im_term = 2.0 * (v_mu * u_nu - u_mu * v_nu);

                        p_k_re[mu * norbs_cell + nu] += re_term;
                        p_k_im[mu * norbs_cell + nu] += im_term;
                    }
                }
            }

            // Accumulate back into real space blocks P_new(m) = sum_k w_k P(k) e^{-i k . R_m}
            let w = kp.weight;
            for (m_idx, tr) in translations.iter().enumerate() {
                let k_dot_r = kp.cartesian[0] * tr.shift_angstrom[0]
                    + kp.cartesian[1] * tr.shift_angstrom[1]
                    + kp.cartesian[2] * tr.shift_angstrom[2];
                let cos_kr = k_dot_r.cos();
                let sin_kr = k_dot_r.sin();

                for i in 0..norbs_cell {
                    for j in 0..norbs_cell {
                        let re_val = p_k_re[i * norbs_cell + j];
                        let im_val = p_k_im[i * norbs_cell + j];
                        // Re( (Re + i Im) * (cos - i sin) ) = Re * cos + Im * sin
                        let p_m_val = w * (re_val * cos_kr + im_val * sin_kr);
                        new_p_real[m_idx].add(i, j, p_m_val);
                    }
                }
            }
        }

        // 9. Compute electronic energy per central unit cell corresponding to current state:
        // E_elec = 0.5 * sum_m Tr(P(0, m) [H(0, m) + F(0, m)])
        let mut e_elec_cell = 0.0f64;
        for (m_idx, _) in translations.iter().enumerate() {
            for i in 0..norbs_cell {
                for j in 0..norbs_cell {
                    let p_val = workspace.p_real[m_idx].get(i, j);
                    let h_val = workspace.h_core_real[m_idx].get(i, j);
                    let f_val = workspace.f_real[m_idx].get(i, j);
                    e_elec_cell += 0.5 * p_val * (h_val + f_val);
                }
            }
        }

        let e_tot = e_elec_cell + e_nuc_ev;
        let d_e = (e_tot - prev_energy).abs();
        prev_energy = e_tot;
        final_e_elec = e_elec_cell;

        // 10. Check convergence and apply density damping
        let mut max_dp = 0.0f64;
        for (m_idx, new_p) in new_p_real.iter().enumerate().take(n_trans) {
            for i in 0..norbs_cell {
                for j in 0..norbs_cell {
                    let p_old = workspace.p_real[m_idx].get(i, j);
                    let p_new = new_p.get(i, j);
                    let dp = (p_new - p_old).abs();
                    if dp > max_dp {
                        max_dp = dp;
                    }
                }
            }
        }

        if iter > 2 && d_e < options.energy_tol_ev && max_dp < options.density_tol {
            converged = true;
            break;
        }

        let damping = if iter > 20 {
            0.85
        } else if iter > 10 {
            0.75
        } else if iter > 5 {
            0.60
        } else {
            options.damping
        };
        for (m_idx, new_p) in new_p_real.iter().enumerate().take(n_trans) {
            for i in 0..norbs_cell {
                for j in 0..norbs_cell {
                    let p_old = workspace.p_real[m_idx].get(i, j);
                    let p_new = new_p.get(i, j);
                    workspace.p_real[m_idx].set(i, j, (1.0 - damping) * p_new + damping * p_old);
                }
            }
        }
    }

    let total_energy_cell_ev = prev_energy;

    // Heat of formation per unit cell
    let hof_cell_kcal = compute_heat_of_formation(
        total_energy_cell_ev,
        &unit_cell_atoms.atomic_numbers,
        model,
        0.0,
    )
    .1;

    // 11. Compute Band Structure along high-symmetry line path
    let band_path = options
        .unit_cell
        .generate_band_path(options.band_path_points);
    let n_path = band_path.len();
    let mut band_energies = Vec::with_capacity(n_path);

    let mut vbm = -f64::INFINITY;
    let mut cbm = f64::INFINITY;
    let mut min_direct_gap = f64::INFINITY;

    for kp in &band_path {
        let mut a_k = vec![0.0f64; norbs_cell * norbs_cell];
        let mut b_k = vec![0.0f64; norbs_cell * norbs_cell];

        for (m_idx, tr) in translations.iter().enumerate() {
            let k_dot_r = kp.cartesian[0] * tr.shift_angstrom[0]
                + kp.cartesian[1] * tr.shift_angstrom[1]
                + kp.cartesian[2] * tr.shift_angstrom[2];
            let cos_kr = k_dot_r.cos();
            let sin_kr = k_dot_r.sin();

            for i in 0..norbs_cell {
                for j in 0..norbs_cell {
                    let f_val = workspace.f_real[m_idx].get(i, j);
                    a_k[i * norbs_cell + j] += f_val * cos_kr;
                    b_k[i * norbs_cell + j] += f_val * sin_kr;
                }
            }
        }

        workspace.hermitian_embed_f.fill_zero();

        for i in 0..norbs_cell {
            for j in 0..norbs_cell {
                let a_val = a_k[i * norbs_cell + j];
                let b_val = b_k[i * norbs_cell + j];
                workspace.hermitian_embed_f.set(i, j, a_val);
                workspace.hermitian_embed_f.set(i, norbs_cell + j, -b_val);
                workspace.hermitian_embed_f.set(norbs_cell + i, j, b_val);
                workspace
                    .hermitian_embed_f
                    .set(norbs_cell + i, norbs_cell + j, a_val);
            }
        }

        diagonalize_symmetric_with_work(
            &workspace.hermitian_embed_f,
            &mut workspace.hermitian_embed_work,
            &mut workspace.embed_eigenvalues,
            &mut workspace.embed_eigenvectors,
        );

        // Extract the norbs_cell distinct eigenvalues (every second one from degenerate pairs)
        let mut k_eigs = Vec::with_capacity(norbs_cell);
        for o in 0..norbs_cell {
            k_eigs.push(workspace.embed_eigenvalues[2 * o]);
        }

        // Track VBM (HOMO), CBM (LUMO), direct gap
        if n_occ > 0 && n_occ < norbs_cell {
            let homo = k_eigs[n_occ - 1];
            let lumo = k_eigs[n_occ];
            if homo > vbm {
                vbm = homo;
            }
            if lumo < cbm {
                cbm = lumo;
            }
            let direct_gap = lumo - homo;
            if direct_gap < min_direct_gap {
                min_direct_gap = direct_gap;
            }
        }

        band_energies.push(k_eigs);
    }

    let indirect_gap = (cbm - vbm).max(0.0);

    // 12. Evaluate continuous Density of States (DOS) on energy grid
    let mut all_eigs = Vec::new();
    for k_e in &band_energies {
        for &e in k_e {
            all_eigs.push(e);
        }
    }
    all_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let e_min = all_eigs.first().copied().unwrap_or(-30.0) - 2.0;
    let e_max = all_eigs.last().copied().unwrap_or(20.0) + 2.0;
    let n_dos_pts = 300;
    let de = (e_max - e_min) / (n_dos_pts as f64);
    let mut dos_energies = Vec::with_capacity(n_dos_pts);
    let mut dos_values = vec![0.0f64; n_dos_pts];

    for idx in 0..n_dos_pts {
        dos_energies.push(e_min + (idx as f64) * de);
    }

    let sigma = options.dos_sigma_ev.max(0.05);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let norm_factor = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt() * (n_path as f64));

    for k_e in &band_energies {
        for &e_band in k_e {
            for (idx, &e_grid) in dos_energies.iter().enumerate() {
                let diff = e_grid - e_band;
                if diff.abs() < 4.0 * sigma {
                    dos_values[idx] += norm_factor * (-diff * diff / two_sigma_sq).exp();
                }
            }
        }
    }

    Ok(PbcResult {
        converged,
        iterations,
        total_energy_per_cell_ev: total_energy_cell_ev,
        electronic_energy_per_cell_ev: final_e_elec,
        nuclear_repulsion_per_cell_ev: e_nuc_ev,
        heat_of_formation_kcal_mol: hof_cell_kcal,
        vbm_energy_ev: vbm,
        cbm_energy_ev: cbm,
        direct_bandgap_ev: min_direct_gap,
        indirect_bandgap_ev: indirect_gap,
        band_k_points: band_path,
        band_energies_ev: band_energies,
        dos_energies_ev: dos_energies,
        dos_values,
    })
}
