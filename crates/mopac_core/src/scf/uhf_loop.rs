//! Unrestricted Hartree-Fock (UHF) SCF Solver Loop for Open-Shell Systems.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Executes open-shell UHF iterations with separate alpha and beta spin channels
//! and zero dynamic heap allocations in the inner iterative cycle.

use crate::fock::fock_builder::{build_fock_spin, build_fock_uhf};
use crate::hamiltonian::build_hcore;
use crate::integrals::core_repulsion::compute_total_core_repulsion;
use crate::parameters::ParameterModel;
use crate::scf::density::{
    compute_spin_density_matrix, compute_uhf_electronic_energy, compute_uhf_s_squared,
    max_density_diff,
};
use crate::scf::diis::DiisWorkspace;
use crate::scf::eigensolver::diagonalize_symmetric;
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch};

/// Summary result of a completed Unrestricted Hartree-Fock (UHF) calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct UhfResult {
    pub converged: bool,
    pub iterations: usize,
    pub total_energy_ev: f64,
    pub electronic_energy_ev: f64,
    pub nuclear_repulsion_ev: f64,
    pub homo_a_energy_ev: f64,
    pub lumo_a_energy_ev: f64,
    pub homo_b_energy_ev: f64,
    pub lumo_b_energy_ev: f64,
    pub s_squared: f64,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub dielectric_energy_ev: Option<f64>,
}

/// Configuration options for the UHF open-shell solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UhfOptions {
    /// Spin multiplicity $2S + 1$ (default: 2 for doublet)
    pub multiplicity: usize,
    /// Molecular charge (default: 0)
    pub charge: i32,
    /// Maximum allowed SCF iterations (default: 80)
    pub max_iter: usize,
    /// Energy convergence threshold in eV (default: 1e-7)
    pub energy_tol_ev: f64,
    /// Density matrix maximum element difference threshold (default: 1e-6)
    pub density_tol: f64,
    /// Linear damping factor applied when DIIS is not yet active (default: 0.5)
    pub damping: f64,
    /// Enable full NDDO 22 diatomic multipoles and rotated attractions (default: true)
    pub use_nddo: bool,
    /// COSMO implicit solvation model parameters (default: None)
    pub cosmo: Option<crate::solvation::CosmoParams>,
}

impl Default for UhfOptions {
    fn default() -> Self {
        Self {
            multiplicity: 2,
            charge: 0,
            max_iter: 80,
            energy_tol_ev: 1e-7,
            density_tol: 1e-6,
            damping: 0.5,
            use_nddo: true,
            cosmo: None,
        }
    }
}

/// Pre-allocated workspace for UHF calculations guaranteed to perform zero heap reallocations.
#[derive(Debug, Clone)]
pub struct UhfWorkspace {
    pub norbs: usize,
    pub h_core: AlignedMatrix<f64>,
    pub fock_a: AlignedMatrix<f64>,
    pub fock_b: AlignedMatrix<f64>,
    pub density_a: AlignedMatrix<f64>,
    pub density_b: AlignedMatrix<f64>,
    pub density_tot: AlignedMatrix<f64>,
    pub eigenvectors_a: AlignedMatrix<f64>,
    pub eigenvectors_b: AlignedMatrix<f64>,
    pub eigenvalues_a: AlignedVec64<f64>,
    pub eigenvalues_b: AlignedVec64<f64>,
    pub tmp_density_a: AlignedMatrix<f64>,
    pub tmp_density_b: AlignedMatrix<f64>,
    pub tmp_mult: AlignedMatrix<f64>,
    pub diis_a: DiisWorkspace,
    pub diis_b: DiisWorkspace,
    pub diatomic_pairs: Vec<crate::integrals::multipoles::DiatomicPairIntegrals>,
}

impl UhfWorkspace {
    /// Allocate pre-sized 64-byte aligned matrix buffers for a system of `norbs` basis functions.
    pub fn new(norbs: usize) -> Self {
        Self {
            norbs,
            h_core: AlignedMatrix::zeroed(norbs, norbs),
            fock_a: AlignedMatrix::zeroed(norbs, norbs),
            fock_b: AlignedMatrix::zeroed(norbs, norbs),
            density_a: AlignedMatrix::zeroed(norbs, norbs),
            density_b: AlignedMatrix::zeroed(norbs, norbs),
            density_tot: AlignedMatrix::zeroed(norbs, norbs),
            eigenvectors_a: AlignedMatrix::zeroed(norbs, norbs),
            eigenvectors_b: AlignedMatrix::zeroed(norbs, norbs),
            eigenvalues_a: AlignedVec64::zeroed(norbs),
            eigenvalues_b: AlignedVec64::zeroed(norbs),
            tmp_density_a: AlignedMatrix::zeroed(norbs, norbs),
            tmp_density_b: AlignedMatrix::zeroed(norbs, norbs),
            tmp_mult: AlignedMatrix::zeroed(norbs, norbs),
            diis_a: DiisWorkspace::allocate(norbs, 6),
            diis_b: DiisWorkspace::allocate(norbs, 6),
            diatomic_pairs: Vec::new(),
        }
    }
}

/// Run an Unrestricted Hartree-Fock (UHF) SCF calculation with explicit options.
pub fn run_uhf_scf_with_options(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    ws: &mut UhfWorkspace,
    options: &UhfOptions,
) -> UhfResult {
    assert_eq!(ws.norbs, batch.norbs);

    // 1. Calculate total valence electrons
    let mut total_core_charge = 0.0;
    for &z in &batch.atomic_numbers {
        if let Some(p) = model.get_element(z) {
            total_core_charge += p.core_charge;
        }
    }
    let nelecs_signed = total_core_charge.round() as i64 - options.charge as i64;
    assert!(
        nelecs_signed > 0,
        "System must have at least one valence electron"
    );
    let nelecs = nelecs_signed as usize;

    let mult = options.multiplicity;
    assert!(mult >= 1, "Spin multiplicity must be >= 1");
    let msdel = mult - 1; // 2 * S

    assert!(
        nelecs >= msdel,
        "Number of valence electrons ({}) must be >= 2*S ({})",
        nelecs,
        msdel
    );
    assert!(
        (nelecs - msdel).is_multiple_of(2),
        "Electron count ({}) is incompatible with spin multiplicity ({})",
        nelecs,
        mult
    );

    let n_beta = (nelecs - msdel) / 2;
    let n_alpha = nelecs - n_beta;
    assert!(
        n_alpha <= ws.norbs,
        "Number of alpha electrons ({}) cannot exceed basis functions ({})",
        n_alpha,
        ws.norbs
    );

    // 2. Nuclear-nuclear core repulsion energy
    let mut enuc = compute_total_core_repulsion(batch, model);

    // Optional COSMO implicit solvation initialization
    let mut cosmo_state = if let Some(cosmo_params) = options.cosmo {
        crate::solvation::CosmoState::initialize(batch, model, cosmo_params).ok()
    } else {
        None
    };

    if let Some(ref cs) = cosmo_state {
        enuc += cs.e_nuc_diel_ev;
    }

    // 3. Build one-electron core Hamiltonian H_core
    if options.use_nddo {
        ws.diatomic_pairs = crate::integrals::multipoles::precompute_diatomic_pairs(batch, model);
        crate::hamiltonian::hcore::build_hcore_nddo(
            batch,
            model,
            &ws.diatomic_pairs,
            &mut ws.h_core,
        );
    } else {
        build_hcore(batch, model, &mut ws.h_core);
    }

    if let Some(ref cs) = cosmo_state {
        cs.apply_nuclear_reaction_field_to_hcore(&mut ws.h_core);
    }

    // 4. Initial guess: diagonalize H_core to generate initial alpha and beta densities
    diagonalize_symmetric(&ws.h_core, &mut ws.eigenvalues_a, &mut ws.eigenvectors_a);
    ws.eigenvectors_b
        .data
        .copy_from_slice(&ws.eigenvectors_a.data);
    ws.eigenvalues_b.copy_from_slice(&ws.eigenvalues_a);

    compute_spin_density_matrix(&ws.eigenvectors_a, n_alpha, &mut ws.density_a);
    compute_spin_density_matrix(&ws.eigenvectors_b, n_beta, &mut ws.density_b);

    // If singlet open-shell (n_alpha == n_beta), break symmetry slightly to allow UHF solution
    if n_alpha == n_beta && ws.norbs > n_alpha {
        let mix_angle = 0.15f64;
        let c = mix_angle.cos();
        let s = mix_angle.sin();
        let h_idx = n_alpha - 1;
        let l_idx = n_alpha;
        for i in 0..ws.norbs {
            let ch = ws.eigenvectors_b.get(i, h_idx);
            let cl = ws.eigenvectors_b.get(i, l_idx);
            ws.eigenvectors_b.set(i, h_idx, c * ch + s * cl);
            ws.eigenvectors_b.set(i, l_idx, -s * ch + c * cl);
        }
        compute_spin_density_matrix(&ws.eigenvectors_b, n_beta, &mut ws.density_b);
    }

    for idx in 0..ws.density_tot.data.len() {
        ws.density_tot.data[idx] = ws.density_a.data[idx] + ws.density_b.data[idx];
    }

    ws.diis_a.reset();
    ws.diis_b.reset();

    let mut prev_energy = 0.0f64;
    let mut converged = false;
    let mut iters_done = 0;
    let mut last_diel_ev: Option<f64> = None;
    let mut final_e_elec = 0.0f64;

    // 5. UHF Iteration Loop
    for iter in 1..=options.max_iter {
        iters_done = iter;

        // Build alpha and beta Fock matrices
        if options.use_nddo {
            build_fock_uhf(
                batch,
                model,
                &ws.diatomic_pairs,
                &ws.h_core,
                &ws.density_tot,
                &ws.density_a,
                &mut ws.fock_a,
            );
            build_fock_uhf(
                batch,
                model,
                &ws.diatomic_pairs,
                &ws.h_core,
                &ws.density_tot,
                &ws.density_b,
                &mut ws.fock_b,
            );
        } else {
            build_fock_spin(
                batch,
                model,
                &ws.h_core,
                &ws.density_tot,
                &ws.density_a,
                &mut ws.fock_a,
            );
            build_fock_spin(
                batch,
                model,
                &ws.h_core,
                &ws.density_tot,
                &ws.density_b,
                &mut ws.fock_b,
            );
        }

        // Apply COSMO reaction field to Fock matrices
        if let Some(ref mut cs) = cosmo_state {
            let ediel = cs.apply_electronic_reaction_field_to_fock(&ws.density_tot, &mut ws.fock_a);
            cs.apply_electronic_reaction_field_to_fock(&ws.density_tot, &mut ws.fock_b);
            last_diel_ev = Some(ediel);
        }

        // Electronic energy before DIIS extrapolation
        let e_elec = compute_uhf_electronic_energy(
            &ws.density_a,
            &ws.density_b,
            &ws.h_core,
            &ws.fock_a,
            &ws.fock_b,
        );
        final_e_elec = e_elec;
        let e_total = e_elec + enuc;

        // Pulay DIIS extrapolation for alpha and beta channels
        let diis_res_a =
            ws.diis_a
                .push_and_extrapolate(&mut ws.fock_a, &ws.density_a, &mut ws.tmp_mult);
        let diis_res_b =
            ws.diis_b
                .push_and_extrapolate(&mut ws.fock_b, &ws.density_b, &mut ws.tmp_mult);

        // Diagonalize alpha and beta Fock matrices
        diagonalize_symmetric(&ws.fock_a, &mut ws.eigenvalues_a, &mut ws.eigenvectors_a);
        diagonalize_symmetric(&ws.fock_b, &mut ws.eigenvalues_b, &mut ws.eigenvectors_b);

        // Compute candidate new spin densities
        compute_spin_density_matrix(&ws.eigenvectors_a, n_alpha, &mut ws.tmp_density_a);
        compute_spin_density_matrix(&ws.eigenvectors_b, n_beta, &mut ws.tmp_density_b);

        // Check convergence
        let delta_e = (e_total - prev_energy).abs();
        let delta_pa = max_density_diff(&ws.tmp_density_a, &ws.density_a);
        let delta_pb = max_density_diff(&ws.tmp_density_b, &ws.density_b);
        let delta_p = delta_pa.max(delta_pb);

        if iter > 1
            && delta_e < options.energy_tol_ev
            && (delta_p < options.density_tol
                || (diis_res_a.max_error < options.density_tol
                    && diis_res_b.max_error < options.density_tol))
        {
            converged = true;
            ws.density_a.data.copy_from_slice(&ws.tmp_density_a.data);
            ws.density_b.data.copy_from_slice(&ws.tmp_density_b.data);
            for idx in 0..ws.density_tot.data.len() {
                ws.density_tot.data[idx] = ws.density_a.data[idx] + ws.density_b.data[idx];
            }
            break;
        }

        prev_energy = e_total;

        // Update densities
        if diis_res_a.extrapolated {
            ws.density_a.data.copy_from_slice(&ws.tmp_density_a.data);
        } else {
            let d = options.damping;
            for idx in 0..ws.density_a.data.len() {
                ws.density_a.data[idx] =
                    (1.0 - d) * ws.tmp_density_a.data[idx] + d * ws.density_a.data[idx];
            }
        }

        if diis_res_b.extrapolated {
            ws.density_b.data.copy_from_slice(&ws.tmp_density_b.data);
        } else {
            let d = options.damping;
            for idx in 0..ws.density_b.data.len() {
                ws.density_b.data[idx] =
                    (1.0 - d) * ws.tmp_density_b.data[idx] + d * ws.density_b.data[idx];
            }
        }

        for idx in 0..ws.density_tot.data.len() {
            ws.density_tot.data[idx] = ws.density_a.data[idx] + ws.density_b.data[idx];
        }
    }

    let homo_a = if n_alpha > 0 {
        ws.eigenvalues_a[n_alpha - 1]
    } else {
        0.0
    };
    let lumo_a = if n_alpha < ws.norbs {
        ws.eigenvalues_a[n_alpha]
    } else {
        0.0
    };

    let homo_b = if n_beta > 0 {
        ws.eigenvalues_b[n_beta - 1]
    } else {
        0.0
    };
    let lumo_b = if n_beta < ws.norbs {
        ws.eigenvalues_b[n_beta]
    } else {
        0.0
    };

    let s_squared = compute_uhf_s_squared(&ws.density_a, &ws.density_b, n_alpha, n_beta);

    UhfResult {
        converged,
        iterations: iters_done,
        total_energy_ev: final_e_elec + enuc,
        electronic_energy_ev: final_e_elec,
        nuclear_repulsion_ev: enuc,
        homo_a_energy_ev: homo_a,
        lumo_a_energy_ev: lumo_a,
        homo_b_energy_ev: homo_b,
        lumo_b_energy_ev: lumo_b,
        s_squared,
        n_alpha,
        n_beta,
        dielectric_energy_ev: last_diel_ev,
    }
}
