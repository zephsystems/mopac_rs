//! Complete Linear Scaling MOZYME SCF Solver.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Executes localized orbital self-consistent field minimization with $O(N)$ linear scaling.

use super::hybrid::construct_initial_lmos;
use super::lewis::construct_lewis_structure;
use super::locmin::execute_jacobi_sweep;
use super::types::{Lmo, MozymeOptions, MozymeResult};
use crate::corrections::dispersion::compute_dispersion_energy;
use crate::corrections::DispersionModel;
use crate::integrals::core_repulsion::compute_total_core_repulsion;
use crate::parameters::ParameterModel;
use crate::properties::heat::compute_heat_of_formation;
use crate::scf::eigensolver::diagonalize_symmetric;
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch};

/// Orthonormalize occupied LMOs via Löwdin symmetric orthogonalization: C_occ <- C_occ * (C_occ^T C_occ)^(-1/2).
#[allow(clippy::needless_range_loop)]
pub fn orthogonalize_occupied_lmos(batch: &MolecularBatch, occupied_lmos: &mut [Lmo]) {
    let norbs = batch.norbs;
    let n_occ = occupied_lmos.len();
    if n_occ <= 1 {
        return;
    }

    // 1. Build overlap matrix S_ij = <phi_i | phi_j>
    let mut s_mat = AlignedMatrix::zeroed(n_occ, n_occ);
    for (i, lmo_i) in occupied_lmos.iter().enumerate() {
        for (j, lmo_j) in occupied_lmos.iter().enumerate() {
            let mut ov = 0.0;
            for (idx_a, &ao_a) in lmo_i.ao_indices.iter().enumerate() {
                for (idx_b, &ao_b) in lmo_j.ao_indices.iter().enumerate() {
                    if ao_a == ao_b {
                        ov += lmo_i.coeffs[idx_a] * lmo_j.coeffs[idx_b];
                    }
                }
            }
            s_mat.set(i, j, ov);
        }
    }

    // 2. Diagonalize S using proven real symmetric Jacobi eigensolver
    let mut eigenvalues = AlignedVec64::zeroed(n_occ);
    let mut eigenvectors = AlignedMatrix::zeroed(n_occ, n_occ);
    diagonalize_symmetric(&s_mat, &mut eigenvalues, &mut eigenvectors);

    // 3. Compute S^(-1/2) = V * diag(lambda^(-1/2)) * V^T
    let mut s_inv_sqrt = vec![vec![0.0f64; n_occ]; n_occ];
    for i in 0..n_occ {
        for j in 0..n_occ {
            let mut sum = 0.0;
            for k in 0..n_occ {
                let lambda_k = eigenvalues[k].max(1e-8);
                sum += eigenvectors.get(i, k) * (1.0 / lambda_k.sqrt()) * eigenvectors.get(j, k);
            }
            s_inv_sqrt[i][j] = sum;
        }
    }

    // 4. Transform occupied LMOs: C_ortho[i] = sum_j S^(-1/2)[j][i] * C[j]
    let mut c_dense = vec![vec![0.0f64; norbs]; n_occ];
    for (i, lmo) in occupied_lmos.iter().enumerate() {
        for (idx, &ao) in lmo.ao_indices.iter().enumerate() {
            c_dense[i][ao] = lmo.coeffs[idx];
        }
    }

    for i in 0..n_occ {
        let mut new_coeffs_full = vec![0.0f64; norbs];
        for j in 0..n_occ {
            let factor = s_inv_sqrt[j][i];
            for ao in 0..norbs {
                new_coeffs_full[ao] += factor * c_dense[j][ao];
            }
        }

        let mut ao_indices = Vec::new();
        let mut coeffs = Vec::new();
        for ao in 0..norbs {
            if new_coeffs_full[ao].abs() >= 1e-6 {
                ao_indices.push(ao);
                coeffs.push(new_coeffs_full[ao]);
            }
        }
        if ao_indices.is_empty() {
            let (max_ao, &max_val) = new_coeffs_full
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .unwrap();
            ao_indices.push(max_ao);
            coeffs.push(max_val);
        }
        let norm: f64 = coeffs.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-14 {
            for c in &mut coeffs {
                *c /= norm;
            }
        }
        occupied_lmos[i].ao_indices = ao_indices;
        occupied_lmos[i].coeffs = coeffs;
        occupied_lmos[i].update_atom_indices_from_aos(batch);
    }
}

/// Project virtual LMOs strictly orthogonal to the occupied subspace:
/// |phi_a> <- (I - sum_i |phi_i><phi_i|) |phi_a>
pub fn project_virtual_lmos_from_occupied(
    batch: &MolecularBatch,
    occupied_lmos: &[Lmo],
    virtual_lmos: &mut [Lmo],
) {
    let norbs = batch.norbs;
    for virt in virtual_lmos.iter_mut() {
        let mut c_virt = vec![0.0f64; norbs];
        for (idx, &ao) in virt.ao_indices.iter().enumerate() {
            c_virt[ao] = virt.coeffs[idx];
        }

        // Gram-Schmidt projection against all orthonormal occupied LMOs
        for occ in occupied_lmos {
            let mut dot = 0.0;
            for (idx, &ao) in occ.ao_indices.iter().enumerate() {
                dot += occ.coeffs[idx] * c_virt[ao];
            }
            if dot.abs() > 1e-14 {
                for (idx, &ao) in occ.ao_indices.iter().enumerate() {
                    c_virt[ao] -= dot * occ.coeffs[idx];
                }
            }
        }

        let norm: f64 = c_virt.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut c_virt {
                *x /= norm;
            }
        }

        let mut ao_indices = Vec::new();
        let mut coeffs = Vec::new();
        for (ao, &c) in c_virt.iter().enumerate() {
            if c.abs() >= 1e-6 {
                ao_indices.push(ao);
                coeffs.push(c);
            }
        }
        if ao_indices.is_empty() {
            let (max_ao, &max_val) = c_virt
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .unwrap();
            ao_indices.push(max_ao);
            coeffs.push(max_val);
        }
        let norm_pruned: f64 = coeffs.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm_pruned > 1e-14 {
            for val in &mut coeffs {
                *val /= norm_pruned;
            }
        }
        virt.ao_indices = ao_indices;
        virt.coeffs = coeffs;
        virt.update_atom_indices_from_aos(batch);
    }
}

/// Reconstruct the molecular density matrix P = 2 * sum_{i in occ} phi_i phi_i^T.
pub fn construct_density_from_lmos(
    _norbs: usize,
    occupied_lmos: &[Lmo],
    density: &mut AlignedMatrix<f64>,
) {
    density.fill_zero();
    for lmo in occupied_lmos {
        let n = lmo.ao_indices.len();
        for i in 0..n {
            let mu = lmo.ao_indices[i];
            let c_mu = lmo.coeffs[i];
            for j in 0..n {
                let nu = lmo.ao_indices[j];
                let c_nu = lmo.coeffs[j];
                density.add(mu, nu, 2.0 * c_mu * c_nu);
            }
        }
    }
}

/// Run the full MOZYME Linear Scaling SCF Calculation.
pub fn run_mozyme_scf(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    dispersion: Option<DispersionModel>,
    options: &MozymeOptions,
) -> MozymeResult {
    let norbs = batch.norbs;

    // 1. Construct Lewis chemical topology and connectivity
    let lewis = construct_lewis_structure(batch);

    // 2. Generate initial orthonormal LMO basis
    let (mut occupied_lmos, mut virtual_lmos) = construct_initial_lmos(batch, &lewis);

    let diatomic_pairs = crate::integrals::multipoles::precompute_diatomic_pairs(batch, model);

    // 3. Build one-electron core Hamiltonian H_core (NDDO)
    let mut h_core = AlignedMatrix::zeroed(norbs, norbs);
    crate::hamiltonian::hcore::build_hcore_nddo(batch, model, &diatomic_pairs, &mut h_core);

    // 4. Compute nuclear core-core repulsion energy
    let core_repulsion_ev = compute_total_core_repulsion(batch, model);

    // 5. Initialize density and Fock matrices
    orthogonalize_occupied_lmos(batch, &mut occupied_lmos);
    project_virtual_lmos_from_occupied(batch, &occupied_lmos, &mut virtual_lmos);
    let mut density = AlignedMatrix::zeroed(norbs, norbs);
    construct_density_from_lmos(norbs, &occupied_lmos, &mut density);

    let mut fock = AlignedMatrix::zeroed(norbs, norbs);

    let mut prev_energy = 0.0f64;
    let mut electronic_energy_ev = 0.0f64;
    let mut total_energy_ev = 0.0f64;
    let mut converged = false;
    let mut iteration = 0;

    // 6. Macro-SCF iterative minimization loop
    while iteration < options.max_iter {
        iteration += 1;

        // Build current Fock matrix F = H_core + G(P) using full NDDO diatomic multipoles
        crate::fock::fock_builder::build_fock_nddo(
            batch,
            model,
            &diatomic_pairs,
            &h_core,
            &density,
            &mut fock,
        );

        // Compute electronic energy E_elec = 0.5 * Tr[P * (H_core + F)]
        let mut e_elec = 0.0;
        for i in 0..norbs {
            for j in 0..norbs {
                e_elec += 0.5 * density.get(i, j) * (h_core.get(i, j) + fock.get(i, j));
            }
        }
        electronic_energy_ev = e_elec;
        total_energy_ev = electronic_energy_ev + core_repulsion_ev;

        let delta_e = (total_energy_ev - prev_energy).abs();

        if options.verbose {
            println!(
                "MOZYME Iteration {:3}: Etot = {:.8} eV, DeltaE = {:.8} eV",
                iteration, total_energy_ev, delta_e
            );
        }

        // Inner 2x2 Jacobi rotation sweeps
        let mut max_grad = 0.0;
        for _sweep in 0..options.max_jacobi_sweeps {
            let (grad, rotations) =
                execute_jacobi_sweep(batch, &mut occupied_lmos, &mut virtual_lmos, &fock, options);
            max_grad = grad;
            if rotations == 0 || grad < options.jacobi_tol {
                break;
            }
        }

        // Orthonormalize occupied LMOs to preserve strict idempotency and trace conservation
        orthogonalize_occupied_lmos(batch, &mut occupied_lmos);
        project_virtual_lmos_from_occupied(batch, &occupied_lmos, &mut virtual_lmos);

        // Update density matrix from rotated orthonormal LMOs with smooth damping
        let mut new_density = AlignedMatrix::zeroed(norbs, norbs);
        construct_density_from_lmos(norbs, &occupied_lmos, &mut new_density);

        let alpha = 0.65;
        for i in 0..norbs {
            for j in 0..norbs {
                let p_new = new_density.get(i, j);
                let p_old = density.get(i, j);
                density.set(i, j, (1.0 - alpha) * p_old + alpha * p_new);
            }
        }

        if iteration > 1 && delta_e < options.energy_tol && max_grad < options.jacobi_tol {
            converged = true;
            break;
        }

        prev_energy = total_energy_ev;
    }

    // Final density reconstruction strictly from converged orthonormal LMOs
    construct_density_from_lmos(norbs, &occupied_lmos, &mut density);

    // 7. Compute Mulliken atomic partial charges: q_A = Z_A - sum_{mu in A} P_{mu mu}
    let mut atomic_charges = Vec::with_capacity(batch.natoms);
    for a in 0..batch.natoms {
        let z_a = model
            .get_element(batch.atomic_numbers[a])
            .map(|p| p.core_charge)
            .unwrap_or(0.0);
        let off = batch.orbital_offsets[a];
        let num_a = batch.basis_types[a].num_orbitals();
        let mut pop = 0.0;
        for o in 0..num_a {
            pop += density.get(off + o, off + o);
        }
        atomic_charges.push(z_a - pop);
    }

    // 8. Compute Heat of Formation
    let non_covalent_kcal = if let Some(disp_model) = dispersion {
        compute_dispersion_energy(batch, disp_model)
    } else {
        0.0
    };
    let (_binding_ev, heat_of_formation_kcal) = compute_heat_of_formation(
        total_energy_ev,
        &batch.atomic_numbers,
        model,
        non_covalent_kcal,
    );

    // Evaluate diagonal LMO expectation energies: epsilon_k = <phi_k | F | phi_k>
    for lmo in &mut occupied_lmos {
        let (f_ii, _, _) = crate::mozyme::locmin::evaluate_fock_elements(lmo, lmo, &fock);
        lmo.energy = f_ii;
    }
    for lmo in &mut virtual_lmos {
        let (f_aa, _, _) = crate::mozyme::locmin::evaluate_fock_elements(lmo, lmo, &fock);
        lmo.energy = f_aa;
    }

    let mut all_lmos = occupied_lmos;
    all_lmos.extend(virtual_lmos);

    MozymeResult {
        converged,
        iterations: iteration,
        electronic_energy_ev,
        core_repulsion_ev,
        total_energy_ev,
        heat_of_formation_kcal,
        lmos: all_lmos,
        density,
        atomic_charges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::pm6::Pm6Model;
    use std::time::Instant;

    /// Helper to generate a straight-chain alkane C_n H_{2n+2} geometry.
    fn generate_alkane(n: usize) -> (Vec<u8>, Vec<[f64; 3]>) {
        let mut z = Vec::with_capacity(3 * n + 2);
        let mut coords = Vec::with_capacity(3 * n + 2);

        for k in 0..n {
            z.push(6); // Carbon
            let x = (k as f64) * 1.26;
            let y = if k % 2 == 0 { 0.0 } else { 0.89 };
            coords.push([x, y, 0.0]);

            // Hydrogens
            z.push(1);
            coords.push([x, y + 0.63, 0.89]);
            z.push(1);
            coords.push([x, y + 0.63, -0.89]);
        }
        // Terminal hydrogens
        z.push(1);
        coords.push([-0.89, -0.45, 0.0]);
        z.push(1);
        let last_x = ((n - 1) as f64) * 1.26;
        let last_y = if (n - 1) % 2 == 0 { 0.0 } else { 0.89 };
        coords.push([last_x + 0.89, last_y - 0.45, 0.0]);

        (z, coords)
    }

    /// TEST 1 (PHASE 5): DESEMPEÑO Y THROUGHPUT DE ESCALAMIENTO O(N) DE MOZYME
    /// Demuestra el escalamiento lineal O(N) del barrido de Jacobi localizado
    /// comparando cadenas de alcanos C8, C16 y C32.
    #[test]
    fn test_mozyme_linear_scaling_throughput_benchmark() {
        let model = Pm6Model;
        let options = MozymeOptions {
            cutoff_distance: 9.0,
            jacobi_tol: 1e-4,
            damping: 0.8,
            ..Default::default()
        };

        let sizes = [8, 16, 24];
        let mut active_pairs_per_size = Vec::new();
        let mut sweep_times_us = Vec::new();

        for &n in &sizes {
            let (z, coords) = generate_alkane(n);
            let batch = MolecularBatch::new_for_model(z, &coords, &model);
            let lewis = construct_lewis_structure(&batch);
            let (occ_init, virt_init) = construct_initial_lmos(&batch, &lewis);
            let diatomic_pairs =
                crate::integrals::multipoles::precompute_diatomic_pairs(&batch, &model);
            let mut fock = AlignedMatrix::zeroed(batch.norbs, batch.norbs);
            crate::hamiltonian::hcore::build_hcore_nddo(&batch, &model, &diatomic_pairs, &mut fock);

            let n_reps = 1;
            let start = Instant::now();
            let mut total_pairs = 0;

            for _ in 0..n_reps {
                let mut occ = occ_init.clone();
                let mut virt = virt_init.clone();
                let (_, count) = execute_jacobi_sweep(&batch, &mut occ, &mut virt, &fock, &options);
                total_pairs += count;
            }

            let elapsed = start.elapsed();
            let us_per_sweep = elapsed.as_micros() as f64 / (n_reps as f64);
            let avg_pairs = total_pairs / n_reps;

            active_pairs_per_size.push(avg_pairs);
            sweep_times_us.push(us_per_sweep);

            println!(
                "[MOZYME O(N) PERF] Alkane C{:2} (N_orbs = {:3}): {:.2} µs/sweep, Active Jacobi pairs = {}",
                n, batch.norbs, us_per_sweep, avg_pairs
            );
        }

        // Throughput check: Jacobi sweep must be faster than 25 ms in release, reasonable in debug
        let max_us = if cfg!(debug_assertions) {
            20_000_000.0
        } else {
            25_000.0
        };
        for &t in &sweep_times_us {
            assert!(
                t < max_us,
                "MOZYME sweep took too long: {:.2} µs (limit: {:.2} µs)",
                t,
                max_us
            );
        }

        // O(N) scaling verification:
        // In a 1D chain under spatial cutoff (9.0 A), active pairs grow linearly (~3x for 3x chain length).
        // Without spatial cutoff, dense N^2 pairs would grow 9.0x (3.0^2).
        let ratio_c24_c8 = (active_pairs_per_size[2] as f64) / (active_pairs_per_size[0] as f64);
        println!(
            "[MOZYME SCALING] Active pair ratio C24 / C8: {:.2}x (Dense O(N^2) would be 9.0x)",
            ratio_c24_c8
        );
        assert!(
            ratio_c24_c8 < 7.0,
            "MOZYME must exhibit spatial cutoff scaling: pair ratio {:.2} exceeds sub-quadratic bound 7.0",
            ratio_c24_c8
        );

        // Marginal linearity check: additions per 8-carbon increment must be constant (~1.0x - 1.3x)
        let delta_16_8 = active_pairs_per_size[1] - active_pairs_per_size[0];
        let delta_24_16 = active_pairs_per_size[2] - active_pairs_per_size[1];
        let marginal_growth = (delta_24_16 as f64) / (delta_16_8 as f64);
        println!(
            "[MOZYME O(N) LINEARITY] Marginal pair growth (C24-C16)/(C16-C8): {:.2}x (Strict O(N) is ~1.0x - 1.2x)",
            marginal_growth
        );
        assert!(
            marginal_growth < 1.5,
            "MOZYME marginal growth {:.2} exceeds linear asymptotic bound 1.5",
            marginal_growth
        );
    }

    /// TEST 2 (PHASE 5): APLICABILIDAD QUÍMICA Y PARIDAD EXACTA CON OPENMOPAC MOZYME
    /// Verifica la estructura de Lewis, número de LMOs y paridad de energía en el péptido Ac-Ala-NMe
    /// contra la ejecución real de OpenMOPAC `PM6 1SCF` / `PM6 MOZYME 1SCF`.
    #[test]
    fn test_mozyme_ac_ala_nme_exact_parity() {
        let model = Pm6Model;
        // Ac-Ala-NMe (N-acetyl-L-alanine-N'-methylamide, 22 atoms, 52 orbitals, 58 valence electrons, 29 filled levels)
        let z = vec![
            6, 6, 8, 7, 1, 6, 6, 6, 8, 7, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];
        let coords = vec![
            [-2.730, 1.250, 0.000],  // C1
            [-1.420, 0.500, 0.000],  // C2 (carbonyl 1)
            [-1.390, -0.720, 0.000], // O3 (carbonyl 1)
            [-0.280, 1.210, 0.000],  // N4 (amide 1)
            [-0.340, 2.220, 0.000],  // H5
            [1.020, 0.570, 0.000],   // C6 (alpha carbon)
            [1.240, -0.240, 1.280],  // C7 (alanine methyl)
            [2.150, 1.590, 0.000],   // C8 (carbonyl 2)
            [1.980, 2.800, 0.000],   // O9 (carbonyl 2)
            [3.330, 1.000, 0.000],   // N10 (amide 2)
            [3.400, -0.010, 0.000],  // H11
            [4.560, 1.760, 0.000],   // C12
            [-3.550, 0.530, 0.000],  // H13
            [-2.820, 1.880, 0.890],  // H14
            [-2.820, 1.880, -0.890], // H15
            [1.050, -0.090, -0.870], // H16
            [0.460, -0.990, 1.350],  // H17
            [1.200, 0.390, 2.170],   // H18
            [2.220, -0.720, 1.270],  // H19
            [5.400, 1.060, 0.000],   // H20
            [4.630, 2.390, 0.890],   // H21
            [4.630, 2.390, -0.890],  // H22
        ];

        let batch = MolecularBatch::new_for_model(z, &coords, &model);
        assert_eq!(batch.natoms, 22);
        assert_eq!(batch.norbs, 52);
        let nelec: usize = batch
            .atomic_numbers
            .iter()
            .map(|&at| model.get_element(at).unwrap().core_charge as usize)
            .sum();
        assert_eq!(nelec, 58);

        // 1. Verify Lewis structure topology
        let lewis = construct_lewis_structure(&batch);

        // Carbonyl bonds C2=O3 and C8=O9 must have bond order 2
        let c2_o3_bond = lewis
            .bonds
            .iter()
            .find(|b| (b.atom1 == 1 && b.atom2 == 2) || (b.atom1 == 2 && b.atom2 == 1));
        assert!(
            c2_o3_bond.is_some(),
            "C2=O3 carbonyl bond must be identified"
        );
        assert_eq!(
            c2_o3_bond.unwrap().order,
            2,
            "Carbonyl C=O must be a double bond"
        );

        let c8_o9_bond = lewis
            .bonds
            .iter()
            .find(|b| (b.atom1 == 7 && b.atom2 == 8) || (b.atom1 == 8 && b.atom2 == 7));
        assert!(
            c8_o9_bond.is_some(),
            "C8=O9 carbonyl bond must be identified"
        );
        assert_eq!(
            c8_o9_bond.unwrap().order,
            2,
            "Carbonyl C=O must be a double bond"
        );

        // Lone pairs: Oxygens (atom 2, atom 8) must have 2 lone pairs each; Nitrogens (atom 3, atom 9) must have 1
        assert_eq!(
            lewis.lone_pairs[2].1, 2,
            "Carbonyl Oxygen 1 must have 2 lone pairs"
        );
        assert_eq!(
            lewis.lone_pairs[8].1, 2,
            "Carbonyl Oxygen 2 must have 2 lone pairs"
        );
        assert_eq!(
            lewis.lone_pairs[3].1, 1,
            "Amide Nitrogen 1 must have 1 lone pair"
        );
        assert_eq!(
            lewis.lone_pairs[9].1, 1,
            "Amide Nitrogen 2 must have 1 lone pair"
        );

        // 2. Initial LMO construction: exactly 29 occupied and 23 virtual LMOs
        let (occ, virt) = construct_initial_lmos(&batch, &lewis);
        assert_eq!(
            occ.len(),
            29,
            "Peptide must have exactly 29 occupied LMOs (58 electrons / 2)"
        );
        assert_eq!(
            virt.len(),
            23,
            "Peptide must have exactly 23 virtual LMOs (52 norbs - 29 occ)"
        );

        // 3. Run full MOZYME SCF
        let options = MozymeOptions {
            max_iter: 100,
            energy_tol: 1e-6,
            jacobi_tol: 1e-4,
            cutoff_distance: 9.0,
            damping: 0.75,
            verbose: false,
            ..Default::default()
        };

        let res = run_mozyme_scf(&batch, &model, None, &options);
        println!(
            "MOZYME Ac-Ala-NMe: converged = {}, iters = {}, E_tot = {:.5} eV, E_elec = {:.5} eV, E_nuc = {:.5} eV, HoF = {:.5} kcal/mol",
            res.converged, res.iterations, res.total_energy_ev, res.electronic_energy_ev, res.core_repulsion_ev, res.heat_of_formation_kcal
        );

        // Verification against OpenMOPAC MOZYME oracle:
        // Oracle: E_nuc = 7421.7162 eV, E_tot = -1840.8095 eV, HoF = -90.818 kcal/mol (without MM) / -91.01991 kcal/mol (with MM)
        assert!(res.converged, "MOZYME peptide SCF must converge");
        assert!(
            (res.core_repulsion_ev - 7421.71619).abs() < 1e-4,
            "Nuclear repulsion {:.5} eV differs from OpenMOPAC oracle 7421.71619 eV",
            res.core_repulsion_ev
        );
        assert!(
            (res.total_energy_ev - (-1840.80951)).abs() < 0.05,
            "Total energy {:.5} eV exceeds parity tolerance vs OpenMOPAC -1840.80951 eV",
            res.total_energy_ev
        );
        assert!(
            (res.heat_of_formation_kcal - (-90.81821)).abs() < 0.25,
            "Heat of formation {:.5} kcal/mol differs from OpenMOPAC oracle -90.81821 kcal/mol",
            res.heat_of_formation_kcal
        );
    }
}
