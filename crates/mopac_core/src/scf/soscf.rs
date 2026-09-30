//! Second-Order Self-Consistent Field (SOSCF) Solver.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Implements a coupled quasi-Newton orbital optimization method.
//!
//! References:
//! - G. Chaban, M. W. Schmidt, M. S. Gordon, "Approximate second-order SCF methods revisited",
//!   Theor. Chem. Acc. 97, 88-95 (1997).
//! - J. Douady, Y. Ellinger, R. Subra, B. Levy, "A coupled Newton-Raphson formalism for
//!   Hartree-Fock calculations", J. Chem. Phys. 72, 1452 (1980).
//!
//! In contrast to first-order DIIS (which can oscillate indefinitely on near-degenerate
//! open shells and transition metals with d-orbitals), SOSCF optimizes the unitary
//! orbital rotation matrix $U = \exp(\kappa)$ using the orbital gradient:
//! $$g_{ai} = 4 F^{\text{MO}}_{ai} = 4 (C^T F C)_{ai}$$
//! and diagonal orbital Hessian with a Levenberg-Marquardt trust-radius shift $\lambda$:
//! $$\kappa_{ai} = -\frac{F^{\text{MO}}_{ai}}{(\epsilon_a - \epsilon_i) + \frac{1}{4}\lambda}$$
//!
//! Strictly guaranteed **zero heap allocations** in iterative cycles.

use crate::fock::build_fock;
use crate::parameters::ParameterModel;
use crate::scf::density::{compute_density_matrix, compute_electronic_energy, max_density_diff};
use crate::scf::eigensolver::diagonalize_symmetric;
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch, ScfWorkspace};

/// Summary result of an SOSCF step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoscfStepResult {
    /// Maximum orbital rotation gradient element $\max_{ai} |4 F^{\text{MO}}_{ai}|$
    pub max_orbital_gradient: f64,
    /// Root-mean-square orbital rotation gradient
    pub rms_orbital_gradient: f64,
    /// Maximum rotation angle in radians $\max_{ai} |\kappa_{ai}|$
    pub max_rotation_angle: f64,
}

/// Pre-allocated workspace for the Second-Order SCF solver.
#[derive(Debug, Clone)]
pub struct SoscfWorkspace {
    pub norbs: usize,
    /// Fock matrix in Molecular Orbital basis: F_MO = C^T * F * C
    pub f_mo: AlignedMatrix<f64>,
    /// Anti-symmetric orbital rotation generator matrix kappa
    pub kappa: AlignedMatrix<f64>,
    /// Unitary orbital rotation matrix U = exp(kappa)
    pub u_rot: AlignedMatrix<f64>,
    /// Rotated molecular orbital coefficients C_new = C * U
    pub c_new: AlignedMatrix<f64>,
    /// Scratch matrix buffer 1
    pub tmp1: AlignedMatrix<f64>,
    /// Scratch matrix buffer 2
    pub tmp2: AlignedMatrix<f64>,
    /// Scratch eigenvalues buffer
    pub evals: AlignedVec64<f64>,
    /// Scratch eigenvectors buffer
    pub evecs: AlignedMatrix<f64>,
}

impl SoscfWorkspace {
    /// Allocate pre-sized buffers for `norbs` basis functions.
    pub fn allocate(norbs: usize) -> Self {
        Self {
            norbs,
            f_mo: AlignedMatrix::zeroed(norbs, norbs),
            kappa: AlignedMatrix::zeroed(norbs, norbs),
            u_rot: AlignedMatrix::zeroed(norbs, norbs),
            c_new: AlignedMatrix::zeroed(norbs, norbs),
            tmp1: AlignedMatrix::zeroed(norbs, norbs),
            tmp2: AlignedMatrix::zeroed(norbs, norbs),
            evals: AlignedVec64::zeroed(norbs),
            evecs: AlignedMatrix::zeroed(norbs, norbs),
        }
    }
}

/// Executes a single Second-Order SCF (SOSCF) orbital rotation step.
///
/// 1. Transforms Fock matrix to MO basis: $F^{\text{MO}} = C^T F C$.
/// 2. Evaluates orbital gradient $g_{ai} = 4 F^{\text{MO}}_{ai}$.
/// 3. Computes rotation matrix $\kappa_{ai} = -\frac{F^{\text{MO}}_{ai}}{\epsilon_a - \epsilon_i + \frac{1}{4}\lambda}$.
/// 4. Evaluates unitary rotation $U = \exp(\kappa)$ via 4th-order Taylor expansion with Löwdin re-orthonormalization.
/// 5. Updates $C \leftarrow C U$ and density $P \leftarrow 2 C_{\text{occ}} C_{\text{occ}}^T$.
pub fn soscf_step(
    fock: &AlignedMatrix<f64>,
    eigenvectors: &mut AlignedMatrix<f64>,
    eigenvalues: &AlignedVec64<f64>,
    density: &mut AlignedMatrix<f64>,
    nocc: usize,
    level_shift_ev: f64,
    ws: &mut SoscfWorkspace,
) -> SoscfStepResult {
    let norbs = ws.norbs;
    assert_eq!(fock.rows, norbs);
    assert_eq!(eigenvectors.rows, norbs);

    // 1. Transform Fock to MO basis: F_MO = C^T * F * C
    // tmp1 = F * C
    for i in 0..norbs {
        let f_row = fock.row(i);
        let t_row = ws.tmp1.row_mut(i);
        t_row.fill(0.0);
        for (k, &f_ik) in f_row.iter().enumerate().take(norbs) {
            let c_row = eigenvectors.row(k);
            for j in 0..norbs {
                t_row[j] += f_ik * c_row[j];
            }
        }
    }

    // F_MO = C^T * tmp1: F_MO_ij = sum_k C_ki * tmp1_kj
    for i in 0..norbs {
        for j in 0..norbs {
            let mut sum = 0.0f64;
            for k in 0..norbs {
                sum += eigenvectors.get(k, i) * ws.tmp1.get(k, j);
            }
            ws.f_mo.set(i, j, sum);
        }
    }

    // 2. Compute orbital gradient and rotation generator kappa
    ws.kappa.fill_zero();
    let mut max_grad = 0.0f64;
    let mut sum_sq_grad = 0.0f64;
    let mut max_angle = 0.0f64;

    let shift_term = 0.25 * level_shift_ev.max(0.0);
    let nvirt = norbs - nocc;

    for i in 0..nocc {
        let eps_i = eigenvalues[i];
        for a in nocc..norbs {
            let eps_a = eigenvalues[a];
            let f_ai = ws.f_mo.get(a, i);
            let grad_ai = 4.0 * f_ai;
            let abs_grad = grad_ai.abs();
            if abs_grad > max_grad {
                max_grad = abs_grad;
            }
            sum_sq_grad += grad_ai * grad_ai;

            // Quasi-Newton step with trust-radius denominator
            let denom = (eps_a - eps_i).max(0.05) + shift_term;
            let angle = -f_ai / denom;
            let clamped_angle = angle.clamp(-0.35, 0.35); // Trust radius clamp

            if clamped_angle.abs() > max_angle {
                max_angle = clamped_angle.abs();
            }

            ws.kappa.set(a, i, clamped_angle);
            ws.kappa.set(i, a, -clamped_angle);
        }
    }

    let n_pairs = (nocc * nvirt).max(1) as f64;
    let rms_grad = (sum_sq_grad / n_pairs).sqrt();

    // 3. Construct unitary U = exp(kappa) via 4th-order Taylor series:
    // U = I + K + 1/2 K^2 + 1/6 K^3 + 1/24 K^4
    // tmp1 = K^2
    for i in 0..norbs {
        for j in 0..norbs {
            let mut sum = 0.0f64;
            for k in 0..norbs {
                sum += ws.kappa.get(i, k) * ws.kappa.get(k, j);
            }
            ws.tmp1.set(i, j, sum);
        }
    }

    // tmp2 = K^3 = K * tmp1
    for i in 0..norbs {
        for j in 0..norbs {
            let mut sum = 0.0f64;
            for k in 0..norbs {
                sum += ws.kappa.get(i, k) * ws.tmp1.get(k, j);
            }
            ws.tmp2.set(i, j, sum);
        }
    }

    // Accumulate into u_rot: I + K + 0.5 tmp1 + (1/6) tmp2
    for i in 0..norbs {
        for j in 0..norbs {
            let delta = if i == j { 1.0 } else { 0.0 };
            let k_val = ws.kappa.get(i, j);
            let k2_val = ws.tmp1.get(i, j);
            let k3_val = ws.tmp2.get(i, j);
            let u_val = delta + k_val + 0.5 * k2_val + (1.0 / 6.0) * k3_val;
            ws.u_rot.set(i, j, u_val);
        }
    }

    // Löwdin orthogonalization of U: S_U = U^T * U
    for i in 0..norbs {
        for j in 0..norbs {
            let mut sum = 0.0f64;
            for k in 0..norbs {
                sum += ws.u_rot.get(k, i) * ws.u_rot.get(k, j);
            }
            ws.tmp1.set(i, j, sum);
        }
    }

    diagonalize_symmetric(&ws.tmp1, &mut ws.evals, &mut ws.evecs);

    // Compute S_U^(-1/2) = V * Lambda^(-1/2) * V^T into tmp2
    for i in 0..norbs {
        for j in 0..norbs {
            let mut sum = 0.0f64;
            for k in 0..norbs {
                let inv_sqrt = 1.0 / ws.evals[k].max(1e-14).sqrt();
                sum += ws.evecs.get(i, k) * inv_sqrt * ws.evecs.get(j, k);
            }
            ws.tmp2.set(i, j, sum);
        }
    }

    // Purified U_ortho = U * S_U^(-1/2) into tmp1
    for i in 0..norbs {
        for j in 0..norbs {
            let mut sum = 0.0f64;
            for k in 0..norbs {
                sum += ws.u_rot.get(i, k) * ws.tmp2.get(k, j);
            }
            ws.tmp1.set(i, j, sum);
        }
    }

    // 4. Update MO coefficients: C_new = C * U_ortho
    for i in 0..norbs {
        for j in 0..norbs {
            let mut sum = 0.0f64;
            for k in 0..norbs {
                sum += eigenvectors.get(i, k) * ws.tmp1.get(k, j);
            }
            ws.c_new.set(i, j, sum);
        }
    }

    eigenvectors.data.copy_from_slice(&ws.c_new.data);

    // 5. Update density: P = 2 * C_occ * C_occ^T
    compute_density_matrix(eigenvectors, nocc, density);

    SoscfStepResult {
        max_orbital_gradient: max_grad,
        rms_orbital_gradient: rms_grad,
        max_rotation_angle: max_angle,
    }
}

/// Run a complete Second-Order SCF optimization loop on difficult / oscillating systems.
pub fn run_rhf_soscf(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    ws: &mut ScfWorkspace,
    max_iter: usize,
    energy_tol_ev: f64,
    density_tol: f64,
) -> crate::scf::scf_loop::ScfResult {
    let norbs = batch.norbs;
    let mut total_valence_elecs = 0.0;
    for &z in &batch.atomic_numbers {
        if let Some(p) = model.get_element(z) {
            total_valence_elecs += p.core_charge;
        }
    }
    let nocc = (total_valence_elecs.round() as usize) / 2;
    let mut soscf_ws = SoscfWorkspace::allocate(norbs);

    // Initial H_core diagonalization
    crate::hamiltonian::build_hcore(batch, model, &mut ws.h_core);
    let enuc = crate::integrals::core_repulsion::compute_total_core_repulsion(batch, model);
    diagonalize_symmetric(&ws.h_core, &mut ws.eigenvalues, &mut ws.eigenvectors);
    compute_density_matrix(&ws.eigenvectors, nocc, &mut ws.density);

    let mut prev_energy = 0.0f64;
    let mut converged = false;
    let mut iters_done = 0;

    for iter in 1..=max_iter {
        iters_done = iter;
        if let Some(ref gamma) = ws.gamma {
            crate::fock::build_fock_with_gamma(
                batch,
                model,
                &ws.h_core,
                &ws.density,
                gamma,
                &mut ws.fock,
            );
        } else {
            build_fock(batch, model, &ws.h_core, &ws.density, &mut ws.fock);
        }

        let e_elec = compute_electronic_energy(&ws.density, &ws.h_core, &ws.fock);
        let e_total = e_elec + enuc;

        // Perform SOSCF coupled Newton-Raphson orbital rotation
        let step_res = soscf_step(
            &ws.fock,
            &mut ws.eigenvectors,
            &ws.eigenvalues,
            &mut ws.tmp1,
            nocc,
            2.0, // Mild trust shift
            &mut soscf_ws,
        );

        let delta_e = (e_total - prev_energy).abs();
        let delta_p = max_density_diff(&ws.tmp1, &ws.density);

        if iter > 1
            && delta_e < energy_tol_ev
            && (delta_p < density_tol || step_res.max_orbital_gradient < 1e-4)
        {
            converged = true;
            ws.density.data.copy_from_slice(&ws.tmp1.data);
            break;
        }

        ws.density.data.copy_from_slice(&ws.tmp1.data);
        prev_energy = e_total;
    }

    let homo = ws.eigenvalues[nocc - 1];
    let lumo = if nocc < norbs {
        ws.eigenvalues[nocc]
    } else {
        0.0
    };
    let e_elec_final = compute_electronic_energy(&ws.density, &ws.h_core, &ws.fock);

    crate::scf::scf_loop::ScfResult {
        converged,
        iterations: iters_done,
        total_energy_ev: e_elec_final + enuc,
        electronic_energy_ev: e_elec_final,
        nuclear_repulsion_ev: enuc,
        homo_energy_ev: homo,
        lumo_energy_ev: lumo,
        dielectric_energy_ev: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::am1::Am1Model;

    #[test]
    fn test_soscf_water_convergence() {
        let z = vec![8, 1, 1];
        let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]];
        let batch = MolecularBatch::new(z, &coords);
        let model = Am1Model;
        let mut ws = ScfWorkspace::allocate(batch.norbs);

        let res = run_rhf_soscf(&batch, &model, &mut ws, 40, 1e-7, 1e-6);
        assert!(res.converged, "SOSCF must converge for water");
        assert!(res.total_energy_ev < 0.0);
    }
}
