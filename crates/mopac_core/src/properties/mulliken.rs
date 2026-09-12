//! Mulliken Population Analysis Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct mathematical translation of OpenMOPAC `mullik.F90` and `mult.F90`.

use crate::integrals::overlap::compute_diatomic_overlap_block;
use crate::parameters::ParameterModel;
use crate::scf::eigensolver::diagonalize_symmetric;
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch};

/// Comprehensive Mulliken population and charge partition result.
#[derive(Debug, Clone, PartialEq)]
pub struct MullikenResult {
    /// Non-orthogonal diatomic overlap matrix $S_{\mu\nu}$ of dimension $N_{\text{orbs}} \times N_{\text{orbs}}$.
    pub overlap: AlignedMatrix<f64>,
    /// Inverse square root overlap matrix $S^{-1/2}$ of dimension $N_{\text{orbs}} \times N_{\text{orbs}}$.
    pub s_inv_sqrt: AlignedMatrix<f64>,
    /// De-orthogonalized density matrix $P'_{\mu\nu} = 2 \sum_i (S^{-1/2} C)_{\mu i} (S^{-1/2} C)_{\nu i}$.
    pub p_prime: AlignedMatrix<f64>,
    /// Mulliken population matrix: $\text{PopMat}_{\mu\nu} = P'_{\mu\nu} S_{\mu\nu}$.
    pub pop_matrix: AlignedMatrix<f64>,
    /// Gross orbital populations: $\text{GrossPop}_\mu = \sum_\nu \text{PopMat}_{\mu\nu} = (P' S)_{\mu\mu}$.
    pub orbital_populations: Vec<f64>,
    /// Gross atomic populations: $\text{Pop}_A = \sum_{\mu \in A} \text{GrossPop}_\mu$.
    pub atomic_populations: Vec<f64>,
    /// Net Mulliken atomic charges: $q_{\text{Mulliken}, A} = Z_{\text{core}, A} - \text{Pop}_A$.
    pub net_charges: Vec<f64>,
    /// Total number of valence electrons accounted for: $\sum_A \text{Pop}_A \equiv N_{\text{electrons}}$.
    pub total_electrons: f64,
}

/// Compute canonical Mulliken population analysis matching OpenMOPAC `mullik.F90`.
///
/// Axiomatic steps:
/// 1. Assemble the non-orthogonal Slater-Type Orbital (STO) overlap matrix $S_{\mu\nu}$:
///    $S_{\mu\mu} = 1.0$, $S_{\mu\nu} = \text{diatomic STO overlap}$.
/// 2. Compute symmetric Löwdin inverse square root $S^{-1/2} = U \Lambda^{-1/2} U^T$ via cyclic Jacobi eigensolver.
/// 3. De-orthogonalize canonical molecular orbital eigenvectors: $V = S^{-1/2} C$.
/// 4. Construct de-orthogonalized density matrix: $P' = 2 \sum_{i=1}^{N_{\text{occ}}} V_{\cdot i} V_{\cdot i}^T$.
/// 5. Compute population matrix $\text{PopMat}_{\mu\nu} = P'_{\mu\nu} S_{\mu\nu}$.
/// 6. Gross atomic populations $\text{Pop}_A = \sum_{\mu \in A} \sum_\nu \text{PopMat}_{\mu\nu}$ and net charges $q_A = Z_{\text{core}, A} - \text{Pop}_A$.
#[allow(clippy::needless_range_loop)]
pub fn compute_mulliken_population<M: ?Sized + ParameterModel>(
    batch: &MolecularBatch,
    model: &M,
    c_mo: &AlignedMatrix<f64>,
    num_occupied: usize,
) -> MullikenResult {
    let norbs = batch.norbs;
    let natoms = batch.natoms;
    assert_eq!(c_mo.rows, norbs);
    assert_eq!(c_mo.cols, norbs);

    // 1. Build overlap matrix S
    let mut s_mat = AlignedMatrix::zeroed(norbs, norbs);
    for i in 0..norbs {
        s_mat.set(i, i, 1.0);
    }

    for a in 0..natoms {
        let za = batch.atomic_numbers[a];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };
        let orb_a_start = batch.orbital_offsets[a];
        let norb_a = batch.basis_types[a].num_orbitals();

        for b in (a + 1)..natoms {
            let zb = batch.atomic_numbers[b];
            let p_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };
            let orb_b_start = batch.orbital_offsets[b];
            let norb_b = batch.basis_types[b].num_orbitals();

            let r_ab = batch.distance(a, b);
            if r_ab < 1e-10 {
                continue;
            }

            let dir = [
                (batch.x[b] - batch.x[a]) / r_ab,
                (batch.y[b] - batch.y[a]) / r_ab,
                (batch.z[b] - batch.z[a]) / r_ab,
            ];

            let mut s_block = [[0.0f64; 4]; 4];
            compute_diatomic_overlap_block(za, zb, &p_a, &p_b, r_ab, dir, &mut s_block);

            for oa in 0..norb_a.min(4) {
                let idx_a = orb_a_start + oa;
                for ob in 0..norb_b.min(4) {
                    let idx_b = orb_b_start + ob;
                    let val = s_block[oa][ob];
                    s_mat.set(idx_a, idx_b, val);
                    s_mat.set(idx_b, idx_a, val);
                }
            }
        }
    }

    // 2. Compute S^{-1/2} = U * diag(lambda^{-1/2}) * U^T
    let mut eigvals = AlignedVec64::zeroed(norbs);
    let mut u_vecs = AlignedMatrix::zeroed(norbs, norbs);
    diagonalize_symmetric(&s_mat, &mut eigvals, &mut u_vecs);

    let mut s_inv_sqrt = AlignedMatrix::zeroed(norbs, norbs);
    for i in 0..norbs {
        for j in 0..norbs {
            let mut sum = 0.0;
            for k in 0..norbs {
                let lam = eigvals[k].abs().max(1e-12);
                let inv_sqrt_lam = 1.0 / lam.sqrt();
                sum += u_vecs.get(i, k) * inv_sqrt_lam * u_vecs.get(j, k);
            }
            s_inv_sqrt.set(i, j, sum);
        }
    }

    // 3. De-orthogonalize eigenvectors: V = S^{-1/2} * C (matching mult.F90)
    let mut v_mat = AlignedMatrix::zeroed(norbs, norbs);
    for j in 0..norbs {
        // MO column j
        for i in 0..norbs {
            let mut sum = 0.0;
            for k in 0..norbs {
                sum += s_inv_sqrt.get(i, k) * c_mo.get(k, j);
            }
            v_mat.set(i, j, sum);
        }
    }

    // 4. Compute de-orthogonalized density matrix P' = 2 \sum_{occ} V_{\cdot i} V_{\cdot i}^T
    let mut p_prime = AlignedMatrix::zeroed(norbs, norbs);
    for i in 0..norbs {
        for j in 0..norbs {
            let mut sum = 0.0;
            for occ in 0..num_occupied {
                sum += 2.0 * v_mat.get(i, occ) * v_mat.get(j, occ);
            }
            p_prime.set(i, j, sum);
        }
    }

    // 5. Mulliken population matrix PopMat_{mu nu} = P'_{mu nu} * S_{mu nu}
    let mut pop_matrix = AlignedMatrix::zeroed(norbs, norbs);
    let mut orbital_populations = vec![0.0; norbs];
    for i in 0..norbs {
        let mut row_sum = 0.0;
        for j in 0..norbs {
            let p_val = p_prime.get(i, j) * s_mat.get(i, j);
            pop_matrix.set(i, j, p_val);
            row_sum += p_val;
        }
        orbital_populations[i] = row_sum;
    }

    // 6. Gross atomic populations and net Mulliken charges
    let mut atomic_populations = vec![0.0; natoms];
    let mut net_charges = vec![0.0; natoms];
    let mut total_electrons = 0.0;

    for a in 0..natoms {
        let z = batch.atomic_numbers[a];
        let p = model
            .get_element(z)
            .unwrap_or_else(|| panic!("Parameters missing for element Z={}", z));
        let orb_start = batch.orbital_offsets[a];
        let norb_a = batch.basis_types[a].num_orbitals();

        let mut pop_a = 0.0;
        for o in 0..norb_a {
            pop_a += orbital_populations[orb_start + o];
        }
        atomic_populations[a] = pop_a;
        net_charges[a] = p.core_charge - pop_a;
        total_electrons += pop_a;
    }

    MullikenResult {
        overlap: s_mat,
        s_inv_sqrt,
        p_prime,
        pop_matrix,
        orbital_populations,
        atomic_populations,
        net_charges,
        total_electrons,
    }
}
