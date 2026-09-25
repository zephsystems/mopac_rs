//! Directional Hybridization and Localized Molecular Orbital (LMO) Constructor.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Generates directional $sp^3, sp^2, sp$ hybrid atomic orbitals (HAOs) and
//! orthonormal Lewis bonding ($\sigma, \pi$), antibonding ($\sigma^*, \pi^*$), and lone-pair LMOs matching OpenMOPAC `hybrid.F90`.

use super::types::{LewisStructure, Lmo, LmoType};
use crate::types::{BasisType, MolecularBatch};

/// Directional Hybrid Atomic Orbital (HAO) on a single atomic center.
#[derive(Debug, Clone)]
pub struct HybridAtomicOrbital {
    /// Atomic site index
    pub atom_index: usize,
    /// Coefficients across the atom's basis functions (e.g. s, px, py, pz)
    pub ao_coeffs: Vec<f64>,
}

/// Construct the full set of initial orthonormal Localized Molecular Orbitals (LMOs).
#[allow(clippy::needless_range_loop)]
pub fn construct_initial_lmos(
    batch: &MolecularBatch,
    lewis: &LewisStructure,
) -> (Vec<Lmo>, Vec<Lmo>) {
    let mut occupied_lmos = Vec::new();
    let mut virtual_lmos = Vec::new();
    let mut lmo_counter = 0;

    // 1. Generate hybrid atomic orbitals (HAOs) for each atom
    let atom_hybrids = generate_atom_hybrids(batch, lewis);

    // Track which pi hybrids have been consumed on each atom
    let mut pi_hybrid_indices = vec![0usize; batch.natoms];

    // 2. Form 2-center sigma and pi bonding and antibonding LMOs
    for (bond_idx, bond) in lewis.bonds.iter().enumerate() {
        let a1 = bond.atom1;
        let a2 = bond.atom2;

        let a1_off = batch.orbital_offsets[a1];
        let a2_off = batch.orbital_offsets[a2];

        // 2a. Sigma bond
        let h1_sigma = &atom_hybrids[a1].bonding_hybrids[bond_idx_on_atom(lewis, a1, bond_idx)];
        let h2_sigma = &atom_hybrids[a2].bonding_hybrids[bond_idx_on_atom(lewis, a2, bond_idx)];

        let mut ao_indices_sigma = Vec::new();
        for (i, _) in h1_sigma.ao_coeffs.iter().enumerate() {
            ao_indices_sigma.push(a1_off + i);
        }
        for (i, _) in h2_sigma.ao_coeffs.iter().enumerate() {
            ao_indices_sigma.push(a2_off + i);
        }

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        // Occupied Sigma Bonding LMO: phi = (h1 + h2) / sqrt(2)
        let mut occ_sigma = Vec::with_capacity(ao_indices_sigma.len());
        for &c in &h1_sigma.ao_coeffs {
            occ_sigma.push(c * inv_sqrt2);
        }
        for &c in &h2_sigma.ao_coeffs {
            occ_sigma.push(c * inv_sqrt2);
        }

        occupied_lmos.push(Lmo {
            index: lmo_counter,
            lmo_type: LmoType::BondingSigma,
            is_occupied: true,
            atom_indices: vec![a1, a2],
            ao_indices: ao_indices_sigma.clone(),
            coeffs: occ_sigma,
            energy: -12.0,
        });
        lmo_counter += 1;

        // Virtual Sigma Antibonding LMO: phi* = (h1 - h2) / sqrt(2)
        let mut virt_sigma = Vec::with_capacity(ao_indices_sigma.len());
        for &c in &h1_sigma.ao_coeffs {
            virt_sigma.push(c * inv_sqrt2);
        }
        for &c in &h2_sigma.ao_coeffs {
            virt_sigma.push(-c * inv_sqrt2);
        }

        virtual_lmos.push(Lmo {
            index: lmo_counter,
            lmo_type: LmoType::AntibondingSigma,
            is_occupied: false,
            atom_indices: vec![a1, a2],
            ao_indices: ao_indices_sigma,
            coeffs: virt_sigma,
            energy: 4.0,
        });
        lmo_counter += 1;

        // 2b. Pi bonds if bond order >= 2 (e.g. 1 pi for double bond, 2 pi for triple bond)
        if bond.order >= 2 {
            let num_pi_bonds = bond.order - 1;
            for _ in 0..num_pi_bonds {
                let pi_idx1 = pi_hybrid_indices[a1];
                let pi_idx2 = pi_hybrid_indices[a2];

                if pi_idx1 < atom_hybrids[a1].pi_hybrids.len()
                    && pi_idx2 < atom_hybrids[a2].pi_hybrids.len()
                {
                    let h1_pi = &atom_hybrids[a1].pi_hybrids[pi_idx1];
                    let h2_pi = &atom_hybrids[a2].pi_hybrids[pi_idx2];

                    let mut ao_indices_pi = Vec::new();
                    for (i, _) in h1_pi.ao_coeffs.iter().enumerate() {
                        ao_indices_pi.push(a1_off + i);
                    }
                    for (i, _) in h2_pi.ao_coeffs.iter().enumerate() {
                        ao_indices_pi.push(a2_off + i);
                    }

                    // Check relative phase alignment between pi orbitals
                    let dot_pi = dot_product(&h1_pi.ao_coeffs, &h2_pi.ao_coeffs);
                    let phase_sign = if dot_pi < 0.0 { -1.0 } else { 1.0 };

                    let mut occ_pi = Vec::with_capacity(ao_indices_pi.len());
                    for &c in &h1_pi.ao_coeffs {
                        occ_pi.push(c * inv_sqrt2);
                    }
                    for &c in &h2_pi.ao_coeffs {
                        occ_pi.push(phase_sign * c * inv_sqrt2);
                    }

                    occupied_lmos.push(Lmo {
                        index: lmo_counter,
                        lmo_type: LmoType::BondingPi,
                        is_occupied: true,
                        atom_indices: vec![a1, a2],
                        ao_indices: ao_indices_pi.clone(),
                        coeffs: occ_pi,
                        energy: -9.0,
                    });
                    lmo_counter += 1;

                    let mut virt_pi = Vec::with_capacity(ao_indices_pi.len());
                    for &c in &h1_pi.ao_coeffs {
                        virt_pi.push(c * inv_sqrt2);
                    }
                    for &c in &h2_pi.ao_coeffs {
                        virt_pi.push(-phase_sign * c * inv_sqrt2);
                    }

                    virtual_lmos.push(Lmo {
                        index: lmo_counter,
                        lmo_type: LmoType::AntibondingPi,
                        is_occupied: false,
                        atom_indices: vec![a1, a2],
                        ao_indices: ao_indices_pi,
                        coeffs: virt_pi,
                        energy: 2.0,
                    });
                    lmo_counter += 1;

                    pi_hybrid_indices[a1] += 1;
                    pi_hybrid_indices[a2] += 1;
                }
            }
        }
    }

    // 3. Form 1-center lone pair LMOs (occupied)
    for (atom_idx, &(_, count)) in lewis.lone_pairs.iter().enumerate() {
        let a_off = batch.orbital_offsets[atom_idx];
        let num_orbs = batch.basis_types[atom_idx].num_orbitals();
        let ao_indices: Vec<usize> = (a_off..(a_off + num_orbs)).collect();

        for lp in 0..count {
            if lp < atom_hybrids[atom_idx].lone_pair_hybrids.len() {
                let h = &atom_hybrids[atom_idx].lone_pair_hybrids[lp];
                occupied_lmos.push(Lmo {
                    index: lmo_counter,
                    lmo_type: LmoType::LonePair,
                    is_occupied: true,
                    atom_indices: vec![atom_idx],
                    ao_indices: ao_indices.clone(),
                    coeffs: h.ao_coeffs.clone(),
                    energy: -10.0,
                });
                lmo_counter += 1;
            }
        }
    }

    // 4. Form any remaining 1-center virtual hybrid LMOs (virtual)
    for atom_idx in 0..batch.natoms {
        let a_off = batch.orbital_offsets[atom_idx];
        let num_orbs = batch.basis_types[atom_idx].num_orbitals();
        let ao_indices: Vec<usize> = (a_off..(a_off + num_orbs)).collect();

        // Any unused pi hybrids become virtual orbitals
        let used_pi = pi_hybrid_indices[atom_idx];
        for h in &atom_hybrids[atom_idx].pi_hybrids[used_pi..] {
            virtual_lmos.push(Lmo {
                index: lmo_counter,
                lmo_type: LmoType::VirtualHybrid,
                is_occupied: false,
                atom_indices: vec![atom_idx],
                ao_indices: ao_indices.clone(),
                coeffs: h.ao_coeffs.clone(),
                energy: 5.0,
            });
            lmo_counter += 1;
        }

        for h in &atom_hybrids[atom_idx].virtual_hybrids {
            virtual_lmos.push(Lmo {
                index: lmo_counter,
                lmo_type: LmoType::VirtualHybrid,
                is_occupied: false,
                atom_indices: vec![atom_idx],
                ao_indices: ao_indices.clone(),
                coeffs: h.ao_coeffs.clone(),
                energy: 6.0,
            });
            lmo_counter += 1;
        }
    }

    (occupied_lmos, virtual_lmos)
}

struct AtomHybrids {
    bonding_hybrids: Vec<HybridAtomicOrbital>,
    pi_hybrids: Vec<HybridAtomicOrbital>,
    lone_pair_hybrids: Vec<HybridAtomicOrbital>,
    virtual_hybrids: Vec<HybridAtomicOrbital>,
}

#[allow(clippy::needless_range_loop)]
fn generate_atom_hybrids(batch: &MolecularBatch, lewis: &LewisStructure) -> Vec<AtomHybrids> {
    let mut atom_hybrids = Vec::with_capacity(batch.natoms);

    for i in 0..batch.natoms {
        let basis = batch.basis_types[i];
        let num_orbs = basis.num_orbitals();

        if basis == BasisType::S {
            // 1s orbital on Hydrogen
            atom_hybrids.push(AtomHybrids {
                bonding_hybrids: vec![HybridAtomicOrbital {
                    atom_index: i,
                    ao_coeffs: vec![1.0],
                }],
                pi_hybrids: Vec::new(),
                lone_pair_hybrids: Vec::new(),
                virtual_hybrids: Vec::new(),
            });
            continue;
        }

        // SP or SPD atom: construct directional p-vectors towards bonded neighbors
        let xi = batch.x[i];
        let yi = batch.y[i];
        let zi = batch.z[i];

        let mut bonded_dirs = Vec::new();
        let mut num_pi_needed = 0;

        for b in &lewis.bonds {
            if b.atom1 == i || b.atom2 == i {
                let other = if b.atom1 == i { b.atom2 } else { b.atom1 };
                let dx = batch.x[other] - xi;
                let dy = batch.y[other] - yi;
                let dz = batch.z[other] - zi;
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
                bonded_dirs.push([dx / len, dy / len, dz / len]);
                if b.order >= 2 {
                    num_pi_needed += b.order - 1;
                }
            }
        }

        let num_bonds = bonded_dirs.len();
        let lp_count = lewis.lone_pairs[i].1;

        // Steric number for sigma framework (bonds + in-plane lone pairs)
        let steric_num = num_bonds + lp_count;

        // Determine hybridization sp^n (sp3 for 4 targets, sp2 for 3 targets, sp for 2 targets)
        let s_weight = if steric_num > 0 {
            (1.0 / steric_num as f64).sqrt()
        } else {
            0.5
        };
        let p_weight = (1.0 - s_weight * s_weight).sqrt().max(0.0);

        let mut raw_hybrids = Vec::with_capacity(num_bonds);
        for dir in &bonded_dirs {
            let mut coeffs = vec![0.0; num_orbs];
            coeffs[0] = s_weight; // s orbital
            coeffs[1] = p_weight * dir[0]; // px
            coeffs[2] = p_weight * dir[1]; // py
            coeffs[3] = p_weight * dir[2]; // pz
            normalize_vector(&mut coeffs);
            raw_hybrids.push(coeffs);
        }

        // Symmetric Löwdin orthogonalization for bonding hybrids
        let ortho_hybrids = lowdin_orthogonalize(&raw_hybrids);
        let mut bonding_hybrids = Vec::with_capacity(num_bonds);
        for coeffs in ortho_hybrids {
            bonding_hybrids.push(HybridAtomicOrbital {
                atom_index: i,
                ao_coeffs: coeffs,
            });
        }

        // Find remaining orthogonal vectors in R^num_orbs to form pi orbitals, lone pairs, and virtual hybrids
        let mut all_hybrids = bonding_hybrids.clone();
        let mut pi_hybrids = Vec::new();
        let mut lone_pair_hybrids = Vec::new();
        let mut virtual_hybrids = Vec::new();

        // Standard canonical unit vectors in AO basis [s, px, py, pz, ...]
        for ao in 0..num_orbs {
            let mut trial = vec![0.0; num_orbs];
            trial[ao] = 1.0;

            // Gram-Schmidt orthogonalization against existing hybrids
            for h in &all_hybrids {
                let dot = dot_product(&trial, &h.ao_coeffs);
                for k in 0..num_orbs {
                    trial[k] -= dot * h.ao_coeffs[k];
                }
            }

            let norm = dot_product(&trial, &trial).sqrt();
            if norm > 1e-4 {
                normalize_vector(&mut trial);
                let new_hybrid = HybridAtomicOrbital {
                    atom_index: i,
                    ao_coeffs: trial,
                };
                all_hybrids.push(new_hybrid.clone());

                if pi_hybrids.len() < num_pi_needed {
                    pi_hybrids.push(new_hybrid);
                } else if lone_pair_hybrids.len() < lp_count {
                    lone_pair_hybrids.push(new_hybrid);
                } else {
                    virtual_hybrids.push(new_hybrid);
                }
            }
        }

        atom_hybrids.push(AtomHybrids {
            bonding_hybrids,
            pi_hybrids,
            lone_pair_hybrids,
            virtual_hybrids,
        });
    }

    atom_hybrids
}

fn bond_idx_on_atom(lewis: &LewisStructure, atom: usize, global_bond_idx: usize) -> usize {
    let mut count = 0;
    for (idx, b) in lewis.bonds.iter().enumerate() {
        if b.atom1 == atom || b.atom2 == atom {
            if idx == global_bond_idx {
                return count;
            }
            count += 1;
        }
    }
    0
}

fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum()
}

fn normalize_vector(v: &mut [f64]) {
    let norm = dot_product(v, v).sqrt();
    if norm > 1e-14 {
        for val in v.iter_mut() {
            *val /= norm;
        }
    }
}

/// Löwdin symmetric orthogonalization: H_ortho = H * S^(-1/2)
#[allow(clippy::needless_range_loop)]
fn lowdin_orthogonalize(hybrids: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = hybrids.len();
    if m <= 1 {
        return hybrids.to_vec();
    }

    let n = hybrids[0].len();
    // Build overlap matrix S (m x m)
    let mut s = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            s[i][j] = dot_product(&hybrids[i], &hybrids[j]);
        }
    }

    // Jacobi eigenvalue decomposition for symmetric m x m matrix (m <= 4)
    let mut v = vec![vec![0.0; m]; m];
    for i in 0..m {
        v[i][i] = 1.0;
    }
    let mut a = s.clone();

    for _sweep in 0..50 {
        let mut max_off = 0.0;
        for p in 0..m {
            for q in (p + 1)..m {
                let val = a[p][q].abs();
                if val > max_off {
                    max_off = val;
                }
                if val > 1e-12 {
                    let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
                    let c = theta.cos();
                    let s = theta.sin();

                    // Apply rotation to A
                    let mut a_new = a.clone();
                    for i in 0..m {
                        a_new[i][p] = c * a[i][p] - s * a[i][q];
                        a_new[i][q] = s * a[i][p] + c * a[i][q];
                    }
                    for i in 0..m {
                        a[p][i] = c * a_new[p][i] - s * a_new[q][i];
                        a[q][i] = s * a_new[p][i] + c * a_new[q][i];
                    }

                    // Accumulate eigenvectors in V
                    for i in 0..m {
                        let vip = v[i][p];
                        let viq = v[i][q];
                        v[i][p] = c * vip - s * viq;
                        v[i][q] = s * vip + c * viq;
                    }
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }
    }

    // Compute S^(-1/2) = V * diag(lambda^(-1/2)) * V^T
    let mut s_inv_sqrt = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..m {
                let lambda_k = a[k][k].max(1e-8);
                sum += v[i][k] * (1.0 / lambda_k.sqrt()) * v[j][k];
            }
            s_inv_sqrt[i][j] = sum;
        }
    }

    // Multiply: H_ortho[i] = sum_j s_inv_sqrt[j][i] * H[j]
    let mut ortho = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..m {
            let factor = s_inv_sqrt[j][i];
            for k in 0..n {
                ortho[i][k] += factor * hybrids[j][k];
            }
        }
        normalize_vector(&mut ortho[i]);
    }

    ortho
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mozyme::lewis::construct_lewis_structure;

    #[test]
    fn test_water_initial_lmos_orthonormality() {
        let z = vec![8, 1, 1];
        let coords = vec![[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]];
        let batch = MolecularBatch::new(z, &coords);
        let lewis = construct_lewis_structure(&batch);
        let (occ, virt) = construct_initial_lmos(&batch, &lewis);

        assert_eq!(
            occ.len(),
            4,
            "Water must have 4 occupied LMOs (2 bonds + 2 lone pairs)"
        );
        assert_eq!(
            virt.len(),
            2,
            "Water must have 2 virtual LMOs (2 antibonds)"
        );

        // Verify full mutual orthonormality between all LMO pairs
        let all_lmos: Vec<&Lmo> = occ.iter().chain(virt.iter()).collect();
        for (i, lmo_i) in all_lmos.iter().enumerate() {
            for (j, lmo_j) in all_lmos.iter().enumerate() {
                let mut ov = 0.0;
                for (idx_a, &ao_a) in lmo_i.ao_indices.iter().enumerate() {
                    for (idx_b, &ao_b) in lmo_j.ao_indices.iter().enumerate() {
                        if ao_a == ao_b {
                            ov += lmo_i.coeffs[idx_a] * lmo_j.coeffs[idx_b];
                        }
                    }
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (ov - expected).abs() < 1e-10,
                    "LMO pair ({}, {}) overlap deviated: got {}, expected {}",
                    i,
                    j,
                    ov,
                    expected
                );
            }
        }
    }
}
