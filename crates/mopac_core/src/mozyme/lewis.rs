//! Chemical Topology and Lewis Structure Builder for MOZYME.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Automatically identifies covalent bonding connectivity, bond orders,
//! formal charges, lone pairs, and coordination geometries matching OpenMOPAC `mbonds.F90`.

use super::types::{LewisBond, LewisStructure};
use crate::types::MolecularBatch;

/// Covalent radius in Angstroms for main-group and transition elements.
pub fn covalent_radius(z: u8) -> f64 {
    match z {
        1 => 0.31,  // H
        2 => 0.28,  // He
        3 => 1.28,  // Li
        4 => 0.96,  // Be
        5 => 0.84,  // B
        6 => 0.76,  // C
        7 => 0.71,  // N
        8 => 0.66,  // O
        9 => 0.57,  // F
        10 => 0.58, // Ne
        11 => 1.66, // Na
        12 => 1.41, // Mg
        13 => 1.21, // Al
        14 => 1.11, // Si
        15 => 1.07, // P
        16 => 1.05, // S
        17 => 1.02, // Cl
        18 => 1.06, // Ar
        30 => 1.22, // Zn
        35 => 1.20, // Br
        53 => 1.39, // I
        _ => 1.20,  // Generic default for extended elements
    }
}

/// Valence electron count for an isolated neutral atom.
pub fn valence_electron_count(z: u8) -> usize {
    match z {
        1 => 1,
        2 => 2,
        3 => 1,
        4 => 2,
        5 => 3,  // B
        6 => 4,  // C
        7 => 5,  // N
        8 => 6,  // O
        9 => 7,  // F
        10 => 8, // Ne
        11 => 1,
        12 => 2,
        13 => 3,
        14 => 4, // Si
        15 => 5, // P
        16 => 6, // S
        17 => 7, // Cl
        18 => 8,
        30 => 2, // Zn (3d is treated as core)
        35 => 7, // Br
        53 => 7, // I
        _ => 4,
    }
}

/// Construct the Lewis structure and covalent connectivity from a MolecularBatch.
#[allow(clippy::needless_range_loop)]
pub fn construct_lewis_structure(batch: &MolecularBatch) -> LewisStructure {
    let natoms = batch.natoms;
    let mut bonds = Vec::new();
    let mut coordination_numbers = vec![0usize; natoms];

    // 1. Identify pairwise covalent bonds based on interatomic distances and covalent radii
    for i in 0..natoms {
        let z_i = batch.atomic_numbers[i];
        let r_cov_i = covalent_radius(z_i);
        let xi = batch.x[i];
        let yi = batch.y[i];
        let zi = batch.z[i];

        for j in (i + 1)..natoms {
            let z_j = batch.atomic_numbers[j];
            let r_cov_j = covalent_radius(z_j);
            let xj = batch.x[j];
            let yj = batch.y[j];
            let zj = batch.z[j];

            let dx = xi - xj;
            let dy = yi - yj;
            let dz = zi - zj;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            // Covalent bonding threshold: R_ij <= 1.20 * (r_cov(i) + r_cov(j)) + 0.20 A
            let max_covalent_dist = 1.20 * (r_cov_i + r_cov_j) + 0.20;

            if dist <= max_covalent_dist && dist > 0.40 {
                // Determine initial bond order based on distance and atom valencies
                let order = determine_bond_order(z_i, z_j, dist, r_cov_i + r_cov_j);
                bonds.push(LewisBond {
                    atom1: i,
                    atom2: j,
                    order,
                    distance: dist,
                });
                coordination_numbers[i] += 1;
                coordination_numbers[j] += 1;
            }
        }
    }

    // 2. Compute lone pairs and formal charges on each atom
    let mut lone_pairs = Vec::with_capacity(natoms);
    let mut formal_charges = vec![0i8; natoms];

    for i in 0..natoms {
        let z_i = batch.atomic_numbers[i];
        let val_e = valence_electron_count(z_i);

        // Sum of bond orders attached to atom i
        let mut total_bond_order = 0;
        for b in &bonds {
            if b.atom1 == i || b.atom2 == i {
                total_bond_order += b.order;
            }
        }

        // Remaining electrons form non-bonding lone pairs
        let non_bonding_electrons = val_e.saturating_sub(total_bond_order);

        let lp_count = non_bonding_electrons / 2;
        lone_pairs.push((i, lp_count));

        // Formal charge = Val_e - (Lone_electrons + Shared_electrons / 2)
        // Shared_electrons / 2 = total_bond_order
        let assigned_electrons = (lp_count * 2) + total_bond_order;
        let fc = val_e as i64 - assigned_electrons as i64;
        formal_charges[i] = fc.clamp(-4, 4) as i8;
    }

    LewisStructure {
        bonds,
        lone_pairs,
        formal_charges,
        coordination_numbers,
    }
}

/// Helper function to estimate bond order (1, 2, or 3) from distance and sum of covalent radii.
fn determine_bond_order(z1: u8, z2: u8, dist: f64, r_sum: f64) -> usize {
    // Hydrogen only forms single bonds
    if z1 == 1 || z2 == 1 {
        return 1;
    }

    // Halogens (F, Cl, Br, I) generally form single bonds with main group atoms
    if matches!(z1, 9 | 17 | 35 | 53) || matches!(z2, 9 | 17 | 35 | 53) {
        return 1;
    }

    // Carbon-Carbon, Carbon-Nitrogen, Carbon-Oxygen, Nitrogen-Nitrogen, Nitrogen-Oxygen bonds
    let ratio = dist / r_sum;
    if ratio < 0.82 {
        // Triple bond region (e.g. C#C ~ 1.20 A / 1.52 = 0.79, C#N ~ 1.16 A / 1.47 = 0.79)
        3
    } else if ratio < 0.92 {
        // Double bond region (e.g. C=C ~ 1.34 A / 1.52 = 0.88, C=O ~ 1.22 A / 1.42 = 0.86)
        2
    } else {
        // Single bond region
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lewis_water() {
        let z = vec![8, 1, 1];
        let coords = vec![[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]];
        let batch = MolecularBatch::new(z, &coords);
        let lewis = construct_lewis_structure(&batch);

        assert_eq!(lewis.bonds.len(), 2, "Water must have 2 O-H bonds");
        assert_eq!(lewis.bonds[0].order, 1);
        assert_eq!(lewis.bonds[1].order, 1);
        assert_eq!(lewis.lone_pairs[0].1, 2, "Oxygen must have 2 lone pairs");
        assert_eq!(lewis.lone_pairs[1].1, 0, "Hydrogen must have 0 lone pairs");
        assert_eq!(lewis.lone_pairs[2].1, 0, "Hydrogen must have 0 lone pairs");
    }

    #[test]
    fn test_lewis_methane() {
        let z = vec![6, 1, 1, 1, 1];
        let r = 1.09;
        let coords = vec![
            [0.0, 0.0, 0.0],
            [r, 0.0, 0.0],
            [-r / 3.0, r * (8.0f64 / 9.0).sqrt(), 0.0],
            [
                -r / 3.0,
                -r * (2.0f64 / 9.0).sqrt(),
                r * (2.0f64 / 3.0).sqrt(),
            ],
            [
                -r / 3.0,
                -r * (2.0f64 / 9.0).sqrt(),
                -r * (2.0f64 / 3.0).sqrt(),
            ],
        ];
        let batch = MolecularBatch::new(z, &coords);
        let lewis = construct_lewis_structure(&batch);

        assert_eq!(lewis.bonds.len(), 4, "Methane must have 4 C-H bonds");
        assert_eq!(lewis.lone_pairs[0].1, 0, "Carbon in CH4 has 0 lone pairs");
    }
}
