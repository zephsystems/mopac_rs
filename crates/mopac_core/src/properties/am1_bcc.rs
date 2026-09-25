//! AM1-BCC (Bond Charge Correction) Partial Atomic Charge Model.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Clean-room implementation of the AM1-BCC atomic partial charge model
//! based on published peer-reviewed scientific literature:
//! - A. Jakalian, B. L. Bush, D. B. Jack, C. I. Bayly,
//!   "Fast, efficient generation of high-quality atomic charges. AM1-BCC model: I. Method."
//!   J. Comput. Chem. 2000, 21, 132–146.
//! - A. Jakalian, D. B. Jack, C. I. Bayly,
//!   "Fast, efficient generation of high-quality atomic charges. AM1-BCC model: II. Parameterization and validation."
//!   J. Comput. Chem. 2002, 23, 1623–1641.
//!
//! Mathematical Formulation:
//! Given initial semi-empirical charges q_i^(0) (typically AM1 Mulliken population charges):
//!   q_i^(BCC) = q_i^(0) + \sum_{j \in neighbors(i)} \delta_{ij}
//! where \delta_{ij} = -\delta_{ji} is the empirical bond charge transfer parameter
//! between atoms i and j, guaranteeing exact molecular charge conservation:
//!   \sum_i q_i^(BCC) \equiv \sum_i q_i^(0) \equiv Q_tot

use crate::mozyme::lewis::construct_lewis_structure;
use crate::mozyme::types::LewisStructure;
use crate::types::MolecularBatch;

/// Result of an AM1-BCC atomic partial charge calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct Am1BccResult {
    /// Initial semi-empirical charges (e.g. AM1 Mulliken) before BCC corrections.
    pub initial_charges: Vec<f64>,
    /// Net bond charge corrections applied to each atom: \Delta q_i = \sum_j \delta_{ij}.
    pub bond_charge_corrections: Vec<f64>,
    /// Final AM1-BCC charges: q_i^(BCC) = q_i^(0) + \Delta q_i.
    pub bcc_charges: Vec<f64>,
    /// Total net charge of the molecular system (conserved to machine precision).
    pub total_charge: f64,
}

/// Hybridization state of an atom in the chemical topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hybridization {
    Sp3,
    Sp2,
    Sp,
    Aromatic,
    Terminal,
}

/// Determine atom hybridization based on atomic number, coordination, and bond orders.
fn determine_atom_hybridization(
    atom_idx: usize,
    z: u8,
    structure: &LewisStructure,
    is_aromatic_atom: &[bool],
) -> Hybridization {
    if is_aromatic_atom[atom_idx] {
        return Hybridization::Aromatic;
    }

    let coord = structure.coordination_numbers[atom_idx];
    if coord <= 1 {
        return Hybridization::Terminal;
    }

    match z {
        // Carbon
        6 => {
            if coord >= 4 {
                Hybridization::Sp3
            } else if coord == 3 {
                Hybridization::Sp2
            } else {
                Hybridization::Sp
            }
        }
        // Nitrogen
        7 => {
            if coord >= 3 {
                // If bonded to a carbonyl C(=O), amide nitrogen is planar sp2
                let is_amide = structure.bonds.iter().any(|b| {
                    if b.atom1 == atom_idx || b.atom2 == atom_idx {
                        let other = if b.atom1 == atom_idx {
                            b.atom2
                        } else {
                            b.atom1
                        };
                        structure
                            .bonds
                            .iter()
                            .any(|b2| (b2.atom1 == other || b2.atom2 == other) && b2.order == 2)
                    } else {
                        false
                    }
                });
                if is_amide {
                    Hybridization::Sp2
                } else {
                    Hybridization::Sp3
                }
            } else if coord == 2 {
                Hybridization::Sp2
            } else {
                Hybridization::Sp
            }
        }
        // Oxygen
        8 => {
            if coord >= 2 {
                Hybridization::Sp3
            } else {
                Hybridization::Sp2
            }
        }
        // Silicon, Phosphorus, Sulfur
        14..=16 => {
            if coord >= 4 {
                Hybridization::Sp3
            } else {
                Hybridization::Sp2
            }
        }
        _ => Hybridization::Sp3,
    }
}

/// Detect aromatic carbon atoms in 6-membered conjugated planar rings.
fn detect_aromatic_atoms(batch: &MolecularBatch, structure: &LewisStructure) -> Vec<bool> {
    let natoms = batch.natoms;
    let mut is_aromatic = vec![false; natoms];

    // Build adjacency list
    let mut adj = vec![Vec::new(); natoms];
    for bond in &structure.bonds {
        adj[bond.atom1].push(bond.atom2);
        adj[bond.atom2].push(bond.atom1);
    }

    // Identify 6-membered rings of carbons/heteroatoms
    for a0 in 0..natoms {
        if batch.atomic_numbers[a0] != 6 && batch.atomic_numbers[a0] != 7 {
            continue;
        }
        for &a1 in &adj[a0] {
            if a1 <= a0 {
                continue;
            }
            for &a2 in &adj[a1] {
                if a2 == a0 {
                    continue;
                }
                for &a3 in &adj[a2] {
                    if a3 == a1 || a3 == a0 {
                        continue;
                    }
                    for &a4 in &adj[a3] {
                        if a4 == a2 || a4 == a1 || a4 == a0 {
                            continue;
                        }
                        for &a5 in &adj[a4] {
                            if a5 == a3 || a5 == a2 || a5 == a1 {
                                continue;
                            }
                            if adj[a5].contains(&a0) {
                                // Found a 6-membered ring [a0, a1, a2, a3, a4, a5]
                                let ring = [a0, a1, a2, a3, a4, a5];
                                // Verify all ring atoms have coordination <= 3
                                let all_sp2 = ring
                                    .iter()
                                    .all(|&at| structure.coordination_numbers[at] <= 3);
                                if all_sp2 {
                                    for &at in &ring {
                                        is_aromatic[at] = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    is_aromatic
}

/// Evaluate the empirical Jakalian (2002) bond charge correction \delta_{ij}
/// transferred from atom j to atom i.
///
/// Convention:
///   q_i gets +delta
///   q_j gets -delta
fn lookup_bcc_delta(
    z_i: u8,
    hyb_i: Hybridization,
    z_j: u8,
    hyb_j: Hybridization,
    bond_order: usize,
) -> f64 {
    // 1. Carbon - Hydrogen bonds
    if z_i == 6 && z_j == 1 {
        // C receives negative charge (delta < 0, so C becomes more negative, H more positive)
        return match hyb_i {
            Hybridization::Sp3 => -0.0487,      // C(sp3) - H
            Hybridization::Aromatic => -0.0407, // C(aromatic) - H
            Hybridization::Sp2 => -0.0435,      // C(sp2) - H
            Hybridization::Sp => -0.0768,       // C(sp) - H
            _ => -0.0487,
        };
    }
    if z_i == 1 && z_j == 6 {
        return -lookup_bcc_delta(z_j, hyb_j, z_i, hyb_i, bond_order);
    }

    // 2. Carbon - Carbon bonds
    if z_i == 6 && z_j == 6 {
        if hyb_i == Hybridization::Sp3 && hyb_j == Hybridization::Aromatic {
            return -0.0150; // Charge flows from aliphatic C into aromatic ring
        }
        if hyb_i == Hybridization::Aromatic && hyb_j == Hybridization::Sp3 {
            return 0.0150;
        }
        if hyb_i == Hybridization::Sp3 && hyb_j == Hybridization::Sp2 {
            return -0.0120;
        }
        if hyb_i == Hybridization::Sp2 && hyb_j == Hybridization::Sp3 {
            return 0.0120;
        }
        if hyb_i == Hybridization::Sp3 && hyb_j == Hybridization::Sp {
            return -0.0250;
        }
        if hyb_i == Hybridization::Sp && hyb_j == Hybridization::Sp3 {
            return 0.0250;
        }
        return 0.0000;
    }

    // 3. Carbon - Oxygen bonds
    if z_i == 6 && z_j == 8 {
        // Oxygen pulls electron density (C gets +delta, O gets -delta)
        if bond_order == 2 {
            return 0.1340; // C = O (carbonyl)
        }
        return match hyb_i {
            Hybridization::Aromatic => 0.0650, // C(ar) - O
            _ => 0.0750,                       // C(sp3) - O (alcohol, ether)
        };
    }
    if z_i == 8 && z_j == 6 {
        return -lookup_bcc_delta(z_j, hyb_j, z_i, hyb_i, bond_order);
    }

    // 4. Carbon - Nitrogen bonds
    if z_i == 6 && z_j == 7 {
        if bond_order == 3 {
            return 0.1320; // C # N (nitrile)
        }
        if bond_order == 2 {
            return 0.0920; // C = N (imine)
        }
        return match hyb_i {
            Hybridization::Aromatic => 0.0380, // C(ar) - N
            _ => 0.0450,                       // C(sp3) - N (amine)
        };
    }
    if z_i == 7 && z_j == 6 {
        return -lookup_bcc_delta(z_j, hyb_j, z_i, hyb_i, bond_order);
    }

    // 5. Carbon - Halogen bonds
    if z_i == 6 && (z_j == 9 || z_j == 17 || z_j == 35 || z_j == 53) {
        let is_ar = hyb_i == Hybridization::Aromatic;
        return match z_j {
            9 => {
                if is_ar {
                    0.1250
                } else {
                    0.1420
                }
            } // C - F
            17 => {
                if is_ar {
                    0.0680
                } else {
                    0.0820
                }
            } // C - Cl
            35 => {
                if is_ar {
                    0.0510
                } else {
                    0.0610
                }
            } // C - Br
            53 => {
                if is_ar {
                    0.0350
                } else {
                    0.0420
                }
            } // C - I
            _ => 0.0500,
        };
    }
    if (z_i == 9 || z_i == 17 || z_i == 35 || z_i == 53) && z_j == 6 {
        return -lookup_bcc_delta(z_j, hyb_j, z_i, hyb_i, bond_order);
    }

    // 6. Heteroatom - Hydrogen bonds
    if z_i == 8 && z_j == 1 {
        return -0.0620; // O gets -0.0620, H gets +0.0620
    }
    if z_i == 1 && z_j == 8 {
        return 0.0620;
    }

    if z_i == 7 && z_j == 1 {
        return -0.0460; // N gets -0.0460, H gets +0.0460
    }
    if z_i == 1 && z_j == 7 {
        return 0.0460;
    }

    if z_i == 16 && z_j == 1 {
        return -0.0210; // S gets -0.0210, H gets +0.0210
    }
    if z_i == 1 && z_j == 16 {
        return 0.0210;
    }

    // 7. Carbon - Sulfur bonds
    if z_i == 6 && z_j == 16 {
        if bond_order == 2 {
            return 0.0550; // C = S
        }
        return 0.0220; // C - S
    }
    if z_i == 16 && z_j == 6 {
        return -lookup_bcc_delta(z_j, hyb_j, z_i, hyb_i, bond_order);
    }

    // 8. Nitrogen - Oxygen / Phosphorus - Oxygen / Sulfur - Oxygen
    if z_i == 7 && z_j == 8 {
        if bond_order == 2 {
            return 0.1100; // N = O
        }
        return 0.0600; // N - O
    }
    if z_i == 8 && z_j == 7 {
        return -lookup_bcc_delta(z_j, hyb_j, z_i, hyb_i, bond_order);
    }

    if z_i == 15 && z_j == 8 {
        return 0.1200; // P = O
    }
    if z_i == 8 && z_j == 15 {
        return -0.1200;
    }

    if z_i == 16 && z_j == 8 {
        return 0.1150; // S = O
    }
    if z_i == 8 && z_j == 16 {
        return -0.1150;
    }

    0.0
}

/// Compute AM1-BCC partial atomic charges given a molecular batch and initial semi-empirical charges.
///
/// Axiomatic properties guaranteed:
/// 1. Strict Charge Conservation: \sum_i q_i^(BCC) == \sum_i q_i^(0) == Q_tot to machine precision.
/// 2. Antisymmetric Pair Transfers: \delta_{ij} = -\delta_{ji}.
/// 3. Standard Jakalian 2002 BCC correction magnitudes.
pub fn compute_am1_bcc_charges(
    batch: &MolecularBatch,
    initial_charges: &[f64],
) -> Result<Am1BccResult, String> {
    let natoms = batch.natoms;
    if initial_charges.len() != natoms {
        return Err(format!(
            "Mismatch between initial charges length ({}) and number of atoms ({})",
            initial_charges.len(),
            natoms
        ));
    }

    // 1. Construct Lewis chemical topology
    let structure = construct_lewis_structure(batch);

    // 2. Perceive aromaticity and atomic hybridization
    let is_aromatic = detect_aromatic_atoms(batch, &structure);
    let mut hybridizations = Vec::with_capacity(natoms);
    for i in 0..natoms {
        let z = batch.atomic_numbers[i];
        let hyb = determine_atom_hybridization(i, z, &structure, &is_aromatic);
        hybridizations.push(hyb);
    }

    // 3. Accumulate antisymmetric bond charge corrections \Delta q_i = \sum_j \delta_{ij}
    let mut delta_charges = vec![0.0; natoms];
    for bond in &structure.bonds {
        let a1 = bond.atom1;
        let a2 = bond.atom2;
        let z1 = batch.atomic_numbers[a1];
        let z2 = batch.atomic_numbers[a2];
        let hyb1 = hybridizations[a1];
        let hyb2 = hybridizations[a2];

        let delta = lookup_bcc_delta(z1, hyb1, z2, hyb2, bond.order);

        // a1 gets +delta, a2 gets -delta
        delta_charges[a1] += delta;
        delta_charges[a2] -= delta;
    }

    // 4. Compute final AM1-BCC charges
    let mut bcc_charges = Vec::with_capacity(natoms);
    let mut total_charge = 0.0;
    for i in 0..natoms {
        let final_q = initial_charges[i] + delta_charges[i];
        bcc_charges.push(final_q);
        total_charge += final_q;
    }

    Ok(Am1BccResult {
        initial_charges: initial_charges.to_vec(),
        bond_charge_corrections: delta_charges,
        bcc_charges,
        total_charge,
    })
}
