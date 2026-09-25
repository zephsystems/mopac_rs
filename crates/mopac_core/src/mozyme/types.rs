//! MOZYME Linear Scaling Data Types and Representations.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Defines Localized Molecular Orbitals (LMOs), Lewis structures, and solver parameters.

use crate::types::AlignedMatrix;

/// Classification of a Localized Molecular Orbital (LMO).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmoType {
    /// 2-center covalent sigma bonding orbital (occupied)
    BondingSigma,
    /// 2-center covalent sigma antibonding orbital (virtual)
    AntibondingSigma,
    /// 2-center covalent pi bonding orbital (occupied)
    BondingPi,
    /// 2-center covalent pi antibonding orbital (virtual)
    AntibondingPi,
    /// 1-center non-bonding lone pair orbital (occupied)
    LonePair,
    /// 1-center core orbital (occupied)
    Core,
    /// 1-center empty/virtual hybrid orbital (virtual)
    VirtualHybrid,
    /// Multi-center delocalized/conjugated pi orbital (occupied or virtual)
    ConjugatedPi,
}

/// A Localized Molecular Orbital (LMO).
///
/// In MOZYME, an LMO is strictly localized over a small subset of atoms (typically 1 or 2),
/// storing only the non-zero atomic orbital indices and their coefficients.
#[derive(Debug, Clone)]
pub struct Lmo {
    /// Global LMO index
    pub index: usize,
    /// Type classification
    pub lmo_type: LmoType,
    /// Occupancy flag: true for occupied (2 electrons in closed-shell RHF), false for virtual
    pub is_occupied: bool,
    /// Indices of participating atoms (e.g., [atom_A, atom_B] for a 2-center bond)
    pub atom_indices: Vec<usize>,
    /// Global atomic orbital (AO) indices participating in this LMO
    pub ao_indices: Vec<usize>,
    /// Expansion coefficients for the participating AOs (normalized: sum c_i^2 = 1.0)
    pub coeffs: Vec<f64>,
    /// Expectation energy: <phi_i | F | phi_i> in eV
    pub energy: f64,
}

impl Lmo {
    /// Evaluate the spatial center of the LMO (average of participating atom positions).
    pub fn center_of_mass(&self, x: &[f64], y: &[f64], z: &[f64]) -> [f64; 3] {
        if self.atom_indices.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &at in &self.atom_indices {
            cx += x[at];
            cy += y[at];
            cz += z[at];
        }
        let n = self.atom_indices.len() as f64;
        [cx / n, cy / n, cz / n]
    }

    /// Calculate the Euclidean distance between the centers of two LMOs.
    pub fn distance_to(&self, other: &Lmo, x: &[f64], y: &[f64], z: &[f64]) -> f64 {
        let c1 = self.center_of_mass(x, y, z);
        let c2 = other.center_of_mass(x, y, z);
        let dx = c1[0] - c2[0];
        let dy = c1[1] - c2[1];
        let dz = c1[2] - c2[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Check if this LMO shares any atomic center with another LMO.
    pub fn shares_atom(&self, other: &Lmo) -> bool {
        for &a1 in &self.atom_indices {
            for &a2 in &other.atom_indices {
                if a1 == a2 {
                    return true;
                }
            }
        }
        false
    }
}

/// A covalent bond detected between two atoms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LewisBond {
    /// First atom index (lower index)
    pub atom1: usize,
    /// Second atom index (higher index)
    pub atom2: usize,
    /// Bond order (1 = single, 2 = double, 3 = triple)
    pub order: usize,
    /// Interatomic distance in Angstroms
    pub distance: f64,
}

/// Lewis chemical topology of a molecule.
#[derive(Debug, Clone, Default)]
pub struct LewisStructure {
    /// List of covalent bonds
    pub bonds: Vec<LewisBond>,
    /// Number of lone pairs on each atom: (atom_index, count)
    pub lone_pairs: Vec<(usize, usize)>,
    /// Formal charges on each atom
    pub formal_charges: Vec<i8>,
    /// Coordination number of each atom
    pub coordination_numbers: Vec<usize>,
}

/// Options controlling the MOZYME linear scaling SCF solver.
#[derive(Debug, Clone)]
pub struct MozymeOptions {
    /// Maximum number of macro-SCF iterations (default: 200)
    pub max_iter: usize,
    /// Maximum number of 2x2 Jacobi rotation sweeps per macro-iteration (default: 50)
    pub max_jacobi_sweeps: usize,
    /// Electronic energy convergence threshold in eV (default: 1.0e-7)
    pub energy_tol: f64,
    /// Maximum off-diagonal Fock matrix element threshold in eV: max |F_ia| (default: 1.0e-4)
    pub jacobi_tol: f64,
    /// Inter-orbital interaction distance cutoff in Angstroms (default: 8.5)
    pub cutoff_distance: f64,
    /// Rotation angle damping factor (default: 1.0)
    pub damping: f64,
    /// Print debug convergence information
    pub verbose: bool,
}

impl Default for MozymeOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            max_jacobi_sweeps: 5,
            energy_tol: 1.0e-6,
            jacobi_tol: 1.0e-4,
            cutoff_distance: 8.5,
            damping: 0.8,
            verbose: false,
        }
    }
}

/// Result of a MOZYME linear scaling calculation.
#[derive(Debug, Clone)]
pub struct MozymeResult {
    /// Did the LMO Jacobi minimization converge within tolerances
    pub converged: bool,
    /// Number of iterations executed
    pub iterations: usize,
    /// Total electronic energy in eV
    pub electronic_energy_ev: f64,
    /// Nuclear core-core repulsion energy in eV
    pub core_repulsion_ev: f64,
    /// Total energy in eV (E_elec + E_nuc)
    pub total_energy_ev: f64,
    /// Standard heat of formation in kcal/mol
    pub heat_of_formation_kcal: f64,
    /// Converged localized molecular orbitals
    pub lmos: Vec<Lmo>,
    /// Reconstructed one-electron density matrix P = 2 * sum_{i in occ} phi_i phi_i^T
    pub density: AlignedMatrix<f64>,
    /// Mulliken partial charges on each atom
    pub atomic_charges: Vec<f64>,
}
