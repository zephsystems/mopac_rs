//! Molecular Properties and Population Analysis Module.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Provides canonical implementations of:
//! - `dipole`: Point-charge and intra-atomic hybridization electric dipole moments matching `dipole.F90`.
//! - `bonds`: Armstrong-Perkins-Stewart / Mayer bond orders and atomic valencies matching `bonds.F90`.
//! - `mulliken`: Mulliken population analysis and Löwdin de-orthogonalization matching `mullik.F90`.

pub mod bonds;
pub mod dipole;
pub mod heat;
pub mod mulliken;

pub use bonds::{compute_bond_orders, BondOrderResult};
pub use dipole::{compute_dipole_moment, DipoleResult, E_ANGSTROM_TO_DEBYE};
pub use heat::{compute_heat_of_formation, get_isolated_atom_energy_and_heat};
pub use mulliken::{compute_mulliken_population, MullikenResult};
