//! Molecular Properties and Population Analysis Module.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Provides canonical implementations of:
//! - `dipole`: Point-charge and intra-atomic hybridization electric dipole moments matching `dipole.F90`.
//! - `bonds`: Armstrong-Perkins-Stewart / Mayer bond orders and atomic valencies matching `bonds.F90`.
//! - `mulliken`: Mulliken population analysis and Löwdin de-orthogonalization matching `mullik.F90`.

pub mod am1_bcc;
pub mod bonds;
pub mod dipole;
pub mod esp;
pub mod heat;
pub mod mulliken;
pub mod polarizability;

pub use am1_bcc::{Am1BccResult, compute_am1_bcc_charges};
pub use bonds::{BondOrderResult, compute_bond_orders};
pub use dipole::{DipoleResult, E_ANGSTROM_TO_DEBYE, compute_dipole_moment};
pub use esp::{EspOptions, EspResult, bondi_vdw_radius_angstrom, compute_esp_charges};
pub use heat::{
    compute_c_triple_bond_c_correction, compute_heat_of_formation, compute_nsp2_correction,
    get_isolated_atom_energy_and_heat,
};
pub use mulliken::{MullikenResult, compute_mulliken_population};
pub use polarizability::{PolarizabilityOptions, PolarizabilityResult, compute_polarizability};
