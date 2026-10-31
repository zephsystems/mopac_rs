//! Periodic Boundary Conditions (PBC) & Band Structure Module.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Implements crystal orbital Bloch SCF for 1D polymers, 2D surfaces, and 3D bulk solids.

pub mod bloch_scf;
pub mod solrot;
pub mod unit_cell;

pub use bloch_scf::{PbcOptions, PbcResult, PbcWorkspace, run_pbc_scf};
pub use solrot::{
    SolrotPairIntegrals, build_solrot_fock, build_solrot_hcore, compute_lattice_bounds,
    compute_solrot_lattice_integrals, run_solrot_pbc_scf, trunk,
};
pub use unit_cell::{KPoint, PeriodicDimension, TranslationIndex, UnitCell};
