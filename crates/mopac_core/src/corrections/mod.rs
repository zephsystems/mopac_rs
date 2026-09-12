//! Post-SCF Empirical Non-Covalent Corrections Module.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Provides canonical implementations of:
//! - `dispersion`: Empirical Van der Waals dispersion (PM6-DH+, PM7) and analytical Cartesian gradients matching OpenMOPAC `H_bond_correction_PM6_DH_Dispersion.F90`.

pub mod dispersion;
pub mod h_bonds4;

pub use dispersion::{
    compute_dispersion_energy, compute_dispersion_energy_and_gradients,
    diatomic_dispersion_parameters, DispersionModel, DISPERSION_C6, DISPERSION_NEFF, DISPERSION_R0,
};
pub use h_bonds4::{
    compute_h4_energy, compute_hh_repulsion_energy_and_gradients, cvalence_contribution,
    hh_repulsion_potential, H4Parameters, COVALENT_RADII,
};
