//! COSMO (COnductor-like Screening MOdel) Implicit Solvation Module.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct translation and rigorous mathematical verification of OpenMOPAC v23.2.5 `cosmo.F90`.
//!
//! Provides canonical implementations of:
//! - `radii`: Authentic Bondi and Klamt solvent-accessible cavity radii.
//! - `tessellation`: Regular icosahedral sphere tessellation (`dvfill`) with exact unit normals and area weights.
//! - `cavity`: Solvent-Accessible Surface (SAS) numerical boundary element cavity generation.
//! - `cosmo`: Electrostatic Boundary Element solver ($A$-matrix Cholesky decomposition, $B$-matrix coupling, dielectric screening, and self-consistent reaction field Fock matrix updates).

pub mod cavity;
pub mod cosmo;
pub mod radii;
pub mod tessellation;

pub use cavity::{CavitySegment, CosmoCavity};
pub use cosmo::{CosmoParams, CosmoState, A0_EV};
pub use radii::{cosmo_atomic_radius, BONDI_VDW_RADII};
pub use tessellation::{generate_sphere_tessellation, SpherePoint};
