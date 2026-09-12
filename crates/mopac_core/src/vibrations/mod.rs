//! Vibrational Frequency & Hessian Analysis Module.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Provides canonical numerical Cartesian second derivatives, mass-weighted Eckart projection,
//! harmonic normal mode decomposition, and full statistical thermodynamics.

pub mod hessian;

pub use hessian::{
    compute_hessian_and_frequencies, HessianOptions, HessianResult, NormalMode,
    ThermodynamicProperties, CM1_TO_KCAL_MOL, CM1_TO_KELVIN, KCAL_MOL_A2_AMU_TO_CM1,
};
