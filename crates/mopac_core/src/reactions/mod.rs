//! Reaction Coordinates and Direct Molecular Dynamics.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Provides:
//! - Intrinsic Reaction Coordinate (IRC) path tracing via the mass-weighted González-Schlegel algorithm.
//! - Dynamic Reaction Coordinate (DRC) direct molecular dynamics via symplectic Velocity-Verlet integration.

pub mod drc;
pub mod irc;

pub use drc::*;
pub use irc::*;
