//! Multi-Electron Configuration Interaction (MECI) and UV-Vis Spectroscopy Module.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").

pub mod meci;
pub mod spectrum;

pub use meci::{
    CiActiveSpace, CiState, MeciOptions, MeciResult, MeciWorkspace, Microstate, StateSpin,
    assign_spin_state, compute_active_mo_two_electron_integrals, compute_ci_state_density,
    compute_meci_nuclear_gradients, compute_meci_numerical_gradients, generate_microstates,
    run_meci,
};
pub use spectrum::{
    EPSILON_PEAK_FACTOR, OSCILLATOR_STRENGTH_FACTOR, UvVisSpectrum,
    compute_transition_dipoles_and_oscillator_strengths, simulate_uv_vis_spectrum,
};
