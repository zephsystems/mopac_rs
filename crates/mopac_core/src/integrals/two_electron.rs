//! Two-Center Two-Electron Integral (TEI) Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Implements Klopman-Ohno / Dewar multipole expansions over Slater orbitals.

use crate::constants::codata2018::EV_ANGSTROM_FACTOR;

/// Evaluate the two-center two-electron monopole integral $\gamma_{AB} = (s_A s_A | s_B s_B)$ in eV.
///
/// Uses the Dewar-Klopman semi-empirical formula:
/// $$\gamma_{AB}(R) = \frac{e^2}{\sqrt{R^2 + (\rho_A + \rho_B)^2}}$$
/// where $\rho_A = \frac{e^2}{2 g_{ss}^A}$ and $\rho_B = \frac{e^2}{2 g_{ss}^B}$.
///
/// Properties:
/// * At $R = 0$ (one-center limit), $\gamma_{AA}(0) = g_{ss}^A$.
/// * At $R \to \infty$ (long-range limit), $\gamma_{AB}(R) \to \frac{e^2}{R}$ (Coulomb law).
#[inline(always)]
pub fn dewar_klopman_monopole(r_angstrom: f64, gss_a: f64, gss_b: f64) -> f64 {
    let rho_a = EV_ANGSTROM_FACTOR / (2.0 * gss_a);
    let rho_b = EV_ANGSTROM_FACTOR / (2.0 * gss_b);
    let rho_sum = rho_a + rho_b;
    EV_ANGSTROM_FACTOR / (r_angstrom * r_angstrom + rho_sum * rho_sum).sqrt()
}
