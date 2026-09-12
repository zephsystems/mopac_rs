//! Solvent-Accessible Cavity Radii for the COSMO Implicit Solvation Model.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Radii values directly transcribed from OpenMOPAC v23.2.5 `atomradii_C.F90` (`atom_radius_cosmo` and `atom_radius_vdw`).
//!
//! Primary Reference:
//! - Klamt, A.; Schüürmann, G. "COSMO: a new approach to dielectric screening in solvents with
//!   explicit expressions for the screening energy and its gradient", J. Chem. Soc., Perkin Trans. 2,
//!   1993, 799-805. <https://doi.org/10.1039/P29930000799>

/// Bondi van der Waals radii for elements Z=1..86 in Angstroms.
///
/// Reference: Bondi, A. J. Phys. Chem. 1964, 68, 441.
#[rustfmt::skip]
pub const BONDI_VDW_RADII: [f64; 86] = [
    // H, He
    1.20, 1.40,
    // Li, Be, B, C, N, O, F, Ne
    1.82, 1.87, 1.69, 1.70, 1.55, 1.52, 1.47, 1.54,
    // Na, Mg, Al, Si, P, S, Cl, Ar
    2.27, 1.73, 2.06, 2.10, 1.80, 1.80, 1.75, 1.88,
    // K, Ca, Sc, Ti, V, Cr, Mn, Fe
    2.75, 2.17, 2.26, 2.26, 2.15, 2.05, 2.10, 2.06,
    // Co, Ni, Cu, Zn, Ga, Ge, As, Se
    2.05, 1.63, 1.40, 1.39, 1.87, 2.10, 1.85, 1.90,
    // Br, Kr
    1.85, 2.02,
    // Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru
    3.23, 2.94, 2.90, 2.85, 2.80, 2.20, 2.20, 2.20,
    // Rh, Pd, Ag, Cd, In, Sn, Sb, Te
    2.20, 1.63, 1.72, 1.58, 1.93, 2.17, 2.16, 2.06,
    // I, Xe
    1.98, 2.16,
    // Cs, Ba, La, Ce, Pr, Nd, Pm, Sm
    3.42, 2.97, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40,
    // Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb
    2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40,
    // Lu, Hf, Ta, W, Re, Os, Ir, Pt
    2.40, 2.20, 2.20, 2.20, 2.20, 2.20, 2.20, 1.75,
    // Au, Hg, Tl, Pb, Bi, Po, At, Rn
    1.66, 1.55, 1.96, 2.02, 2.26, 2.26, 2.25, 2.30,
];

/// Andreas Klamt's canonical COSMO cavity radii in Angstroms from OpenMOPAC `atomradii_C.F90`.
///
/// For elements with specific optimized parameters:
/// - H: 1.30 Å
/// - B: 2.0475 Å
/// - C: 2.00 Å
/// - N: 1.83 Å
/// - O: 1.72 Å
/// - F: 1.72 Å
/// - Si: 2.457 Å
/// - P: 2.106 Å
/// - S: 2.16 Å
/// - Cl: 2.05 Å
/// - As: 2.223 Å
/// - Se: 2.223 Å
/// - Br: 2.16 Å
/// - I: 2.32 Å
///
/// For other elements, scales the Bondi van der Waals radius by $1.17\times$ (or $1.20\times$) matching Klamt's standard rule.
#[inline]
pub fn cosmo_atomic_radius(z: u8) -> f64 {
    match z {
        1 => 1.30,
        5 => 2.0475,
        6 => 2.00,
        7 => 1.83,
        8 => 1.72,
        9 => 1.72,
        14 => 2.457,
        15 => 2.106,
        16 => 2.16,
        17 => 2.05,
        33 => 2.223,
        34 => 2.223,
        35 => 2.16,
        53 => 2.32,
        _ => {
            if z >= 1 && (z as usize) <= BONDI_VDW_RADII.len() {
                BONDI_VDW_RADII[z as usize - 1] * 1.17
            } else {
                2.00 // standard default radius
            }
        }
    }
}
