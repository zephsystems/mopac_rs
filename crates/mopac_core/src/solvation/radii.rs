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

/// Andreas Klamt's canonical COSMO cavity van der Waals radii in Angstroms for elements Z=1..107.
///
/// Verbatim transcription from OpenMOPAC v23.2.5 `cosmo.F90:1738-1752` (`rvdw` data array).
#[rustfmt::skip]
pub const OPENMOPAC_COSMO_VDW_RADII: [f64; 107] = [
    // 1..8: H, He, Li, Be, B, C, N, O
    1.30, 1.64, 2.13, 2.19, 2.05, 2.00, 1.83, 1.72,
    // 9..16: F, Ne, Na, Mg, Al, Si, P, S
    1.72, 1.80, 2.66, 2.02, 2.41, 2.46, 2.11, 2.16,
    // 17..24: Cl, Ar, K, Ca, Sc, Ti, V, Cr
    2.05, 2.20, 3.22, 2.54, 2.64, 2.64, 2.52, 2.40,
    // 25..32: Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge
    2.46, 2.41, 2.40, 1.91, 1.64, 1.63, 2.19, 2.46,
    // 33..40: As, Se, Br, Kr, Rb, Sr, Y, Zr
    2.22, 2.22, 2.16, 2.36, 3.78, 3.44, 3.39, 3.33,
    // 41..48: Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd
    3.28, 2.57, 2.57, 2.57, 2.57, 1.91, 2.01, 1.85,
    // 49..56: In, Sn, Sb, Te, I, Xe, Cs, Ba
    2.26, 2.54, 2.53, 2.41, 2.32, 2.53, 4.00, 3.47,
    // 57..64: La..Gd
    2.81, 2.81, 2.81, 2.81, 2.81, 2.81, 2.81, 2.81,
    // 65..72: Tb..Hf
    2.81, 2.81, 2.81, 2.81, 2.81, 2.81, 2.81, 2.57,
    // 73..80: Ta..Hg
    2.57, 2.57, 2.57, 2.57, 2.57, 2.05, 1.94, 1.81,
    // 81..88: Tl..Ra
    2.29, 2.36, 2.64, 2.64, 2.63, 2.69, 2.57, 2.57,
    // 89..96: Ac..Cm
    2.57, 2.57, 2.57, 2.18, 2.57, 2.57, 2.57, 2.57,
    // 97..102: Bk..No
    2.57, 2.57, 2.57, 2.57, 2.57, 2.57,
    // 103..107: Lr..Bh
    2.00, 2.00, 2.00, 2.00, 2.00,
];

/// Returns the Andreas Klamt canonical COSMO cavity solute radius for element `z` in Angstroms.
#[inline]
pub fn cosmo_atomic_radius(z: u8) -> f64 {
    if z >= 1 && (z as usize) <= OPENMOPAC_COSMO_VDW_RADII.len() {
        OPENMOPAC_COSMO_VDW_RADII[z as usize - 1]
    } else {
        2.00
    }
}
