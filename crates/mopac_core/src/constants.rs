//! Physical and quantum chemistry constants for MOPAC_RS.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Preserves both modern 2018 CODATA and legacy MOPAC constants for backward parity.

/// Mathematical constant Pi ($\pi$)
pub const PI: f64 = std::f64::consts::PI;

/// Two Pi ($2\pi$)
pub const TWO_PI: f64 = 2.0 * PI;

/// CODATA 2018 Fundamental Constants
pub mod codata2018 {
    /// Elementary charge $e$ in $10^{-19}$ Coulombs
    pub const ELEMENTARY_CHARGE: f64 = 1.602176634;
    /// Bohr radius $a_0$ in Ångströms (0.529177210903 Å)
    pub const BOHR_RADIUS_ANGSTROMS: f64 = 0.529177210903;
    /// 1 atomic unit of energy (Hartree) in electron-volts (eV)
    pub const HARTREE_TO_EV: f64 = 27.211386245988;
    /// Electrostatic conversion factor: $a_0 \times \text{Hartree}$ in eV·Å
    pub const EV_ANGSTROM_FACTOR: f64 = 14.399645478456;
    /// Conversion from eV to kcal/mol
    pub const EV_TO_KCAL_MOL: f64 = 23.060547830619029;
    /// Gas constant R in cal/(mol·K)
    pub const GAS_CONSTANT_CAL: f64 = 1.987204258640832;
    /// Avogadro constant $N_A$ in $\text{mol}^{-1}$
    pub const AVOGADRO: f64 = 6.02214076e23;
}

/// Legacy MOPAC Constants (historical 1980s constants used in early MNDO/AM1 parameterizations)
pub mod legacy {
    pub const BOHR_RADIUS_ANGSTROMS: f64 = 0.529167;
    pub const HARTREE_TO_EV: f64 = 27.21;
    pub const EV_ANGSTROM_FACTOR: f64 = 14.399;
    pub const EV_TO_KCAL_MOL: f64 = 23.061;
}

/// Convert distance in Ångströms to Bohr (atomic units).
#[inline(always)]
pub fn angstrom_to_bohr(r_angstrom: f64, codata: bool) -> f64 {
    let a0 = if codata {
        codata2018::BOHR_RADIUS_ANGSTROMS
    } else {
        legacy::BOHR_RADIUS_ANGSTROMS
    };
    r_angstrom / a0
}

/// Convert distance in Bohr to Ångströms.
#[inline(always)]
pub fn bohr_to_angstrom(r_bohr: f64, codata: bool) -> f64 {
    let a0 = if codata {
        codata2018::BOHR_RADIUS_ANGSTROMS
    } else {
        legacy::BOHR_RADIUS_ANGSTROMS
    };
    r_bohr * a0
}

/// Convert energy in electron-volts (eV) to kcal/mol.
#[inline(always)]
pub fn ev_to_kcal_mol(ev: f64, codata: bool) -> f64 {
    let factor = if codata {
        codata2018::EV_TO_KCAL_MOL
    } else {
        legacy::EV_TO_KCAL_MOL
    };
    ev * factor
}
