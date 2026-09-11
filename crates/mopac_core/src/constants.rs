//! Physical, Quantum Chemistry and Metrological Constants for MOPAC_RS.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! This module documents both modern SI / CODATA values and historical 1970s–1980s constants
//! used during the original parameterization of semi-empirical Hamiltonians (MNDO, AM1, PM3).
//!
//! # Historical Context & Citations
//!
//! ## 1. Legacy 1970s–1980s Constants (The AM1 / MNDO Origin)
//! When Michael J. S. Dewar, Walter Thiel, and James J. P. Stewart parameterized MNDO (1977) and AM1 (1985),
//! computational chemistry relied on the 1973 and 1986 CODATA adjustments:
//! * Cohen, E. R., & Taylor, B. N. (1973). "The 1973 Least-Squares Adjustment of the Fundamental Physical Constants",
//!   *Journal of Physical and Chemical Reference Data*, 2(4), 663–734. DOI: 10.1063/1.3253130
//! * Cohen, E. R., & Taylor, B. N. (1987). "The 1986 adjustment of the fundamental physical constants",
//!   *Reviews of Modern Physics*, 59(4), 1121–1148. DOI: 10.1103/RevModPhys.59.1121
//! * Dewar, M. J. S., Zoebisch, E. G., Healy, E. F., & Stewart, J. J. P. (1985). "Development and use of
//!   quantum mechanical molecular models. 76. AM1: a new general purpose quantum mechanical molecular model",
//!   *Journal of the American Chemical Society*, 107(13), 3902–3909. DOI: 10.1021/ja00299a024
//!
//! Under these historical definitions:
//! * Bohr radius: $a_0 = 0.529167 \text{ \AA}$ (truncated/rounded)
//! * 1 a.u. in eV (Hartree): $27.21 \text{ eV}$ (truncated from 27.2116 eV)
//! * $a_0 \times \text{Hartree}$: $14.399 \text{ eV}\cdot\text{\AA}$
//! * 1 eV in kcal/mol: $23.061 \text{ kcal/mol}$
//!
//! Because semi-empirical parameters ($\zeta, \beta, U_{ss}$) were fitted against experimental heats of
//! formation using these truncated constants, using modern constants without recalibration causes small
//! systematic offsets (~0.05–0.2 kcal/mol). MOPAC historically preserved these under the `OLDENS` keyword.
//!
//! ## 2. 2018 & 2022 CODATA Constants and the 2019 SI Redefinition
//! In November 2018 (effective May 20, 2019), the 26th General Conference on Weights and Measures (CGPM)
//! fundamentally redefined the International System of Units (SI):
//! * Bureau International des Poids et Mesures (BIPM) (2019). "The International System of Units (SI)",
//!   9th Edition. Sèvres, France.
//! * Tiesinga, E., Mohr, P. J., Newell, D. B., & Taylor, B. N. (2021). "CODATA recommended values of the
//!   fundamental physical constants: 2018", *Reviews of Modern Physics*, 93(2), 025010. DOI: 10.1103/RevModPhys.93.025010
//! * Mohr, P. J., Newell, D. B., Taylor, B. N., & Tiesinga, E. (2024). "CODATA Recommended Values of the
//!   Fundamental Physical Constants: 2022", *NIST Special Publication 961*.
//!
//! Under the revised SI, the elementary charge $e$, Planck constant $h$, Boltzmann constant $k$, and
//! Avogadro constant $N_A$ are defined **exactly** with zero experimental uncertainty.

/// Mathematical constant Pi ($\pi$).
pub const PI: f64 = std::f64::consts::PI;

/// Two Pi ($2\pi$).
pub const TWO_PI: f64 = 2.0 * PI;

/// Version selection for physical constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConstantsVersion {
    /// 2018 CODATA recommended values (Default in modern MOPAC v22/v23)
    #[default]
    Codata2018,
    /// 2022 CODATA recommended values (NIST / CODATA 2024)
    Codata2022,
    /// Legacy 1973/1986 CODATA values (Used in original MNDO/AM1 parameterizations and MOPAC `OLDENS`)
    Legacy1986,
}

/// Modern CODATA 2018 Fundamental Physical Constants.
///
/// Reference: Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021).
pub mod codata2018 {
    /// Elementary charge $e$ in $10^{-19}$ Coulombs (exact by 2019 SI definition).
    /// Value: $1.602\,176\,634 \times 10^{-19} \text{ C}$.
    pub const ELEMENTARY_CHARGE: f64 = 1.602176634;

    /// Speed of light in vacuum $c$ in cm/s (exact by definition).
    /// Value: $299\,792\,458 \text{ m/s} = 2.997\,924\,58 \times 10^{10} \text{ cm/s}$.
    pub const SPEED_OF_LIGHT_CM_S: f64 = 2.99792458e10;

    /// Planck constant $h$ in erg·s ($10^{-27}$ erg·s, exact by definition).
    /// Value: $6.626\,070\,15 \times 10^{-34} \text{ J}\cdot\text{s} = 6.626\,070\,15 \times 10^{-27} \text{ erg}\cdot\text{s}$.
    pub const PLANCK_CONSTANT_ERG_S: f64 = 6.62607015e-27;

    /// Boltzmann constant $k_B$ in J/K (exact by definition).
    /// Value: $1.380\,649 \times 10^{-23} \text{ J/K} = 1.380\,649 \times 10^{-16} \text{ erg/K}$.
    pub const BOLTZMANN_CONSTANT_J_K: f64 = 1.380649e-23;

    /// Avogadro constant $N_A$ in $\text{mol}^{-1}$ (exact by definition).
    /// Value: $6.022\,140\,76 \times 10^{23} \text{ mol}^{-1}$.
    pub const AVOGADRO: f64 = 6.02214076e23;

    /// Thermochemical calorie in Joules (exact by definition: $1 \text{ cal} \equiv 4.184 \text{ J}$).
    pub const JOULES_PER_CALORIE: f64 = 4.184;

    /// Molar gas constant $R = N_A k_B$ in cal/(mol·K).
    /// Evaluated as: $(N_A \times k_B) / 4.184 = 1.987204258640832 \text{ cal/(mol}\cdot\text{K)}$.
    pub const GAS_CONSTANT_CAL: f64 = 1.987204258640832;

    /// Bohr radius $a_0 = \frac{4\pi \varepsilon_0 \hbar^2}{m_e e^2}$ in Ångströms.
    /// CODATA 2018 value: $0.529177210903 \text{ \AA}$ (relative standard uncertainty: $1.5 \times 10^{-10}$).
    pub const BOHR_RADIUS_ANGSTROMS: f64 = 0.529177210903;

    /// 1 atomic unit of energy (Hartree, $E_h = \frac{\hbar^2}{m_e a_0^2}$) in electron-volts (eV).
    /// CODATA 2018 value: $27.211386245988 \text{ eV}$ (relative standard uncertainty: $1.9 \times 10^{-12}$).
    pub const HARTREE_TO_EV: f64 = 27.211386245988;

    /// Electrostatic conversion factor: $a_0 \times \text{Hartree} = \frac{e^2}{4\pi\varepsilon_0}$ in eV·Å.
    /// Evaluated as: $0.529177210903 \times 27.211386245988 = 14.399645478456 \text{ eV}\cdot\text{\AA}$.
    pub const EV_ANGSTROM_FACTOR: f64 = 14.399645478456;

    /// Conversion factor from electron-volts (eV) to kcal/mol:
    /// Evaluated as: $\frac{e \times N_A \times 10^{-3}}{4.184} = 23.060547830619029 \text{ kcal/(mol}\cdot\text{eV)}$.
    pub const EV_TO_KCAL_MOL: f64 = 23.060_547_830_619_03;
}

/// Modern CODATA 2022 Fundamental Physical Constants (Released May 2024).
///
/// Reference: Mohr, Newell, Taylor, & Tiesinga, NIST SP 961 (2024).
pub mod codata2022 {
    pub use super::codata2018::{
        AVOGADRO, BOLTZMANN_CONSTANT_J_K, ELEMENTARY_CHARGE, JOULES_PER_CALORIE,
        PLANCK_CONSTANT_ERG_S, SPEED_OF_LIGHT_CM_S,
    };

    /// Bohr radius $a_0$ in Ångströms.
    /// CODATA 2022 value: $0.529177210903(80) \text{ \AA}$.
    pub const BOHR_RADIUS_ANGSTROMS: f64 = 0.529177210903;

    /// 1 atomic unit of energy (Hartree) in eV.
    /// CODATA 2022 value: $27.211386245988(53) \text{ eV}$.
    pub const HARTREE_TO_EV: f64 = 27.211386245988;

    /// Electrostatic conversion factor: $a_0 \times \text{Hartree}$ in eV·Å.
    pub const EV_ANGSTROM_FACTOR: f64 = 14.399645478456;

    /// Conversion factor from eV to kcal/mol.
    pub const EV_TO_KCAL_MOL: f64 = 23.060_547_830_619_03;

    /// Gas constant R in cal/(mol·K).
    pub const GAS_CONSTANT_CAL: f64 = 1.987204258640832;
}

/// Historical Legacy 1973/1986 CODATA Constants.
///
/// References:
/// * Cohen & Taylor, J. Phys. Chem. Ref. Data 2, 663 (1973).
/// * Cohen & Taylor, Rev. Mod. Phys. 59, 1121 (1987).
/// * Dewar, Zoebisch, Healy, & Stewart, J. Am. Chem. Soc. 107, 3902 (1985).
///
/// Retained for exact backwards-compatibility with historical MOPAC outputs (`OLDENS`).
pub mod legacy1986 {
    /// Historical Bohr radius used in Dewar's original code: $0.529167 \text{ \AA}$.
    pub const BOHR_RADIUS_ANGSTROMS: f64 = 0.529167;

    /// Historical Hartree-to-eV conversion factor: truncated to $27.21 \text{ eV}$.
    pub const HARTREE_TO_EV: f64 = 27.21;

    /// Historical electrostatic factor: truncated to $14.399 \text{ eV}\cdot\text{\AA}$.
    pub const EV_ANGSTROM_FACTOR: f64 = 14.399;

    /// Historical eV-to-kcal/mol factor: truncated to $23.061 \text{ kcal/(mol}\cdot\text{eV)}$.
    pub const EV_TO_KCAL_MOL: f64 = 23.061;

    /// Historical gas constant R: $1.98726 \text{ cal/(mol}\cdot\text{K)}$.
    pub const GAS_CONSTANT_CAL: f64 = 1.98726;

    /// Historical Avogadro constant: $6.02205 \times 10^{23} \text{ mol}^{-1}$.
    pub const AVOGADRO: f64 = 6.02205e23;

    /// Historical speed of light: $2.99776 \times 10^{10} \text{ cm/s}$.
    pub const SPEED_OF_LIGHT_CM_S: f64 = 2.99776e10;

    /// Historical elementary charge: $1.60217733 \times 10^{-19} \text{ C}$.
    pub const ELEMENTARY_CHARGE: f64 = 1.60217733;

    /// Historical Planck constant: $6.626 \times 10^{-27} \text{ erg}\cdot\text{s}$.
    pub const PLANCK_CONSTANT_ERG_S: f64 = 6.626e-27;

    /// Historical Boltzmann constant: $1.3807 \times 10^{-16} \text{ erg/K}$.
    pub const BOLTZMANN_CONSTANT_J_K: f64 = 1.3807e-23;
}

/// Convert distance in Ångströms to Bohr (atomic units).
#[inline(always)]
pub fn angstrom_to_bohr(r_angstrom: f64, version: ConstantsVersion) -> f64 {
    let a0 = match version {
        ConstantsVersion::Codata2018 => codata2018::BOHR_RADIUS_ANGSTROMS,
        ConstantsVersion::Codata2022 => codata2022::BOHR_RADIUS_ANGSTROMS,
        ConstantsVersion::Legacy1986 => legacy1986::BOHR_RADIUS_ANGSTROMS,
    };
    r_angstrom / a0
}

/// Convert distance in Bohr to Ångströms.
#[inline(always)]
pub fn bohr_to_angstrom(r_bohr: f64, version: ConstantsVersion) -> f64 {
    let a0 = match version {
        ConstantsVersion::Codata2018 => codata2018::BOHR_RADIUS_ANGSTROMS,
        ConstantsVersion::Codata2022 => codata2022::BOHR_RADIUS_ANGSTROMS,
        ConstantsVersion::Legacy1986 => legacy1986::BOHR_RADIUS_ANGSTROMS,
    };
    r_bohr * a0
}

/// Convert energy in electron-volts (eV) to kcal/mol.
#[inline(always)]
pub fn ev_to_kcal_mol(ev: f64, version: ConstantsVersion) -> f64 {
    let factor = match version {
        ConstantsVersion::Codata2018 => codata2018::EV_TO_KCAL_MOL,
        ConstantsVersion::Codata2022 => codata2022::EV_TO_KCAL_MOL,
        ConstantsVersion::Legacy1986 => legacy1986::EV_TO_KCAL_MOL,
    };
    ev * factor
}
