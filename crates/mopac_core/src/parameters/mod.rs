//! Semi-empirical Hamiltonian parameter models for MOPAC_RS.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").

pub mod am1;
pub mod pm3;
pub mod pm6;
pub mod rm1;

pub use am1::Am1Model;
pub use pm3::Pm3Model;
pub use pm6::Pm6Model;
pub use rm1::Rm1Model;

/// A Gaussian core-core repulsion correction term:
/// $\Delta E_{AB}^{\text{Gauss}} = \frac{Z_A Z_B}{R_{AB}} \left[ \sum_k a_k e^{-b_k (R_{AB} - c_k)^2} \right]$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianCoreCorrection {
    /// Amplitude factor $a_k$ (dimensionless or energy-scaled)
    pub a: f64,
    /// Exponential decay width $b_k$ in $\text{\AA}^{-2}$
    pub b: f64,
    /// Centroid interatomic separation $c_k$ in $\text{\AA}$
    pub c: f64,
}

/// Fundamental semi-empirical atomic parameters for an element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemiEmpiricalElementParams {
    /// Atomic number Z (1..118)
    pub z: u8,
    /// Valence core charge $Z_{\text{core}}$ (e.g. 1.0 for H, 4.0 for C, 6.0 for O)
    pub core_charge: f64,
    /// One-center one-electron energy $U_{ss}$ (eV)
    pub uss: f64,
    /// One-center one-electron energy $U_{pp}$ (eV)
    pub upp: f64,
    /// One-center one-electron energy $U_{dd}$ (eV)
    pub udd: f64,
    /// Slater orbital exponent $\zeta_s$ (a.u.)
    pub zs: f64,
    /// Slater orbital exponent $\zeta_p$ (a.u.)
    pub zp: f64,
    /// Slater orbital exponent $\zeta_d$ (a.u.)
    pub zd: f64,
    /// Resonance parameter $\beta_s$ (eV)
    pub betas: f64,
    /// Resonance parameter $\beta_p$ (eV)
    pub betap: f64,
    /// Resonance parameter $\beta_d$ (eV)
    pub betad: f64,
    /// Core-core repulsion exponential parameter $\alpha$ ($\text{\AA}^{-1}$)
    pub alpha: f64,
    /// One-center two-electron integral $(ss|ss)$ (eV)
    pub gss: f64,
    /// One-center two-electron integral $(ss|pp)$ (eV)
    pub gsp: f64,
    /// One-center two-electron integral $(pp|pp)$ (eV)
    pub gpp: f64,
    /// One-center two-electron integral $(pp|p'p')$ (eV)
    pub gp2: f64,
    /// One-center two-electron exchange integral $(sp|sp)$ (eV)
    pub hsp: f64,
    /// Gaussian core corrections
    pub gaussians: [GaussianCoreCorrection; 4],
    /// Active number of Gaussian terms (0..4)
    pub num_gaussians: usize,
}

/// Trait implemented by semi-empirical Hamiltonian models (AM1, PM3, PM6, RM1, MNDO).
pub trait ParameterModel: Send + Sync {
    /// Retrieve parameters for an element by atomic number $Z$.
    fn get_element(&self, z: u8) -> Option<SemiEmpiricalElementParams>;

    /// Return model identifier name, e.g. "AM1", "PM6", "RM1".
    fn name(&self) -> &'static str {
        "AM1"
    }

    /// Compute pairwise core-core nuclear repulsion energy between atom A and atom B in eV.
    fn pair_core_repulsion(
        &self,
        r_angstrom: f64,
        elem_a: &SemiEmpiricalElementParams,
        elem_b: &SemiEmpiricalElementParams,
    ) -> f64 {
        crate::integrals::core_repulsion::compute_pair_core_repulsion(r_angstrom, elem_a, elem_b)
    }
}
