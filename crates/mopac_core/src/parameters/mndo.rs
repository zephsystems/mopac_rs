//! MNDO (Modified Neglect of Diatomic Overlap) Hamiltonian parameters.
//!
//! Authentic parameters extracted directly from OpenMOPAC v23.2.5 reference library.
//! Dewar, M. J. S., Thiel, W., J. Am. Chem. Soc. 99, 4899-4907 (1977).
//! Licensed under the Apache License, Version 2.0 (the "License").

use super::{GaussianCoreCorrection, ParameterModel, SemiEmpiricalElementParams};

/// The canonical MNDO semi-empirical Hamiltonian model.
#[derive(Debug, Clone, Copy, Default)]
pub struct MndoModel;

#[allow(clippy::approx_constant)]
impl ParameterModel for MndoModel {
    fn name(&self) -> &'static str {
        "MNDO"
    }

    fn get_element(&self, z: u8) -> Option<SemiEmpiricalElementParams> {
        match z {
            // Element 1: Hydrogen
            1 => Some(SemiEmpiricalElementParams {
                z: 1,
                core_charge: 1.0,
                uss: -11.906276,
                upp: 0.0,
                udd: 0.0,
                zs: 1.331967,
                zp: 0.0,
                zd: 0.0,
                betas: -6.989064,
                betap: 0.0,
                betad: 0.0,
                alpha: 2.544134,
                gss: 12.848000,
                gsp: 0.0,
                gpp: 0.0,
                gp2: 0.0,
                hsp: 0.0,
                gaussians: [GaussianCoreCorrection {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }; 4],
                num_gaussians: 0,
            }),

            // Element 6: Carbon
            6 => Some(SemiEmpiricalElementParams {
                z: 6,
                core_charge: 4.0,
                uss: -52.279745,
                upp: -39.205558,
                udd: 0.0,
                zs: 1.787537,
                zp: 1.787537,
                zd: 0.0,
                betas: -18.985044,
                betap: -7.934122,
                betad: 0.0,
                alpha: 2.546380,
                gss: 12.230000,
                gsp: 11.470000,
                gpp: 11.080000,
                gp2: 9.840000,
                hsp: 2.430000,
                gaussians: [GaussianCoreCorrection {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }; 4],
                num_gaussians: 0,
            }),

            // Element 7: Nitrogen
            7 => Some(SemiEmpiricalElementParams {
                z: 7,
                core_charge: 5.0,
                uss: -71.932122,
                upp: -57.172319,
                udd: 0.0,
                zs: 2.255614,
                zp: 2.255614,
                zd: 0.0,
                betas: -20.495758,
                betap: -20.495758,
                betad: 0.0,
                alpha: 2.861342,
                gss: 13.590000,
                gsp: 12.660000,
                gpp: 12.980000,
                gp2: 11.590000,
                hsp: 3.140000,
                gaussians: [GaussianCoreCorrection {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }; 4],
                num_gaussians: 0,
            }),

            // Element 8: Oxygen
            8 => Some(SemiEmpiricalElementParams {
                z: 8,
                core_charge: 6.0,
                uss: -99.644309,
                upp: -77.797472,
                udd: 0.0,
                zs: 2.699905,
                zp: 2.699905,
                zd: 0.0,
                betas: -32.688082,
                betap: -32.688082,
                betad: 0.0,
                alpha: 3.160604,
                gss: 15.420000,
                gsp: 14.480000,
                gpp: 14.520000,
                gp2: 12.980000,
                hsp: 3.940000,
                gaussians: [GaussianCoreCorrection {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }; 4],
                num_gaussians: 0,
            }),

            // Element 9: Fluorine
            9 => Some(SemiEmpiricalElementParams {
                z: 9,
                core_charge: 7.0,
                uss: -131.071548,
                upp: -105.782137,
                udd: 0.0,
                zs: 2.848487,
                zp: 2.848487,
                zd: 0.0,
                betas: -48.290466,
                betap: -36.508540,
                betad: 0.0,
                alpha: 3.419661,
                gss: 16.920000,
                gsp: 17.250000,
                gpp: 16.710000,
                gp2: 14.910000,
                hsp: 4.830000,
                gaussians: [GaussianCoreCorrection {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }; 4],
                num_gaussians: 0,
            }),

            // Element 15: Phosphorus
            15 => Some(SemiEmpiricalElementParams {
                z: 15,
                core_charge: 5.0,
                uss: -56.143360,
                upp: -42.851080,
                udd: 0.0,
                zs: 2.108720,
                zp: 1.785810,
                zd: 0.0,
                betas: -6.791600,
                betap: -6.791600,
                betad: 0.0,
                alpha: 2.415280,
                gss: 11.560000,
                gsp: 10.080000,
                gpp: 8.640000,
                gp2: 7.680000,
                hsp: 1.920000,
                gaussians: [GaussianCoreCorrection {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }; 4],
                num_gaussians: 0,
            }),

            // Element 16: Sulfur
            16 => Some(SemiEmpiricalElementParams {
                z: 16,
                core_charge: 6.0,
                uss: -72.242281,
                upp: -56.973207,
                udd: 0.0,
                zs: 2.312962,
                zp: 2.009146,
                zd: 0.0,
                betas: -10.761670,
                betap: -10.108433,
                betad: 0.0,
                alpha: 2.478026,
                gss: 12.880000,
                gsp: 11.260000,
                gpp: 9.900000,
                gp2: 8.830000,
                hsp: 2.260000,
                gaussians: [GaussianCoreCorrection {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }; 4],
                num_gaussians: 0,
            }),

            // Element 17: Chlorine
            17 => Some(SemiEmpiricalElementParams {
                z: 17,
                core_charge: 7.0,
                uss: -100.227166,
                upp: -77.378667,
                udd: 0.0,
                zs: 3.784645,
                zp: 2.036263,
                zd: 0.0,
                betas: -14.262320,
                betap: -14.262320,
                betad: 0.0,
                alpha: 2.542201,
                gss: 15.030000,
                gsp: 13.160000,
                gpp: 11.300000,
                gp2: 9.970000,
                hsp: 2.420000,
                gaussians: [GaussianCoreCorrection {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }; 4],
                num_gaussians: 0,
            }),

            // Element 35: Bromine
            35 => Some(SemiEmpiricalElementParams {
                z: 35,
                core_charge: 7.0,
                uss: -99.986441,
                upp: -75.671307,
                udd: 0.0,
                zs: 3.854302,
                zp: 2.199209,
                zd: 0.0,
                betas: -8.917107,
                betap: -9.943740,
                betad: 0.0,
                alpha: 2.445705,
                gss: 15.036440,
                gsp: 13.034682,
                gpp: 11.276325,
                gp2: 9.854425,
                hsp: 2.455868,
                gaussians: [GaussianCoreCorrection {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }; 4],
                num_gaussians: 0,
            }),

            // Element 53: Iodine
            53 => Some(SemiEmpiricalElementParams {
                z: 53,
                core_charge: 7.0,
                uss: -100.003054,
                upp: -74.611469,
                udd: 0.0,
                zs: 2.272961,
                zp: 2.169498,
                zd: 0.0,
                betas: -7.414451,
                betap: -6.196781,
                betad: 0.0,
                alpha: 2.207320,
                gss: 15.040449,
                gsp: 13.056558,
                gpp: 11.147784,
                gp2: 9.914091,
                hsp: 2.456382,
                gaussians: [GaussianCoreCorrection {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }; 4],
                num_gaussians: 0,
            }),

            _ => None,
        }
    }
}
