//! PM3 (Parametric Method 3) Hamiltonian parameters.
//!
//! Authentic parameters extracted directly from OpenMOPAC v23.2.5 reference library.
//! Stewart, J. J. P., J. Comput. Chem. 10, 209-220 (1989).
//! Licensed under the Apache License, Version 2.0 (the "License").

use super::{GaussianCoreCorrection, ParameterModel, SemiEmpiricalElementParams};

/// The PM3 semi-empirical Hamiltonian model.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pm3Model;

impl ParameterModel for Pm3Model {
    fn name(&self) -> &'static str {
        "PM3"
    }

    fn get_element(&self, z: u8) -> Option<SemiEmpiricalElementParams> {
        match z {
            // Element 1: Hydrogen
            1 => Some(SemiEmpiricalElementParams {
                z: 1,
                core_charge: 1.0,
                uss: -13.073321,
                upp: 0.0,
                udd: 0.0,
                zs: 0.967807,
                zp: 0.0,
                zd: 0.0,
                betas: -5.626512,
                betap: 0.0,
                betad: 0.0,
                alpha: 3.356386,
                gss: 14.794208,
                gsp: 0.0,
                gpp: 0.0,
                gp2: 0.0,
                hsp: 0.0,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 1.128750,
                        b: 5.096282,
                        c: 1.537465,
                    },
                    GaussianCoreCorrection {
                        a: -1.060329,
                        b: 6.003788,
                        c: 1.570189,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            // Element 5: Boron
            5 => Some(SemiEmpiricalElementParams {
                z: 5,
                core_charge: 3.0,
                uss: -50.4776829,
                upp: -37.4119835,
                udd: 0.0,
                zs: 1.5312597,
                zp: 1.1434597,
                zd: 0.0,
                betas: -10.5497263,
                betap: -3.9995953,
                betad: 0.0,
                alpha: 2.2104163,
                gss: 18.2782796,
                gsp: 15.3330673,
                gpp: 12.3158582,
                gp2: 11.1785351,
                hsp: 0.5997885,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.3518407,
                        b: 3.0008621,
                        c: 0.8241176,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 1,
            }),

            // Element 6: Carbon
            6 => Some(SemiEmpiricalElementParams {
                z: 6,
                core_charge: 4.0,
                uss: -47.270320,
                upp: -36.266918,
                udd: 0.0,
                zs: 1.565085,
                zp: 1.842345,
                zd: 0.0,
                betas: -11.910015,
                betap: -9.802755,
                betad: 0.0,
                alpha: 2.707807,
                gss: 11.200708,
                gsp: 10.265027,
                gpp: 10.796292,
                gp2: 9.042566,
                hsp: 2.290980,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.050107,
                        b: 6.003165,
                        c: 1.642214,
                    },
                    GaussianCoreCorrection {
                        a: 0.050733,
                        b: 6.002979,
                        c: 0.892488,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            // Element 7: Nitrogen
            7 => Some(SemiEmpiricalElementParams {
                z: 7,
                core_charge: 5.0,
                uss: -49.335672,
                upp: -47.509736,
                udd: 0.0,
                zs: 2.028094,
                zp: 2.313728,
                zd: 0.0,
                betas: -14.062521,
                betap: -20.043848,
                betad: 0.0,
                alpha: 2.830545,
                gss: 11.904787,
                gsp: 7.348565,
                gpp: 11.754672,
                gp2: 10.807277,
                hsp: 1.136713,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 1.501674,
                        b: 5.901148,
                        c: 1.710740,
                    },
                    GaussianCoreCorrection {
                        a: -1.505772,
                        b: 6.004658,
                        c: 1.716149,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            // Element 8: Oxygen
            8 => Some(SemiEmpiricalElementParams {
                z: 8,
                core_charge: 6.0,
                uss: -86.993002,
                upp: -71.879580,
                udd: 0.0,
                zs: 3.796544,
                zp: 2.389402,
                zd: 0.0,
                betas: -45.202651,
                betap: -24.752515,
                betad: 0.0,
                alpha: 3.217102,
                gss: 15.755760,
                gsp: 10.621160,
                gpp: 13.654016,
                gp2: 12.406095,
                hsp: 0.593883,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -1.131128,
                        b: 6.002477,
                        c: 1.607311,
                    },
                    GaussianCoreCorrection {
                        a: 1.137891,
                        b: 5.950512,
                        c: 1.598395,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            // Element 9: Fluorine
            9 => Some(SemiEmpiricalElementParams {
                z: 9,
                core_charge: 7.0,
                uss: -110.435303,
                upp: -105.685047,
                udd: 0.0,
                zs: 4.708555,
                zp: 2.491178,
                zd: 0.0,
                betas: -48.405939,
                betap: -27.744660,
                betad: 0.0,
                alpha: 3.358921,
                gss: 10.496667,
                gsp: 16.073689,
                gpp: 14.817256,
                gp2: 14.418393,
                hsp: 0.727763,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.012166,
                        b: 6.023574,
                        c: 1.856859,
                    },
                    GaussianCoreCorrection {
                        a: -0.002852,
                        b: 6.003717,
                        c: 2.636158,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            // Element 14: Silicon
            14 => Some(SemiEmpiricalElementParams {
                z: 14,
                core_charge: 4.0,
                uss: -26.7634830,
                upp: -22.8136350,
                udd: 0.0,
                zs: 1.6350750,
                zp: 1.3130880,
                zd: 0.0,
                betas: -2.8621450,
                betap: -3.9331480,
                betad: 0.0,
                alpha: 2.1358090,
                gss: 5.0471960,
                gsp: 5.9490570,
                gpp: 6.7593670,
                gp2: 5.1612970,
                hsp: 0.9198320,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.3906000,
                        b: 6.0000540,
                        c: 0.6322620,
                    },
                    GaussianCoreCorrection {
                        a: 0.0572590,
                        b: 6.0071830,
                        c: 2.0199870,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            // Element 15: Phosphorus
            15 => Some(SemiEmpiricalElementParams {
                z: 15,
                core_charge: 5.0,
                uss: -40.413096,
                upp: -29.593052,
                udd: 0.0,
                zs: 2.017563,
                zp: 1.504732,
                zd: 0.0,
                betas: -12.615879,
                betap: -4.160040,
                betad: 0.0,
                alpha: 1.940534,
                gss: 7.801615,
                gsp: 5.186949,
                gpp: 6.618478,
                gp2: 6.062002,
                hsp: 1.542809,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.611421,
                        b: 1.997272,
                        c: 0.794624,
                    },
                    GaussianCoreCorrection {
                        a: -0.093935,
                        b: 1.998360,
                        c: 1.910677,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            // Element 16: Sulfur
            16 => Some(SemiEmpiricalElementParams {
                z: 16,
                core_charge: 6.0,
                uss: -49.895371,
                upp: -44.392583,
                udd: 0.0,
                zs: 1.891185,
                zp: 1.658972,
                zd: 0.0,
                betas: -8.827465,
                betap: -8.091415,
                betad: 0.0,
                alpha: 2.269706,
                gss: 8.964667,
                gsp: 6.785936,
                gpp: 9.968164,
                gp2: 7.970247,
                hsp: 4.041836,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.399191,
                        b: 6.000669,
                        c: 0.962123,
                    },
                    GaussianCoreCorrection {
                        a: -0.054899,
                        b: 6.001845,
                        c: 1.579944,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            // Element 17: Chlorine
            17 => Some(SemiEmpiricalElementParams {
                z: 17,
                core_charge: 7.0,
                uss: -100.626747,
                upp: -53.614396,
                udd: 0.0,
                zs: 2.246210,
                zp: 2.151010,
                zd: 0.0,
                betas: -27.528560,
                betap: -11.593922,
                betad: 0.0,
                alpha: 2.517296,
                gss: 16.013601,
                gsp: 8.048115,
                gpp: 7.522215,
                gp2: 7.504154,
                hsp: 3.481153,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.171591,
                        b: 6.000802,
                        c: 1.087502,
                    },
                    GaussianCoreCorrection {
                        a: -0.013458,
                        b: 1.966618,
                        c: 2.292891,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            // Element 35: Bromine
            35 => Some(SemiEmpiricalElementParams {
                z: 35,
                core_charge: 7.0,
                uss: -116.619311,
                upp: -74.227129,
                udd: 0.0,
                zs: 5.348457,
                zp: 2.127590,
                zd: 0.0,
                betas: -31.171342,
                betap: -6.814013,
                betad: 0.0,
                alpha: 2.511842,
                gss: 15.943425,
                gsp: 16.061680,
                gpp: 8.282763,
                gp2: 7.816849,
                hsp: 0.578869,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.960458,
                        b: 5.976508,
                        c: 2.321654,
                    },
                    GaussianCoreCorrection {
                        a: -0.954916,
                        b: 5.944703,
                        c: 2.328142,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            // Element 53: Iodine
            53 => Some(SemiEmpiricalElementParams {
                z: 53,
                core_charge: 7.0,
                uss: -96.454037,
                upp: -61.091582,
                udd: 0.0,
                zs: 7.001013,
                zp: 2.454354,
                zd: 0.0,
                betas: -14.494234,
                betap: -5.894703,
                betad: 0.0,
                alpha: 1.990185,
                gss: 13.631943,
                gsp: 14.990406,
                gpp: 7.288330,
                gp2: 5.966407,
                hsp: 2.630035,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.131481,
                        b: 5.206417,
                        c: 1.748824,
                    },
                    GaussianCoreCorrection {
                        a: -0.036897,
                        b: 6.010117,
                        c: 2.710373,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 2,
            }),

            _ => None,
        }
    }
}
