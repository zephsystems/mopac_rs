//! PM6 (Parameterized Model number 6) Hamiltonian Parameters.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Authentic parameter values extracted from OpenMOPAC `parameters_for_PM6_C.F90`.

use super::{GaussianCoreCorrection, ParameterModel, SemiEmpiricalElementParams};

/// The PM6 Parameter Model.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pm6Model;

impl Pm6Model {
    /// Retrieve diatomic pairwise bond parameters (alpb, xfac) for PM6 core-core interaction.
    pub fn get_pair_params(z1: u8, z2: u8) -> (f64, f64) {
        let (za, zb) = if z1 >= z2 { (z1, z2) } else { (z2, z1) };
        match (za, zb) {
            // H pairs
            (1, 1) => (3.540942, 2.243587),
            (6, 1) => (1.027806, 0.216506),
            (7, 1) => (0.969406, 0.175506),
            (8, 1) => (1.260942, 0.192295),
            (9, 1) => (3.136740, 0.815802),
            (15, 1) => (1.926537, 1.234986),
            (16, 1) => (2.215975, 0.849712),
            (17, 1) => (2.402886, 0.754831),

            // C pairs
            (6, 6) => (2.613713, 0.813510),
            (7, 6) => (2.686108, 0.859949),
            (8, 6) => (2.889607, 0.990211),
            (9, 6) => (3.027600, 0.732968),
            (15, 6) => (1.994653, 0.979512),
            (16, 6) => (2.210305, 0.666849),
            (17, 6) => (2.162197, 0.515787),

            // N pairs
            (7, 7) => (2.574502, 0.675313),
            (8, 7) => (2.784292, 0.764756),
            (9, 7) => (2.856646, 0.635854),
            (15, 7) => (2.147042, 0.972154),
            (16, 7) => (2.289990, 0.738710),
            (17, 7) => (2.172134, 0.520745),

            // O pairs
            (8, 8) => (2.623998, 0.535112),
            (9, 8) => (3.015444, 0.674251),
            (15, 8) => (2.220768, 0.878705),
            (16, 8) => (2.383289, 0.747215),
            (17, 8) => (2.323236, 0.585510),

            // F pairs
            (9, 9) => (3.175759, 0.681343),
            (15, 9) => (2.234356, 0.514575),
            (16, 9) => (2.187186, 0.375251),
            (17, 9) => (2.313270, 0.411124),

            // Si pairs
            (14, 1) => (1.896950, 0.924196),
            (14, 6) => (1.984498, 0.785745),
            (14, 7) => (1.818988, 0.592972),
            (14, 8) => (1.923600, 0.751095),
            (14, 9) => (2.131028, 0.543516),
            (14, 14) => (1.329000, 0.273477),
            (15, 14) => (3.313466, 13.239121),
            (16, 14) => (1.885916, 0.876658),
            (17, 14) => (1.684978, 0.513000),
            (35, 14) => (1.570825, 0.589511),
            (53, 14) => (1.559579, 0.700299),

            // P pairs
            (15, 15) => (1.505792, 0.902501),
            (16, 15) => (1.595325, 0.562266),
            (17, 15) => (1.468306, 0.352361),

            // S pairs
            (16, 16) => (1.794556, 0.473856),
            (17, 16) => (1.715435, 0.356971),

            // Cl pairs
            (17, 17) => (1.823239, 0.332919),

            // Br pairs
            (35, 1) => (2.192803, 0.850378),
            (35, 6) => (2.015086, 0.570686),
            (35, 7) => (4.224901, 30.000133),
            (35, 8) => (2.283046, 0.706584),
            (35, 9) => (2.031765, 0.293500),
            (35, 15) => (1.402139, 0.456521),
            (35, 16) => (1.509874, 0.286688),
            (35, 17) => (1.710331, 0.389238),
            (35, 35) => (1.758146, 0.615308),

            // I pairs
            (53, 1) => (2.139913, 0.981898),
            (53, 6) => (2.068710, 0.810156),
            (53, 7) => (1.677518, 0.264903),
            (53, 8) => (2.288919, 0.866204),
            (53, 9) => (2.203580, 0.392425),
            (53, 15) => (2.131593, 3.047207),
            (53, 16) => (1.855110, 0.709929),
            (53, 17) => (1.574161, 0.310474),
            (53, 35) => (1.579376, 0.483054),
            (53, 53) => (1.519925, 0.510542),

            _ => (1.2, 0.0),
        }
    }
}

impl ParameterModel for Pm6Model {
    fn name(&self) -> &'static str {
        "PM6"
    }

    fn pair_core_repulsion(
        &self,
        r_angstrom: f64,
        elem_a: &SemiEmpiricalElementParams,
        elem_b: &SemiEmpiricalElementParams,
    ) -> f64 {
        crate::integrals::core_repulsion::compute_pair_core_repulsion_pm6(
            r_angstrom, elem_a, elem_b,
        )
    }

    fn get_element(&self, z: u8) -> Option<SemiEmpiricalElementParams> {
        match z {
            // Element 1: Hydrogen
            1 => Some(SemiEmpiricalElementParams {
                z: 1,
                core_charge: 1.0,
                uss: -11.246958,
                upp: 0.0,
                udd: 0.0,
                zs: 1.268641,
                zp: 0.0,
                zd: 0.0,
                betas: -8.352984,
                betap: 0.0,
                betad: 0.0,
                alpha: 3.540942,
                gss: 14.448686,
                gsp: 0.0,
                gpp: 0.0,
                gp2: 0.0,
                hsp: 0.0,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.024184,
                        b: 3.055953,
                        c: 1.786011,
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
                uss: -51.089653,
                upp: -39.937920,
                udd: 0.0,
                zs: 2.047558,
                zp: 1.702841,
                zd: 0.0,
                betas: -15.385236,
                betap: -7.471929,
                betad: 0.0,
                alpha: 2.613713,
                gss: 13.335519,
                gsp: 11.528134,
                gpp: 10.778326,
                gp2: 9.486212,
                hsp: 0.717322,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.046302,
                        b: 2.100206,
                        c: 1.333959,
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

            // Element 7: Nitrogen
            7 => Some(SemiEmpiricalElementParams {
                z: 7,
                core_charge: 5.0,
                uss: -57.784823,
                upp: -49.893036,
                udd: 0.0,
                zs: 2.380406,
                zp: 1.999246,
                zd: 0.0,
                betas: -17.979377,
                betap: -15.055017,
                betad: 0.0,
                alpha: 2.574502,
                gss: 12.357026,
                gsp: 9.636190,
                gpp: 12.570756,
                gp2: 10.576425,
                hsp: 2.871545,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.001436,
                        b: 0.495196,
                        c: 1.704857,
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

            // Element 8: Oxygen
            8 => Some(SemiEmpiricalElementParams {
                z: 8,
                core_charge: 6.0,
                uss: -91.678761,
                upp: -70.460949,
                udd: 0.0,
                zs: 5.421751,
                zp: 2.270960,
                zd: 0.0,
                betas: -65.635137,
                betap: -21.622604,
                betad: 0.0,
                alpha: 2.623998,
                gss: 11.304042,
                gsp: 15.807424,
                gpp: 13.618205,
                gp2: 10.332765,
                hsp: 5.010801,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.017771,
                        b: 3.058310,
                        c: 1.896435,
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

            // Element 9: Fluorine
            9 => Some(SemiEmpiricalElementParams {
                z: 9,
                core_charge: 7.0,
                uss: -140.225626,
                upp: -98.778044,
                udd: 0.0,
                zs: 6.043849,
                zp: 2.906722,
                zd: 0.0,
                betas: -69.922593,
                betap: -30.448165,
                betad: 0.0,
                alpha: 3.175759,
                gss: 12.446818,
                gsp: 18.496082,
                gpp: 8.417366,
                gp2: 12.179816,
                hsp: 2.604382,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.010792,
                        b: 6.004648,
                        c: 1.847724,
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

            // Element 14: Silicon
            14 => Some(SemiEmpiricalElementParams {
                z: 14,
                core_charge: 4.0,
                uss: -27.358058,
                upp: -20.490578,
                udd: -22.751900,
                zs: 1.752741,
                zp: 1.198413,
                zd: 2.128593,
                betas: -8.686909,
                betap: -1.856482,
                betad: -6.360627,
                alpha: 1.329000,
                gss: 5.194805,
                gsp: 5.090534,
                gpp: 5.185150,
                gp2: 4.769775,
                hsp: 1.425012,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.208571,
                        b: 6.000483,
                        c: 1.185245,
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

            // Element 15: Phosphorus
            15 => Some(SemiEmpiricalElementParams {
                z: 15,
                core_charge: 5.0,
                uss: -48.729905,
                upp: -40.354689,
                udd: -7.349246,
                zs: 2.158033,
                zp: 1.805343,
                zd: 1.230358,
                betas: -14.583780,
                betap: -11.744725,
                betad: -20.099893,
                alpha: 1.505792,
                gss: 8.758856,
                gsp: 8.483679,
                gpp: 8.662754,
                gp2: 7.734264,
                hsp: 0.871681,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.034320,
                        b: 6.001394,
                        c: 2.296737,
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

            // Element 16: Sulfur
            16 => Some(SemiEmpiricalElementParams {
                z: 16,
                core_charge: 6.0,
                uss: -47.530706,
                upp: -39.191045,
                udd: -46.306944,
                zs: 2.192844,
                zp: 1.841078,
                zd: 3.109401,
                betas: -13.827440,
                betap: -7.664613,
                betad: -9.986172,
                alpha: 1.794556,
                gss: 9.170350,
                gsp: 5.944296,
                gpp: 8.165473,
                gp2: 7.301878,
                hsp: 5.005404,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.036928,
                        b: 1.795067,
                        c: 2.082618,
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

            // Element 17: Chlorine
            17 => Some(SemiEmpiricalElementParams {
                z: 17,
                core_charge: 7.0,
                uss: -61.389930,
                upp: -54.482801,
                udd: -38.258155,
                zs: 2.637050,
                zp: 2.118146,
                zd: 1.324033,
                betas: -2.367988,
                betap: -13.802139,
                betad: -4.037751,
                alpha: 1.823239,
                gss: 11.142654,
                gsp: 7.487881,
                gpp: 9.551886,
                gp2: 8.128436,
                hsp: 5.004267,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.013213,
                        b: 3.687022,
                        c: 2.544635,
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

            // Element 35: Bromine
            35 => Some(SemiEmpiricalElementParams {
                z: 35,
                core_charge: 7.0,
                uss: -45.834364,
                upp: -50.293675,
                udd: 7.086738,
                zs: 4.670684,
                zp: 2.035626,
                zd: 1.521031,
                betas: -32.131665,
                betap: -9.514484,
                betad: -9.839124,
                alpha: 1.758146,
                gss: 7.616791,
                gsp: 5.010425,
                gpp: 9.649216,
                gp2: 8.343792,
                hsp: 4.996553,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.004996,
                        b: 6.001292,
                        c: 2.895153,
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

            // Element 53: Iodine
            53 => Some(SemiEmpiricalElementParams {
                z: 53,
                core_charge: 7.0,
                uss: -59.973232,
                upp: -56.459835,
                udd: -28.822603,
                zs: 4.498653,
                zp: 1.917072,
                zd: 1.875175,
                betas: -30.522481,
                betap: -5.942120,
                betad: -7.676107,
                alpha: 1.519925,
                gss: 7.234759,
                gsp: 9.154406,
                gpp: 9.877466,
                gp2: 8.035916,
                hsp: 5.004215,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.035519,
                        b: 1.744389,
                        c: 1.223844,
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

            _ => None,
        }
    }
}
