//! RM1 (Recife Model 1) Hamiltonian Parameters.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Authentic parameter values extracted from OpenMOPAC `parameters_for_RM1_C.F90`.

use super::{GaussianCoreCorrection, ParameterModel, SemiEmpiricalElementParams};

/// The RM1 Parameter Model.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rm1Model;

impl ParameterModel for Rm1Model {
    fn name(&self) -> &'static str {
        "RM1"
    }

    fn get_element(&self, z: u8) -> Option<SemiEmpiricalElementParams> {
        match z {
            // Element 1: Hydrogen
            1 => Some(SemiEmpiricalElementParams {
                z: 1,
                core_charge: 1.0,
                uss: -11.9606770,
                upp: 0.0,
                udd: 0.0,
                zs: 1.0826737,
                zp: 0.0,
                zd: 0.0,
                betas: -5.7654447,
                betap: 0.0,
                betad: 0.0,
                alpha: 3.0683595,
                gss: 13.9832130,
                gsp: 0.0,
                gpp: 0.0,
                gp2: 0.0,
                hsp: 0.0,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.1028888,
                        b: 5.9017227,
                        c: 1.1750118,
                    },
                    GaussianCoreCorrection {
                        a: 0.0645745,
                        b: 6.4178567,
                        c: 1.9384448,
                    },
                    GaussianCoreCorrection {
                        a: -0.0356739,
                        b: 2.8047313,
                        c: 1.6365524,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 3,
            }),

            // Element 6: Carbon
            6 => Some(SemiEmpiricalElementParams {
                z: 6,
                core_charge: 4.0,
                uss: -51.7255603,
                upp: -39.4072894,
                udd: 0.0,
                zs: 1.8501880,
                zp: 1.7683009,
                zd: 0.0,
                betas: -15.4593243,
                betap: -8.2360864,
                betad: 0.0,
                alpha: 2.7928208,
                gss: 13.0531244,
                gsp: 11.3347939,
                gpp: 10.9511374,
                gp2: 9.7239510,
                hsp: 1.5521513,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.0746227,
                        b: 5.7392160,
                        c: 1.0439698,
                    },
                    GaussianCoreCorrection {
                        a: 0.0117705,
                        b: 6.9240173,
                        c: 1.6615957,
                    },
                    GaussianCoreCorrection {
                        a: 0.0372066,
                        b: 6.2615894,
                        c: 1.6315872,
                    },
                    GaussianCoreCorrection {
                        a: -0.0027066,
                        b: 9.0000373,
                        c: 2.7955790,
                    },
                ],
                num_gaussians: 4,
            }),

            // Element 7: Nitrogen
            7 => Some(SemiEmpiricalElementParams {
                z: 7,
                core_charge: 5.0,
                uss: -70.8512372,
                upp: -57.9773092,
                udd: 0.0,
                zs: 2.3744716,
                zp: 1.9781257,
                zd: 0.0,
                betas: -20.8712455,
                betap: -16.6717185,
                betad: 0.0,
                alpha: 2.9642254,
                gss: 13.0873623,
                gsp: 13.2122683,
                gpp: 13.6992432,
                gp2: 11.9410395,
                hsp: 5.0000085,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.0607338,
                        b: 4.5889295,
                        c: 1.3787388,
                    },
                    GaussianCoreCorrection {
                        a: 0.0243856,
                        b: 4.6273052,
                        c: 2.0837070,
                    },
                    GaussianCoreCorrection {
                        a: -0.0228343,
                        b: 2.0527466,
                        c: 1.8676382,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 3,
            }),

            // Element 8: Oxygen
            8 => Some(SemiEmpiricalElementParams {
                z: 8,
                core_charge: 6.0,
                uss: -96.9494807,
                upp: -77.8909298,
                udd: 0.0,
                zs: 3.1793691,
                zp: 2.5536191,
                zd: 0.0,
                betas: -29.8510121,
                betap: -29.1510131,
                betad: 0.0,
                alpha: 4.1719672,
                gss: 14.0024279,
                gsp: 14.9562504,
                gpp: 14.1451514,
                gp2: 12.7032550,
                hsp: 3.9321716,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.2309355,
                        b: 5.2182874,
                        c: 0.9036356,
                    },
                    GaussianCoreCorrection {
                        a: 0.0585987,
                        b: 7.4293293,
                        c: 1.5175461,
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
                uss: -134.1836959,
                upp: -107.8466092,
                udd: 0.0,
                zs: 4.4033791,
                zp: 2.6484156,
                zd: 0.0,
                betas: -70.0000051,
                betap: -32.6798271,
                betad: 0.0,
                alpha: 6.0000006,
                gss: 16.7209132,
                gsp: 16.7614263,
                gpp: 15.2258103,
                gp2: 14.8657868,
                hsp: 1.9976617,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.4030203,
                        b: 7.2044196,
                        c: 0.8165301,
                    },
                    GaussianCoreCorrection {
                        a: 0.0708583,
                        b: 9.0000156,
                        c: 1.4380238,
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
                uss: -41.8153318,
                upp: -34.3834253,
                udd: 0.0,
                zs: 2.1224012,
                zp: 1.7432795,
                zd: 0.0,
                betas: -6.1351497,
                betap: -5.9444213,
                betad: 0.0,
                alpha: 1.9099329,
                gss: 11.0805926,
                gsp: 5.6833920,
                gpp: 7.6041756,
                gp2: 7.4026518,
                hsp: 1.1618179,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.4106347,
                        b: 6.0875283,
                        c: 1.3165026,
                    },
                    GaussianCoreCorrection {
                        a: -0.1629929,
                        b: 7.0947260,
                        c: 1.9072132,
                    },
                    GaussianCoreCorrection {
                        a: -0.0488713,
                        b: 8.9997931,
                        c: 2.6585778,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 3,
            }),

            // Element 16: Sulfur
            16 => Some(SemiEmpiricalElementParams {
                z: 16,
                core_charge: 6.0,
                uss: -55.1677512,
                upp: -46.5293042,
                udd: 0.0,
                zs: 2.1334431,
                zp: 1.8746065,
                zd: 0.0,
                betas: -1.9591072,
                betap: -8.7743065,
                betad: 0.0,
                alpha: 2.4401564,
                gss: 12.4882841,
                gsp: 8.5691057,
                gpp: 8.5230117,
                gp2: 7.6686330,
                hsp: 3.8897893,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.7460106,
                        b: 4.8103800,
                        c: 0.5938013,
                    },
                    GaussianCoreCorrection {
                        a: -0.0651929,
                        b: 7.2076086,
                        c: 1.2949201,
                    },
                    GaussianCoreCorrection {
                        a: -0.0065598,
                        b: 9.0000018,
                        c: 1.8006015,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 3,
            }),

            // Element 17: Chlorine
            17 => Some(SemiEmpiricalElementParams {
                z: 17,
                core_charge: 7.0,
                uss: -118.4730692,
                upp: -76.3533034,
                udd: 0.0,
                zs: 3.8649107,
                zp: 1.8959314,
                zd: 0.0,
                betas: -19.9243043,
                betap: -11.5293520,
                betad: 0.0,
                alpha: 3.6935883,
                gss: 15.3602310,
                gsp: 13.3067117,
                gpp: 12.5650264,
                gp2: 9.6639708,
                hsp: 1.7648990,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.1294711,
                        b: 2.9772442,
                        c: 1.4674978,
                    },
                    GaussianCoreCorrection {
                        a: 0.0028890,
                        b: 7.0982759,
                        c: 2.5000272,
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
                uss: -113.4839818,
                upp: -76.1872002,
                udd: 0.0,
                zs: 5.7315721,
                zp: 2.0314758,
                zd: 0.0,
                betas: -1.3413984,
                betap: -8.2022599,
                betad: 0.0,
                alpha: 2.8671053,
                gss: 17.1156307,
                gsp: 15.6241925,
                gpp: 10.7354629,
                gp2: 8.8605620,
                hsp: 2.2351276,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.9868994,
                        b: 4.2848419,
                        c: 2.0001970,
                    },
                    GaussianCoreCorrection {
                        a: -0.9273125,
                        b: 4.5400591,
                        c: 2.0161770,
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
                uss: -74.8999784,
                upp: -51.4102380,
                udd: 0.0,
                zs: 2.5300375,
                zp: 2.3173868,
                zd: 0.0,
                betas: -4.1931615,
                betap: -4.4003841,
                betad: 0.0,
                alpha: 2.1415709,
                gss: 19.9997413,
                gsp: 7.6895767,
                gpp: 7.3048834,
                gp2: 6.8542461,
                hsp: 1.4160294,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.0814772,
                        b: 1.5606507,
                        c: 2.0000206,
                    },
                    GaussianCoreCorrection {
                        a: 0.0591499,
                        b: 5.7611127,
                        c: 2.2048880,
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
