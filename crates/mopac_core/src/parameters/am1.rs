//! AM1 (Austin Model 1) Hamiltonian Parameters.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Exact values extracted from OpenMOPAC `parameters_for_AM1_C.F90`.

use super::{GaussianCoreCorrection, ParameterModel, SemiEmpiricalElementParams};

/// The AM1 Parameter Model.
#[derive(Debug, Clone, Copy, Default)]
pub struct Am1Model;

impl ParameterModel for Am1Model {
    fn name(&self) -> &'static str {
        "AM1"
    }

    fn get_element(&self, z: u8) -> Option<SemiEmpiricalElementParams> {
        match z {
            // Element 1: Hydrogen
            1 => Some(SemiEmpiricalElementParams {
                z: 1,
                core_charge: 1.0,
                uss: -11.3964270,
                upp: 0.0,
                udd: 0.0,
                zs: 1.1880780,
                zp: 0.0,
                zd: 0.0,
                betas: -6.1737870,
                betap: 0.0,
                betad: 0.0,
                alpha: 2.8823240,
                gss: 12.8480000,
                gsp: 0.0,
                gpp: 0.0,
                gp2: 0.0,
                hsp: 0.0,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.1227960,
                        b: 5.0,
                        c: 1.2,
                    },
                    GaussianCoreCorrection {
                        a: 0.0050900,
                        b: 5.0,
                        c: 1.8,
                    },
                    GaussianCoreCorrection {
                        a: -0.0183360,
                        b: 2.0,
                        c: 2.1,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 3,
            }),

            // Element 5: Boron
            5 => Some(SemiEmpiricalElementParams {
                z: 5,
                core_charge: 3.0,
                uss: -34.492870,
                upp: -22.631525,
                udd: 0.0,
                zs: 1.611709,
                zp: 1.555385,
                zd: 0.0,
                betas: -9.599114,
                betap: -6.273757,
                betad: 0.0,
                alpha: 2.446909,
                gss: 10.590000,
                gsp: 9.560000,
                gpp: 8.860000,
                gp2: 7.860000,
                hsp: 1.810000,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.182613,
                        b: 6.000000,
                        c: 0.727592,
                    },
                    GaussianCoreCorrection {
                        a: 0.118587,
                        b: 6.000000,
                        c: 1.466639,
                    },
                    GaussianCoreCorrection {
                        a: -0.073280,
                        b: 5.000000,
                        c: 1.570975,
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
                uss: -52.0286580,
                upp: -39.6142390,
                udd: 0.0,
                zs: 1.8086650,
                zp: 1.6851160,
                zd: 0.0,
                betas: -15.7157830,
                betap: -7.7192830,
                betad: 0.0,
                alpha: 2.6482740,
                gss: 12.2300000,
                gsp: 11.4700000,
                gpp: 11.0800000,
                gp2: 9.8400000,
                hsp: 2.4300000,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.0113550,
                        b: 5.0,
                        c: 1.60,
                    },
                    GaussianCoreCorrection {
                        a: 0.0459240,
                        b: 5.0,
                        c: 1.85,
                    },
                    GaussianCoreCorrection {
                        a: -0.0200610,
                        b: 5.0,
                        c: 2.05,
                    },
                    GaussianCoreCorrection {
                        a: -0.0012600,
                        b: 5.0,
                        c: 2.65,
                    },
                ],
                num_gaussians: 4,
            }),

            // Element 7: Nitrogen
            7 => Some(SemiEmpiricalElementParams {
                z: 7,
                core_charge: 5.0,
                uss: -71.8600000,
                upp: -57.1675810,
                udd: 0.0,
                zs: 2.3154100,
                zp: 2.1579400,
                zd: 0.0,
                betas: -20.2991100,
                betap: -18.2386660,
                betad: 0.0,
                alpha: 2.9472860,
                gss: 13.5900000,
                gsp: 12.6600000,
                gpp: 12.6600000,
                gp2: 11.0800000,
                hsp: 2.4300000,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.0500000,
                        b: 5.0,
                        c: 1.45,
                    },
                    GaussianCoreCorrection {
                        a: 0.0500000,
                        b: 5.0,
                        c: 1.70,
                    },
                    GaussianCoreCorrection {
                        a: -0.0200000,
                        b: 5.0,
                        c: 2.10,
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
                uss: -97.8300000,
                upp: -78.2623800,
                udd: 0.0,
                zs: 3.1080320,
                zp: 2.5240390,
                zd: 0.0,
                betas: -29.2727730,
                betap: -29.2727730,
                betad: 0.0,
                alpha: 4.4553710,
                gss: 15.4200000,
                gsp: 14.4800000,
                gpp: 14.5200000,
                gp2: 12.9800000,
                hsp: 3.9400000,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.2809620,
                        b: 5.0,
                        c: 0.8479180,
                    },
                    GaussianCoreCorrection {
                        a: 0.0814300,
                        b: 7.0,
                        c: 1.4450710,
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
                uss: -136.1055790,
                upp: -104.8898850,
                udd: 0.0,
                zs: 3.7700820,
                zp: 2.4946700,
                zd: 0.0,
                betas: -69.5902770,
                betap: -27.9223600,
                betad: 0.0,
                alpha: 5.5178000,
                gss: 16.9200000,
                gsp: 17.2500000,
                gpp: 16.7100000,
                gp2: 14.9100000,
                hsp: 4.8300000,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.2420790,
                        b: 4.8000000,
                        c: 0.9300000,
                    },
                    GaussianCoreCorrection {
                        a: 0.0036070,
                        b: 4.6000000,
                        c: 1.6600000,
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
                uss: -33.9536220,
                upp: -28.9347490,
                udd: 0.0,
                zs: 1.8306970,
                zp: 1.2849530,
                zd: 0.0,
                betas: -3.7848520,
                betap: -1.9681230,
                betad: 0.0,
                alpha: 2.2578160,
                gss: 9.8200000,
                gsp: 8.3600000,
                gpp: 7.3100000,
                gp2: 6.5400000,
                hsp: 1.3200000,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.2500000,
                        b: 9.0000000,
                        c: 0.9114530,
                    },
                    GaussianCoreCorrection {
                        a: 0.0615130,
                        b: 5.0000000,
                        c: 1.9955690,
                    },
                    GaussianCoreCorrection {
                        a: 0.0207890,
                        b: 5.0000000,
                        c: 2.9906100,
                    },
                    GaussianCoreCorrection {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    },
                ],
                num_gaussians: 3,
            }),

            // Element 15: Phosphorus
            15 => Some(SemiEmpiricalElementParams {
                z: 15,
                core_charge: 5.0,
                uss: -42.0298630,
                upp: -34.0307090,
                udd: 0.0,
                zs: 1.9812800,
                zp: 1.8751500,
                zd: 0.0,
                betas: -6.3537640,
                betap: -6.5907090,
                betad: 0.0,
                alpha: 2.4553220,
                gss: 11.5600050,
                gsp: 5.2374490,
                gpp: 7.8775890,
                gp2: 7.3076480,
                hsp: 0.7792380,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.0318270,
                        b: 6.0000000,
                        c: 1.4743230,
                    },
                    GaussianCoreCorrection {
                        a: 0.0184700,
                        b: 7.0000000,
                        c: 1.7793540,
                    },
                    GaussianCoreCorrection {
                        a: 0.0332900,
                        b: 9.0000000,
                        c: 3.0065760,
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
                uss: -56.6940560,
                upp: -48.7170490,
                udd: 0.0,
                zs: 2.3665150,
                zp: 1.6672630,
                zd: 0.0,
                betas: -3.9205660,
                betap: -7.9052780,
                betad: 0.0,
                alpha: 2.4616480,
                gss: 11.7863290,
                gsp: 8.6631270,
                gpp: 10.0393080,
                gp2: 7.7816880,
                hsp: 2.5321370,
                gaussians: [
                    GaussianCoreCorrection {
                        a: -0.5091950,
                        b: 4.5936910,
                        c: 0.7706650,
                    },
                    GaussianCoreCorrection {
                        a: -0.0118630,
                        b: 5.8657310,
                        c: 1.5033130,
                    },
                    GaussianCoreCorrection {
                        a: 0.0123340,
                        b: 13.5573360,
                        c: 2.0091730,
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
                uss: -111.6139480,
                upp: -76.6401070,
                udd: 0.0,
                zs: 3.6313760,
                zp: 2.0767990,
                zd: 0.0,
                betas: -24.5946700,
                betap: -14.6372160,
                betad: 0.0,
                alpha: 2.9193680,
                gss: 15.0300000,
                gsp: 13.1600000,
                gpp: 11.3000000,
                gp2: 9.9700000,
                hsp: 2.4200000,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.0942430,
                        b: 4.0000000,
                        c: 1.3000000,
                    },
                    GaussianCoreCorrection {
                        a: 0.0271680,
                        b: 4.0000000,
                        c: 2.1000000,
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
                uss: -104.6560630,
                upp: -74.9300520,
                udd: 0.0,
                zs: 3.0641330,
                zp: 2.0383330,
                zd: 0.0,
                betas: -19.3998800,
                betap: -8.9571950,
                betad: 0.0,
                alpha: 2.5765460,
                gss: 15.0364395,
                gsp: 13.0346824,
                gpp: 11.2763254,
                gp2: 9.8544255,
                hsp: 2.4558683,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.0666850,
                        b: 4.0000000,
                        c: 1.5000000,
                    },
                    GaussianCoreCorrection {
                        a: 0.0255680,
                        b: 4.0000000,
                        c: 2.3000000,
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
                uss: -103.5896630,
                upp: -74.4299970,
                udd: 0.0,
                zs: 2.1028580,
                zp: 2.1611530,
                zd: 0.0,
                betas: -8.4433270,
                betap: -6.3234050,
                betad: 0.0,
                alpha: 2.2994240,
                gss: 15.0404486,
                gsp: 13.0565580,
                gpp: 11.1477837,
                gp2: 9.9140907,
                hsp: 2.4563820,
                gaussians: [
                    GaussianCoreCorrection {
                        a: 0.0043610,
                        b: 2.3000000,
                        c: 1.8000000,
                    },
                    GaussianCoreCorrection {
                        a: 0.0157060,
                        b: 3.0000000,
                        c: 2.2400000,
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
