//! AM1 (Austin Model 1) Hamiltonian Parameters.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Exact values extracted from OpenMOPAC `parameters_for_AM1_C.F90`.

use super::{GaussianCoreCorrection, ParameterModel, SemiEmpiricalElementParams};

/// The AM1 Parameter Model.
#[derive(Debug, Clone, Copy, Default)]
pub struct Am1Model;

impl ParameterModel for Am1Model {
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
                    GaussianCoreCorrection { a: 0.1227960, b: 5.0, c: 1.2 },
                    GaussianCoreCorrection { a: 0.0050900, b: 5.0, c: 1.8 },
                    GaussianCoreCorrection { a: -0.0183360, b: 2.0, c: 2.1 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
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
                    GaussianCoreCorrection { a: 0.0113550, b: 5.0, c: 1.60 },
                    GaussianCoreCorrection { a: 0.0459240, b: 5.0, c: 1.85 },
                    GaussianCoreCorrection { a: -0.0200610, b: 5.0, c: 2.05 },
                    GaussianCoreCorrection { a: -0.0012600, b: 5.0, c: 2.65 },
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
                    GaussianCoreCorrection { a: 0.0500000, b: 5.0, c: 1.45 },
                    GaussianCoreCorrection { a: 0.0500000, b: 5.0, c: 1.70 },
                    GaussianCoreCorrection { a: -0.0200000, b: 5.0, c: 2.10 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
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
                    GaussianCoreCorrection { a: 0.2809620, b: 5.0, c: 0.8479180 },
                    GaussianCoreCorrection { a: 0.0814300, b: 7.0, c: 1.4450710 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                ],
                num_gaussians: 2,
            }),

            _ => None,
        }
    }
}
