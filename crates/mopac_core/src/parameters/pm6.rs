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
            // (1, 1): H - H
            (1, 1) => (3.540942, 2.243587),
            // (6, 1): C - H
            (6, 1) => (1.027806, 0.216506),
            // (6, 6): C - C
            (6, 6) => (2.613713, 0.813510),
            // (7, 1): N - H
            (7, 1) => (0.969406, 0.175506),
            // (7, 6): N - C
            (7, 6) => (2.686108, 0.859949),
            // (7, 7): N - N
            (7, 7) => (2.574502, 0.675313),
            // (8, 1): O - H
            (8, 1) => (1.260942, 0.192295),
            // (8, 6): O - C
            (8, 6) => (2.889607, 0.990211),
            // (8, 7): O - N
            (8, 7) => (2.784292, 0.764756),
            // (8, 8): O - O
            (8, 8) => (2.623998, 0.535112),
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
        crate::integrals::core_repulsion::compute_pair_core_repulsion_pm6(r_angstrom, elem_a, elem_b)
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
                    GaussianCoreCorrection { a: 0.024184, b: 3.055953, c: 1.786011 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
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
                    GaussianCoreCorrection { a: 0.046302, b: 2.100206, c: 1.333959 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
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
                    GaussianCoreCorrection { a: -0.001436, b: 0.495196, c: 1.704857 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
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
                    GaussianCoreCorrection { a: -0.017771, b: 3.058310, c: 1.896435 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                    GaussianCoreCorrection { a: 0.0, b: 0.0, c: 0.0 },
                ],
                num_gaussians: 1,
            }),

            _ => None,
        }
    }
}
