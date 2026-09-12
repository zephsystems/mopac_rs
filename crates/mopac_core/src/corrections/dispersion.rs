//! Empirical Van der Waals Dispersion Corrections (PM6-DH+ and PM7).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct mathematical translation of OpenMOPAC `H_bond_correction_PM6_DH_Dispersion.F90`.
//! Reference:
//! - Korth, M., Pitonak, M., Rezac, J., Hobza, P., "A Transferable H-bonding Correction for Semiempirical
//!   Quantum-Chemical Methods", *J. Chem. Theory Comput.* 6, 344-352 (2010).
//! - Stewart, J. J. P., "Optimization of Parameters for Semiempirical Methods VI: More Modifications to the
//!   NDDO Approximations and Re-optimization of Parameters", *J. Mol. Model.* 19, 1-32 (2013).

use crate::types::MolecularBatch;

/// Dispersion correction model parameter sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DispersionModel {
    /// PM6-DH+ dispersion model (Korth et al. 2010)
    Pm6DhPlus,
    /// PM7 dispersion model (Stewart 2013)
    Pm7,
}

impl DispersionModel {
    /// Damping steepness parameter alpha.
    #[inline(always)]
    pub fn alpha(self) -> f64 {
        match self {
            Self::Pm6DhPlus => 20.0,
            Self::Pm7 => 15.450118,
        }
    }

    /// Scaling factor s for cut-off distance R0.
    #[inline(always)]
    pub fn s(self) -> f64 {
        match self {
            Self::Pm6DhPlus => 1.04,
            Self::Pm7 => 1.226593,
        }
    }

    /// Overall empirical energy scaling constant cscale.
    #[inline(always)]
    pub fn cscale(self) -> f64 {
        match self {
            Self::Pm6DhPlus => 0.89,
            Self::Pm7 => 2.286419,
        }
    }
}

/// Dispersion atomic C6 coefficients (J * nm^6 / mol) for elements Z=1..86.
/// Extracted from OpenMOPAC `H_bond_correction_PM6_DH_Dispersion.F90` lines 74-85.
#[rustfmt::skip]
pub const DISPERSION_C6: [f64; 86] = [
    // H, He, Li, Be, B, C, N, O
    0.16, 0.084, 0.00, 0.00, 5.79, 1.65, 1.11, 0.70,
    // F, Ne, Na, Mg, Al, Si, P, S
    0.57, 0.45, 0.00, 0.00, 0.00, 0.00, 3.25, 5.79,
    // Cl, Ar, K, Ca, Sc, Ti, V, Cr
    5.97, 3.71, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    // Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge
    0.00, 0.00, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00,
    // As, Se, Br, Kr, Rb, Sr, Y, Zr
    0.00, 0.00, 11.60, 4.47, 0.00, 0.00, 0.00, 0.00,
    // Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    // In, Sn, Sb, Te, I, Xe, Cs, Ba
    0.00, 0.00, 0.00, 0.00, 25.80, 16.50, 0.00, 0.00,
    // La - Gd (Z=57..64)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    // Tb - Hf (Z=65..72)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    // Ta - Hg (Z=73..80)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    // Tl - Rn (Z=81..86)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
];

/// Dispersion atomic cut-off radii R0 (pm) for elements Z=1..86.
/// Extracted from OpenMOPAC `H_bond_correction_PM6_DH_Dispersion.F90` lines 88-99.
#[rustfmt::skip]
pub const DISPERSION_R0: [f64; 86] = [
    // H, He, Li, Be, B, C, N, O
    156.0, 140.0, 0.0, 0.0, 180.0, 170.0, 155.0, 152.0,
    // F, Ne, Na, Mg, Al, Si, P, S
    147.0, 154.0, 0.0, 0.0, 0.0, 0.0, 180.0, 180.0,
    // Cl, Ar, K, Ca, Sc, Ti, V, Cr
    175.0, 188.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge
    0.0, 0.0, 140.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // As, Se, Br, Kr, Rb, Sr, Y, Zr
    0.0, 0.0, 185.0, 202.0, 0.0, 0.0, 0.0, 0.0,
    // Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // In, Sn, Sb, Te, I, Xe, Cs, Ba
    0.0, 0.0, 0.0, 0.0, 198.0, 216.0, 0.0, 0.0,
    // La - Gd (Z=57..64)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // Tb - Hf (Z=65..72)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // Ta - Hg (Z=73..80)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // Tl - Rn (Z=81..86)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// Slater-Kirkwood effective number of electrons Neff for elements Z=1..86.
/// Extracted from OpenMOPAC `H_bond_correction_PM6_DH_Dispersion.F90` lines 102-113.
#[rustfmt::skip]
pub const DISPERSION_NEFF: [f64; 86] = [
    // H, He, Li, Be, B, C, N, O
    0.80, 1.42, 0.00, 0.00, 2.16, 2.50, 2.82, 3.15,
    // F, Ne, Na, Mg, Al, Si, P, S
    3.48, 3.81, 0.00, 0.00, 0.00, 0.00, 4.50, 4.80,
    // Cl, Ar, K, Ca, Sc, Ti, V, Cr
    5.10, 5.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    // Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge
    0.00, 0.00, 2.90, 0.00, 0.00, 0.00, 0.00, 0.00,
    // As, Se, Br, Kr, Rb, Sr, Y, Zr
    0.00, 0.00, 6.00, 6.30, 0.00, 0.00, 0.00, 0.00,
    // Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    // In, Sn, Sb, Te, I, Xe, Cs, Ba
    0.00, 0.00, 0.00, 0.00, 6.95, 7.25, 0.00, 0.00,
    // La - Gd (Z=57..64)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    // Tb - Hf (Z=65..72)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    // Ta - Hg (Z=73..80)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    // Tl - Rn (Z=81..86)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
];

/// Evaluate diatomic dispersion parameters $C_{6,AB}$ (in J*nm^6/mol) and $R_{0,AB}$ (in nm).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn diatomic_dispersion_parameters(
    za: u8,
    zb: u8,
    c6_a: f64,
    c6_b: f64,
    r0_a: f64,
    r0_b: f64,
    n_a: f64,
    n_b: f64,
) -> Option<(f64, f64)> {
    if za == 0 || za > 86 || zb == 0 || zb > 86 {
        return None;
    }
    if c6_a <= 0.0 || c6_b <= 0.0 || r0_a <= 0.0 || r0_b <= 0.0 || n_a <= 0.0 || n_b <= 0.0 {
        return None;
    }

    // Slater-Kirkwood combining rule for C6:
    // C6_AB = 2 * (C6_A^2 * C6_B^2 * N_A * N_B)^(1/3) / [ (C6_A * N_B^2)^(1/3) + (C6_B * N_A^2)^(1/3) ]
    let num = 2.0 * (c6_a * c6_a * c6_b * c6_b * n_a * n_b).powf(1.0 / 3.0);
    let den = (c6_a * n_b * n_b).powf(1.0 / 3.0) + (c6_b * n_a * n_a).powf(1.0 / 3.0);
    let c6_ab = num / den;

    // R0 combining rule (pm converted to nm: * 2.0 * 1e-3):
    // R0_AB = 2.0 * (R_A^3 + R_B^3) / (R_A^2 + R_B^2) * 1e-3 nm
    let r0_ab = 2.0 * (r0_a.powi(3) + r0_b.powi(3)) / (r0_a.powi(2) + r0_b.powi(2)) * 1e-3;

    Some((c6_ab, r0_ab))
}

/// Compute empirical dispersion energy in kcal/mol matching OpenMOPAC `PM6_DH_Disp`.
#[allow(clippy::needless_range_loop)]
pub fn compute_dispersion_energy(batch: &MolecularBatch, model: DispersionModel) -> f64 {
    let natoms = batch.natoms;
    let alpha = model.alpha();
    let s = model.s();
    let cscale = model.cscale();

    let mut e_disp_tot = 0.0;

    for i in 0..natoms {
        let zi = batch.atomic_numbers[i];
        if zi == 0 || zi > 86 {
            continue;
        }
        let mut c6_i = DISPERSION_C6[(zi - 1) as usize];
        let r0_i = DISPERSION_R0[(zi - 1) as usize];
        let n_i = DISPERSION_NEFF[(zi - 1) as usize];

        if c6_i == 0.0 || r0_i == 0.0 || n_i == 0.0 {
            continue;
        }

        // Carbon special rule: if sp3 tetrahedral, C6 = 0.95, else 1.65
        if zi == 6 {
            // Check coordination number in batch
            let mut coord_num = 0;
            for k in 0..natoms {
                if k != i && batch.distance(i, k) < 1.85 {
                    coord_num += 1;
                }
            }
            if coord_num == 4 {
                c6_i = 0.95;
            } else {
                c6_i = 1.65;
            }
        }

        for j in (i + 1)..natoms {
            let zj = batch.atomic_numbers[j];
            if zj == 0 || zj > 86 {
                continue;
            }
            let mut c6_j = DISPERSION_C6[(zj - 1) as usize];
            let r0_j = DISPERSION_R0[(zj - 1) as usize];
            let n_j = DISPERSION_NEFF[(zj - 1) as usize];

            if c6_j == 0.0 || r0_j == 0.0 || n_j == 0.0 {
                continue;
            }

            if zj == 6 {
                let mut coord_num = 0;
                for k in 0..natoms {
                    if k != j && batch.distance(j, k) < 1.85 {
                        coord_num += 1;
                    }
                }
                if coord_num == 4 {
                    c6_j = 0.95;
                } else {
                    c6_j = 1.65;
                }
            }

            if let Some((c6_ab, r0_ab)) =
                diatomic_dispersion_parameters(zi, zj, c6_i, c6_j, r0_i, r0_j, n_i, n_j)
            {
                // Distance R_ij in Angstroms converted to nm (* 0.1)
                let rij_angstrom = batch.distance(i, j);
                let rij = rij_angstrom * 0.1; // in nm

                if rij > 1e-6 {
                    let damp = 1.0 / (1.0 + (-alpha * (rij / (s * r0_ab) - 1.0)).exp());
                    // 1 kcal = 4184.0 J
                    let e_pair = (c6_ab / rij.powi(6)) * damp / (1000.0 * 4.184);
                    e_disp_tot -= e_pair;
                }
            }
        }
    }

    e_disp_tot * cscale
}

/// Compute empirical dispersion energy and analytical Cartesian gradients in kcal/mol and kcal/(mol * A).
#[allow(clippy::needless_range_loop)]
pub fn compute_dispersion_energy_and_gradients(
    batch: &MolecularBatch,
    model: DispersionModel,
    gradients: &mut [[f64; 3]],
) -> f64 {
    let natoms = batch.natoms;
    assert_eq!(gradients.len(), natoms);

    let alpha = model.alpha();
    let s = model.s();
    let cscale = model.cscale();

    let mut e_disp_tot = 0.0;

    for i in 0..natoms {
        let zi = batch.atomic_numbers[i];
        if zi == 0 || zi > 86 {
            continue;
        }
        let mut c6_i = DISPERSION_C6[(zi - 1) as usize];
        let r0_i = DISPERSION_R0[(zi - 1) as usize];
        let n_i = DISPERSION_NEFF[(zi - 1) as usize];

        if c6_i == 0.0 || r0_i == 0.0 || n_i == 0.0 {
            continue;
        }

        if zi == 6 {
            let mut coord_num = 0;
            for k in 0..natoms {
                if k != i && batch.distance(i, k) < 1.85 {
                    coord_num += 1;
                }
            }
            if coord_num == 4 {
                c6_i = 0.95;
            } else {
                c6_i = 1.65;
            }
        }

        for j in (i + 1)..natoms {
            let zj = batch.atomic_numbers[j];
            if zj == 0 || zj > 86 {
                continue;
            }
            let mut c6_j = DISPERSION_C6[(zj - 1) as usize];
            let r0_j = DISPERSION_R0[(zj - 1) as usize];
            let n_j = DISPERSION_NEFF[(zj - 1) as usize];

            if c6_j == 0.0 || r0_j == 0.0 || n_j == 0.0 {
                continue;
            }

            if zj == 6 {
                let mut coord_num = 0;
                for k in 0..natoms {
                    if k != j && batch.distance(j, k) < 1.85 {
                        coord_num += 1;
                    }
                }
                if coord_num == 4 {
                    c6_j = 0.95;
                } else {
                    c6_j = 1.65;
                }
            }

            if let Some((c6_ab, r0_ab)) =
                diatomic_dispersion_parameters(zi, zj, c6_i, c6_j, r0_i, r0_j, n_i, n_j)
            {
                let dx = batch.x[i] - batch.x[j];
                let dy = batch.y[i] - batch.y[j];
                let dz = batch.z[i] - batch.z[j];
                let rij_angstrom = (dx * dx + dy * dy + dz * dz).sqrt();

                if rij_angstrom > 1e-6 {
                    let rij = rij_angstrom * 0.1; // in nm
                    let exp_term = (-alpha * (rij / (s * r0_ab) - 1.0)).exp();
                    let damp = 1.0 / (1.0 + exp_term);

                    // E_disp_pair in kcal/mol
                    let inv_r6 = 1.0 / rij.powi(6);
                    let e_pair = (c6_ab * inv_r6) * damp / (1000.0 * 4.184);
                    e_disp_tot -= e_pair;

                    // Analytical derivative d(E_pair)/d(rij_angstrom):
                    // E_pair(rij) = - cscale * (C6 / rij^6) * damp / (4184.0)
                    // d(E_pair)/d(rij_nm) = cscale * (C6 / (4184.0 * rij^6)) * [ 6/rij * damp - alpha/(s*r0) * damp * (1 - damp) ]
                    // Note rij = 0.1 * rij_angstrom, so d/d(rij_angstrom) = 0.1 * d/d(rij_nm)
                    let d_damp_d_rij = (alpha / (s * r0_ab)) * damp * (1.0 - damp);
                    let de_d_rij_nm = (cscale * c6_ab / (4184.0 * rij.powi(6)))
                        * (6.0 / rij * damp - d_damp_d_rij);
                    let de_d_rij_angstrom = de_d_rij_nm * 0.1;

                    // Cartesian force projection on atom i: F_x = dE/dR * (dx / R)
                    let gx = de_d_rij_angstrom * (dx / rij_angstrom);
                    let gy = de_d_rij_angstrom * (dy / rij_angstrom);
                    let gz = de_d_rij_angstrom * (dz / rij_angstrom);

                    gradients[i][0] += gx;
                    gradients[i][1] += gy;
                    gradients[i][2] += gz;

                    // Newton's 3rd law: -F on atom j
                    gradients[j][0] -= gx;
                    gradients[j][1] -= gy;
                    gradients[j][2] -= gz;
                }
            }
        }
    }

    e_disp_tot * cscale
}
