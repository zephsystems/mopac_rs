//! Rezac-Hobza H4 Hydrogen Bond and H-H Short-Range Repulsion Correction.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct translation and rigorous verification of OpenMOPAC v23.2.5 `H_bonds4.F90`.
//!
//! Primary Reference:
//! - Rezac, J.; Hobza, P. "Advanced Corrections of Hydrogen Bonding and Dispersion for
//!   Semiempirical Quantum Mechanical Methods", J. Chem. Theory Comput. 2012, 8, 1, 141-151.
//!   <https://doi.org/10.1021/ct200751e>

#![allow(clippy::excessive_precision, clippy::needless_range_loop)]

use crate::types::MolecularBatch;
use std::f64::consts::PI;


/// Authentic covalent radii table for elements Z=1..118 from OpenMOPAC `radii_C` in `H_bonds4.F90`.
pub const COVALENT_RADII: [f64; 118] = [
    0.37, 0.32, 1.34, 0.90, 0.82, 0.77, 0.75, 0.73, 0.71, 0.69, // 1-10
    1.54, 1.30, 1.18, 1.11, 1.06, 1.02, 0.99, 0.97, 1.96, 1.74, // 11-20
    1.44, 1.36, 1.25, 1.27, 1.39, 1.25, 1.26, 1.21, 1.38, 1.31, // 21-30
    1.26, 1.22, 1.19, 1.16, 1.14, 1.10, 2.11, 1.92, 1.62, 1.48, // 31-40
    1.37, 1.45, 1.56, 1.26, 1.35, 1.31, 1.53, 1.48, 1.44, 1.41, // 41-50
    1.38, 1.35, 1.33, 1.30, 2.25, 1.98, 1.69, 0.00, 0.00, 0.00, // 51-60
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, // 61-70
    1.60, 1.50, 1.38, 1.46, 1.59, 1.28, 1.37, 1.28, 1.44, 1.49, // 71-80
    0.00, 0.00, 1.46, 0.00, 0.00, 1.45, 0.00, 0.00, 0.00, 0.00, // 81-90
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, // 91-100
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, // 101-110
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,             // 111-118
];

/// Get covalent radius for element Z in Angstroms.
#[inline]
pub fn covalent_radius(z: u8) -> f64 {
    if z >= 1 && (z as usize) <= COVALENT_RADII.len() {
        COVALENT_RADII[z as usize - 1]
    } else {
        0.0
    }
}

/// Smooth covalent valence contribution $V_{AB}(R)$ between two atoms matching OpenMOPAC `cvalence_contribution`.
///
/// Uses a 7th-order polynomial switching function $S(x)$ between $r_0 = r_A + r_B$ and $r_1 = 1.6 r_0$:
/// $$x = \frac{r - r_0}{r_1 - r_0}$$
/// $$S(x) = -20 x^7 + 70 x^6 - 84 x^5 + 35 x^4$$
/// $$V_{AB}(r) = 1.0 - S(x)$$
#[inline]
pub fn cvalence_contribution(r: f64, z_a: u8, z_b: u8) -> f64 {
    let r0 = covalent_radius(z_a) + covalent_radius(z_b);
    let r1 = r0 * 1.6;
    if r <= 0.0 || r >= r1 {
        0.0
    } else if r <= r0 {
        1.0
    } else {
        let x = (r - r0) / (r1 - r0);
        let x2 = x * x;
        let x4 = x2 * x2;
        let s = x4 * (-20.0 * x2 * x + 70.0 * x2 - 84.0 * x + 35.0);
        1.0 - s
    }
}

/// H4 Hydrogen Bond Model Parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H4Parameters {
    pub para_oh_o: f64,
    pub para_oh_n: f64,
    pub para_nh_o: f64,
    pub para_nh_n: f64,
    pub multiplier_wh_o: f64,
    pub multiplier_nh4: f64,
    pub multiplier_coo: f64,
}

impl Default for H4Parameters {
    fn default() -> Self {
        Self {
            para_oh_o: 2.32,
            para_oh_n: 3.10,
            para_nh_o: 1.07,
            para_nh_n: 2.01,
            multiplier_wh_o: 0.42,
            multiplier_nh4: 3.61,
            multiplier_coo: 1.41,
        }
    }
}

/// Radial polynomial function $E_{\text{rad}}(R_{DA})$ for donor-acceptor distance $R_{DA} \le 5.5\text{ \AA}$.
#[inline]
pub fn h4_radial(rda: f64) -> f64 {
    if rda > 5.5 {
        return 0.0;
    }
    -0.00303407407407313510 * rda.powi(7)
        + 0.07357629629627092382 * rda.powi(6)
        - 0.70087111111082800452 * rda.powi(5)
        + 3.25309629629461749545 * rda.powi(4)
        - 7.20687407406838786983 * rda.powi(3)
        + 5.31754666665572184314 * rda.powi(2)
        + 3.40736000001102778967 * rda
        - 4.68512000000450434811
}

/// Angular polynomial factor $E_{\text{ang}}(\theta)$ where $\theta = \pi - \angle(D, H, A)$.
#[inline]
pub fn h4_angular(angle_rad: f64) -> f64 {
    if angle_rad >= PI / 2.0 || angle_rad <= 0.0 {
        return 0.0;
    }
    let a = angle_rad / (PI / 2.0);
    let a2 = a * a;
    let a4 = a2 * a2;
    let x = a4 * (-20.0 * a2 * a + 70.0 * a2 - 84.0 * a + 35.0);
    1.0 - x * x
}

/// Compute total H4 hydrogen bond correction energy in kcal/mol.
pub fn compute_h4_energy(batch: &MolecularBatch, params: &H4Parameters) -> f64 {
    let natoms = batch.natoms;
    let z = &batch.atomic_numbers;

    let mut e_h4_sum = 0.0;

    // Identify candidate donors and acceptors: Oxygen (8) and Nitrogen (7)
    for i in 0..natoms {
        let zi = z[i];
        if zi != 7 && zi != 8 {
            continue;
        }

        for j in (i + 1)..natoms {
            let zj = z[j];
            if zj != 7 && zj != 8 {
                continue;
            }

            let rda = batch.distance(i, j);
            if rda > 5.5 {
                continue;
            }

            let e_radial = h4_radial(rda);

            // Iterate over all hydrogen atoms
            for h in 0..natoms {
                if z[h] != 1 {
                    continue;
                }

                let rih = batch.distance(i, h);
                let rjh = batch.distance(j, h);

                if rih < 1e-12 || rjh < 1e-12 {
                    continue;
                }

                let dx_ih = batch.x[i] - batch.x[h];
                let dy_ih = batch.y[i] - batch.y[h];
                let dz_ih = batch.z[i] - batch.z[h];

                let dx_jh = batch.x[j] - batch.x[h];
                let dy_jh = batch.y[j] - batch.y[h];
                let dz_jh = batch.z[j] - batch.z[h];

                // Angle at H between i and j: cos(ang) = (v_hi . v_hj) / (rih * rjh)
                let dot = dx_ih * dx_jh + dy_ih * dy_jh + dz_ih * dz_jh;
                let cos_ang = (dot / (rih * rjh)).clamp(-1.0, 1.0);
                let ang = cos_ang.acos();
                let angle = PI - ang;

                if angle >= PI / 2.0 {
                    continue;
                }

                // Determine donor (closer to H) and acceptor (farther from H)
                let (d_i, a_i, rdh, rah) = if rih < rjh {
                    (i, j, rih, rjh)
                } else {
                    (j, i, rjh, rih)
                };

                let zd = z[d_i];
                let za = z[a_i];

                let e_para = match (zd, za) {
                    (8, 8) => params.para_oh_o,
                    (8, 7) => params.para_oh_n,
                    (7, 8) => params.para_nh_o,
                    (7, 7) => params.para_nh_n,
                    _ => 0.0,
                };

                let e_angular = h4_angular(angle);

                // Bond switching
                let e_bond_switch = if rdh > 1.15 {
                    let rdhs = rdh - 1.15;
                    let ravgs = 0.5 * rdh + 0.5 * rah - 1.15;
                    if ravgs > 1e-12 {
                        let x = rdhs / ravgs;
                        let x2 = x * x;
                        let x4 = x2 * x2;
                        let s = x4 * (-20.0 * x2 * x + 70.0 * x2 - 84.0 * x + 35.0);
                        1.0 - s
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                // Water scaling
                let mut e_scale_w = 1.0;
                if zd == 8 && za == 8 {
                    let mut hydrogens = 0.0;
                    let mut others = 0.0;
                    for k in 0..natoms {
                        if k == d_i {
                            continue;
                        }
                        let r_dk = batch.distance(d_i, k);
                        let v = cvalence_contribution(r_dk, zd, z[k]);
                        if z[k] == 1 {
                            hydrogens += v;
                        } else {
                            others += v;
                        }
                    }

                    if hydrogens >= 1.0 {
                        let slope = params.multiplier_wh_o - 1.0;
                        let fv = if hydrogens > 1.0 && hydrogens <= 2.0 {
                            hydrogens - 1.0
                        } else if hydrogens > 2.0 && hydrogens < 3.0 {
                            3.0 - hydrogens
                        } else {
                            0.0
                        };
                        let fv2 = (1.0 - others).max(0.0);
                        e_scale_w = 1.0 + slope * fv * fv2;
                    }
                }

                // Charged groups (NR4+ and COO-)
                let mut e_scale_chd = 1.0;
                if zd == 7 {
                    let slope = params.multiplier_nh4 - 1.0;
                    let mut v_sum = 0.0;
                    for k in 0..natoms {
                        if k == d_i {
                            continue;
                        }
                        let r = batch.distance(d_i, k);
                        v_sum += cvalence_contribution(r, zd, z[k]);
                    }
                    let v = if v_sum > 3.0 { v_sum - 3.0 } else { 0.0 };
                    e_scale_chd = 1.0 + slope * v;
                }

                let mut e_scale_cha = 1.0;
                if za == 8 {
                    let slope = params.multiplier_coo - 1.0;
                    let o1 = a_i;
                    let mut cdist = 1e10;
                    let mut cv_o1 = 0.0;
                    let mut cc: Option<usize> = None;

                    for k in 0..natoms {
                        if k == o1 {
                            continue;
                        }
                        let r = batch.distance(o1, k);
                        let v = cvalence_contribution(r, 8, z[k]);
                        cv_o1 += v;
                        if v > 0.0 && z[k] == 6 && r < cdist {
                            cdist = r;
                            cc = Some(k);
                        }
                    }

                    if let Some(c_idx) = cc {
                        let mut odist = 1e10;
                        let mut cv_cc = 0.0;
                        let mut o2: Option<usize> = None;

                        for k in 0..natoms {
                            if k == c_idx {
                                continue;
                            }
                            let r = batch.distance(c_idx, k);
                            let v = cvalence_contribution(r, 6, z[k]);
                            cv_cc += v;
                            if v > 0.0 && k != o1 && z[k] == 8 && r < odist {
                                odist = r;
                                o2 = Some(k);
                            }
                        }

                        if let Some(o2_idx) = o2 {
                            let mut cv_o2 = 0.0;
                            for k in 0..natoms {
                                if k == o2_idx {
                                    continue;
                                }
                                let r = batch.distance(o2_idx, k);
                                cv_o2 += cvalence_contribution(r, 8, z[k]);
                            }

                            let f_o1 = (1.0 - (1.0 - cv_o1).abs()).max(0.0);
                            let f_o2 = (1.0 - (1.0 - cv_o2).abs()).max(0.0);
                            let f_cc = (1.0 - (3.0 - cv_cc).abs()).max(0.0);
                            e_scale_cha = 1.0 + slope * f_o1 * f_o2 * f_cc;
                        }
                    }
                }

                let e_corr = e_para
                    * e_radial
                    * e_angular
                    * e_bond_switch
                    * e_scale_w
                    * e_scale_chd
                    * e_scale_cha;

                e_h4_sum += e_corr;
            }
        }
    }

    e_h4_sum
}

/// Hydrogen-Hydrogen short-range repulsion potential and exact analytical derivative.
///
/// Matches `poly(r, l_grad, dpoly)` from OpenMOPAC v23.2.5 `H_bonds4.F90`.
/// Returns `(energy_kcal_mol, dE_dr)`.
#[inline]
pub fn hh_repulsion_potential(r: f64) -> (f64, f64) {
    if r <= 1.0 {
        (25.46293603147693, 0.0)
    } else if r < 1.5 {
        let r2 = r * r;
        let r3 = r2 * r;
        let r4 = r2 * r2;
        let r5 = r4 * r;

        let energy = -2714.952351603469651 * r5
            + 17103.650110591705015 * r4
            - 42511.857982217959943 * r3
            + 52063.196799138342612 * r2
            - 31430.658335972289933 * r
            + 7516.084696095140316;

        let d_energy = -2714.952351603469651 * 5.0 * r4
            + 17103.650110591705015 * 4.0 * r3
            - 42511.857982217959943 * 3.0 * r2
            + 52063.196799138342612 * 2.0 * r
            - 31430.658335972289933;

        (energy, d_energy)
    } else {
        let r_pow = r.powf(1.72905);
        let exp_factor = (-1.53965 * r_pow).exp();
        let energy = 118.7326 * exp_factor;
        let d_energy = -1.53965 * 1.72905 * r.powf(0.72905) * energy;
        (energy, d_energy)
    }
}

/// Compute total Hydrogen-Hydrogen short-range repulsion energy and analytical Cartesian gradients.
///
/// Gradients are returned in kcal/(mol * Angstrom).
pub fn compute_hh_repulsion_energy_and_gradients(
    batch: &MolecularBatch,
) -> (f64, Vec<[f64; 3]>) {
    let natoms = batch.natoms;
    let z = &batch.atomic_numbers;

    let mut total_e = 0.0;
    let mut grads = vec![[0.0; 3]; natoms];

    for i in 0..natoms {
        if z[i] != 1 {
            continue;
        }

        for j in 0..i {
            if z[j] != 1 {
                continue;
            }

            let dx = batch.x[i] - batch.x[j];
            let dy = batch.y[i] - batch.y[j];
            let dz = batch.z[i] - batch.z[j];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();

            if r < 1e-12 {
                continue;
            }

            let (e_pair, d_rad) = hh_repulsion_potential(r);
            total_e += e_pair;

            let inv_r = 1.0 / r;
            let gx = (dx * inv_r) * d_rad;
            let gy = (dy * inv_r) * d_rad;
            let gz = (dz * inv_r) * d_rad;

            // Forces satisfy Newton's 3rd Law: F_i = -F_j
            grads[i][0] += gx;
            grads[i][1] += gy;
            grads[i][2] += gz;

            grads[j][0] -= gx;
            grads[j][1] -= gy;
            grads[j][2] -= gz;
        }
    }

    (total_e, grads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hh_potential_continuity() {
        // Test value at boundary r = 1.0
        let (e1_left, _) = hh_repulsion_potential(1.0);
        let (e1_right, _) = hh_repulsion_potential(1.00000001);
        assert!(
            (e1_left - e1_right).abs() < 1e-4,
            "Discontinuity at r=1.0: left={}, right={}",
            e1_left,
            e1_right
        );

        // Test value at boundary r = 1.5
        let (e15_left, _) = hh_repulsion_potential(1.49999999);
        let (e15_right, _) = hh_repulsion_potential(1.50000001);
        assert!(
            (e15_left - e15_right).abs() < 1e-4,
            "Discontinuity at r=1.5: left={}, right={}",
            e15_left,
            e15_right
        );
    }

    #[test]
    fn test_hh_analytical_gradients_vs_finite_difference() {
        let coords = vec![
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
            [3.500, 0.586, 0.0],
            [2.200, 0.400, 0.0],
        ];
        let z = vec![1, 1, 1, 1];
        let batch = MolecularBatch::new(z.clone(), &coords);

        let (_energy, analytical_grads) = compute_hh_repulsion_energy_and_gradients(&batch);

        // Central finite difference check with optimal step size
        let h = 1e-5;
        for atom in 0..4 {
            for comp in 0..3 {
                let mut coords_plus = coords.clone();
                let mut coords_minus = coords.clone();
                coords_plus[atom][comp] += h;
                coords_minus[atom][comp] -= h;

                let b_plus = MolecularBatch::new(z.clone(), &coords_plus);
                let b_minus = MolecularBatch::new(z.clone(), &coords_minus);

                let (e_plus, _) = compute_hh_repulsion_energy_and_gradients(&b_plus);
                let (e_minus, _) = compute_hh_repulsion_energy_and_gradients(&b_minus);

                let num_grad = (e_plus - e_minus) / (2.0 * h);
                let ana_grad = analytical_grads[atom][comp];

                assert!(
                    (num_grad - ana_grad).abs() < 5e-5,
                    "Atom {} comp {}: num={}, ana={}, diff={}",
                    atom,
                    comp,
                    num_grad,
                    ana_grad,
                    (num_grad - ana_grad).abs()
                );
            }
        }

    }

    #[test]
    fn test_h4_water_dimer_energy_parity() {
        // Exact water dimer geometry
        let coords = vec![
            [0.000, 0.000, 0.000],  // O
            [0.757, 0.586, 0.000],  // H
            [-0.757, 0.586, 0.000], // H
            [2.900, 0.000, 0.000],  // O
            [3.500, 0.586, 0.000],  // H
            [2.200, 0.400, 0.000],  // H
        ];
        let z = vec![8, 1, 1, 8, 1, 1];
        let batch = MolecularBatch::new(z, &coords);

        let params = H4Parameters::default();
        let e_h4 = compute_h4_energy(&batch, &params);
        let (e_hh, _) = compute_hh_repulsion_energy_and_gradients(&batch);

        // Validated against authentic OpenMOPAC v23.2.5 H_bonds4.F90
        assert!(
            (e_h4 - (-1.333486)).abs() < 1e-4,
            "H4 energy mismatch: got {}, expected -1.333486",
            e_h4
        );
        assert!(
            (e_hh - 24.343855).abs() < 1e-4,
            "HH repulsion mismatch: got {}, expected 24.343855",
            e_hh
        );
    }
}
