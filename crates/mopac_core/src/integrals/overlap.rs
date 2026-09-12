//! Analytical Slater-Type Orbital (STO) Overlap Integrals.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Evaluates diatomic overlap integrals $S_{\mu\nu}(R) = \int \phi_\mu^*(\vec{r}) \phi_\nu(\vec{r}) d^3r$
//! between $s, p_x, p_y, p_z$ Slater orbitals in arbitrary 3D orientations with zero heap allocations.
//!
//! Reference:
//! - MOPAC `diat.F90`, `diat2`, `set.F90`.
//! - Mulliken, R. S., Rieke, C. A., Orloff, D., & Orloff, H. (1949). "Formulas and Numerical Tables
//!   for Overlap Integrals", *J. Chem. Phys.* 17(12), 1248-1267.

use crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS;
use crate::parameters::SemiEmpiricalElementParams;

/// Fill table of auxiliary integrals $A_k(\alpha) = \int_1^\infty x^k e^{-\alpha x} dx$ for $k = 0..k_{\max}$.
#[inline(always)]
pub fn fill_aux_a(alpha: f64, k_max: usize, a: &mut [f64]) {
    debug_assert!(a.len() > k_max);
    if alpha < 1e-12 {
        for (i, val) in a.iter_mut().enumerate().take(k_max + 1) {
            *val = 1.0 / (i as f64 + 1.0);
        }
        return;
    }
    let c = (-alpha).exp();
    a[0] = c / alpha;
    for i in 1..=k_max {
        a[i] = (a[i - 1] * (i as f64) + c) / alpha;
    }
}

/// Fill table of auxiliary integrals $B_k(\beta) = \int_{-1}^1 x^k e^{-\beta x} dx$ for $k = 0..k_{\max}$.
#[inline(always)]
pub fn fill_aux_b(beta: f64, k_max: usize, b: &mut [f64]) {
    debug_assert!(b.len() > k_max);
    let abs_beta = beta.abs();
    if abs_beta <= 1e-6 {
        for (i, val) in b.iter_mut().enumerate().take(k_max + 1) {
            *val = if (i + 1) % 2 == 0 {
                0.0
            } else {
                2.0 / (i as f64 + 1.0)
            };
        }
        return;
    }

    let expx = beta.exp();
    let expmx = 1.0 / expx;
    b[0] = (expx - expmx) / beta;
    for i in 1..=k_max {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        b[i] = ((i as f64) * b[i - 1] + sign * expx - expmx) / beta;
    }
}

/// Helper to configure spheroidal coordinate parameters $\alpha, \beta$ and fill $A, B$ tables.
#[inline(always)]
fn set_ab_tables(s1: f64, s2: f64, rab: f64, a_tab: &mut [f64; 9], b_tab: &mut [f64; 9]) {
    let alpha = 0.5 * rab * (s1 + s2);
    let beta = 0.5 * rab * (s2 - s1);
    fill_aux_a(alpha, 8, a_tab);
    fill_aux_b(beta, 8, b_tab);
}

/// Diatomic overlap between two $1s$ Slater-type orbitals with exponents $\zeta_1, \zeta_2$ at separation $R$ in Ångströms.
pub fn overlap_1s_1s(r_angstrom: f64, zeta1: f64, zeta2: f64) -> f64 {
    let a0 = BOHR_RADIUS_ANGSTROMS;
    let r_bohr = r_angstrom / a0;
    if r_bohr < 1e-12 {
        return if (zeta1 - zeta2).abs() < 1e-12 {
            1.0
        } else {
            let p1 = zeta1.powf(1.5);
            let p2 = zeta2.powf(1.5);
            8.0 * (p1 * p2) / (zeta1 + zeta2).powi(3)
        };
    }

    let mut a = [0.0f64; 9];
    let mut b = [0.0f64; 9];
    set_ab_tables(zeta1, zeta2, r_bohr, &mut a, &mut b);

    let w = 0.25 * (zeta1 * zeta2 * r_bohr * r_bohr).powf(1.5);
    w * (a[2] * b[0] - b[2] * a[0])
}

/// Compute complete 3D Cartesian diatomic overlap matrix between atom A and atom B.
///
/// Output: `s_mat[oa][ob]` where:
/// - Index 0: $s$ orbital
/// - Index 1: $p_x$ orbital
/// - Index 2: $p_y$ orbital
/// - Index 3: $p_z$ orbital
///
/// Rotation into the 3D laboratory Cartesian frame is performed via direction cosines:
/// $\mathbf{l} = \frac{\vec{R}_B - \vec{R}_A}{\|\vec{R}_B - \vec{R}_A\|}$.
///
/// Mathematically exact and guaranteed **0 heap allocations**.
pub fn compute_diatomic_overlap_block(
    za: u8,
    zb: u8,
    p_a: &SemiEmpiricalElementParams,
    p_b: &SemiEmpiricalElementParams,
    r_angstrom: f64,
    dir_cosines: [f64; 3], // (l_x, l_y, l_z) pointing from A to B
    s_mat: &mut [[f64; 4]; 4],
) {
    for row in s_mat.iter_mut() {
        row.fill(0.0);
    }

    if !(1e-10..=20.0).contains(&r_angstrom) {
        return;
    }

    let a0 = BOHR_RADIUS_ANGSTROMS;
    let rab = r_angstrom / a0;
    let lx = dir_cosines[0];
    let ly = dir_cosines[1];
    let lz = dir_cosines[2];

    let mut a = [0.0f64; 9];
    let mut b = [0.0f64; 9];

    if za == 1 && zb == 1 {
        // Case 1: H - H (1s - 1s)
        s_mat[0][0] = overlap_1s_1s(r_angstrom, p_a.zs, p_b.zs);
    } else if za == 1 && zb > 1 {
        // Case 2a: H (Atom A, 1s) with 2nd row (Atom B: C, N, O, 2s, 2p)
        set_ab_tables(p_a.zs, p_b.zs, rab, &mut a, &mut b);
        let rab4 = rab.powi(4) * 0.125;
        let w = (p_a.zs.powi(3) * p_b.zs.powi(5)).sqrt() * rab4;
        let rt3 = 1.0 / 3.0f64.sqrt();
        let s_ss = w * rt3 * (a[3] * b[0] - b[3] * a[0] + a[2] * b[1] - b[2] * a[1]);
        s_mat[0][0] = s_ss;

        // 1s_A with 2p_B (with aa = -1.0 phase for atom B p-orbital)
        set_ab_tables(p_a.zs, p_b.zp, rab, &mut a, &mut b);
        let wp = (p_a.zs.powi(3) * p_b.zp.powi(5)).sqrt() * rab4;
        let s_sp = wp * (a[2] * b[0] - b[2] * a[0] + a[3] * b[1] - b[3] * a[1]);

        s_mat[0][1] = lx * s_sp;
        s_mat[0][2] = ly * s_sp;
        s_mat[0][3] = lz * s_sp;
    } else if za > 1 && zb == 1 {
        // Case 2b: 2nd row (Atom A: C, N, O, 2s, 2p) with H (Atom B, 1s)
        set_ab_tables(p_b.zs, p_a.zs, rab, &mut a, &mut b);
        let rab4 = rab.powi(4) * 0.125;
        let w = (p_b.zs.powi(3) * p_a.zs.powi(5)).sqrt() * rab4;
        let rt3 = 1.0 / 3.0f64.sqrt();
        let s_ss = w * rt3 * (a[3] * b[0] - b[3] * a[0] + a[2] * b[1] - b[2] * a[1]);
        s_mat[0][0] = s_ss;

        // 2p_A with 1s_B (atom B is s, so aa = +1.0)
        set_ab_tables(p_b.zs, p_a.zp, rab, &mut a, &mut b);
        let wp = (p_b.zs.powi(3) * p_a.zp.powi(5)).sqrt() * rab4;
        let s_ps = wp * (a[2] * b[0] - b[2] * a[0] + a[3] * b[1] - b[3] * a[1]);

        s_mat[1][0] = lx * s_ps;
        s_mat[2][0] = ly * s_ps;
        s_mat[3][0] = lz * s_ps;
    } else {
        // Case 4: 2nd row with 2nd row (C, N, O with C, N, O)
        let rab5 = rab.powi(5) * 0.0625;

        // 1. s_A - s_B
        set_ab_tables(p_a.zs, p_b.zs, rab, &mut a, &mut b);
        let w_ss = (p_a.zs * p_b.zs).powi(5).sqrt() * rab5;
        let s_ss = w_ss * (a[4] * b[0] + b[4] * a[0] - 2.0 * a[2] * b[2]) / 3.0;
        s_mat[0][0] = s_ss;

        // 2. p - p overlaps: sigma and pi (with aa = -1.0 for atom B p-orbital)
        set_ab_tables(p_a.zp, p_b.zp, rab, &mut a, &mut b);
        let w_pp = (p_a.zp * p_b.zp).powi(5).sqrt() * rab5;
        let s_sigma = w_pp * (b[2] * (a[4] + a[0]) - a[2] * (b[4] + b[0]));
        let s_pi =
            0.5 * w_pp * (a[4] * (b[0] - b[2]) - b[4] * (a[0] - a[2]) - a[2] * b[0] + b[2] * a[0]);

        let l = [lx, ly, lz];
        for i in 0..3 {
            for j in 0..3 {
                let delta = if i == j { 1.0 } else { 0.0 };
                s_mat[1 + i][1 + j] = l[i] * l[j] * s_sigma + (delta - l[i] * l[j]) * s_pi;
            }
        }

        // 3. s_A - p_sigma_B (directed along -R_AB)
        set_ab_tables(p_a.zs, p_b.zp, rab, &mut a, &mut b);
        let w_sp = (p_a.zs * p_b.zp).powi(5).sqrt() * rab5;
        let rt3 = 1.0 / 3.0f64.sqrt();
        let d = a[3] * (b[0] - b[2]) - a[1] * (b[2] - b[4]);
        let e = b[3] * (a[0] - a[2]) - b[1] * (a[2] - a[4]);
        let s_s_psigma = -w_sp * rt3 * (d + e);

        s_mat[0][1] = lx * s_s_psigma;
        s_mat[0][2] = ly * s_s_psigma;
        s_mat[0][3] = lz * s_s_psigma;

        // 4. p_sigma_A - s_B (atom B is s, so aa = +1.0)
        set_ab_tables(p_a.zp, p_b.zs, rab, &mut a, &mut b);
        let w_ps = (p_a.zp * p_b.zs).powi(5).sqrt() * rab5;
        let d2 = a[3] * (b[0] - b[2]) - a[1] * (b[2] - b[4]);
        let e2 = b[3] * (a[0] - a[2]) - b[1] * (a[2] - a[4]);
        let s_psigma_s = w_ps * rt3 * (d2 - e2);

        s_mat[1][0] = lx * s_psigma_s;
        s_mat[2][0] = ly * s_psigma_s;
        s_mat[3][0] = lz * s_psigma_s;
    }
}
