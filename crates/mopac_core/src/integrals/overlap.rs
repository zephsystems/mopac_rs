//! Analytical Slater-Type Orbital (STO) Overlap Integrals.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Evaluates diatomic overlap integrals $S_{\mu\nu}(R) = \int \phi_\mu^*(\vec{r}) \phi_\nu(\vec{r}) d^3r$
//! between $s, p_x, p_y, p_z, d$ Slater orbitals in arbitrary 3D orientations with zero heap allocations.
//!
//! References:
//! - OpenMOPAC `diat.F90`, `diat2`, `ss`, `coe.F90`, `bfn.F90`, `parameters_C.F90`.
//! - Mulliken, R. S., Rieke, C. A., Orloff, D., & Orloff, H. (1949). "Formulas and Numerical Tables
//!   for Overlap Integrals", *J. Chem. Phys.* 17(12), 1248-1267.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of
)]

use crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS;
use crate::parameters::SemiEmpiricalElementParams;

/// Principal Quantum Number Table for elements 0..=107 (s, p, d shells).
/// Authentic parameters from OpenMOPAC `parameters_C.F90`.
pub const NPQ: [[u8; 3]; 108] = [
    [0, 0, 0], // Z = 0
    [1, 1, 0], // Z = 1
    [1, 2, 0], // Z = 2
    [2, 2, 0], // Z = 3
    [2, 2, 0], // Z = 4
    [2, 2, 0], // Z = 5
    [2, 2, 0], // Z = 6
    [2, 2, 0], // Z = 7
    [2, 2, 0], // Z = 8
    [2, 2, 0], // Z = 9
    [3, 2, 0], // Z = 10
    [3, 3, 3], // Z = 11
    [3, 3, 3], // Z = 12
    [3, 3, 3], // Z = 13
    [3, 3, 3], // Z = 14
    [3, 3, 3], // Z = 15
    [3, 3, 3], // Z = 16
    [3, 3, 3], // Z = 17
    [4, 3, 4], // Z = 18
    [4, 4, 3], // Z = 19
    [4, 4, 3], // Z = 20
    [4, 4, 3], // Z = 21
    [4, 4, 3], // Z = 22
    [4, 4, 3], // Z = 23
    [4, 4, 3], // Z = 24
    [4, 4, 3], // Z = 25
    [4, 4, 3], // Z = 26
    [4, 4, 3], // Z = 27
    [4, 4, 3], // Z = 28
    [4, 4, 3], // Z = 29
    [4, 4, 4], // Z = 30
    [4, 4, 4], // Z = 31
    [4, 4, 4], // Z = 32
    [4, 4, 4], // Z = 33
    [4, 4, 4], // Z = 34
    [4, 4, 4], // Z = 35
    [5, 4, 5], // Z = 36
    [5, 5, 4], // Z = 37
    [5, 5, 4], // Z = 38
    [5, 5, 4], // Z = 39
    [5, 5, 4], // Z = 40
    [5, 5, 4], // Z = 41
    [5, 5, 4], // Z = 42
    [5, 5, 4], // Z = 43
    [5, 5, 4], // Z = 44
    [5, 5, 4], // Z = 45
    [5, 5, 4], // Z = 46
    [5, 5, 4], // Z = 47
    [5, 5, 5], // Z = 48
    [5, 5, 5], // Z = 49
    [5, 5, 5], // Z = 50
    [5, 5, 5], // Z = 51
    [5, 5, 5], // Z = 52
    [5, 5, 5], // Z = 53
    [6, 5, 6], // Z = 54
    [6, 6, 5], // Z = 55
    [6, 6, 5], // Z = 56
    [6, 6, 5], // Z = 57
    [6, 6, 5], // Z = 58
    [6, 6, 5], // Z = 59
    [6, 6, 5], // Z = 60
    [6, 6, 5], // Z = 61
    [6, 6, 5], // Z = 62
    [6, 6, 5], // Z = 63
    [6, 6, 5], // Z = 64
    [6, 6, 5], // Z = 65
    [6, 6, 5], // Z = 66
    [6, 6, 5], // Z = 67
    [6, 6, 5], // Z = 68
    [6, 6, 5], // Z = 69
    [6, 6, 5], // Z = 70
    [6, 6, 5], // Z = 71
    [6, 6, 5], // Z = 72
    [6, 6, 5], // Z = 73
    [6, 6, 5], // Z = 74
    [6, 6, 5], // Z = 75
    [6, 6, 5], // Z = 76
    [6, 6, 5], // Z = 77
    [6, 6, 5], // Z = 78
    [6, 6, 5], // Z = 79
    [6, 6, 6], // Z = 80
    [6, 6, 6], // Z = 81
    [6, 6, 6], // Z = 82
    [6, 6, 6], // Z = 83
    [6, 6, 6], // Z = 84
    [6, 6, 6], // Z = 85
    [7, 6, 7], // Z = 86
    [0, 0, 0], // Z = 87
    [0, 0, 0], // Z = 88
    [0, 0, 0], // Z = 89
    [0, 0, 0], // Z = 90
    [0, 0, 0], // Z = 91
    [0, 0, 0], // Z = 92
    [0, 0, 0], // Z = 93
    [0, 0, 0], // Z = 94
    [0, 0, 0], // Z = 95
    [0, 0, 0], // Z = 96
    [0, 0, 0], // Z = 97
    [1, 0, 0], // Z = 98
    [0, 0, 0], // Z = 99
    [0, 0, 0], // Z = 100
    [0, 0, 0], // Z = 101
    [3, 0, 0], // Z = 102
    [0, 0, 0], // Z = 103
    [0, 0, 0], // Z = 104
    [0, 0, 0], // Z = 105
    [0, 0, 0], // Z = 106
    [0, 0, 0], // Z = 107
];

/// Exact factorials for overlap normalization ($n!$ for $n \in 0..=20$).
const FACT: [f64; 21] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    8.71782912e10,
    1.307674368e12,
    2.092278989e13,
    3.556874281e14,
    6.402373706e15,
    1.216451004e17,
    2.432902008e18,
];

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

/// Forms the B auxiliary integrals for general Slater overlap calculation.
/// Port of OpenMOPAC `bfn.F90`.
pub fn bfn(x: f64, bf: &mut [f64; 14]) {
    let absx = x.abs();
    let k = 12;
    if absx <= 3.0 {
        let last = if absx > 2.0 {
            15
        } else if absx > 1.0 {
            12
        } else if absx > 0.5 {
            7
        } else if absx <= 1e-6 {
            for (i, val) in bf.iter_mut().enumerate().take(k + 1) {
                *val = (2.0 * (((i + 1) % 2) as f64)) / (i as f64 + 1.0);
            }
            return;
        } else {
            6
        };
        for (i, val) in bf.iter_mut().enumerate().take(k + 1) {
            let mut y = 0.0;
            let mut neg_x_pow = 1.0;
            for m in 0..=last {
                let xf = if m == 0 { 1.0 } else { FACT[m] };
                let term =
                    neg_x_pow * (2.0 * (((m + i + 1) % 2) as f64)) / (xf * (m + i + 1) as f64);
                y += term;
                neg_x_pow *= -x;
            }
            *val = y;
        }
        return;
    }

    let expx = x.exp();
    let expmx = 1.0 / expx;
    bf[0] = (expx - expmx) / x;
    for i in 1..=k {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        bf[i] = ((i as f64) * bf[i - 1] + sign * expx - expmx) / x;
    }
}

/// Evaluates general Slater two-center overlap integral $S(n_A, l_A, n_B, l_B, M)$.
/// Port of OpenMOPAC `ss.F90`.
pub fn slater_overlap_ss(
    na: usize,
    nb: usize,
    la1: usize,
    lb1: usize,
    m1: usize,
    ua: f64,
    ub: f64,
    r1_angstrom: f64,
) -> f64 {
    let a0 = BOHR_RADIUS_ANGSTROMS;
    let r = r1_angstrom / a0;
    if r < 1e-12 {
        return 0.0;
    }

    let m = m1 - 1;
    let lb = lb1 - 1;
    let la = la1 - 1;

    let mut bi = [[0.0f64; 13]; 13];
    for i in 0..=12 {
        bi[i][0] = 1.0;
        bi[i][i] = 1.0;
    }
    for i in 0..12 {
        for j in 1..=i {
            bi[i + 1][j] = bi[i][j] + bi[i][j - 1];
        }
    }

    let mut aff = [[[0.0f64; 3]; 3]; 3];
    aff[0][0][0] = 1.0;
    aff[1][0][0] = 1.0;
    aff[1][1][0] = (0.5f64).sqrt();
    aff[2][0][0] = 1.5;
    aff[2][1][0] = (1.5f64).sqrt();
    aff[2][2][0] = (0.375f64).sqrt();
    aff[2][0][2] = -0.5;

    let p = (ua + ub) * r * 0.5;
    let b = (ua - ub) * r * 0.5;
    let quo = 1.0 / p;
    let mut af = [0.0f64; 20];
    af[0] = quo * (-p).exp();
    for n in 1..20 {
        af[n] = (n as f64) * quo * af[n - 1] + af[0];
    }

    let mut bf = [0.0f64; 14];
    bfn(b, &mut bf);

    let mut total_sum = 0.0;
    if la < m || lb < m {
        return 0.0;
    }
    let lam1 = la - m;
    let lbm1 = lb - m;

    let mut i = 0;
    while i <= lam1 {
        let ia = na + i - la;
        let ic = la - i - m;
        let mut j = 0;
        while j <= lbm1 {
            let ib = nb + j - lb;
            let id = lb - j - m;
            let mut sum1 = 0.0;
            let iab = ia + ib;
            for k1 in 0..=ia {
                for k2 in 0..=ib {
                    for k3 in 0..=ic {
                        for k4 in 0..=id {
                            for k5 in 0..=m {
                                let iaf = iab - k1 - k2 + k3 + k4 + 2 * k5;
                                for k6 in 0..=m {
                                    let ibf = k1 + k2 + k3 + k4 + 2 * k6;
                                    let term = bi[id][k4]
                                        * bi[ic][k3]
                                        * bi[ib][k2]
                                        * bi[ia][k1]
                                        * bi[m][k5]
                                        * bi[m][k6];
                                    let sign = if (m + k2 + k4 + k5 + k6) % 2 == 0 {
                                        1.0
                                    } else {
                                        -1.0
                                    };
                                    if iaf < 20 && ibf < 14 {
                                        sum1 += term * sign * af[iaf] * bf[ibf];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if la < 3 && m < 3 && i < 3 && lb < 3 && j < 3 {
                total_sum += sum1 * aff[la][m][i] * aff[lb][m][j];
            }
            j += 2;
        }
        i += 2;
    }

    let norm = r.powi((na + nb + 1) as i32) * ua.powi(na as i32) * ub.powi(nb as i32) / 2.0;
    let f_2na = if 2 * na < FACT.len() {
        FACT[2 * na]
    } else {
        1.0
    };
    let f_2nb = if 2 * nb < FACT.len() {
        FACT[2 * nb]
    } else {
        1.0
    };
    let pre = (ua * ub / (f_2na * f_2nb) * (((2 * la + 1) * (2 * lb + 1)) as f64)).sqrt();
    total_sum * norm * pre
}

/// Rotates local-frame diatomic overlaps into 3D Cartesian frame via direction cosines.
/// Port of OpenMOPAC `coe.F90`.
pub fn diatomic_rotation_coe(
    x2: f64,
    y2: f64,
    z2: f64,
    norbi: usize,
    norbj: usize,
    c_3d: &mut [[[f64; 6]; 6]; 4],
    r_out: &mut f64,
) {
    let rt34 = 0.86602540378444f64;
    let rt13 = 0.57735026918963f64;
    let xy_sq = x2 * x2 + y2 * y2;
    let r = (xy_sq + z2 * z2).sqrt();
    *r_out = r;
    let xy = xy_sq.sqrt();

    let (ca, cb, sa, sb) = if xy >= 1e-10 {
        (x2 / xy, z2 / r, y2 / xy, xy / r)
    } else if z2 < 0.0 {
        (-1.0, -1.0, 0.0, 0.0)
    } else if z2 == 0.0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        (1.0, 1.0, 0.0, 0.0)
    };

    let mut c = [0.0f64; 76];
    let nij = norbi.max(norbj);
    c[37] = 1.0;

    if nij >= 2 {
        c[56] = ca * cb;
        c[41] = ca * sb;
        c[26] = -sa;
        c[53] = -sb;
        c[38] = cb;
        c[23] = 0.0;
        c[50] = sa * cb;
        c[35] = sa * sb;
        c[20] = ca;

        if nij >= 5 {
            let c2a = 2.0 * ca * ca - 1.0;
            let c2b = 2.0 * cb * cb - 1.0;
            let s2a = 2.0 * sa * ca;
            let s2b = 2.0 * sb * cb;

            c[75] = c2a * cb * cb + 0.5 * c2a * sb * sb;
            c[60] = 0.5 * c2a * s2b;
            c[45] = rt34 * c2a * sb * sb;
            c[30] = -s2a * sb;
            c[15] = -s2a * cb;
            c[72] = -0.5 * ca * s2b;
            c[57] = ca * c2b;
            c[42] = rt34 * ca * s2b;
            c[27] = -sa * cb;
            c[12] = sa * sb;
            c[69] = rt13 * sb * sb * 1.5;
            c[54] = -rt34 * s2b;
            c[39] = cb * cb - 0.5 * sb * sb;
            c[66] = -0.5 * sa * s2b;
            c[51] = sa * c2b;
            c[36] = rt34 * sa * s2b;
            c[21] = ca * cb;
            c[6] = -ca * sb;
            c[63] = s2a * cb * cb + 0.5 * s2a * sb * sb;
            c[48] = 0.5 * s2a * s2b;
            c[33] = rt34 * s2a * sb * sb;
            c[18] = c2a * sb;
            c[3] = c2a * cb;
        }
    }

    for i in 1..4 {
        for k in 1..6 {
            for m in 1..6 {
                let idx = (i - 1) + 3 * (k - 1) + 15 * (m - 1) + 1;
                c_3d[i][k][m] = c[idx];
            }
        }
    }
}

/// Analytical overlap fast path for main-group row 1 and 2 elements ($Z \le 17$).
pub fn compute_diatomic_overlap_block_analytical(
    za: u8,
    zb: u8,
    p_a: &SemiEmpiricalElementParams,
    p_b: &SemiEmpiricalElementParams,
    r_angstrom: f64,
    dir_cosines: [f64; 3],
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
        s_mat[0][0] = overlap_1s_1s(r_angstrom, p_a.zs, p_b.zs);
    } else if za == 1 && zb > 1 {
        set_ab_tables(p_a.zs, p_b.zs, rab, &mut a, &mut b);
        let rab4 = rab.powi(4) * 0.125;
        let w = (p_a.zs.powi(3) * p_b.zs.powi(5)).sqrt() * rab4;
        let rt3 = 1.0 / 3.0f64.sqrt();
        let s_ss = w * rt3 * (a[3] * b[0] - b[3] * a[0] + a[2] * b[1] - b[2] * a[1]);
        s_mat[0][0] = s_ss;

        set_ab_tables(p_a.zs, p_b.zp, rab, &mut a, &mut b);
        let wp = (p_a.zs.powi(3) * p_b.zp.powi(5)).sqrt() * rab4;
        let s_sp = wp * (a[2] * b[0] - b[2] * a[0] + a[3] * b[1] - b[3] * a[1]);

        s_mat[0][1] = lx * s_sp;
        s_mat[0][2] = ly * s_sp;
        s_mat[0][3] = lz * s_sp;
    } else if za > 1 && zb == 1 {
        set_ab_tables(p_b.zs, p_a.zs, rab, &mut a, &mut b);
        let rab4 = rab.powi(4) * 0.125;
        let w = (p_b.zs.powi(3) * p_a.zs.powi(5)).sqrt() * rab4;
        let rt3 = 1.0 / 3.0f64.sqrt();
        let s_ss = w * rt3 * (a[3] * b[0] - b[3] * a[0] + a[2] * b[1] - b[2] * a[1]);
        s_mat[0][0] = s_ss;

        set_ab_tables(p_b.zs, p_a.zp, rab, &mut a, &mut b);
        let wp = (p_b.zs.powi(3) * p_a.zp.powi(5)).sqrt() * rab4;
        let s_ps = wp * (a[2] * b[0] - b[2] * a[0] + a[3] * b[1] - b[3] * a[1]);

        s_mat[1][0] = lx * s_ps;
        s_mat[2][0] = ly * s_ps;
        s_mat[3][0] = lz * s_ps;
    } else {
        let rab5 = rab.powi(5) * 0.0625;

        set_ab_tables(p_a.zs, p_b.zs, rab, &mut a, &mut b);
        let w_ss = (p_a.zs * p_b.zs).powi(5).sqrt() * rab5;
        let s_ss = w_ss * (a[4] * b[0] + b[4] * a[0] - 2.0 * a[2] * b[2]) / 3.0;
        s_mat[0][0] = s_ss;

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

        set_ab_tables(p_a.zs, p_b.zp, rab, &mut a, &mut b);
        let w_sp = (p_a.zs * p_b.zp).powi(5).sqrt() * rab5;
        let rt3 = 1.0 / 3.0f64.sqrt();
        let d = a[3] * (b[0] - b[2]) - a[1] * (b[2] - b[4]);
        let e = b[3] * (a[0] - a[2]) - b[1] * (a[2] - a[4]);
        let s_s_psigma = -w_sp * rt3 * (d + e);

        s_mat[0][1] = lx * s_s_psigma;
        s_mat[0][2] = ly * s_s_psigma;
        s_mat[0][3] = lz * s_s_psigma;

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

/// Compute complete 3D Cartesian diatomic overlap matrix between atom A and atom B (up to 9x9).
///
/// Output: `s_mat[oa][ob]` where:
/// - 0: s
/// - 1: px
/// - 2: py
/// - 3: pz
/// - 4: d(x^2 - y^2)
/// - 5: d(xz)
/// - 6: d(z^2)
/// - 7: d(yz)
/// - 8: d(xy)
pub fn compute_diatomic_overlap_matrix_9x9(
    za: u8,
    zb: u8,
    norb_a: usize,
    norb_b: usize,
    p_a: &SemiEmpiricalElementParams,
    p_b: &SemiEmpiricalElementParams,
    dx: f64,
    dy: f64,
    dz: f64,
    r_angstrom: f64,
    s_mat: &mut [[f64; 9]; 9],
) {
    for row in s_mat.iter_mut() {
        row.fill(0.0);
    }
    if !(1e-10..=20.0).contains(&r_angstrom) {
        return;
    }

    // Fast path: main group elements 1..=17 without d-orbitals
    let use_diat2_a = (1..=17).contains(&za) && za != 2 && za != 10 && norb_a <= 4;
    let use_diat2_b = (1..=17).contains(&zb) && zb != 2 && zb != 10 && norb_b <= 4;

    if use_diat2_a && use_diat2_b {
        let mut s4 = [[0.0f64; 4]; 4];
        let dir = [dx / r_angstrom, dy / r_angstrom, dz / r_angstrom];
        compute_diatomic_overlap_block_analytical(za, zb, p_a, p_b, r_angstrom, dir, &mut s4);
        for i in 0..norb_a.min(4) {
            for j in 0..norb_b.min(4) {
                s_mat[i][j] = s4[i][j];
            }
        }
        return;
    }

    // General Slater overlap via OpenMOPAC diat/ss/coe/bfn
    let za_usize = za as usize;
    let zb_usize = zb as usize;
    if za_usize >= 108 || zb_usize >= 108 {
        return;
    }

    let pq1 = NPQ[za_usize][0] as usize;
    let pq2 = NPQ[zb_usize][0] as usize;
    if pq1 == 0 || pq2 == 0 {
        return;
    }

    let mut c_3d = [[[0.0f64; 6]; 6]; 4];
    let mut r_coe = 0.0;
    diatomic_rotation_coe(dx, dy, dz, norb_a, norb_b, &mut c_3d, &mut r_coe);
    if r_coe < 1e-3 {
        return;
    }

    let ia = (pq1 + 1).min(3);
    let ib = (pq2 + 1).min(3);
    let a = ia - 1;
    let b = ib - 1;

    let ul1 = [0.0, p_a.zs, p_a.zp, p_a.zd.max(0.3)];
    let ul2 = [0.0, p_b.zs, p_b.zp, p_b.zd.max(0.3)];

    let newk = a.min(b);
    let nk1 = newk + 1;
    let mut s = [[[0.0f64; 4]; 4]; 4];

    for i in 1..=ia {
        let iss = i;
        let pq1_i = NPQ[za_usize][i - 1] as usize;
        for j in 1..=(b + 1) {
            let jss = j;
            let pq2_j = NPQ[zb_usize][j - 1] as usize;
            for k in 1..=nk1 {
                if k > i || k > j {
                    continue;
                }
                let pi = pq1_i.max(iss);
                let pj = pq2_j.max(jss);
                s[i][j][k] = slater_overlap_ss(pi, pj, iss, jss, k, ul1[i], ul2[j], r_angstrom);
            }
        }
    }

    // OpenMOPAC ival mapping: (i in 1..3, k in 1..5) -> 1..9 (0..8 in 0-based)
    const IVAL: [[usize; 6]; 4] = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 3, 4, 2, 0],
        [0, 9, 8, 7, 6, 5],
    ];

    for i in 1..=ia {
        let kmin = 4 - i;
        let kmax = 2 + i;
        for j in 1..=ib {
            let (aa, bb) = if j == 2 {
                (-1.0, 1.0)
            } else if j == 3 {
                (1.0, -1.0)
            } else {
                (1.0, 1.0)
            };
            let lmin = 4 - j;
            let lmax = 2 + j;
            for k in kmin..=kmax {
                for l in lmin..=lmax {
                    let ii = IVAL[i][k];
                    let jj = IVAL[j][l];
                    if ii == 0 || jj == 0 || ii > 9 || jj > 9 {
                        continue;
                    }
                    let term1 = s[i][j][1] * (c_3d[i][k][3] * c_3d[j][l][3]) * aa;
                    let term2 = s[i][j][2]
                        * (c_3d[i][k][4] * c_3d[j][l][4] + c_3d[i][k][2] * c_3d[j][l][2])
                        * bb;
                    let term3 = s[i][j][3]
                        * (c_3d[i][k][5] * c_3d[j][l][5] + c_3d[i][k][1] * c_3d[j][l][1]);
                    s_mat[ii - 1][jj - 1] += term1 + term2 + term3;
                }
            }
        }
    }
}

/// Compute complete 3D Cartesian diatomic overlap matrix between atom A and atom B (4x4 submatrix).
///
/// Output: `s_mat[oa][ob]` where:
/// - Index 0: $s$ orbital
/// - Index 1: $p_x$ orbital
/// - Index 2: $p_y$ orbital
/// - Index 3: $p_z$ orbital
///
/// Rotation into the 3D laboratory Cartesian frame is performed via direction cosines:
/// $\mathbf{l} = \frac{\vec{R}_B - \vec{R}_A}{\|\vec{R}_B - \vec{R}_A\|}$.
/// Guaranteed **0 heap allocations**.
pub fn compute_diatomic_overlap_block(
    za: u8,
    zb: u8,
    p_a: &SemiEmpiricalElementParams,
    p_b: &SemiEmpiricalElementParams,
    r_angstrom: f64,
    dir_cosines: [f64; 3],
    s_mat: &mut [[f64; 4]; 4],
) {
    let mut di = [[0.0f64; 9]; 9];
    let dx = dir_cosines[0] * r_angstrom;
    let dy = dir_cosines[1] * r_angstrom;
    let dz = dir_cosines[2] * r_angstrom;
    compute_diatomic_overlap_matrix_9x9(za, zb, 4, 4, p_a, p_b, dx, dy, dz, r_angstrom, &mut di);
    for i in 0..4 {
        for j in 0..4 {
            s_mat[i][j] = di[i][j];
        }
    }
}
