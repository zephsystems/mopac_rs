//! NDDO Diatomic Multipole Integrals and 3D Frame Rotation Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct mathematical translation of OpenMOPAC `mndod.F90`, `rotate.F90`,
//! `jab.F90`, and `kab.F90`.

use crate::constants::codata2018::{BOHR_RADIUS_ANGSTROMS as A0_BOHR, HARTREE_TO_EV as EV_HARTREE};
use crate::parameters::{ParameterModel, SemiEmpiricalElementParams};
use crate::types::{AlignedMatrix, MolecularBatch};

/// Derived multipole parameters for an element ($dd$, $qq$, $ad$, $aq$, $am$, $po$).
///
/// Derived from fundamental semi-empirical parameters via `calpar.F90` and `inid.F90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivedMultipoleParams {
    pub z: u8,
    /// Dipole charge separation $D_1$ (Bohr)
    pub dd: f64,
    /// Quadrupole charge separation $D_2$ (Bohr)
    pub qq: f64,
    /// Monopole additive Klopman-Ohno parameter $A_M$ (a.u. = $g_{ss} / \text{eV}$)
    pub am: f64,
    /// Dipole additive Klopman-Ohno parameter $A_D$ (a.u.)
    pub ad: f64,
    /// Quadrupole additive Klopman-Ohno parameter $A_Q$ (a.u.)
    pub aq: f64,
    /// Klopman-Ohno core/electron additive parameters $PO$ (Bohr)
    /// `po[0]` = Monopole radius $\rho_s$ ($0.5 / A_M$)
    /// `po[1]` = Dipole radius $\rho_p$ ($0.5 / A_D$)
    /// `po[2]` = Quadrupole radius $\rho_d$ ($0.5 / A_Q$)
    pub po: [f64; 3],
}

impl DerivedMultipoleParams {
    /// Compute derived multipole parameters from element parameters matching `calpar.F90`.
    pub fn from_element(p: &SemiEmpiricalElementParams) -> Self {
        let ev = EV_HARTREE;
        if p.z == 1 {
            let am = p.gss / ev;
            let po0 = 0.5 / am;
            return Self {
                z: 1,
                dd: 0.0,
                qq: 0.0,
                am,
                ad: am,
                aq: am,
                po: [po0, 0.0, 0.0],
            };
        }

        // Valence shell principal quantum number n (calpar.F90 nspqn)
        let qn = match p.z {
            1..=2 => 1.0,
            3..=10 => 2.0,
            11..=18 => 3.0,
            19..=36 => 4.0,
            37..=54 => 5.0,
            _ => 6.0,
        };

        // dd and qq in atomic units (Bohr)
        let zs = p.zs;
        let zp = p.zp;
        let dd = (2.0 * qn + 1.0) * (4.0 * zs * zp).powf(qn + 0.5)
            / (zs + zp).powf(2.0 * qn + 2.0)
            / 3.0f64.sqrt();
        let qq = ((4.0 * qn * qn + 6.0 * qn + 2.0) / 20.0).sqrt() / zp;

        // One-center exchange and Coulomb differences
        let hpp = (0.5 * (p.gpp - p.gp2)).max(0.1);
        let hsp = p.hsp.max(1e-7);

        // Secant iteration for ad (calpar.F90 lines 159-171)
        let gdd1 = (hsp / (ev * dd * dd)).powf(1.0 / 3.0);
        let mut d1 = gdd1;
        let mut d2 = gdd1 + 0.04;
        for _ in 0..10 {
            let df = d2 - d1;
            let hsp1 = 0.5 * d1 - 0.5 / (4.0 * dd * dd + 1.0 / (d1 * d1)).sqrt();
            let hsp2 = 0.5 * d2 - 0.5 / (4.0 * dd * dd + 1.0 / (d2 * d2)).sqrt();
            if (hsp2 - hsp1).abs() < 1e-25 {
                break;
            }
            let d3 = d1 + df * (hsp / ev - hsp1) / (hsp2 - hsp1);
            d1 = d2;
            d2 = d3;
        }
        let ad = d2;

        // Secant iteration for aq (calpar.F90 lines 172-185)
        let gqq = (4.0 * hpp / (ev * 48.0 * qq.powi(4))).powf(0.2);
        let mut q1 = gqq;
        let mut q2 = gqq + 0.04;
        for _ in 0..10 {
            let qf = q2 - q1;
            let hpp1 = 0.25 * q1 - 0.5 / (4.0 * qq * qq + 1.0 / (q1 * q1)).sqrt()
                + 0.25 / (8.0 * qq * qq + 1.0 / (q1 * q1)).sqrt();
            let hpp2 = 0.25 * q2 - 0.5 / (4.0 * qq * qq + 1.0 / (q2 * q2)).sqrt()
                + 0.25 / (8.0 * qq * qq + 1.0 / (q2 * q2)).sqrt();
            if (hpp2 - hpp1).abs() < 1e-25 {
                break;
            }
            let q3 = q1 + qf * (hpp / ev - hpp1) / (hpp2 - hpp1);
            q1 = q2;
            q2 = q3;
        }
        let aq = q2;

        let am = p.gss / ev;
        let po0 = 0.5 / am;
        let po1 = if ad > 1e-5 { 0.5 / ad } else { 0.0 };
        let po2 = if aq > 1e-5 { 0.5 / aq } else { 0.0 };

        Self {
            z: p.z,
            dd,
            qq,
            am,
            ad,
            aq,
            po: [po0, po1, po2],
        }
    }
}

/// Evaluation of the 22 NDDO two-center two-electron integrals over local coordinates ($R_I$) in eV.
///
/// Port of OpenMOPAC `reppd` (`mndod.F90` lines 426-843).
///
/// Output indexing matches MOPAC `ri[0..22]` (0-indexed in Rust):
///  0: (ss|ss)     1: (so|ss)     2: (oo|ss)     3: (pp|ss)     4: (ss|os)
///  5: (so|so)     6: (sp|sp)     7: (oo|so)     8: (pp|so)     9: (po|sp)
/// 10: (ss|oo)    11: (ss|pp)    12: (so|oo)    13: (so|pp)    14: (sp|op)
/// 15: (oo|oo)    16: (pp|oo)    17: (oo|pp)    18: (pp|pp)    19: (po|po)
/// 20: (pp|p*p*)  21: (p*p|p*p)
pub fn compute_22_multipoles(
    params_a: &DerivedMultipoleParams,
    params_b: &DerivedMultipoleParams,
    r_angstrom: f64,
) -> ([f64; 22], f64) {
    let ev = EV_HARTREE;
    let a0 = A0_BOHR;
    let r = r_angstrom / a0; // distance in Bohr

    let ev1 = ev / 2.0;
    let ev2 = ev1 / 2.0;
    let ev3 = ev2 / 2.0;
    let ev4 = ev3 / 2.0;

    let mut ri = [0.0f64; 22];

    // Core-core parameter gab
    let po_core_sum = params_a.po[0] + params_b.po[0];
    let aee_core = po_core_sum * po_core_sum;
    let gab = ev / (r * r + aee_core).sqrt();

    let aee_val = 0.5 / params_a.am + 0.5 / params_b.am;
    let aee = aee_val * aee_val;

    let si = params_a.z >= 3;
    let sj = params_b.z >= 3;

    if !si && !sj {
        // H - H
        ri[0] = ev / (r * r + aee).sqrt();
    } else if si && !sj {
        // Heavy - H
        let da = params_a.dd;
        let qa = params_a.qq * 2.0;
        let ade_val = 0.5 / params_a.ad + 0.5 / params_b.am;
        let ade = ade_val * ade_val;
        let aqe_val = 0.5 / params_a.aq + 0.5 / params_b.am;
        let aqe = aqe_val * aqe_val;

        let rsq = r * r;
        let sqr1 = (rsq + aee).sqrt();
        let sqr2 = ((r + da) * (r + da) + ade).sqrt();
        let sqr3 = ((r - da) * (r - da) + ade).sqrt();
        let sqr4 = ((r + qa) * (r + qa) + aqe).sqrt();
        let sqr5 = ((r - qa) * (r - qa) + aqe).sqrt();
        let sqr6 = (rsq + aqe).sqrt();
        let sqr7 = (rsq + aqe + qa * qa).sqrt();

        let ee = ev / sqr1;
        ri[0] = ee;
        ri[1] = ev1 / sqr2 - ev1 / sqr3;
        ri[2] = ee + ev2 / sqr4 + ev2 / sqr5 - ev1 / sqr6;
        ri[3] = ee + ev1 / sqr7 - ev1 / sqr6;
    } else if !si && sj {
        // H - Heavy
        let db = params_b.dd;
        let qb = params_b.qq * 2.0;
        let aed_val = 0.5 / params_a.am + 0.5 / params_b.ad;
        let aed = aed_val * aed_val;
        let aeq_val = 0.5 / params_a.am + 0.5 / params_b.aq;
        let aeq = aeq_val * aeq_val;

        let rsq = r * r;
        let sqr1 = (rsq + aee).sqrt();
        let sqr2 = ((r - db) * (r - db) + aed).sqrt();
        let sqr3 = ((r + db) * (r + db) + aed).sqrt();
        let sqr4 = ((r - qb) * (r - qb) + aeq).sqrt();
        let sqr5 = ((r + qb) * (r + qb) + aeq).sqrt();
        let sqr6 = (rsq + aeq).sqrt();
        let sqr7 = (rsq + aeq + qb * qb).sqrt();

        let ee = ev / sqr1;
        ri[0] = ee;
        ri[4] = ev1 / sqr2 - ev1 / sqr3;
        ri[10] = ee + ev2 / sqr4 + ev2 / sqr5 - ev1 / sqr6;
        ri[11] = ee + ev1 / sqr7 - ev1 / sqr6;
    } else {
        // Heavy - Heavy
        let da = params_a.dd;
        let db = params_b.dd;
        let qa = params_a.qq * 2.0;
        let qb = params_b.qq * 2.0;

        let ade = (0.5 / params_a.ad + 0.5 / params_b.am).powi(2);
        let aqe = (0.5 / params_a.aq + 0.5 / params_b.am).powi(2);
        let aed = (0.5 / params_a.am + 0.5 / params_b.ad).powi(2);
        let aeq = (0.5 / params_a.am + 0.5 / params_b.aq).powi(2);
        let axx = (0.5 / params_a.ad + 0.5 / params_b.ad).powi(2);
        let adq = (0.5 / params_a.ad + 0.5 / params_b.aq).powi(2);
        let aqd = (0.5 / params_a.aq + 0.5 / params_b.ad).powi(2);
        let aqq = (0.5 / params_a.aq + 0.5 / params_b.aq).powi(2);

        let rsq = r * r;
        let mut arg = [0.0f64; 73]; // 1-based indexing for parity with Fortran

        arg[1] = rsq + aee;
        arg[2] = (r + da) * (r + da) + ade;
        arg[3] = (r - da) * (r - da) + ade;
        arg[4] = (r - qa) * (r - qa) + aqe;
        arg[5] = (r + qa) * (r + qa) + aqe;
        arg[6] = rsq + aqe;
        arg[7] = arg[6] + qa * qa;

        arg[8] = (r - db) * (r - db) + aed;
        arg[9] = (r + db) * (r + db) + aed;
        arg[10] = (r - qb) * (r - qb) + aeq;
        arg[11] = (r + qb) * (r + qb) + aeq;
        arg[12] = rsq + aeq;
        arg[13] = arg[12] + qb * qb;

        arg[14] = rsq + axx + (da - db) * (da - db);
        arg[15] = rsq + axx + (da + db) * (da + db);
        arg[16] = (r + da - db) * (r + da - db) + axx;
        arg[17] = (r - da + db) * (r - da + db) + axx;
        arg[18] = (r - da - db) * (r - da - db) + axx;
        arg[19] = (r + da + db) * (r + da + db) + axx;

        arg[20] = (r + da) * (r + da) + adq;
        arg[21] = arg[20] + qb * qb;
        arg[22] = (r - da) * (r - da) + adq;
        arg[23] = arg[22] + qb * qb;

        arg[24] = (r - db) * (r - db) + aqd;
        arg[25] = arg[24] + qa * qa;
        arg[26] = (r + db) * (r + db) + aqd;
        arg[27] = arg[26] + qa * qa;

        arg[28] = (r + da - qb) * (r + da - qb) + adq;
        arg[29] = (r - da - qb) * (r - da - qb) + adq;
        arg[30] = (r + da + qb) * (r + da + qb) + adq;
        arg[31] = (r - da + qb) * (r - da + qb) + adq;

        arg[32] = (r + qa - db) * (r + qa - db) + aqd;
        arg[33] = (r + qa + db) * (r + qa + db) + aqd;
        arg[34] = (r - qa - db) * (r - qa - db) + aqd;
        arg[35] = (r - qa + db) * (r - qa + db) + aqd;

        arg[36] = rsq + aqq;
        arg[37] = arg[36] + (qa - qb) * (qa - qb);
        arg[38] = arg[36] + (qa + qb) * (qa + qb);
        arg[39] = arg[36] + qa * qa;
        arg[40] = arg[36] + qb * qb;
        arg[41] = arg[39] + qb * qb;

        arg[42] = (r - qb) * (r - qb) + aqq;
        arg[43] = arg[42] + qa * qa;
        arg[44] = (r + qb) * (r + qb) + aqq;
        arg[45] = arg[44] + qa * qa;

        arg[46] = (r + qa) * (r + qa) + aqq;
        arg[47] = arg[46] + qb * qb;
        arg[48] = (r - qa) * (r - qa) + aqq;
        arg[49] = arg[48] + qb * qb;

        arg[50] = (r + qa - qb) * (r + qa - qb) + aqq;
        arg[51] = (r + qa + qb) * (r + qa + qb) + aqq;
        arg[52] = (r - qa - qb) * (r - qa - qb) + aqq;
        arg[53] = (r - qa + qb) * (r - qa + qb) + aqq;

        let qa1 = params_a.qq;
        let qb1 = params_b.qq;
        let xxx_dq = (da - qb1) * (da - qb1);
        let yyy_rq = (r - qb1) * (r - qb1);
        let zzz_dq = (da + qb1) * (da + qb1);
        let www_rq = (r + qb1) * (r + qb1);

        arg[54] = xxx_dq + yyy_rq + adq;
        arg[55] = xxx_dq + www_rq + adq;
        arg[56] = zzz_dq + yyy_rq + adq;
        arg[57] = zzz_dq + www_rq + adq;

        let xxx_qd = (qa1 - db) * (qa1 - db);
        let yyy_qd = (qa1 + db) * (qa1 + db);
        let zzz_rq = (r + qa1) * (r + qa1);
        let www_rq2 = (r - qa1) * (r - qa1);

        arg[58] = zzz_rq + xxx_qd + aqd;
        arg[59] = www_rq2 + xxx_qd + aqd;
        arg[60] = zzz_rq + yyy_qd + aqd;
        arg[61] = www_rq2 + yyy_qd + aqd;

        let xxx_qq = (qa1 - qb1) * (qa1 - qb1);
        arg[62] = arg[36] + 2.0 * xxx_qq;
        let yyy_qq = (qa1 + qb1) * (qa1 + qb1);
        arg[63] = arg[36] + 2.0 * yyy_qq;
        arg[64] = arg[36] + 2.0 * (qa1 * qa1 + qb1 * qb1);

        let zzz_rqq1 = (r + qa1 - qb1) * (r + qa1 - qb1);
        arg[65] = zzz_rqq1 + xxx_qq + aqq;
        arg[66] = zzz_rqq1 + yyy_qq + aqq;

        let zzz_rqq2 = (r + qa1 + qb1) * (r + qa1 + qb1);
        arg[67] = zzz_rqq2 + xxx_qq + aqq;
        arg[68] = zzz_rqq2 + yyy_qq + aqq;

        let zzz_rqq3 = (r - qa1 - qb1) * (r - qa1 - qb1);
        arg[69] = zzz_rqq3 + xxx_qq + aqq;
        arg[70] = zzz_rqq3 + yyy_qq + aqq;

        let zzz_rqq4 = (r - qa1 + qb1) * (r - qa1 + qb1);
        arg[71] = zzz_rqq4 + xxx_qq + aqq;
        arg[72] = zzz_rqq4 + yyy_qq + aqq;

        let mut sqr = [0.0f64; 73];
        for k in 1..=72 {
            sqr[k] = arg[k].sqrt();
        }

        let ee = ev / sqr[1];
        let dze = -ev1 / sqr[2] + ev1 / sqr[3];
        let qzze = ev2 / sqr[4] + ev2 / sqr[5] - ev1 / sqr[6];
        let qxxe = ev1 / sqr[7] - ev1 / sqr[6];
        let edz = -ev1 / sqr[8] + ev1 / sqr[9];
        let eqzz = ev2 / sqr[10] + ev2 / sqr[11] - ev1 / sqr[12];
        let eqxx = ev1 / sqr[13] - ev1 / sqr[12];
        let dxdx = ev1 / sqr[14] - ev1 / sqr[15];
        let dzdz = ev2 / sqr[16] + ev2 / sqr[17] - ev2 / sqr[18] - ev2 / sqr[19];
        let dzqxx = ev2 / sqr[20] - ev2 / sqr[21] - ev2 / sqr[22] + ev2 / sqr[23];
        let qxxdz = ev2 / sqr[24] - ev2 / sqr[25] - ev2 / sqr[26] + ev2 / sqr[27];
        let dzqzz = -ev3 / sqr[28] + ev3 / sqr[29] - ev3 / sqr[30] + ev3 / sqr[31] - ev2 / sqr[22]
            + ev2 / sqr[20];
        let qzzdz = -ev3 / sqr[32] + ev3 / sqr[33] - ev3 / sqr[34] + ev3 / sqr[35] + ev2 / sqr[24]
            - ev2 / sqr[26];
        let qxxqxx = ev3 / sqr[37] + ev3 / sqr[38] - ev2 / sqr[39] - ev2 / sqr[40] + ev2 / sqr[36];
        let qxxqyy = ev2 / sqr[41] - ev2 / sqr[39] - ev2 / sqr[40] + ev2 / sqr[36];
        let qxxqzz = ev3 / sqr[43] + ev3 / sqr[45] - ev3 / sqr[42] - ev3 / sqr[44] - ev2 / sqr[39]
            + ev2 / sqr[36];
        let qzzqxx = ev3 / sqr[47] + ev3 / sqr[49] - ev3 / sqr[46] - ev3 / sqr[48] - ev2 / sqr[40]
            + ev2 / sqr[36];
        let qzzqzz = ev4 / sqr[50] + ev4 / sqr[51] + ev4 / sqr[52] + ev4 / sqr[53]
            - ev3 / sqr[48]
            - ev3 / sqr[46]
            - ev3 / sqr[42]
            - ev3 / sqr[44]
            + ev2 / sqr[36];
        let dxqxz = -ev2 / sqr[54] + ev2 / sqr[55] + ev2 / sqr[56] - ev2 / sqr[57];
        let qxzdx = -ev2 / sqr[58] + ev2 / sqr[59] + ev2 / sqr[60] - ev2 / sqr[61];
        let qxzqxz = ev3 / sqr[65] - ev3 / sqr[67] - ev3 / sqr[69] + ev3 / sqr[71] - ev3 / sqr[66]
            + ev3 / sqr[68]
            + ev3 / sqr[70]
            - ev3 / sqr[72];

        ri[0] = ee;
        ri[1] = -dze;
        ri[2] = ee + qzze;
        ri[3] = ee + qxxe;
        ri[4] = -edz;
        ri[5] = dzdz;
        ri[6] = dxdx;
        ri[7] = -edz - qzzdz;
        ri[8] = -edz - qxxdz;
        ri[9] = -qxzdx;
        ri[10] = ee + eqzz;
        ri[11] = ee + eqxx;
        ri[12] = -dze - dzqzz;
        ri[13] = -dze - dzqxx;
        ri[14] = -dxqxz;
        ri[15] = ee + eqzz + qzze + qzzqzz;
        ri[16] = ee + eqzz + qxxe + qxxqzz;
        ri[17] = ee + eqxx + qzze + qzzqxx;
        ri[18] = ee + eqxx + qxxe + qxxqxx;
        ri[19] = qxzqxz;
        ri[20] = ee + eqxx + qxxe + qxxqyy;
        ri[21] = 0.5 * (qxxqxx - qxxqyy);
    }

    // Sign alternation vector nri (mndod.F90 line 489)
    const NRI: [f64; 22] = [
        1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    for k in 0..22 {
        ri[k] *= NRI[k];
    }

    (ri, gab)
}

/// Diatomic 3D Rotation Matrix and Products ($P$, $PP$).
///
/// Port of `rotmat` (`mndod.F90` lines 1382-1568).
#[derive(Debug, Clone, Copy)]
pub struct DiatomicRotation3D {
    pub p: [[f64; 3]; 3],
    pub pp: [[[f64; 3]; 3]; 6],
}

impl DiatomicRotation3D {
    /// Compute diatomic rotation matrix from vector $\vec{R}_{AB} = \vec{R}_B - \vec{R}_A$.
    pub fn new(dx: f64, dy: f64, dz: f64, r: f64) -> Self {
        let b = dx * dx + dy * dy;
        let sqb = b.sqrt();
        let sb = sqb / r;

        let (ca, sa, cb) = if sb > 1e-7 {
            (dx / sqb, dy / sqb, dz / r)
        } else {
            let ca = if dz < 0.0 {
                -1.0
            } else if dz > 0.0 {
                1.0
            } else {
                0.0
            };
            let cb = ca;
            (ca, 0.0, cb)
        };

        // 1-based Fortran indexing:
        // p(1,1)=ca*sb, p(2,1)=ca*cb, p(3,1)=-sa
        // p(1,2)=sa*sb, p(2,2)=sa*cb, p(3,2)=ca
        // p(1,3)=cb,    p(2,3)=-sb,   p(3,3)=0
        let mut p = [[0.0; 3]; 3];
        p[0][0] = ca * sb;
        p[1][0] = ca * cb;
        p[2][0] = -sa;

        p[0][1] = sa * sb;
        p[1][1] = sa * cb;
        p[2][1] = ca;

        p[0][2] = cb;
        p[1][2] = -sb;
        p[2][2] = 0.0;

        let mut pp = [[[0.0; 3]; 3]; 6];
        for k in 0..3 {
            pp[0][k][k] = p[k][0] * p[k][0];
            pp[1][k][k] = p[k][1] * p[k][1];
            pp[2][k][k] = p[k][2] * p[k][2];
            pp[3][k][k] = p[k][0] * p[k][1];
            pp[4][k][k] = p[k][0] * p[k][2];
            pp[5][k][k] = p[k][1] * p[k][2];

            if k > 0 {
                for l in 0..k {
                    pp[0][k][l] = 2.0 * p[k][0] * p[l][0];
                    pp[1][k][l] = 2.0 * p[k][1] * p[l][1];
                    pp[2][k][l] = 2.0 * p[k][2] * p[l][2];
                    pp[3][k][l] = p[k][0] * p[l][1] + p[k][1] * p[l][0];
                    pp[4][k][l] = p[k][0] * p[l][2] + p[k][2] * p[l][0];
                    pp[5][k][l] = p[k][1] * p[l][2] + p[k][2] * p[l][1];
                }
            }
        }

        Self { p, pp }
    }
}

/// Rotates the 22 local-frame integrals into the 3D molecular Cartesian frame ($W$).
///
/// Port of `tx` and `rotatd` (`mndod.F90` lines 1184-1262 and 1870-1980).
pub fn rotate_multipoles_to_w(
    norb_a: usize,
    norb_b: usize,
    ri: &[f64; 22],
    rot: &DiatomicRotation3D,
    w_out: &mut [f64],
) {
    if norb_a == 1 && norb_b == 1 {
        w_out[0] = ri[0];
        return;
    }

    // IPOS mapping array (mndod.F90 line 881)
    const IPOS: [usize; 34] = [
        0, 4, 10, 11, 11, 1, 5, 12, 13, 13, 2, 7, 15, 17, 17, 6, 14, 9, 19, 3, 8, 16, 18, 20, 6,
        14, 9, 19, 21, 3, 8, 16, 20, 18,
    ];
    let mut rep = [0.0f64; 35];
    for k in 0..34 {
        rep[k + 1] = ri[IPOS[k]];
    }

    // Pair conventions in lower triangle
    // indexd(i, j) = -(j*(j-1))/2 + i + 9*(j-1)
    let indexd = |i: usize, j: usize| -> usize {
        let (mi, mj) = if i >= j { (i, j) } else { (j, i) };
        (-(mj as isize * (mj as isize - 1)) / 2 + mi as isize + 9 * (mj as isize - 1)) as usize
    };

    let indx = |i: usize, j: usize| -> usize {
        let (mi, mj) = if i >= j { (i, j) } else { (j, i) };
        (mi * (mi - 1)) / 2 + mj
    };

    // ind2 table (mndod.F90 lines 2563-2596)
    let mut ind2 = [[0usize; 26]; 26];
    ind2[1][1] = 1;
    ind2[1][2] = 2;
    ind2[1][10] = 3;
    ind2[1][18] = 4;
    ind2[1][25] = 5;
    ind2[2][1] = 6;
    ind2[2][2] = 7;
    ind2[2][10] = 8;
    ind2[2][18] = 9;
    ind2[2][25] = 10;
    ind2[10][1] = 11;
    ind2[10][2] = 12;
    ind2[10][10] = 13;
    ind2[10][18] = 14;
    ind2[10][25] = 15;
    ind2[3][3] = 16;
    ind2[3][11] = 17;
    ind2[11][3] = 18;
    ind2[11][11] = 19;
    ind2[18][1] = 20;
    ind2[18][2] = 21;
    ind2[18][10] = 22;
    ind2[18][18] = 23;
    ind2[18][25] = 24;
    ind2[4][4] = 25;
    ind2[4][12] = 26;
    ind2[12][4] = 27;
    ind2[12][12] = 28;
    ind2[19][19] = 29;
    ind2[25][1] = 30;
    ind2[25][2] = 31;
    ind2[25][10] = 32;
    ind2[25][18] = 33;
    ind2[25][25] = 34;

    const MET: [usize; 11] = [0, 1, 2, 3, 2, 3, 3, 2, 3, 3, 3];

    let limkl = indx(norb_b, norb_b);
    let mut v = [[0.0f64; 11]; 26];
    let mut logv = [[false; 11]; 26];

    // Subroutine TX: rotate second atom B
    for i1 in 1..=norb_a {
        for j1 in 1..=i1 {
            let ij = indexd(i1, j1);
            for k1 in 1..=norb_b {
                for l1 in 1..=k1 {
                    let kl = indexd(k1, l1);
                    let nd = ind2[ij][kl];
                    if nd == 0 {
                        continue;
                    }
                    let wrepp = rep[nd];
                    let ll = indx(k1, l1);
                    let mm = MET[ll];

                    match mm {
                        1 => {
                            v[ij][1] = wrepp;
                            logv[ij][1] = true;
                        }
                        2 => {
                            let k = k1 - 2; // 0-based p orbital
                            v[ij][2] += rot.p[k][0] * wrepp;
                            v[ij][4] += rot.p[k][1] * wrepp;
                            v[ij][7] += rot.p[k][2] * wrepp;
                            logv[ij][2] = true;
                            logv[ij][4] = true;
                            logv[ij][7] = true;
                        }
                        3 => {
                            let k = k1 - 2;
                            let l = l1 - 2;
                            v[ij][3] += rot.pp[0][k][l] * wrepp;
                            v[ij][6] += rot.pp[1][k][l] * wrepp;
                            v[ij][10] += rot.pp[2][k][l] * wrepp;
                            v[ij][5] += rot.pp[3][k][l] * wrepp;
                            v[ij][8] += rot.pp[4][k][l] * wrepp;
                            v[ij][9] += rot.pp[5][k][l] * wrepp;
                            logv[ij][3] = true;
                            logv[ij][5] = true;
                            logv[ij][6] = true;
                            logv[ij][8] = true;
                            logv[ij][9] = true;
                            logv[ij][10] = true;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Subroutine ROTATD step 2: rotate first atom A
    let indw = |i: usize, j: usize, kl: usize| -> usize { (indx(i, j) - 1) * limkl + (kl - 1) };

    let n_total_w = indx(norb_a, norb_a) * limkl;
    for x in &mut w_out[..n_total_w] {
        *x = 0.0;
    }

    for i1 in 1..=norb_a {
        for j1 in 1..=i1 {
            let ij = indexd(i1, j1);
            let jj = indx(i1, j1);
            let mm = MET[jj];

            for k in 1..=norb_b {
                for l in 1..=k {
                    let kl = indx(k, l);
                    if !logv[ij][kl] {
                        continue;
                    }
                    let wrepp = v[ij][kl];

                    match mm {
                        1 => {
                            let iw = indw(1, 1, kl);
                            w_out[iw] = wrepp;
                        }
                        2 => {
                            for i in 1..=3 {
                                let iw = indw(i + 1, 1, kl);
                                w_out[iw] += rot.p[i1 - 2][i - 1] * wrepp;
                            }
                        }
                        3 => {
                            for i in 1..=3 {
                                let cc = rot.pp[i - 1][i1 - 2][j1 - 2];
                                let iw = indw(i + 1, i + 1, kl);
                                w_out[iw] += cc * wrepp;
                                if i > 1 {
                                    for j in 1..i {
                                        let cc = rot.pp[i + j][i1 - 2][j1 - 2];
                                        let iw = indw(i + 1, j + 1, kl);
                                        w_out[iw] += cc * wrepp;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

/// Compute electron-nuclear attraction integrals $E_{1B}$ and $E_{2A}$.
///
/// Port of `spcore` and `elenuc` (`mndod.F90` lines 1683-1775 and 2367-2488).
#[allow(clippy::too_many_arguments)]
pub fn compute_electron_nuclear_attraction(
    norb_a: usize,
    norb_b: usize,
    params_a: &DerivedMultipoleParams,
    params_b: &DerivedMultipoleParams,
    core_charge_a: f64,
    core_charge_b: f64,
    r_angstrom: f64,
    rot: &DiatomicRotation3D,
    e1b: &mut [f64],
    e2a: &mut [f64],
    feather: bool,
) {
    let ev = EV_HARTREE;
    let a0 = A0_BOHR;
    let r = r_angstrom / a0;
    let r2 = r * r;

    let ssi = (params_a.po[0] + params_b.po[0]).powi(2);
    let ssj = (params_b.po[0] + params_a.po[0]).powi(2);

    let mut core = [[0.0f64; 2]; 5]; // 1-based indexing for core: 1..4
    core[1][0] = -core_charge_b * ev / (r2 + ssj).sqrt();
    core[1][1] = -core_charge_a * ev / (r2 + ssi).sqrt();

    const PXY: [f64; 7] = [1.0, -0.5, -0.5, 0.5, 0.25, 0.25, 0.5];

    if params_a.z >= 3 {
        let ppj = (params_b.po[0] + params_a.po[0]).powi(2);
        let da = params_a.dd;
        let qa = params_a.qq;
        let twoqa = 2.0 * qa;
        let adj = (params_a.po[1] + params_b.po[0]).powi(2);
        let aqj = (params_a.po[2] + params_b.po[0]).powi(2);

        let mut xj = [0.0f64; 7];
        xj[0] = r2 + ppj;
        xj[1] = r2 + aqj;
        xj[2] = (r + da).powi(2) + adj;
        xj[3] = (r - da).powi(2) + adj;
        xj[4] = (r - twoqa).powi(2) + aqj;
        xj[5] = (r + twoqa).powi(2) + aqj;
        xj[6] = r2 + twoqa * twoqa + aqj;

        let mut xj_term = [0.0f64; 7];
        for k in 0..7 {
            xj_term[k] = PXY[k] / xj[k].sqrt();
        }

        let aj2 = (xj_term[2] + xj_term[3]) * ev;
        let aj3 = (xj_term[0] + xj_term[1] + xj_term[4] + xj_term[5]) * ev;
        let aj4 = (xj_term[0] + xj_term[1] + xj_term[6]) * ev;

        core[2][0] = -core_charge_b * aj2;
        core[3][0] = -core_charge_b * aj3;
        core[4][0] = -core_charge_b * aj4;
    }

    if params_b.z >= 3 {
        let ppi = (params_a.po[0] + params_b.po[0]).powi(2);
        let db = params_b.dd;
        let qb = params_b.qq;
        let twoqb = 2.0 * qb;
        let adi = (params_b.po[1] + params_a.po[0]).powi(2);
        let aqi = (params_b.po[2] + params_a.po[0]).powi(2);

        let mut xi = [0.0f64; 7];
        xi[0] = r2 + ppi;
        xi[1] = r2 + aqi;
        xi[2] = (r + db).powi(2) + adi;
        xi[3] = (r - db).powi(2) + adi;
        xi[4] = (r - twoqb).powi(2) + aqi;
        xi[5] = (r + twoqb).powi(2) + aqi;
        xi[6] = r2 + twoqb * twoqb + aqi;

        let mut xi_term = [0.0f64; 7];
        for k in 0..7 {
            xi_term[k] = PXY[k] / xi[k].sqrt();
        }

        let ai2 = -(xi_term[2] + xi_term[3]) * ev;
        let ai3 = (xi_term[0] + xi_term[1] + xi_term[4] + xi_term[5]) * ev;
        let ai4 = (xi_term[0] + xi_term[1] + xi_term[6]) * ev;

        core[2][1] = -core_charge_a * ai2;
        core[3][1] = -core_charge_a * ai3;
        core[4][1] = -core_charge_a * ai4;
    }

    if feather {
        const TRUNC_1: f64 = 7.0;
        const TRUNC_2: f64 = 0.22;
        let const_val = if r_angstrom < TRUNC_1 {
            let dr = r_angstrom - TRUNC_1;
            1.0 - (-dr * dr * TRUNC_2).exp()
        } else {
            0.0
        };
        let inv_r = crate::constants::codata2018::EV_ANGSTROM_FACTOR / r_angstrom;
        let pt_b = -core_charge_b * inv_r;
        core[1][0] = core[1][0] * const_val + (1.0 - const_val) * pt_b;
        core[2][0] *= const_val;
        core[3][0] = core[3][0] * const_val + (1.0 - const_val) * pt_b;
        core[4][0] = core[4][0] * const_val + (1.0 - const_val) * pt_b;

        let pt_a = -core_charge_a * inv_r;
        core[1][1] = core[1][1] * const_val + (1.0 - const_val) * pt_a;
        core[2][1] *= const_val;
        core[3][1] = core[3][1] * const_val + (1.0 - const_val) * pt_a;
        core[4][1] = core[4][1] * const_val + (1.0 - const_val) * pt_a;
    }

    // ELENUC: Rotate local-frame core attraction into Cartesian frame
    // For Atom A:
    e1b[0] = core[1][0]; // (s, s)
    if norb_a >= 4 {
        for i in 0..3 {
            let idx_s = ((i + 1) * (i + 2)) / 2;
            e1b[idx_s] = core[2][0] * rot.p[0][i]; // (p_i, s)
            for j in 0..=i {
                let idx_pp = idx_s + j + 1; // (p_i, p_j)
                let term_sigma = rot.p[0][i] * rot.p[0][j] * core[3][0];
                let term_pi = (rot.p[1][i] * rot.p[1][j] + rot.p[2][i] * rot.p[2][j]) * core[4][0];
                e1b[idx_pp] = term_sigma + term_pi;
            }
        }
    }

    // For Atom B:
    e2a[0] = core[1][1]; // (s, s)
    if norb_b >= 4 {
        for i in 0..3 {
            let idx_s = ((i + 1) * (i + 2)) / 2;
            e2a[idx_s] = core[2][1] * rot.p[0][i]; // (p_i, s)
            for j in 0..=i {
                let idx_pp = idx_s + j + 1; // (p_i, p_j)
                let term_sigma = rot.p[0][i] * rot.p[0][j] * core[3][1];
                let term_pi = (rot.p[1][i] * rot.p[1][j] + rot.p[2][i] * rot.p[2][j]) * core[4][1];
                e2a[idx_pp] = term_sigma + term_pi;
            }
        }
    }
}

/// Contract two-center two-electron Coulomb block $W$ with atomic density blocks $P_A$ and $P_B$ (`JAB`).
///
/// Direct port of OpenMOPAC `jab.F90`. Adds Coulomb repulsion into diagonal blocks of Fock matrix.
pub fn contract_jab(
    pja: &[f64; 16],
    pjb: &[f64; 16],
    w: &[f64],
    f_block_a: &mut [f64; 10],
    f_block_b: &mut [f64; 10],
) {
    let mut suma = [0.0f64; 10];
    let mut sumb = [0.0f64; 10];

    // Offsets in w corresponding to Fortran 1-based columns: 1, 11, 31, 61, 11, 21, 41, 71, 31, 41, 51, 81, 61, 71, 81, 91
    const W_OFFSETS_A: [usize; 16] = [
        0, 10, 30, 60, 10, 20, 40, 70, 30, 40, 50, 80, 60, 70, 80, 90,
    ];

    for i in 0..10 {
        let mut sa = 0.0;
        for p in 0..16 {
            sa += pja[p] * w[W_OFFSETS_A[p] + i];
        }
        suma[i] = sa;
    }

    const W_OFFSETS_B: [[usize; 16]; 10] = [
        [0, 1, 3, 6, 1, 2, 4, 7, 3, 4, 5, 8, 6, 7, 8, 9],
        [
            10, 11, 13, 16, 11, 12, 14, 17, 13, 14, 15, 18, 16, 17, 18, 19,
        ],
        [
            20, 21, 23, 26, 21, 22, 24, 27, 23, 24, 25, 28, 26, 27, 28, 29,
        ],
        [
            30, 31, 33, 36, 31, 32, 34, 37, 33, 34, 35, 38, 36, 37, 38, 39,
        ],
        [
            40, 41, 43, 46, 41, 42, 44, 47, 43, 44, 45, 48, 46, 47, 48, 49,
        ],
        [
            50, 51, 53, 56, 51, 52, 54, 57, 53, 54, 55, 58, 56, 57, 58, 59,
        ],
        [
            60, 61, 63, 66, 61, 62, 64, 67, 63, 64, 65, 68, 66, 67, 68, 69,
        ],
        [
            70, 71, 73, 76, 71, 72, 74, 77, 73, 74, 75, 78, 76, 77, 78, 79,
        ],
        [
            80, 81, 83, 86, 81, 82, 84, 87, 83, 84, 85, 88, 86, 87, 88, 89,
        ],
        [
            90, 91, 93, 96, 91, 92, 94, 97, 93, 94, 95, 98, 96, 97, 98, 99,
        ],
    ];

    for i in 0..10 {
        let mut sb = 0.0;
        for p in 0..16 {
            sb += pjb[p] * w[W_OFFSETS_B[i][p]];
        }
        sumb[i] = sb;
    }

    for i in 0..10 {
        f_block_a[i] += sumb[i];
        f_block_b[i] += suma[i];
    }
}

/// Contract two-center two-electron exchange block $W$ with diatomic density block $P_{AB}$ (`KAB`).
///
/// Direct port of OpenMOPAC `kab.F90`. Subtracts exchange from off-diagonal block $F_{AB}$.
pub fn contract_kab(pk: &[f64; 16], w: &[f64], f_ab: &mut [f64; 16]) {
    const K_COLS: [[usize; 16]; 16] = [
        [0, 1, 3, 6, 10, 11, 13, 16, 30, 31, 33, 36, 60, 61, 63, 66],
        [1, 2, 4, 7, 11, 12, 14, 17, 31, 32, 34, 37, 61, 62, 64, 67],
        [3, 4, 5, 8, 13, 14, 15, 18, 33, 34, 35, 38, 63, 64, 65, 68],
        [6, 7, 8, 9, 16, 17, 18, 19, 36, 37, 38, 39, 66, 67, 68, 69],
        [
            10, 11, 13, 16, 20, 21, 23, 26, 40, 41, 43, 46, 70, 71, 73, 76,
        ],
        [
            11, 12, 14, 17, 21, 22, 24, 27, 41, 42, 44, 47, 71, 72, 74, 77,
        ],
        [
            13, 14, 15, 18, 23, 24, 25, 28, 43, 44, 45, 48, 73, 74, 75, 78,
        ],
        [
            16, 17, 18, 19, 26, 27, 28, 29, 46, 47, 48, 49, 76, 77, 78, 79,
        ],
        [
            30, 31, 33, 36, 40, 41, 43, 46, 50, 51, 53, 56, 80, 81, 83, 86,
        ],
        [
            31, 32, 34, 37, 41, 42, 44, 47, 51, 52, 54, 57, 81, 82, 84, 87,
        ],
        [
            33, 34, 35, 38, 43, 44, 45, 48, 53, 54, 55, 58, 83, 84, 85, 88,
        ],
        [
            36, 37, 38, 39, 46, 47, 48, 49, 56, 57, 58, 59, 86, 87, 88, 89,
        ],
        [
            60, 61, 63, 66, 70, 71, 73, 76, 80, 81, 83, 86, 90, 91, 93, 96,
        ],
        [
            61, 62, 64, 67, 71, 72, 74, 77, 81, 82, 84, 87, 91, 92, 94, 97,
        ],
        [
            63, 64, 65, 68, 73, 74, 75, 78, 83, 84, 85, 88, 93, 94, 95, 98,
        ],
        [
            66, 67, 68, 69, 76, 77, 78, 79, 86, 87, 88, 89, 96, 97, 98, 99,
        ],
    ];

    for m in 0..16 {
        let mut s = 0.0;
        for p in 0..16 {
            s += pk[p] * w[K_COLS[m][p]];
        }
        f_ab[m] -= s;
    }
}

/// Precomputed diatomic integrals for an atom pair $(A, B)$ with $A > B$.
///
/// Stores rotated two-center two-electron repulsion tensor $W$ and electron-nuclear
/// attraction vectors $E_{1B}, E_{2A}$ directly in the molecular Cartesian frame.
#[derive(Debug, Clone)]
pub struct DiatomicPairIntegrals {
    pub atom_a: usize,
    pub atom_b: usize,
    pub norb_a: usize,
    pub norb_b: usize,
    pub orb_start_a: usize,
    pub orb_start_b: usize,
    pub w: Vec<f64>,
    pub e1b: Vec<f64>,
    pub e2a: Vec<f64>,
}

/// Precompute all diatomic multipole pairs $(A, B)$ with $A > B$ for a molecular batch.
pub fn precompute_diatomic_pairs(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
) -> Vec<DiatomicPairIntegrals> {
    let mut pairs = Vec::with_capacity((batch.natoms * (batch.natoms - 1)) / 2);

    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };
        let norb_a = batch.basis_types[i].num_orbitals();
        let orb_start_a = batch.orbital_offsets[i];
        let params_a = DerivedMultipoleParams::from_element(&p_a);

        for j in 0..i {
            let zb = batch.atomic_numbers[j];
            let p_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };
            let norb_b = batch.basis_types[j].num_orbitals();
            let orb_start_b = batch.orbital_offsets[j];
            let params_b = DerivedMultipoleParams::from_element(&p_b);

            let r_angstrom = batch.distance(i, j);
            if r_angstrom < 1e-10 {
                continue;
            }

            let dx = batch.x[j] - batch.x[i];
            let dy = batch.y[j] - batch.y[i];
            let dz = batch.z[j] - batch.z[i];

            let (mut ri, _gab) = compute_22_multipoles(&params_a, &params_b, r_angstrom);
            if model.use_feathering() {
                const TRUNC_1: f64 = 7.0;
                const TRUNC_2: f64 = 0.22;
                let (point, const_val) = if r_angstrom < TRUNC_1 {
                    let dr = r_angstrom - TRUNC_1;
                    let c = 1.0 - (-dr * dr * TRUNC_2).exp();
                    let pt = crate::constants::codata2018::EV_ANGSTROM_FACTOR / r_angstrom;
                    (pt, c)
                } else {
                    let pt = crate::constants::codata2018::EV_ANGSTROM_FACTOR / r_angstrom;
                    (pt, 0.0)
                };

                for (k, val) in ri.iter_mut().enumerate() {
                    match k {
                        0 | 2 | 3 | 10 | 11 | 15 | 16 | 17 | 18 | 20 => {
                            *val = *val * const_val + (1.0 - const_val) * point;
                        }
                        _ => {
                            *val *= const_val;
                        }
                    }
                }
            }

            let has_d_a = norb_a >= 9;
            let has_d_b = norb_b >= 9;

            let (w, e1b, e2a) = if has_d_a || has_d_b {
                let r_bohr = r_angstrom / A0_BOHR;
                let rot_d =
                    crate::integrals::d_orbitals::DOrbitalRotation3D::new(dx, dy, dz, r_angstrom);

                let (po_a, ddp_a) = match model.get_d_element_params(za) {
                    Some(dp) => (dp.po, dp.ddp),
                    None => {
                        let mut po = [0.0f64; 10];
                        let mut ddp = [0.0f64; 7];
                        po[1] = params_a.po[0];
                        po[2] = params_a.po[1];
                        po[3] = params_a.po[2];
                        po[7] = params_a.po[0];
                        po[9] = params_a.po[0];
                        ddp[2] = params_a.dd;
                        ddp[3] = params_a.qq * 2.0f64.sqrt();
                        (po, ddp)
                    }
                };
                let (po_b, ddp_b) = match model.get_d_element_params(zb) {
                    Some(dp) => (dp.po, dp.ddp),
                    None => {
                        let mut po = [0.0f64; 10];
                        let mut ddp = [0.0f64; 7];
                        po[1] = params_b.po[0];
                        po[2] = params_b.po[1];
                        po[3] = params_b.po[2];
                        po[7] = params_b.po[0];
                        po[9] = params_b.po[0];
                        ddp[2] = params_b.dd;
                        ddp[3] = params_b.qq * 2.0f64.sqrt();
                        (po, ddp)
                    }
                };

                let r2 = r_bohr * r_bohr;
                let ssi = (po_a[9] + po_b[1]).powi(2);
                let ssj = (po_b[9] + po_a[1]).powi(2);

                let mut cored = [[0.0f64; 2]; 10];
                cored[0][0] = -p_b.core_charge * EV_HARTREE / (r2 + ssj).sqrt();
                cored[0][1] = -p_a.core_charge * EV_HARTREE / (r2 + ssi).sqrt();

                const PXY: [f64; 7] = [1.0, -0.5, -0.5, 0.5, 0.25, 0.25, 0.5];

                if za >= 3 {
                    let ppj = (po_b[9] + po_a[7]).powi(2);
                    let da = ddp_a[2];
                    let qa = ddp_a[3] / 2.0f64.sqrt();
                    let twoqa = 2.0 * qa;
                    let adj = (po_a[2] + po_b[9]).powi(2);
                    let aqj = (po_a[3] + po_b[9]).powi(2);

                    let mut xj = [0.0f64; 7];
                    xj[0] = r2 + ppj;
                    xj[1] = r2 + aqj;
                    xj[2] = (r_bohr + da).powi(2) + adj;
                    xj[3] = (r_bohr - da).powi(2) + adj;
                    xj[4] = (r_bohr - twoqa).powi(2) + aqj;
                    xj[5] = (r_bohr + twoqa).powi(2) + aqj;
                    xj[6] = r2 + twoqa * twoqa + aqj;

                    let mut xj_term = [0.0f64; 7];
                    for k in 0..7 {
                        xj_term[k] = PXY[k] / xj[k].sqrt();
                    }

                    let aj2 = (xj_term[2] + xj_term[3]) * EV_HARTREE;
                    let aj3 = (xj_term[0] + xj_term[1] + xj_term[4] + xj_term[5]) * EV_HARTREE;
                    let aj4 = (xj_term[0] + xj_term[1] + xj_term[6]) * EV_HARTREE;

                    cored[1][0] = -p_b.core_charge * aj2;
                    cored[2][0] = -p_b.core_charge * aj3;
                    cored[3][0] = -p_b.core_charge * aj4;
                }

                if zb >= 3 {
                    let ppi = (po_a[9] + po_b[7]).powi(2);
                    let db = ddp_b[2];
                    let qb = ddp_b[3] / 2.0f64.sqrt();
                    let twoqb = 2.0 * qb;
                    let adi = (po_b[2] + po_a[9]).powi(2);
                    let aqi = (po_b[3] + po_a[9]).powi(2);

                    let mut xi = [0.0f64; 7];
                    xi[0] = r2 + ppi;
                    xi[1] = r2 + aqi;
                    xi[2] = (r_bohr + db).powi(2) + adi;
                    xi[3] = (r_bohr - db).powi(2) + adi;
                    xi[4] = (r_bohr - twoqb).powi(2) + aqi;
                    xi[5] = (r_bohr + twoqb).powi(2) + aqi;
                    xi[6] = r2 + twoqb * twoqb + aqi;

                    let mut xi_term = [0.0f64; 7];
                    for k in 0..7 {
                        xi_term[k] = PXY[k] / xi[k].sqrt();
                    }

                    let ai2 = -(xi_term[2] + xi_term[3]) * EV_HARTREE;
                    let ai3 = (xi_term[0] + xi_term[1] + xi_term[4] + xi_term[5]) * EV_HARTREE;
                    let ai4 = (xi_term[0] + xi_term[1] + xi_term[6]) * EV_HARTREE;

                    cored[1][1] = -p_a.core_charge * ai2;
                    cored[2][1] = -p_a.core_charge * ai3;
                    cored[3][1] = -p_a.core_charge * ai4;
                }

                let mut rep = [0.0f64; 492];
                crate::integrals::d_orbitals::compute_reppd2(
                    norb_a,
                    norb_b,
                    r_bohr,
                    &ri,
                    &po_a,
                    &ddp_a,
                    &po_b,
                    &ddp_b,
                    p_a.core_charge,
                    p_b.core_charge,
                    &mut rep,
                    &mut cored,
                    model.use_feathering(),
                );

                if model.use_feathering() {
                    let const_val = if r_angstrom < 7.0 {
                        let dr = r_angstrom - 7.0;
                        1.0 - (-dr * dr * 0.22).exp()
                    } else {
                        0.0
                    };

                    let point_b = -(EV_HARTREE / r_bohr) * p_b.core_charge;
                    cored[0][0] = cored[0][0] * const_val + (1.0 - const_val) * point_b;
                    cored[1][0] *= const_val;
                    cored[2][0] = cored[2][0] * const_val + (1.0 - const_val) * point_b;
                    cored[3][0] = cored[3][0] * const_val + (1.0 - const_val) * point_b;
                    cored[4][0] *= const_val;
                    cored[5][0] *= const_val;
                    cored[6][0] = cored[6][0] * const_val + (1.0 - const_val) * point_b;
                    cored[7][0] *= const_val;
                    cored[8][0] = cored[8][0] * const_val + (1.0 - const_val) * point_b;
                    cored[9][0] = cored[9][0] * const_val + (1.0 - const_val) * point_b;

                    let point_a = -(EV_HARTREE / r_bohr) * p_a.core_charge;
                    cored[0][1] = cored[0][1] * const_val + (1.0 - const_val) * point_a;
                    cored[1][1] *= const_val;
                    cored[2][1] = cored[2][1] * const_val + (1.0 - const_val) * point_a;
                    cored[3][1] = cored[3][1] * const_val + (1.0 - const_val) * point_a;
                    cored[4][1] *= const_val;
                    cored[5][1] *= const_val;
                    cored[6][1] = cored[6][1] * const_val + (1.0 - const_val) * point_a;
                    cored[7][1] *= const_val;
                    cored[8][1] = cored[8][1] * const_val + (1.0 - const_val) * point_a;
                    cored[9][1] = cored[9][1] * const_val + (1.0 - const_val) * point_a;
                }

                let mut logv = [[false; 45]; 45];
                let mut v = [[0.0f64; 45]; 45];
                crate::integrals::d_orbitals::tx(norb_a, norb_b, &rep, &mut logv, &mut v, &rot_d);

                let mut ww = [0.0f64; 2025];
                crate::integrals::d_orbitals::rotatd_step2(
                    norb_a, norb_b, &v, &logv, &rot_d, &mut ww,
                );

                if model.use_feathering() {
                    crate::integrals::d_orbitals::apply_pm7_d_correction(
                        norb_a, norb_b, has_d_a, has_d_b, &mut ww, &mut cored,
                    );
                }

                let limij = (norb_a * (norb_a + 1)) / 2;
                let limkl = (norb_b * (norb_b + 1)) / 2;
                let mut w = vec![0.0f64; limij * limkl];
                w.copy_from_slice(&ww[..limij * limkl]);

                let mut e1b = vec![0.0f64; limij];
                for i in 0..norb_a {
                    for j in 0..=i {
                        let val = if i == 0 && j == 0 {
                            cored[0][0]
                        } else if i < 4 && j == 0 {
                            rot_d.sp[0][i - 1] * cored[1][0]
                        } else if i < 4 && j < 4 {
                            let ipp = crate::integrals::d_orbitals::INDPP[i - 1][j - 1] - 1;
                            cored[2][0] * rot_d.pp[ipp][0][0]
                                + cored[3][0] * (rot_d.pp[ipp][1][1] + rot_d.pp[ipp][2][2])
                        } else if i >= 4 && j == 0 {
                            rot_d.sd[0][i - 4] * cored[4][0]
                        } else if i >= 4 && j < 4 {
                            let idp = crate::integrals::d_orbitals::INDDP[i - 4][j - 1] - 1;
                            cored[5][0] * rot_d.dp[idp][0][0]
                                + cored[7][0] * (rot_d.dp[idp][1][1] + rot_d.dp[idp][2][2])
                        } else {
                            let idd = crate::integrals::d_orbitals::INDDD[i - 4][j - 4] - 1;
                            cored[6][0] * rot_d.d_d[idd][0][0]
                                + cored[8][0] * (rot_d.d_d[idd][1][1] + rot_d.d_d[idd][2][2])
                                + cored[9][0] * (rot_d.d_d[idd][3][3] + rot_d.d_d[idd][4][4])
                        };
                        e1b[(i * (i + 1)) / 2 + j] = val;
                    }
                }

                let mut e2a = vec![0.0f64; limkl];
                for i in 0..norb_b {
                    for j in 0..=i {
                        let val = if i == 0 && j == 0 {
                            cored[0][1]
                        } else if i < 4 && j == 0 {
                            rot_d.sp[0][i - 1] * cored[1][1]
                        } else if i < 4 && j < 4 {
                            let ipp = crate::integrals::d_orbitals::INDPP[i - 1][j - 1] - 1;
                            cored[2][1] * rot_d.pp[ipp][0][0]
                                + cored[3][1] * (rot_d.pp[ipp][1][1] + rot_d.pp[ipp][2][2])
                        } else if i >= 4 && j == 0 {
                            rot_d.sd[0][i - 4] * cored[4][1]
                        } else if i >= 4 && j < 4 {
                            let idp = crate::integrals::d_orbitals::INDDP[i - 4][j - 1] - 1;
                            cored[5][1] * rot_d.dp[idp][0][0]
                                + cored[7][1] * (rot_d.dp[idp][1][1] + rot_d.dp[idp][2][2])
                        } else {
                            let idd = crate::integrals::d_orbitals::INDDD[i - 4][j - 4] - 1;
                            cored[6][1] * rot_d.d_d[idd][0][0]
                                + cored[8][1] * (rot_d.d_d[idd][1][1] + rot_d.d_d[idd][2][2])
                                + cored[9][1] * (rot_d.d_d[idd][3][3] + rot_d.d_d[idd][4][4])
                        };
                        e2a[(i * (i + 1)) / 2 + j] = val;
                    }
                }

                (w, e1b, e2a)
            } else {
                let rot = DiatomicRotation3D::new(dx, dy, dz, r_angstrom);
                let mut w = vec![0.0f64; 100];
                rotate_multipoles_to_w(norb_a, norb_b, &ri, &rot, &mut w);

                let mut e1b = vec![0.0f64; 10];
                let mut e2a = vec![0.0f64; 10];
                compute_electron_nuclear_attraction(
                    norb_a,
                    norb_b,
                    &params_a,
                    &params_b,
                    p_a.core_charge,
                    p_b.core_charge,
                    r_angstrom,
                    &rot,
                    &mut e1b,
                    &mut e2a,
                    model.use_feathering(),
                );
                (w, e1b, e2a)
            };

            pairs.push(DiatomicPairIntegrals {
                atom_a: i,
                atom_b: j,
                norb_a,
                norb_b,
                orb_start_a,
                orb_start_b,
                w,
                e1b,
                e2a,
            });
        }
    }

    pairs
}

#[allow(clippy::needless_range_loop)]
fn assemble_nddo_pair_dorbs(
    pair: &DiatomicPairIntegrals,
    p_coul: &AlignedMatrix<f64>,
    p_exch: &AlignedMatrix<f64>,
    scale_exch: f64,
    fock: &mut AlignedMatrix<f64>,
) {
    let ia = pair.orb_start_a;
    let ja = pair.orb_start_b;
    let na = pair.norb_a;
    let nb = pair.norb_b;
    let w = &pair.w;

    let mut delta_f_ab = [[0.0f64; 9]; 9];
    let mut kr = 0;

    for i in 0..na {
        for j in 0..=i {
            let aa = if i == j { 1.0 } else { 2.0 };
            for k in 0..nb {
                for l in 0..=k {
                    let bb = if k == l { 1.0 } else { 2.0 };
                    let a = w[kr];
                    kr += 1;

                    let p_kl = p_coul.get(ja + k, ja + l);
                    let p_ij = p_coul.get(ia + i, ia + j);
                    let term_a = bb * a * p_kl;
                    let term_b = aa * a * p_ij;

                    let cur_a = fock.get(ia + i, ia + j);
                    fock.set(ia + i, ia + j, cur_a + term_a);
                    if i != j {
                        fock.set(ia + j, ia + i, cur_a + term_a);
                    }

                    let cur_b = fock.get(ja + k, ja + l);
                    fock.set(ja + k, ja + l, cur_b + term_b);
                    if k != l {
                        fock.set(ja + l, ja + k, cur_b + term_b);
                    }

                    let exch_a = a * aa * bb * 0.25 * scale_exch;
                    delta_f_ab[i][k] -= exch_a * p_exch.get(ia + j, ja + l);
                    delta_f_ab[i][l] -= exch_a * p_exch.get(ia + j, ja + k);
                    delta_f_ab[j][k] -= exch_a * p_exch.get(ia + i, ja + l);
                    delta_f_ab[j][l] -= exch_a * p_exch.get(ia + i, ja + k);
                }
            }
        }
    }

    for i in 0..na {
        for k in 0..nb {
            let val = delta_f_ab[i][k];
            let cur = fock.get(ia + i, ja + k);
            fock.set(ia + i, ja + k, cur + val);
            fock.set(ja + k, ia + i, cur + val);
        }
    }
}

/// Assemble full two-center two-electron NDDO Coulomb and Exchange into the Fock matrix.
///
/// Direct port of OpenMOPAC `fock2.F90`.
pub fn assemble_nddo_two_center_fock(
    pairs: &[DiatomicPairIntegrals],
    density: &AlignedMatrix<f64>,
    fock: &mut AlignedMatrix<f64>,
) {
    for pair in pairs {
        let ia = pair.orb_start_a;
        let ja = pair.orb_start_b;
        let na = pair.norb_a;
        let nb = pair.norb_b;
        let w = &pair.w;

        if na >= 9 || nb >= 9 {
            assemble_nddo_pair_dorbs(pair, density, density, 0.5, fock);
        } else if na >= 4 && nb >= 4 {
            let mut pja = [0.0f64; 16];
            let mut pjb = [0.0f64; 16];
            let mut pk = [0.0f64; 16];

            for r in 0..4 {
                for c in 0..4 {
                    pja[r * 4 + c] = density.get(ia + r, ia + c);
                    pjb[r * 4 + c] = density.get(ja + r, ja + c);
                    pk[r * 4 + c] = 0.5 * density.get(ia + r, ja + c);
                }
            }

            let mut f_block_a = [0.0f64; 10];
            let mut f_block_b = [0.0f64; 10];
            contract_jab(&pja, &pjb, w, &mut f_block_a, &mut f_block_b);

            for r in 0..4 {
                for c in 0..=r {
                    let idx = (r * (r + 1)) / 2 + c;
                    let val_a = f_block_a[idx];
                    let cur_a = fock.get(ia + r, ia + c);
                    fock.set(ia + r, ia + c, cur_a + val_a);
                    if r != c {
                        fock.set(ia + c, ia + r, cur_a + val_a);
                    }

                    let val_b = f_block_b[idx];
                    let cur_b = fock.get(ja + r, ja + c);
                    fock.set(ja + r, ja + c, cur_b + val_b);
                    if r != c {
                        fock.set(ja + c, ja + r, cur_b + val_b);
                    }
                }
            }

            let mut f_ab = [0.0f64; 16];
            contract_kab(&pk, w, &mut f_ab);

            for r in 0..4 {
                for c in 0..4 {
                    let val = f_ab[r * 4 + c];
                    let cur = fock.get(ia + r, ja + c);
                    fock.set(ia + r, ja + c, cur + val);
                    fock.set(ja + c, ia + r, cur + val);
                }
            }
        } else if na >= 4 && nb == 1 {
            let p_b = density.get(ja, ja);
            for r in 0..4 {
                for c in 0..=r {
                    let idx = (r * (r + 1)) / 2 + c;
                    let val = p_b * w[idx];
                    let cur = fock.get(ia + r, ia + c);
                    fock.set(ia + r, ia + c, cur + val);
                    if r != c {
                        fock.set(ia + c, ia + r, cur + val);
                    }
                }
            }

            let mut sumdia = 0.0;
            let mut sumoff = 0.0;
            for r in 0..4 {
                let idx_dia = (r * (r + 1)) / 2 + r;
                sumdia += density.get(ia + r, ia + r) * w[idx_dia];
                for c in 0..r {
                    let idx_off = (r * (r + 1)) / 2 + c;
                    sumoff += density.get(ia + r, ia + c) * w[idx_off];
                }
            }
            let cur_b = fock.get(ja, ja);
            fock.set(ja, ja, cur_b + sumdia + 2.0 * sumoff);

            for r in 0..4 {
                let mut s = 0.0;
                for c in 0..4 {
                    let idx = if r >= c {
                        (r * (r + 1)) / 2 + c
                    } else {
                        (c * (c + 1)) / 2 + r
                    };
                    s += density.get(ia + c, ja) * w[idx];
                }
                let cur = fock.get(ia + r, ja);
                fock.set(ia + r, ja, cur - 0.5 * s);
                fock.set(ja, ia + r, cur - 0.5 * s);
            }
        } else if na == 1 && nb >= 4 {
            let p_a = density.get(ia, ia);
            for r in 0..4 {
                for c in 0..=r {
                    let idx = (r * (r + 1)) / 2 + c;
                    let val = p_a * w[idx];
                    let cur = fock.get(ja + r, ja + c);
                    fock.set(ja + r, ja + c, cur + val);
                    if r != c {
                        fock.set(ja + c, ja + r, cur + val);
                    }
                }
            }

            let mut sumdia = 0.0;
            let mut sumoff = 0.0;
            for r in 0..4 {
                let idx_dia = (r * (r + 1)) / 2 + r;
                sumdia += density.get(ja + r, ja + r) * w[idx_dia];
                for c in 0..r {
                    let idx_off = (r * (r + 1)) / 2 + c;
                    sumoff += density.get(ja + r, ja + c) * w[idx_off];
                }
            }
            let cur_a = fock.get(ia, ia);
            fock.set(ia, ia, cur_a + sumdia + 2.0 * sumoff);

            for r in 0..4 {
                let mut s = 0.0;
                for c in 0..4 {
                    let idx = if r >= c {
                        (r * (r + 1)) / 2 + c
                    } else {
                        (c * (c + 1)) / 2 + r
                    };
                    s += density.get(ia, ja + c) * w[idx];
                }
                let cur = fock.get(ia, ja + r);
                fock.set(ia, ja + r, cur - 0.5 * s);
                fock.set(ja + r, ia, cur - 0.5 * s);
            }
        } else if na == 1 && nb == 1 {
            let w0 = w[0];
            let p_a = density.get(ia, ia);
            let p_b = density.get(ja, ja);
            let p_ab = density.get(ia, ja);

            let cur_a = fock.get(ia, ia);
            fock.set(ia, ia, cur_a + p_b * w0);
            let cur_b = fock.get(ja, ja);
            fock.set(ja, ja, cur_b + p_a * w0);

            let cur_ab = fock.get(ia, ja);
            fock.set(ia, ja, cur_ab - 0.5 * p_ab * w0);
            fock.set(ja, ia, cur_ab - 0.5 * p_ab * w0);
        }
    }
}

/// Assemble full two-center two-electron NDDO Coulomb and Exchange into the Fock matrix for UHF.
///
/// Uses total density `p_tot` ($P^\alpha + P^\beta$) for Coulomb interactions and spin-channel
/// density `p_spin` ($P^\alpha$ or $P^\beta$) for exchange interactions, matching OpenMOPAC `fock2.F90`.
pub fn assemble_nddo_two_center_fock_spin(
    pairs: &[DiatomicPairIntegrals],
    p_tot: &AlignedMatrix<f64>,
    p_spin: &AlignedMatrix<f64>,
    fock: &mut AlignedMatrix<f64>,
) {
    for pair in pairs {
        let ia = pair.orb_start_a;
        let ja = pair.orb_start_b;
        let na = pair.norb_a;
        let nb = pair.norb_b;
        let w = &pair.w;

        if na >= 9 || nb >= 9 {
            assemble_nddo_pair_dorbs(pair, p_tot, p_spin, 1.0, fock);
        } else if na >= 4 && nb >= 4 {
            let mut pja = [0.0f64; 16];
            let mut pjb = [0.0f64; 16];
            let mut pk = [0.0f64; 16];

            for r in 0..4 {
                for c in 0..4 {
                    pja[r * 4 + c] = p_tot.get(ia + r, ia + c);
                    pjb[r * 4 + c] = p_tot.get(ja + r, ja + c);
                    pk[r * 4 + c] = p_spin.get(ia + r, ja + c);
                }
            }

            let mut f_block_a = [0.0f64; 10];
            let mut f_block_b = [0.0f64; 10];
            contract_jab(&pja, &pjb, w, &mut f_block_a, &mut f_block_b);

            for r in 0..4 {
                for c in 0..=r {
                    let idx = (r * (r + 1)) / 2 + c;
                    let val_a = f_block_a[idx];
                    let cur_a = fock.get(ia + r, ia + c);
                    fock.set(ia + r, ia + c, cur_a + val_a);
                    if r != c {
                        fock.set(ia + c, ia + r, cur_a + val_a);
                    }

                    let val_b = f_block_b[idx];
                    let cur_b = fock.get(ja + r, ja + c);
                    fock.set(ja + r, ja + c, cur_b + val_b);
                    if r != c {
                        fock.set(ja + c, ja + r, cur_b + val_b);
                    }
                }
            }

            let mut f_ab = [0.0f64; 16];
            contract_kab(&pk, w, &mut f_ab);

            for r in 0..4 {
                for c in 0..4 {
                    let val = f_ab[r * 4 + c];
                    let cur = fock.get(ia + r, ja + c);
                    fock.set(ia + r, ja + c, cur + val);
                    fock.set(ja + c, ia + r, cur + val);
                }
            }
        } else if na >= 4 && nb == 1 {
            let p_b = p_tot.get(ja, ja);
            for r in 0..4 {
                for c in 0..=r {
                    let idx = (r * (r + 1)) / 2 + c;
                    let val = p_b * w[idx];
                    let cur = fock.get(ia + r, ia + c);
                    fock.set(ia + r, ia + c, cur + val);
                    if r != c {
                        fock.set(ia + c, ia + r, cur + val);
                    }
                }
            }

            let mut sumdia = 0.0;
            let mut sumoff = 0.0;
            for r in 0..4 {
                let idx_dia = (r * (r + 1)) / 2 + r;
                sumdia += p_tot.get(ia + r, ia + r) * w[idx_dia];
                for c in 0..r {
                    let idx_off = (r * (r + 1)) / 2 + c;
                    sumoff += p_tot.get(ia + r, ia + c) * w[idx_off];
                }
            }
            let cur_b = fock.get(ja, ja);
            fock.set(ja, ja, cur_b + sumdia + 2.0 * sumoff);

            for r in 0..4 {
                let mut s = 0.0;
                for c in 0..4 {
                    let idx = if r >= c {
                        (r * (r + 1)) / 2 + c
                    } else {
                        (c * (c + 1)) / 2 + r
                    };
                    s += p_spin.get(ia + c, ja) * w[idx];
                }
                let cur = fock.get(ia + r, ja);
                fock.set(ia + r, ja, cur - s);
                fock.set(ja, ia + r, cur - s);
            }
        } else if na == 1 && nb >= 4 {
            let p_a = p_tot.get(ia, ia);
            for r in 0..4 {
                for c in 0..=r {
                    let idx = (r * (r + 1)) / 2 + c;
                    let val = p_a * w[idx];
                    let cur = fock.get(ja + r, ja + c);
                    fock.set(ja + r, ja + c, cur + val);
                    if r != c {
                        fock.set(ja + c, ja + r, cur + val);
                    }
                }
            }

            let mut sumdia = 0.0;
            let mut sumoff = 0.0;
            for r in 0..4 {
                let idx_dia = (r * (r + 1)) / 2 + r;
                sumdia += p_tot.get(ja + r, ja + r) * w[idx_dia];
                for c in 0..r {
                    let idx_off = (r * (r + 1)) / 2 + c;
                    sumoff += p_tot.get(ja + r, ja + c) * w[idx_off];
                }
            }
            let cur_a = fock.get(ia, ia);
            fock.set(ia, ia, cur_a + sumdia + 2.0 * sumoff);

            for r in 0..4 {
                let mut s = 0.0;
                for c in 0..4 {
                    let idx = if r >= c {
                        (r * (r + 1)) / 2 + c
                    } else {
                        (c * (c + 1)) / 2 + r
                    };
                    s += p_spin.get(ia, ja + c) * w[idx];
                }
                let cur = fock.get(ia, ja + r);
                fock.set(ia, ja + r, cur - s);
                fock.set(ja + r, ia, cur - s);
            }
        } else if na == 1 && nb == 1 {
            let w0 = w[0];
            let p_a = p_tot.get(ia, ia);
            let p_b = p_tot.get(ja, ja);
            let p_ab = p_spin.get(ia, ja);

            let cur_a = fock.get(ia, ia);
            fock.set(ia, ia, cur_a + p_b * w0);
            let cur_b = fock.get(ja, ja);
            fock.set(ja, ja, cur_b + p_a * w0);

            let cur_ab = fock.get(ia, ja);
            fock.set(ia, ja, cur_ab - p_ab * w0);
            fock.set(ja, ia, cur_ab - p_ab * w0);
        }
    }
}

/// Apply rotated electron-nuclear attractions $E_{1B}$ and $E_{2A}$ to $H^{\text{core}}$.
///
/// Direct port of OpenMOPAC `hcore.F90` lines 270-300.
pub fn apply_electron_nuclear_attractions(
    pairs: &[DiatomicPairIntegrals],
    h_core: &mut AlignedMatrix<f64>,
) {
    for pair in pairs {
        let ia = pair.orb_start_a;
        let ja = pair.orb_start_b;
        let na = pair.norb_a;
        let nb = pair.norb_b;

        for r in 0..na {
            for c in 0..=r {
                let idx = (r * (r + 1)) / 2 + c;
                let val = pair.e1b[idx];
                let cur = h_core.get(ia + r, ia + c);
                h_core.set(ia + r, ia + c, cur + val);
                if r != c {
                    h_core.set(ia + c, ia + r, cur + val);
                }
            }
        }

        for r in 0..nb {
            for c in 0..=r {
                let idx = (r * (r + 1)) / 2 + c;
                let val = pair.e2a[idx];
                let cur = h_core.get(ja + r, ja + c);
                h_core.set(ja + r, ja + c, cur + val);
                if r != c {
                    h_core.set(ja + c, ja + r, cur + val);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::am1::Am1Model;
    use crate::parameters::ParameterModel;

    #[test]
    fn test_derived_multipole_params_am1() {
        let model = Am1Model;
        let h_param = model.get_element(1).unwrap();
        let c_param = model.get_element(6).unwrap();

        let h_derived = DerivedMultipoleParams::from_element(&h_param);
        assert_eq!(h_derived.dd, 0.0);
        assert_eq!(h_derived.qq, 0.0);
        assert!((h_derived.am - h_param.gss / EV_HARTREE).abs() < 1e-12);

        let c_derived = DerivedMultipoleParams::from_element(&c_param);
        assert!(
            c_derived.dd > 0.0,
            "Carbon dipole separation must be positive"
        );
        assert!(
            c_derived.qq > 0.0,
            "Carbon quadrupole separation must be positive"
        );
        assert!(c_derived.ad > 0.0, "Carbon Ad parameter must be positive");
        assert!(c_derived.aq > 0.0, "Carbon Aq parameter must be positive");
    }

    #[test]
    fn test_22_multipoles_asymptotics() {
        let model = Am1Model;
        let h_param = model.get_element(1).unwrap();
        let c_param = model.get_element(6).unwrap();

        let h_derived = DerivedMultipoleParams::from_element(&h_param);
        let c_derived = DerivedMultipoleParams::from_element(&c_param);

        // At R = 1.0 A
        let (ri_hh, gab_hh) = compute_22_multipoles(&h_derived, &h_derived, 1.0);
        assert!(ri_hh[0] > 0.0);
        assert!(gab_hh > 0.0);

        // At large R (e.g. 100.0 A), (ss|ss) should match e^2 / R = 14.399645 / 100 = 0.143996 eV
        let (ri_large, _) = compute_22_multipoles(&c_derived, &c_derived, 100.0);
        let expected_coulomb = 14.399645478456 / 100.0;
        assert!(
            (ri_large[0] - expected_coulomb).abs() < 1e-4,
            "ri[0] at 100 A should approach 1/R: got {}, expected {}",
            ri_large[0],
            expected_coulomb
        );
        // All dipole and quadrupole terms should decay much faster than monopole (at least ~ 1/R^2 or 1/R^3)
        assert!(ri_large[1].abs() < 1e-3); // (so|ss) dipole
        assert!(ri_large[5].abs() < 1e-4); // (so|so) dipole-dipole
    }

    #[test]
    fn test_rotation_matrix_orthonormality() {
        let rot = DiatomicRotation3D::new(
            0.5,
            0.7,
            0.9,
            (0.5f64.powi(2) + 0.7f64.powi(2) + 0.9f64.powi(2)).sqrt(),
        );
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += rot.p[i][k] * rot.p[j][k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "Row dot product failed for {} {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_precompute_and_assemble_nddo_water() {
        let model = Am1Model;
        // Water molecule: O at (0, 0, 0), H1 at (0.757, 0.586, 0), H2 at (-0.757, 0.586, 0)
        let coords = vec![[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]];
        let batch = MolecularBatch::new(vec![8, 1, 1], &coords);
        assert_eq!(batch.norbs, 6); // O: 4, H: 1, H: 1

        let pairs = precompute_diatomic_pairs(&batch, &model);
        assert_eq!(pairs.len(), 3); // (O, H1), (O, H2), (H1, H2)

        // Test electron-nuclear attraction assembly
        let mut h_core = AlignedMatrix::zeroed(batch.norbs, batch.norbs);
        apply_electron_nuclear_attractions(&pairs, &mut h_core);
        // H_core must be strictly symmetric: H_ij == H_ji
        for i in 0..batch.norbs {
            for j in 0..batch.norbs {
                assert!(
                    (h_core.get(i, j) - h_core.get(j, i)).abs() < 1e-14,
                    "H_core symmetry violated at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Test Fock matrix assembly from idempotent identity-like density
        let mut density = AlignedMatrix::zeroed(batch.norbs, batch.norbs);
        for i in 0..batch.norbs {
            density.set(i, i, 1.333); // total valence population
        }
        let mut fock = AlignedMatrix::zeroed(batch.norbs, batch.norbs);
        assemble_nddo_two_center_fock(&pairs, &density, &mut fock);

        // Fock must be strictly symmetric
        for i in 0..batch.norbs {
            for j in 0..batch.norbs {
                assert!(
                    (fock.get(i, j) - fock.get(j, i)).abs() < 1e-14,
                    "Fock symmetry violated at ({}, {})",
                    i,
                    j
                );
            }
        }
        println!(
            "[OK] NDDO Precomputation & Fock Assembly verified with strict Hermiticity on H2O!"
        );
    }
}
