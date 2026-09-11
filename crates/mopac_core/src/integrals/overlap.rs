//! Analytical Slater-Type Orbital (STO) Overlap Integrals.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Evaluates $S_{\mu\nu}(R) = \int \phi_\mu^*(\vec{r}) \phi_\nu(\vec{r}) d^3r$ via auxiliary integrals.

use crate::constants::angstrom_to_bohr;

/// Auxiliary integral $A_k(p) = \int_1^\infty x^k e^{-p x} dx$.
///
/// Evaluated via recurrence relation:
/// $A_0(p) = \frac{e^{-p}}{p}$
/// $A_k(p) = \frac{e^{-p} + k A_{k-1}(p)}{p}$
#[inline(always)]
pub fn aux_a(k: usize, p: f64) -> f64 {
    if p < 1e-12 {
        return 1.0 / (k as f64 + 1.0);
    }
    let exp_neg_p = (-p).exp();
    let mut a = exp_neg_p / p;
    for i in 1..=k {
        a = (exp_neg_p + (i as f64) * a) / p;
    }
    a
}

/// Auxiliary integral $B_k(\alpha) = \int_{-1}^1 x^k e^{-\alpha x} dx$.
///
/// Evaluated via expansion or recurrence:
/// $B_0(\alpha) = \frac{e^\alpha - e^{-\alpha}}{\alpha} = \frac{2 \sinh(\alpha)}{\alpha}$
#[inline(always)]
pub fn aux_b(k: usize, alpha: f64) -> f64 {
    if alpha.abs() < 1e-8 {
        // Taylor series for small alpha: \int_{-1}^1 x^k (1 - alpha*x + alpha^2*x^2/2 - ...) dx
        return if k % 2 == 0 {
            2.0 / (k as f64 + 1.0)
        } else {
            -2.0 * alpha / (k as f64 + 2.0)
        };
    }
    let e_pos = alpha.exp();
    let e_neg = (-alpha).exp();
    let mut b = (e_pos - e_neg) / alpha;
    if k == 0 {
        return b;
    }
    for i in 1..=k {
        let term = if i % 2 == 0 {
            e_pos - e_neg
        } else {
            -e_pos - e_neg
        };
        b = -(term - (i as f64) * b) / alpha;
    }
    b
}

/// Diatomic overlap between two $1s$ Slater-type orbitals with exponents $\zeta_1, \zeta_2$ at separation $R$ in Ångströms.
pub fn overlap_1s_1s(r_angstrom: f64, zeta1: f64, zeta2: f64) -> f64 {
    let r_bohr = angstrom_to_bohr(r_angstrom, true);
    if r_bohr < 1e-12 {
        // One-center limit: orthogonal if different, 1.0 if identical exponents
        return if (zeta1 - zeta2).abs() < 1e-12 {
            1.0
        } else {
            let p1 = zeta1.powf(1.5);
            let p2 = zeta2.powf(1.5);
            8.0 * (p1 * p2) / (zeta1 + zeta2).powi(3)
        };
    }

    let p = 0.5 * (zeta1 + zeta2) * r_bohr;
    let t = (zeta1 - zeta2) / (zeta1 + zeta2);
    let alpha = p * t;

    // S(1s, 1s) in terms of A and B functions in spheroidal coordinates:
    // S = (p^3 / 4) * (1 - t^2)^{3/2} * [ A_2(p) B_0(alpha) - A_0(p) B_2(alpha) ]
    let prefactor = 0.25 * (p * p * p) * (1.0 - t * t).powf(1.5);
    let a0 = aux_a(0, p);
    let a2 = aux_a(2, p);
    let b0 = aux_b(0, alpha);
    let b2 = aux_b(2, alpha);

    prefactor * (a2 * b0 - a0 * b2)
}
