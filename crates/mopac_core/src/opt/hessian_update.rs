//! Hessian Matrix Update Formulas for Transition State & Minima Geometry Optimization.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Provides rank-1 and rank-2 quasi-Newton Hessian updates matching OpenMOPAC `updhes` (ef.F90):
//! * Powell Symmetric Dual (PSB / Powell) - standard for transition state searches.
//! * Murtagh-Sargent (SR1) - symmetric rank-one update.
//! * Bofill - optimal convex combination of SR1 and PSB for transition state searches.
//! * BFGS - Broyden-Fletcher-Goldfarb-Shanno for local minima searches.

use crate::types::AlignedMatrix;

/// Hessian update schemes supported by the optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianUpdateScheme {
    /// Powell Symmetric Dual (PSB / Powell) update (OpenMOPAC iupd = 1).
    Powell,
    /// Bofill update: linear combination of Murtagh-Sargent and Powell.
    Bofill,
    /// Murtagh-Sargent (SR1) symmetric rank-one update.
    MurtaghSargent,
    /// Broyden-Fletcher-Goldfarb-Shanno update (OpenMOPAC iupd = 2).
    Bfgs,
}

/// Preallocated workspace for Hessian updates ensuring zero heap allocations in iterative cycles.
#[derive(Debug, Clone)]
pub struct HessianUpdateWorkspace {
    /// Product vector H * dx of dimension N
    pub h_dx: Vec<f64>,
    /// Secant defect vector xi = dg - H * dx of dimension N
    pub xi: Vec<f64>,
}

impl HessianUpdateWorkspace {
    /// Allocate workspace for dimension `n`.
    pub fn allocate(n: usize) -> Self {
        Self {
            h_dx: vec![0.0; n],
            xi: vec![0.0; n],
        }
    }

    /// Resize workspace if needed.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.h_dx.len() < n {
            self.h_dx.resize(n, 0.0);
            self.xi.resize(n, 0.0);
        }
    }
}

/// Perform in-place Hessian update: $H \leftarrow H + \Delta H$.
///
/// Modifies `hessian` in-place using coordinate displacement `dx` and gradient change `dg`.
///
/// # Invariants
/// * Preserves exact symmetry $H_{ij} = H_{ji}$.
/// * Satisfies secant condition $H_{k+1} \Delta x = \Delta g$ (to machine precision for rank-1/rank-2 updates).
/// * Zero dynamic heap allocations when reusing `ws`.
#[allow(clippy::needless_range_loop)]
pub fn update_cartesian_hessian(
    hessian: &mut AlignedMatrix<f64>,
    dx: &[f64],
    dg: &[f64],
    scheme: HessianUpdateScheme,
    ws: &mut HessianUpdateWorkspace,
) {
    let n = hessian.rows;
    assert_eq!(hessian.cols, n);
    assert_eq!(dx.len(), n);
    assert_eq!(dg.len(), n);
    ws.ensure_capacity(n);

    // Compute norm squared of displacement: dds = dx^T dx
    let mut dds = 0.0;
    for i in 0..n {
        dds += dx[i] * dx[i];
    }
    if dds < 1e-24 {
        // Step too small; skip update to prevent numerical blowup
        return;
    }

    // Compute H * dx: ws.h_dx = H * dx
    for i in 0..n {
        let mut sum = 0.0;
        let row_i = hessian.row(i);
        for j in 0..n {
            sum += row_i[j] * dx[j];
        }
        ws.h_dx[i] = sum;
    }

    // Compute secant defect: xi = dg - H * dx
    let mut xi_dot_dx = 0.0;
    let mut xi_norm_sq = 0.0;
    for i in 0..n {
        let diff = dg[i] - ws.h_dx[i];
        ws.xi[i] = diff;
        xi_dot_dx += diff * dx[i];
        xi_norm_sq += diff * diff;
    }

    match scheme {
        HessianUpdateScheme::Powell => {
            // Powell Symmetric Dual (PSB) update:
            // Delta H = (xi * dx^T + dx * xi^T) / dds - (xi^T dx) * (dx * dx^T) / (dds^2)
            let ddtd = xi_dot_dx / dds;
            for i in 0..n {
                let dxi = dx[i];
                let xii = ws.xi[i];
                // Diagonal element
                let diag_term = (dxi * (2.0 * xii - dxi * ddtd)) / dds;
                hessian.add(i, i, diag_term);

                // Off-diagonal elements (symmetric)
                for j in (i + 1)..n {
                    let dxj = dx[j];
                    let xij = ws.xi[j];
                    let term = (xii * dxj + dxi * xij - dxi * ddtd * dxj) / dds;
                    hessian.add(i, j, term);
                    hessian.set(j, i, hessian.get(i, j));
                }
            }
        }
        HessianUpdateScheme::MurtaghSargent => {
            // Symmetric Rank-One (SR1) / Murtagh-Sargent update:
            // Delta H = (xi * xi^T) / (xi^T dx)
            if xi_dot_dx.abs() > 1e-14 * dds.sqrt() * xi_norm_sq.sqrt() {
                let inv_denom = 1.0 / xi_dot_dx;
                for i in 0..n {
                    let xii = ws.xi[i];
                    hessian.add(i, i, xii * xii * inv_denom);
                    for j in (i + 1)..n {
                        let term = xii * ws.xi[j] * inv_denom;
                        hessian.add(i, j, term);
                        hessian.set(j, i, hessian.get(i, j));
                    }
                }
            }
        }
        HessianUpdateScheme::Bofill => {
            // Bofill update:
            // phi = 1 - (xi^T dx)^2 / ( (dx^T dx) * (xi^T xi) )
            // Delta H = (1 - phi) * Delta H_MS + phi * Delta H_PSB
            if xi_norm_sq < 1e-24 {
                return;
            }
            let cos_sq = (xi_dot_dx * xi_dot_dx) / (dds * xi_norm_sq);
            let phi = (1.0 - cos_sq).clamp(0.0, 1.0);

            let ddtd = xi_dot_dx / dds;
            let inv_denom_sr1 = if xi_dot_dx.abs() > 1e-14 * dds.sqrt() * xi_norm_sq.sqrt() {
                Some(1.0 / xi_dot_dx)
            } else {
                None
            };

            for i in 0..n {
                let dxi = dx[i];
                let xii = ws.xi[i];

                // Powell contribution
                let psb_diag = (dxi * (2.0 * xii - dxi * ddtd)) / dds;
                let ms_diag = match inv_denom_sr1 {
                    Some(inv) => xii * xii * inv,
                    None => psb_diag,
                };
                let diag_term = (1.0 - phi) * ms_diag + phi * psb_diag;
                hessian.add(i, i, diag_term);

                for j in (i + 1)..n {
                    let dxj = dx[j];
                    let xij = ws.xi[j];

                    let psb_term = (xii * dxj + dxi * xij - dxi * ddtd * dxj) / dds;
                    let ms_term = match inv_denom_sr1 {
                        Some(inv) => xii * xij * inv,
                        None => psb_term,
                    };
                    let term = (1.0 - phi) * ms_term + phi * psb_term;
                    hessian.add(i, j, term);
                    hessian.set(j, i, hessian.get(i, j));
                }
            }
        }
        HessianUpdateScheme::Bfgs => {
            // BFGS update:
            // Delta H = (dg * dg^T) / (dg^T dx) - (H dx * (H dx)^T) / (dx^T H dx)
            let mut dg_dot_dx = 0.0;
            let mut dx_dot_h_dx = 0.0;
            for i in 0..n {
                dg_dot_dx += dg[i] * dx[i];
                dx_dot_h_dx += dx[i] * ws.h_dx[i];
            }

            if dg_dot_dx.abs() > 1e-20 && dx_dot_h_dx.abs() > 1e-20 {
                let inv_dg_dx = 1.0 / dg_dot_dx;
                let inv_dx_hdx = 1.0 / dx_dot_h_dx;

                for i in 0..n {
                    let dgi = dg[i];
                    let h_dxi = ws.h_dx[i];
                    let diag_term = dgi * dgi * inv_dg_dx - h_dxi * h_dxi * inv_dx_hdx;
                    hessian.add(i, i, diag_term);

                    for j in (i + 1)..n {
                        let term = (dgi * dg[j]) * inv_dg_dx - (h_dxi * ws.h_dx[j]) * inv_dx_hdx;
                        hessian.add(i, j, term);
                        hessian.set(j, i, hessian.get(i, j));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn test_powell_update_secant_condition_and_symmetry() {
        let n = 3;
        let mut h = AlignedMatrix::zeroed(n, n);
        // Initial diagonal Hessian
        h.set(0, 0, 2.0);
        h.set(1, 1, 3.0);
        h.set(2, 2, 4.0);

        let dx = vec![0.1, -0.05, 0.2];
        let dg = vec![0.35, -0.12, 0.95];

        let mut ws = HessianUpdateWorkspace::allocate(n);
        update_cartesian_hessian(&mut h, &dx, &dg, HessianUpdateScheme::Powell, &mut ws);

        // 1. Verify exact symmetry: H_ij = H_ji
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (h.get(i, j) - h.get(j, i)).abs() < 1e-14,
                    "Symmetry violated at ({}, {})",
                    i,
                    j
                );
            }
        }

        // 2. Verify secant condition: H * dx = dg
        for i in 0..n {
            let mut h_dx_i = 0.0;
            for j in 0..n {
                h_dx_i += h.get(i, j) * dx[j];
            }
            assert!(
                (h_dx_i - dg[i]).abs() < 1e-12,
                "Secant condition violated at row {}: got {}, expected {}",
                i,
                h_dx_i,
                dg[i]
            );
        }
    }

    #[test]
    fn test_bofill_update_secant_condition_and_interpolation() {
        let n = 4;
        let mut h = AlignedMatrix::zeroed(n, n);
        for i in 0..n {
            h.set(i, i, 1.0 + i as f64);
        }

        let dx = vec![0.05, -0.02, 0.08, -0.04];
        let dg = vec![0.15, -0.08, 0.32, -0.19];

        let mut ws = HessianUpdateWorkspace::allocate(n);
        update_cartesian_hessian(&mut h, &dx, &dg, HessianUpdateScheme::Bofill, &mut ws);

        // 1. Symmetry check
        for i in 0..n {
            for j in 0..n {
                assert!((h.get(i, j) - h.get(j, i)).abs() < 1e-14);
            }
        }

        // 2. Secant condition check
        for i in 0..n {
            let mut h_dx_i = 0.0;
            for j in 0..n {
                h_dx_i += h.get(i, j) * dx[j];
            }
            assert!(
                (h_dx_i - dg[i]).abs() < 1e-11,
                "Secant condition failed in Bofill: got {}, expected {}",
                h_dx_i,
                dg[i]
            );
        }
    }

    #[test]
    fn test_bfgs_update_secant_condition() {
        let n = 3;
        let mut h = AlignedMatrix::zeroed(n, n);
        h.set(0, 0, 5.0);
        h.set(1, 1, 5.0);
        h.set(2, 2, 5.0);

        let dx = vec![0.1, 0.1, 0.1];
        // Ensure dg^T dx > 0 for positive definiteness
        let dg = vec![0.6, 0.7, 0.8];

        let mut ws = HessianUpdateWorkspace::allocate(n);
        update_cartesian_hessian(&mut h, &dx, &dg, HessianUpdateScheme::Bfgs, &mut ws);

        // 1. Symmetry
        for i in 0..n {
            for j in 0..n {
                assert!((h.get(i, j) - h.get(j, i)).abs() < 1e-14);
            }
        }

        // 2. Secant condition
        for i in 0..n {
            let mut h_dx_i = 0.0;
            for j in 0..n {
                h_dx_i += h.get(i, j) * dx[j];
            }
            assert!(
                (h_dx_i - dg[i]).abs() < 1e-12,
                "Secant condition failed in BFGS"
            );
        }
    }
}
