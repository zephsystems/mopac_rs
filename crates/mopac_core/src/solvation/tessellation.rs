//! Regular Icosahedral Sphere Tessellation for COSMO Cavity Generation.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct translation of OpenMOPAC v23.2.5 `dvfill` subroutine in `cosmo.F90`.
//!
//! Generates regular icosahedron-derived polyhedral grids on the unit sphere with:
//! - $N = 12$ points (used for Hydrogen atoms)
//! - $N = 42$ points (standard segments for non-Hydrogen atoms)
//! - $N = 1082$ points (fine numerical integration and SAS intersection grid)
//!
//! Each grid point provides a 3D unit normal vector $\hat{u}_i$ and a surface area weight $w_i$
//! such that $\sum_{i=1}^N w_i \equiv 4\pi$.

use std::f64::consts::PI;

/// A directional point on the unit sphere with its associated surface area weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpherePoint {
    pub dir: [f64; 3],
    pub weight: f64,
}

/// The 30 edges of the base icosahedron (1-indexed in OpenMOPAC `kset`).
const KSET: [[usize; 2]; 30] = [
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
    [11, 10], [11, 9], [11, 8], [11, 7], [11, 6],
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 1],
    [6, 7], [7, 8], [8, 9], [9, 10], [10, 6],
    [1, 6], [6, 2], [2, 7], [7, 3], [3, 8],
    [8, 4], [4, 9], [9, 5], [5, 10], [10, 1],
];

/// The 20 triangular faces of the base icosahedron (1-indexed in OpenMOPAC `fset`).
const FSET: [[usize; 3]; 20] = [
    [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
    [11, 10, 9], [11, 9, 8], [11, 8, 7], [11, 7, 6], [11, 6, 10],
    [1, 2, 6], [2, 3, 7], [3, 4, 8], [4, 5, 9], [5, 1, 10],
    [6, 7, 2], [7, 8, 3], [8, 9, 4], [9, 10, 5], [10, 6, 1],
];

/// Generate icosahedral sphere tessellation of size `nppa`.
///
/// Supported canonical sizes: 12, 42, 1082.
#[allow(clippy::needless_range_loop)]
pub fn generate_sphere_tessellation(nppa: usize) -> Vec<SpherePoint> {
    assert!(
        nppa == 12 || nppa == 42 || nppa == 1082,
        "nppa must be 12, 42, or 1082"
    );

    let mut dirvec = vec![[0.0f64; 4]; nppa];

    // 1. Initialize the 12 base vertices of the icosahedron
    dirvec[0][0] = -1.0;
    dirvec[0][1] = 0.0;
    dirvec[0][2] = 0.0;

    let mut nd = 0;
    let r = (0.8f64).sqrt();
    let h = (0.2f64).sqrt();

    for &i_sign in &[-1.0f64, 1.0f64] {
        for j in 1..=5 {
            nd += 1;
            let beta = 1.0 + (j as f64) * 0.4 * PI + (i_sign + 1.0) * 0.1 * PI;
            dirvec[nd][0] = i_sign * h;
            dirvec[nd][1] = r * beta.cos();
            dirvec[nd][2] = r * beta.sin();
        }
    }

    dirvec[11][0] = 1.0;
    dirvec[11][1] = 0.0;
    dirvec[11][2] = 0.0;

    // Apply rotation around z-axis by 1.0 radian matching Fortran dvfill
    let cphi = (1.0f64).cos();
    let sphi = (1.0f64).sin();
    for i in 0..12 {
        let xx = dirvec[i][0];
        let yy = dirvec[i][1];
        dirvec[i][0] = cphi * xx + sphi * yy;
        dirvec[i][1] = -sphi * xx + cphi * yy;
    }

    // 2. Subdivide edges and faces if nppa > 12
    if nppa > 12 {
        nd = 11;
        let mut m2 = (nppa - 2) / 10;
        let mut m = (m2 as f64).sqrt().round() as usize;
        let mut k = 0;
        if m2 != m * m {
            k = 1;
            m2 /= 3;
            m = (m2 as f64).sqrt().round() as usize;
        }

        assert_eq!(
            10 * (3usize.pow(k as u32)) * m * m + 2,
            nppa,
            "Invalid nppa configuration"
        );

        // Subdivide 30 edges
        for i in 0..30 {
            let na = KSET[i][0];
            let nb = KSET[i][1];
            for j in 1..m {
                nd += 1;
                for ix in 0..3 {
                    dirvec[nd][ix] = dirvec[na][ix] * ((m - j) as f64) + dirvec[nb][ix] * (j as f64);
                }
            }
        }

        // Subdivide 20 triangular faces
        for i in 0..20 {
            let na = FSET[i][0];
            let nb = FSET[i][1];
            let nc = FSET[i][2];
            for j1 in 1..m {
                for j2 in 1..(m - j1) {
                    nd += 1;
                    for ix in 0..3 {
                        dirvec[nd][ix] = dirvec[na][ix] * ((m - j1 - j2) as f64)
                            + dirvec[nb][ix] * (j1 as f64)
                            + dirvec[nc][ix] * (j2 as f64);
                    }
                }
            }

            if k != 0 {
                let t = 1.0 / 3.0;
                for j1 in 0..m {
                    for j2 in 0..(m - j1) {
                        nd += 1;
                        for ix in 0..3 {
                            dirvec[nd][ix] = dirvec[na][ix] * ((m - j1 - j2) as f64 - 2.0 * t)
                                + dirvec[nb][ix] * (j1 as f64 + t)
                                + dirvec[nc][ix] * (j2 as f64 + t);
                        }
                    }
                }
                let t2 = 2.0 / 3.0;
                if m >= 2 {
                    for j1 in 0..(m - 1) {
                        for j2 in 0..(m - j1 - 1) {
                            nd += 1;
                            for ix in 0..3 {
                                dirvec[nd][ix] = dirvec[na][ix] * ((m - j1 - j2) as f64 - 2.0 * t2)
                                    + dirvec[nb][ix] * (j1 as f64 + t2)
                                    + dirvec[nc][ix] * (j2 as f64 + t2);
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(nd + 1, nppa, "Point count mismatch in dvfill");
    }

    // 3. Normalize all vectors to unit sphere and assign area weights
    let m_val = if nppa > 12 {
        let m2 = (nppa - 2) / 10;
        let mut m = (m2 as f64).sqrt().round() as usize;
        if m2 != m * m {
            m = ((m2 / 3) as f64).sqrt().round() as usize;
        }
        m as f64
    } else {
        1.0
    };

    let mut sumar = 0.0f64;
    for i in 0..nppa {
        let dist = (dirvec[i][0].powi(2) + dirvec[i][1].powi(2) + dirvec[i][2].powi(2)).sqrt();
        let inv_dist = 1.0 / dist;
        let dist2 = (m_val * inv_dist).powi(2);

        dirvec[i][0] *= inv_dist;
        dirvec[i][1] *= inv_dist;
        dirvec[i][2] *= inv_dist;

        let ar = if i < 12 { 5.0 } else { 6.0 * dist2 };
        dirvec[i][3] = ar;
        sumar += ar;
    }

    // Normalize weights to integrate exactly to 4 * pi
    let norm_factor = 4.0 * PI / sumar;
    let mut points = Vec::with_capacity(nppa);
    for i in 0..nppa {
        points.push(SpherePoint {
            dir: [dirvec[i][0], dirvec[i][1], dirvec[i][2]],
            weight: dirvec[i][3] * norm_factor,
        });
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tessellation_unit_norms_and_surface_integrals() {
        for &nppa in &[12, 42, 1082] {
            let pts = generate_sphere_tessellation(nppa);
            assert_eq!(pts.len(), nppa);

            let mut sum_weight = 0.0;
            for pt in &pts {
                let norm = (pt.dir[0].powi(2) + pt.dir[1].powi(2) + pt.dir[2].powi(2)).sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-12,
                    "Tessellation vector is not normalized: norm = {}",
                    norm
                );
                sum_weight += pt.weight;
            }

            let expected_4pi = 4.0 * PI;
            assert!(
                (sum_weight - expected_4pi).abs() < 1e-10,
                "nppa = {}: total surface area weight {} differs from 4*pi ({})",
                nppa,
                sum_weight,
                expected_4pi
            );
        }
    }
}
