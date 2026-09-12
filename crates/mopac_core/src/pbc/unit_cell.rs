//! Periodic Unit Cell, Translation Vectors & Reciprocal Lattice Geometry.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Implements direct/reciprocal lattice geometry, metric tensors, and Monkhorst-Pack $k$-space sampling.

use std::f64::consts::PI;

/// Dimensionality of the periodic boundary condition system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicDimension {
    /// 1D Periodic Polymer (1 translation vector $T_1$)
    OneD = 1,
    /// 2D Periodic Surface / Layer / Slab (2 translation vectors $T_1, T_2$)
    TwoD = 2,
    /// 3D Periodic Bulk Solid / Crystal (3 translation vectors $T_1, T_2, T_3$)
    ThreeD = 3,
}

/// A periodic translation vector $\vec{R}_m = n_1 \vec{a}_1 + n_2 \vec{a}_2 + n_3 \vec{a}_3$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranslationIndex {
    pub n1: i32,
    pub n2: i32,
    pub n3: i32,
    /// Shift vector in Cartesian Ångströms $[x, y, z]$
    pub shift_angstrom: [f64; 3],
    /// Norm of translation vector in Ångströms
    pub distance_angstrom: f64,
}

/// A sampling point in the first Brillouin zone.
#[derive(Debug, Clone, PartialEq)]
pub struct KPoint {
    /// Fractional coordinates in units of reciprocal lattice vectors $\vec{b}_1, \vec{b}_2, \vec{b}_3$
    pub fractional: [f64; 3],
    /// Cartesian coordinates in units of $\text{Å}^{-1}$
    pub cartesian: [f64; 3],
    /// Symmetry / quadrature integration weight ($w_{\vec{k}} \in [0, 1]$, $\sum w_k = 1$)
    pub weight: f64,
    /// High symmetry label if applicable (e.g. "Gamma", "X", "M", "K", "Z")
    pub label: Option<String>,
}

/// Complete representation of a periodic unit cell.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitCell {
    /// Dimensionality (1D, 2D, or 3D)
    pub dimension: PeriodicDimension,
    /// Direct translation lattice vectors $\vec{a}_1, \vec{a}_2, \vec{a}_3$ in Ångströms
    pub direct_vectors: [[f64; 3]; 3],
    /// Reciprocal lattice vectors $\vec{b}_1, \vec{b}_2, \vec{b}_3$ in $\text{Å}^{-1}$ satisfying $\vec{a}_i \cdot \vec{b}_j = 2\pi \delta_{ij}$
    pub reciprocal_vectors: [[f64; 3]; 3],
    /// Unit cell volume ($\text{Å}^3$ for 3D), area ($\text{Å}^2$ for 2D), or length ($\text{Å}$ for 1D)
    pub cell_measure: f64,
}

impl UnitCell {
    /// Construct a unit cell from 1, 2, or 3 translation vectors in Ångströms.
    pub fn from_translation_vectors(vectors: &[[f64; 3]]) -> Result<Self, String> {
        let n_vec = vectors.len();
        if n_vec == 0 || n_vec > 3 {
            return Err(format!(
                "Unit cell requires 1, 2, or 3 translation vectors, got {}",
                n_vec
            ));
        }

        let dimension = match n_vec {
            1 => PeriodicDimension::OneD,
            2 => PeriodicDimension::TwoD,
            _ => PeriodicDimension::ThreeD,
        };

        let mut direct = [[0.0f64; 3]; 3];
        for (i, v) in vectors.iter().enumerate() {
            direct[i] = *v;
        }

        let (reciprocal, cell_measure) = match dimension {
            PeriodicDimension::OneD => {
                let a1 = direct[0];
                let norm_sq = a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2];
                let len = norm_sq.sqrt();
                if len < 1e-6 {
                    return Err("1D translation vector length is virtually zero".to_string());
                }
                let factor = 2.0 * PI / norm_sq;
                let b1 = [a1[0] * factor, a1[1] * factor, a1[2] * factor];
                ([b1, [0.0; 3], [0.0; 3]], len)
            }
            PeriodicDimension::TwoD => {
                let a1 = direct[0];
                let a2 = direct[1];
                let n = cross_product(&a1, &a2);
                let area = norm(&n);
                if area < 1e-6 {
                    return Err("2D translation vectors are collinear (area = 0)".to_string());
                }
                let a2_cross_n = cross_product(&a2, &n);
                let n_cross_a1 = cross_product(&n, &a1);
                let factor = 2.0 * PI / (area * area);
                let b1 = [
                    a2_cross_n[0] * factor,
                    a2_cross_n[1] * factor,
                    a2_cross_n[2] * factor,
                ];
                let b2 = [
                    n_cross_a1[0] * factor,
                    n_cross_a1[1] * factor,
                    n_cross_a1[2] * factor,
                ];
                ([b1, b2, [0.0; 3]], area)
            }
            PeriodicDimension::ThreeD => {
                let a1 = direct[0];
                let a2 = direct[1];
                let a3 = direct[2];
                let a2_cross_a3 = cross_product(&a2, &a3);
                let a3_cross_a1 = cross_product(&a3, &a1);
                let a1_cross_a2 = cross_product(&a1, &a2);
                let volume = dot_product(&a1, &a2_cross_a3);
                if volume.abs() < 1e-6 {
                    return Err("3D translation vectors are coplanar (volume = 0)".to_string());
                }
                let factor = 2.0 * PI / volume;
                let b1 = [
                    a2_cross_a3[0] * factor,
                    a2_cross_a3[1] * factor,
                    a2_cross_a3[2] * factor,
                ];
                let b2 = [
                    a3_cross_a1[0] * factor,
                    a3_cross_a1[1] * factor,
                    a3_cross_a1[2] * factor,
                ];
                let b3 = [
                    a1_cross_a2[0] * factor,
                    a1_cross_a2[1] * factor,
                    a1_cross_a2[2] * factor,
                ];
                ([b1, b2, b3], volume.abs())
            }
        };

        Ok(Self {
            dimension,
            direct_vectors: direct,
            reciprocal_vectors: reciprocal,
            cell_measure,
        })
    }

    /// Generate real-space translation vectors $\vec{R}_m = n_1 \vec{a}_1 + n_2 \vec{a}_2 + n_3 \vec{a}_3$
    /// for a Born-von Kármán supercell defined by `mers: [n1, n2, n3]`.
    pub fn generate_translation_indices(&self, mers: [usize; 3]) -> Vec<TranslationIndex> {
        let (n1_max, n2_max, n3_max) = match self.dimension {
            PeriodicDimension::OneD => (mers[0] as i32, 1, 1),
            PeriodicDimension::TwoD => (mers[0] as i32, mers[1] as i32, 1),
            PeriodicDimension::ThreeD => (mers[0] as i32, mers[1] as i32, mers[2] as i32),
        };

        let half1 = (n1_max - 1) / 2;
        let half2 = (n2_max - 1) / 2;
        let half3 = (n3_max - 1) / 2;

        let mut indices = Vec::with_capacity((n1_max * n2_max * n3_max) as usize);

        for n1 in -half1..=half1 {
            for n2 in -half2..=half2 {
                for n3 in -half3..=half3 {
                    let shift = [
                        (n1 as f64) * self.direct_vectors[0][0]
                            + (n2 as f64) * self.direct_vectors[1][0]
                            + (n3 as f64) * self.direct_vectors[2][0],
                        (n1 as f64) * self.direct_vectors[0][1]
                            + (n2 as f64) * self.direct_vectors[1][1]
                            + (n3 as f64) * self.direct_vectors[2][1],
                        (n1 as f64) * self.direct_vectors[0][2]
                            + (n2 as f64) * self.direct_vectors[1][2]
                            + (n3 as f64) * self.direct_vectors[2][2],
                    ];
                    let dist = norm(&shift);
                    indices.push(TranslationIndex {
                        n1,
                        n2,
                        n3,
                        shift_angstrom: shift,
                        distance_angstrom: dist,
                    });
                }
            }
        }

        // Sort so that central cell (0, 0, 0) is always index 0
        indices.sort_by(|a, b| {
            a.distance_angstrom
                .partial_cmp(&b.distance_angstrom)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        indices
    }

    /// Generate a regular Monkhorst-Pack $k$-point grid in the first Brillouin Zone.
    pub fn generate_monkhorst_pack_grid(&self, nk: [usize; 3]) -> Vec<KPoint> {
        let (nk1, nk2, nk3) = match self.dimension {
            PeriodicDimension::OneD => (nk[0].max(1), 1, 1),
            PeriodicDimension::TwoD => (nk[0].max(1), nk[1].max(1), 1),
            PeriodicDimension::ThreeD => (nk[0].max(1), nk[1].max(1), nk[2].max(1)),
        };

        let total_k = nk1 * nk2 * nk3;
        let weight = 1.0 / (total_k as f64);
        let mut k_points = Vec::with_capacity(total_k);

        for i1 in 0..nk1 {
            let frac1 = if nk1 == 1 {
                0.0
            } else {
                (2.0 * (i1 as f64) - (nk1 as f64) + 1.0) / (2.0 * (nk1 as f64))
            };

            for i2 in 0..nk2 {
                let frac2 = if nk2 == 1 {
                    0.0
                } else {
                    (2.0 * (i2 as f64) - (nk2 as f64) + 1.0) / (2.0 * (nk2 as f64))
                };

                for i3 in 0..nk3 {
                    let frac3 = if nk3 == 1 {
                        0.0
                    } else {
                        (2.0 * (i3 as f64) - (nk3 as f64) + 1.0) / (2.0 * (nk3 as f64))
                    };

                    let cart = [
                        frac1 * self.reciprocal_vectors[0][0]
                            + frac2 * self.reciprocal_vectors[1][0]
                            + frac3 * self.reciprocal_vectors[2][0],
                        frac1 * self.reciprocal_vectors[0][1]
                            + frac2 * self.reciprocal_vectors[1][1]
                            + frac3 * self.reciprocal_vectors[2][1],
                        frac1 * self.reciprocal_vectors[0][2]
                            + frac2 * self.reciprocal_vectors[1][2]
                            + frac3 * self.reciprocal_vectors[2][2],
                    ];

                    let is_gamma = frac1.abs() < 1e-6 && frac2.abs() < 1e-6 && frac3.abs() < 1e-6;
                    let label = if is_gamma {
                        Some("Gamma".to_string())
                    } else {
                        None
                    };

                    k_points.push(KPoint {
                        fractional: [frac1, frac2, frac3],
                        cartesian: cart,
                        weight,
                        label,
                    });
                }
            }
        }

        k_points
    }

    /// Generate high-symmetry line path in Brillouin Zone for band structure rendering.
    pub fn generate_band_path(&self, n_points_per_segment: usize) -> Vec<KPoint> {
        let n_pts = n_points_per_segment.max(10);
        let mut path = Vec::new();

        match self.dimension {
            PeriodicDimension::OneD => {
                // Path: Gamma (0) -> X (0.5)
                for i in 0..=n_pts {
                    let frac = 0.5 * (i as f64) / (n_pts as f64);
                    let cart = [
                        frac * self.reciprocal_vectors[0][0],
                        frac * self.reciprocal_vectors[0][1],
                        frac * self.reciprocal_vectors[0][2],
                    ];
                    let label = if i == 0 {
                        Some("Gamma".to_string())
                    } else if i == n_pts {
                        Some("X".to_string())
                    } else {
                        None
                    };
                    path.push(KPoint {
                        fractional: [frac, 0.0, 0.0],
                        cartesian: cart,
                        weight: 0.0,
                        label,
                    });
                }
            }
            PeriodicDimension::TwoD => {
                // Path: Gamma (0,0) -> X (0.5, 0) -> M (0.5, 0.5) -> Gamma (0,0)
                let waypoints = [
                    ([0.0, 0.0, 0.0], "Gamma"),
                    ([0.5, 0.0, 0.0], "X"),
                    ([0.5, 0.5, 0.0], "M"),
                    ([0.0, 0.0, 0.0], "Gamma"),
                ];

                for seg in 0..(waypoints.len() - 1) {
                    let (p_start, label_start) = waypoints[seg];
                    let (p_end, label_end) = waypoints[seg + 1];

                    for step in 0..n_pts {
                        let t = (step as f64) / (n_pts as f64);
                        let frac = [
                            p_start[0] + t * (p_end[0] - p_start[0]),
                            p_start[1] + t * (p_end[1] - p_start[1]),
                            0.0,
                        ];
                        let cart = [
                            frac[0] * self.reciprocal_vectors[0][0]
                                + frac[1] * self.reciprocal_vectors[1][0],
                            frac[0] * self.reciprocal_vectors[0][1]
                                + frac[1] * self.reciprocal_vectors[1][1],
                            frac[0] * self.reciprocal_vectors[0][2]
                                + frac[1] * self.reciprocal_vectors[1][2],
                        ];
                        let label = if step == 0 {
                            Some(label_start.to_string())
                        } else {
                            None
                        };
                        path.push(KPoint {
                            fractional: frac,
                            cartesian: cart,
                            weight: 0.0,
                            label,
                        });
                    }
                    if seg == waypoints.len() - 2 {
                        let frac = p_end;
                        let cart = [
                            frac[0] * self.reciprocal_vectors[0][0]
                                + frac[1] * self.reciprocal_vectors[1][0],
                            frac[0] * self.reciprocal_vectors[0][1]
                                + frac[1] * self.reciprocal_vectors[1][1],
                            frac[0] * self.reciprocal_vectors[0][2]
                                + frac[1] * self.reciprocal_vectors[1][2],
                        ];
                        path.push(KPoint {
                            fractional: frac,
                            cartesian: cart,
                            weight: 0.0,
                            label: Some(label_end.to_string()),
                        });
                    }
                }
            }
            PeriodicDimension::ThreeD => {
                // Path: Gamma (0,0,0) -> X (0.5,0,0) -> M (0.5,0.5,0) -> Gamma (0,0,0) -> Z (0,0,0.5)
                let waypoints = [
                    ([0.0, 0.0, 0.0], "Gamma"),
                    ([0.5, 0.0, 0.0], "X"),
                    ([0.5, 0.5, 0.0], "M"),
                    ([0.0, 0.0, 0.0], "Gamma"),
                    ([0.0, 0.0, 0.5], "Z"),
                ];

                for seg in 0..(waypoints.len() - 1) {
                    let (p_start, label_start) = waypoints[seg];
                    let (p_end, label_end) = waypoints[seg + 1];

                    for step in 0..n_pts {
                        let t = (step as f64) / (n_pts as f64);
                        let frac = [
                            p_start[0] + t * (p_end[0] - p_start[0]),
                            p_start[1] + t * (p_end[1] - p_start[1]),
                            p_start[2] + t * (p_end[2] - p_start[2]),
                        ];
                        let cart = [
                            frac[0] * self.reciprocal_vectors[0][0]
                                + frac[1] * self.reciprocal_vectors[1][0]
                                + frac[2] * self.reciprocal_vectors[2][0],
                            frac[0] * self.reciprocal_vectors[0][1]
                                + frac[1] * self.reciprocal_vectors[1][1]
                                + frac[2] * self.reciprocal_vectors[2][1],
                            frac[0] * self.reciprocal_vectors[0][2]
                                + frac[1] * self.reciprocal_vectors[1][2]
                                + frac[2] * self.reciprocal_vectors[2][2],
                        ];
                        let label = if step == 0 {
                            Some(label_start.to_string())
                        } else {
                            None
                        };
                        path.push(KPoint {
                            fractional: frac,
                            cartesian: cart,
                            weight: 0.0,
                            label,
                        });
                    }
                    if seg == waypoints.len() - 2 {
                        let frac = p_end;
                        let cart = [
                            frac[0] * self.reciprocal_vectors[0][0]
                                + frac[1] * self.reciprocal_vectors[1][0]
                                + frac[2] * self.reciprocal_vectors[2][0],
                            frac[0] * self.reciprocal_vectors[0][1]
                                + frac[1] * self.reciprocal_vectors[1][1]
                                + frac[2] * self.reciprocal_vectors[2][1],
                            frac[0] * self.reciprocal_vectors[0][2]
                                + frac[1] * self.reciprocal_vectors[1][2]
                                + frac[2] * self.reciprocal_vectors[2][2],
                        ];
                        path.push(KPoint {
                            fractional: frac,
                            cartesian: cart,
                            weight: 0.0,
                            label: Some(label_end.to_string()),
                        });
                    }
                }
            }
        }

        path
    }
}

#[inline(always)]
fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn norm(a: &[f64; 3]) -> f64 {
    dot_product(a, a).sqrt()
}
