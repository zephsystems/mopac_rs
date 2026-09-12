//! Solvent-Accessible Surface (SAS) Cavity Construction for COSMO.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct translation and vectorized refinement of OpenMOPAC v23.2.5 `coscav` subroutine in `cosmo.F90`.

use crate::solvation::radii::cosmo_atomic_radius;
use crate::solvation::tessellation::generate_sphere_tessellation;
use crate::types::MolecularBatch;


/// A discrete boundary segment on the solvent-accessible cavity surface.
#[derive(Debug, Clone, PartialEq)]
pub struct CavitySegment {
    /// 3D Cartesian coordinates of the segment center in Angstroms.
    pub position: [f64; 3],
    /// Surface area of the segment in square Angstroms (A^2).
    pub area: f64,
    /// Outward unit normal vector on the cavity surface.
    pub normal: [f64; 3],
    /// Index of the parent atom (0-indexed).
    pub atom_index: usize,
}

/// Geometric properties of the constructed COSMO solute cavity.
#[derive(Debug, Clone, PartialEq)]
pub struct CosmoCavity {
    pub segments: Vec<CavitySegment>,
    pub total_area_angstrom2: f64,
    pub total_volume_angstrom3: f64,
}

impl CosmoCavity {
    /// Construct the discrete COSMO cavity surface for a molecular batch.
    ///
    /// Uses Andreas Klamt's solvent probe radius `rsolv` (default: 1.30005 A) and
    /// 1082-point icosahedral sphere tessellation.
    pub fn construct(batch: &MolecularBatch, rsolv: f64) -> Self {
        let natoms = batch.natoms;
        assert!(natoms > 0, "Molecular batch cannot be empty for COSMO");

        // Generate static icosahedral grids
        let fine_grid = generate_sphere_tessellation(1082);
        let basic_heavy = generate_sphere_tessellation(42);
        let basic_hydro = generate_sphere_tessellation(12);

        let mut segments = Vec::new();
        let mut total_area = 0.0f64;
        let mut total_volume = 0.0f64;

        for i in 0..natoms {
            let zi = batch.atomic_numbers[i];
            let ri = cosmo_atomic_radius(zi);
            let ri2 = ri * ri;
            let xi = [batch.x[i], batch.y[i], batch.z[i]];

            let basic_grid = if zi == 1 {
                &basic_hydro
            } else {
                &basic_heavy
            };
            let num_basic = basic_grid.len();

            // Group fine grid points into basic segment clusters
            let mut segment_fine_points: Vec<Vec<usize>> = vec![Vec::new(); num_basic];

            for (k, pt) in fine_grid.iter().enumerate() {
                // Test point on the solvent-exclusion surface: x_k = x_i + u * (r_i + rsolv)
                let r_probe = ri + rsolv;
                let xk = [
                    xi[0] + pt.dir[0] * r_probe,
                    xi[1] + pt.dir[1] * r_probe,
                    xi[2] + pt.dir[2] * r_probe,
                ];

                // Check overlap with all other atoms j != i
                let mut accessible = true;
                for j in 0..natoms {
                    if j == i {
                        continue;
                    }
                    let rj = cosmo_atomic_radius(batch.atomic_numbers[j]);
                    let dx = xk[0] - batch.x[j];
                    let dy = xk[1] - batch.y[j];
                    let dz = xk[2] - batch.z[j];
                    let dist2 = dx * dx + dy * dy + dz * dz;
                    let cutoff = rj + rsolv;
                    if dist2 < cutoff * cutoff {
                        accessible = false;
                        break;
                    }
                }

                if !accessible {
                    continue;
                }

                // Find the closest basic segment direction
                let mut max_dot = -2.0f64;
                let mut best_seg = 0;
                for (seg_idx, bpt) in basic_grid.iter().enumerate() {
                    let dot = pt.dir[0] * bpt.dir[0] + pt.dir[1] * bpt.dir[1] + pt.dir[2] * bpt.dir[2];
                    if dot > max_dot {
                        max_dot = dot;
                        best_seg = seg_idx;
                    }
                }

                segment_fine_points[best_seg].push(k);
            }

            // Create valid segments from non-empty clusters
            for fine_indices in segment_fine_points {
                if fine_indices.is_empty() {
                    continue;
                }

                let mut seg_area = 0.0f64;
                let mut avg_dir = [0.0f64; 3];

                for &k in &fine_indices {
                    let w = fine_grid[k].weight * ri2;
                    seg_area += w;
                    avg_dir[0] += fine_grid[k].dir[0] * w;
                    avg_dir[1] += fine_grid[k].dir[1] * w;
                    avg_dir[2] += fine_grid[k].dir[2] * w;
                }

                let dir_norm = (avg_dir[0].powi(2) + avg_dir[1].powi(2) + avg_dir[2].powi(2)).sqrt();
                let normal = if dir_norm > 1e-12 {
                    [
                        avg_dir[0] / dir_norm,
                        avg_dir[1] / dir_norm,
                        avg_dir[2] / dir_norm,
                    ]
                } else {
                    [1.0, 0.0, 0.0]
                };

                let pos = [
                    xi[0] + normal[0] * ri,
                    xi[1] + normal[1] * ri,
                    xi[2] + normal[2] * ri,
                ];

                total_area += seg_area;
                // Volume contribution: (1/3) * sum (r . n) * dS
                let r_dot_n = pos[0] * normal[0] + pos[1] * normal[1] + pos[2] * normal[2];
                total_volume += (1.0 / 3.0) * r_dot_n * seg_area;

                segments.push(CavitySegment {
                    position: pos,
                    area: seg_area,
                    normal,
                    atom_index: i,
                });
            }
        }

        CosmoCavity {
            segments,
            total_area_angstrom2: total_area,
            total_volume_angstrom3: total_volume,
        }
    }

    /// Number of discrete surface segments on the cavity.
    #[inline]
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_atom_sphere_area_and_volume() {
        // Single oxygen atom at origin: R = 1.72 A
        let coords = vec![[0.0, 0.0, 0.0]];
        let z = vec![8];
        let batch = MolecularBatch::new(z, &coords);

        let cavity = CosmoCavity::construct(&batch, 1.30005);
        let r = 1.72f64;
        let expected_area = 4.0 * std::f64::consts::PI * r * r;
        let expected_vol = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3);

        assert!(
            (cavity.total_area_angstrom2 - expected_area).abs() < 0.1,
            "Single sphere area {} differs from 4*pi*R^2 = {}",
            cavity.total_area_angstrom2,
            expected_area
        );
        assert!(
            (cavity.total_volume_angstrom3 - expected_vol).abs() < 0.1,
            "Single sphere volume {} differs from 4/3*pi*R^3 = {}",
            cavity.total_volume_angstrom3,
            expected_vol
        );
    }
}
