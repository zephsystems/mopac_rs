//! Z-Matrix Internal Coordinates to 3D Cartesian Conversion (`intxyz.F90` / `xyzint.F90`).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Reconstructs 3D Cartesian coordinates $(x, y, z)$ from standard Z-matrix internal coordinates
//! (bond lengths $r$, bond angles $\theta$, and torsion dihedral angles $\phi$) using robust
//! trigonometric local frame triangulation (NeRF / Natural Extension of Reference Frames).

use std::f64::consts::PI;

/// Representation of a single Z-matrix line specification.
#[derive(Debug, Clone, PartialEq)]
pub struct ZMatrixAtom {
    /// Atomic symbol (e.g. "C", "H", "O") or atomic number Z
    pub symbol: String,
    pub atomic_number: u8,
    /// Bond length in Ångströms to atom at index `bond_to` (0-indexed)
    pub bond_length: f64,
    /// 0-based index of atom defining bond length
    pub bond_to: Option<usize>,
    /// Bond angle in degrees (0..180) subtended with `angle_to`
    pub bond_angle_deg: f64,
    /// 0-based index of atom defining bond angle
    pub angle_to: Option<usize>,
    /// Dihedral angle in degrees (-180..180) with `dihedral_to`
    pub dihedral_deg: f64,
    /// 0-based index of atom defining dihedral angle
    pub dihedral_to: Option<usize>,
    /// Optimization flags for distance, angle, dihedral
    pub opt_bond: bool,
    pub opt_angle: bool,
    pub opt_dihedral: bool,
}

/// Convert a list of Z-matrix atoms into contiguous 3D Cartesian coordinates in Ångströms.
pub fn zmatrix_to_cartesian(zmat: &[ZMatrixAtom]) -> Result<Vec<[f64; 3]>, String> {
    let n = zmat.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut coords: Vec<[f64; 3]> = Vec::with_capacity(n);

    // Atom 0: Origin (0, 0, 0)
    coords.push([0.0, 0.0, 0.0]);
    if n == 1 {
        return Ok(coords);
    }

    // Atom 1: Placed along X axis at (r_01, 0, 0)
    let r1 = zmat[1].bond_length;
    coords.push([r1, 0.0, 0.0]);
    if n == 2 {
        return Ok(coords);
    }

    // Atom 2: Placed in XY plane
    let r2 = zmat[2].bond_length;
    let theta2_rad = zmat[2].bond_angle_deg * (PI / 180.0);
    let j2 = zmat[2].bond_to.unwrap_or(1);
    let k2 = zmat[2].angle_to.unwrap_or(0);

    let origin2 = coords[j2];
    let ref2 = coords[k2];
    let mut v_base = [
        ref2[0] - origin2[0],
        ref2[1] - origin2[1],
        ref2[2] - origin2[2],
    ];
    let d_base = (v_base[0] * v_base[0] + v_base[1] * v_base[1] + v_base[2] * v_base[2]).sqrt();
    if d_base > 1e-12 {
        v_base[0] /= d_base;
        v_base[1] /= d_base;
        v_base[2] /= d_base;
    } else {
        v_base = [-1.0, 0.0, 0.0];
    }

    // Orthogonal vector in XY plane
    let u_base = [-v_base[1], v_base[0], 0.0];
    let d_u = (u_base[0] * u_base[0] + u_base[1] * u_base[1]).sqrt();
    let u_norm = if d_u > 1e-12 {
        [u_base[0] / d_u, u_base[1] / d_u, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    let cos_t2 = theta2_rad.cos();
    let sin_t2 = theta2_rad.sin();
    coords.push([
        origin2[0] + r2 * (cos_t2 * v_base[0] + sin_t2 * u_norm[0]),
        origin2[1] + r2 * (cos_t2 * v_base[1] + sin_t2 * u_norm[1]),
        origin2[2] + r2 * (cos_t2 * v_base[2] + sin_t2 * u_norm[2]),
    ]);

    // Atom i >= 3: General NeRF 3D triangulation
    for (i, entry) in zmat.iter().enumerate().skip(3) {
        let j = entry
            .bond_to
            .ok_or_else(|| format!("Atom {} missing bond_to index", i + 1))?;
        let k = entry
            .angle_to
            .ok_or_else(|| format!("Atom {} missing angle_to index", i + 1))?;
        let l = entry
            .dihedral_to
            .ok_or_else(|| format!("Atom {} missing dihedral_to index", i + 1))?;

        if j >= i || k >= i || l >= i {
            return Err(format!(
                "Atom {} references forward atom indices (j={}, k={}, l={})",
                i + 1,
                j + 1,
                k + 1,
                l + 1
            ));
        }

        let r = entry.bond_length;
        let theta_rad = entry.bond_angle_deg * (PI / 180.0);
        let phi_rad = entry.dihedral_deg * (PI / 180.0);

        let p_j = coords[j];
        let p_k = coords[k];
        let p_l = coords[l];

        // Vector k -> j (primary bond direction)
        let mut v1 = [p_j[0] - p_k[0], p_j[1] - p_k[1], p_j[2] - p_k[2]];
        let d1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
        if d1 > 1e-12 {
            v1[0] /= d1;
            v1[1] /= d1;
            v1[2] /= d1;
        } else {
            v1 = [1.0, 0.0, 0.0];
        }

        // Vector l -> k (reference for plane definition)
        let v2 = [p_k[0] - p_l[0], p_k[1] - p_l[1], p_k[2] - p_l[2]];

        // Normal vector to j-k-l plane: n = v2 x v1
        let mut n_vec = [
            v2[1] * v1[2] - v2[2] * v1[1],
            v2[2] * v1[0] - v2[0] * v1[2],
            v2[0] * v1[1] - v2[1] * v1[0],
        ];
        let d_n = (n_vec[0] * n_vec[0] + n_vec[1] * n_vec[1] + n_vec[2] * n_vec[2]).sqrt();
        if d_n > 1e-12 {
            n_vec[0] /= d_n;
            n_vec[1] /= d_n;
            n_vec[2] /= d_n;
        } else {
            // Collinear fallback: pick perpendicular vector to v1
            n_vec = if v1[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let dot_n = n_vec[0] * v1[0] + n_vec[1] * v1[1] + n_vec[2] * v1[2];
            n_vec[0] -= dot_n * v1[0];
            n_vec[1] -= dot_n * v1[1];
            n_vec[2] -= dot_n * v1[2];
            let norm_n = (n_vec[0] * n_vec[0] + n_vec[1] * n_vec[1] + n_vec[2] * n_vec[2]).sqrt();
            n_vec[0] /= norm_n;
            n_vec[1] /= norm_n;
            n_vec[2] /= norm_n;
        }

        // Orthogonal in-plane vector: u = n x v1
        let u_vec = [
            n_vec[1] * v1[2] - n_vec[2] * v1[1],
            n_vec[2] * v1[0] - n_vec[0] * v1[2],
            n_vec[0] * v1[1] - n_vec[1] * v1[0],
        ];

        // Local spherical displacement:
        // Angle theta is subtended with atom k (direction -v1)
        let cos_t = theta_rad.cos();
        let sin_t = theta_rad.sin();
        let cos_p = phi_rad.cos();
        let sin_p = phi_rad.sin();

        let dx = -r * cos_t * v1[0] + r * sin_t * (cos_p * u_vec[0] + sin_p * n_vec[0]);
        let dy = -r * cos_t * v1[1] + r * sin_t * (cos_p * u_vec[1] + sin_p * n_vec[1]);
        let dz = -r * cos_t * v1[2] + r * sin_t * (cos_p * u_vec[2] + sin_p * n_vec[2]);

        coords.push([p_j[0] + dx, p_j[1] + dy, p_j[2] + dz]);
    }

    Ok(coords)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zmatrix_water_geometry() {
        // Water molecule in Z-matrix:
        // O
        // H 1 0.96
        // H 1 0.96 2 104.5
        let zmat = vec![
            ZMatrixAtom {
                symbol: "O".to_string(),
                atomic_number: 8,
                bond_length: 0.0,
                bond_to: None,
                bond_angle_deg: 0.0,
                angle_to: None,
                dihedral_deg: 0.0,
                dihedral_to: None,
                opt_bond: false,
                opt_angle: false,
                opt_dihedral: false,
            },
            ZMatrixAtom {
                symbol: "H".to_string(),
                atomic_number: 1,
                bond_length: 0.96,
                bond_to: Some(0),
                bond_angle_deg: 0.0,
                angle_to: None,
                dihedral_deg: 0.0,
                dihedral_to: None,
                opt_bond: true,
                opt_angle: false,
                opt_dihedral: false,
            },
            ZMatrixAtom {
                symbol: "H".to_string(),
                atomic_number: 1,
                bond_length: 0.96,
                bond_to: Some(0),
                bond_angle_deg: 104.5,
                angle_to: Some(1),
                dihedral_deg: 0.0,
                dihedral_to: None,
                opt_bond: true,
                opt_angle: true,
                opt_dihedral: false,
            },
        ];

        let coords = zmatrix_to_cartesian(&zmat).expect("Z-matrix conversion failed");
        assert_eq!(coords.len(), 3);

        // Check O-H1 distance
        let d_oh1 = ((coords[1][0] - coords[0][0]).powi(2)
            + (coords[1][1] - coords[0][1]).powi(2)
            + (coords[1][2] - coords[0][2]).powi(2))
        .sqrt();
        assert!((d_oh1 - 0.96).abs() < 1e-6);

        // Check O-H2 distance
        let d_oh2 = ((coords[2][0] - coords[0][0]).powi(2)
            + (coords[2][1] - coords[0][1]).powi(2)
            + (coords[2][2] - coords[0][2]).powi(2))
        .sqrt();
        assert!((d_oh2 - 0.96).abs() < 1e-6);

        // Check H1-O-H2 angle: dot product v1 . v2 = cos(104.5 deg)
        let v1 = [
            (coords[1][0] - coords[0][0]) / d_oh1,
            (coords[1][1] - coords[0][1]) / d_oh1,
            (coords[1][2] - coords[0][2]) / d_oh1,
        ];
        let v2 = [
            (coords[2][0] - coords[0][0]) / d_oh2,
            (coords[2][1] - coords[0][1]) / d_oh2,
            (coords[2][2] - coords[0][2]) / d_oh2,
        ];
        let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        let angle_deg = dot.acos() * (180.0 / PI);
        assert!(
            (angle_deg - 104.5).abs() < 1e-4,
            "Calculated angle {:.4} != expected 104.5",
            angle_deg
        );
    }
}
