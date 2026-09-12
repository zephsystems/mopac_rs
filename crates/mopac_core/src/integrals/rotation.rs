//! 3D Diatomic Frame Rotation Transformations.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Projects atomic orbitals from Cartesian molecular frames $(x, y, z)$ into local diatomic frames $(x', y', z')$.

/// Diatomic reference frame orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiatomicRotationFrame {
    /// Interatomic separation $R$ in Ångströms
    pub r: f64,
    /// Direction cosine along internuclear axis $z'$: $l = \Delta x / R$
    pub l: f64,
    /// Direction cosine along internuclear axis $z'$: $m = \Delta y / R$
    pub m: f64,
    /// Direction cosine along internuclear axis $z'$: $n = \Delta z / R$
    pub n: f64,
    /// Rotation matrix $D^{(1)}$ ($3 \times 3$) for p-orbitals: [row0, row1, row2]
    pub d1: [[f64; 3]; 3],
}

impl DiatomicRotationFrame {
    /// Compute the rotation frame between atom A at $r_A$ and atom B at $r_B$.
    /// The internuclear axis $z'$ points from atom A to atom B.
    pub fn compute(ra: &[f64; 3], rb: &[f64; 3]) -> Self {
        let dx = rb[0] - ra[0];
        let dy = rb[1] - ra[1];
        let dz = rb[2] - ra[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let r = r2.sqrt();

        if r < 1e-12 {
            // Degenerate case (same position)
            return Self {
                r: 0.0,
                l: 0.0,
                m: 0.0,
                n: 1.0,
                d1: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            };
        }

        let l = dx / r;
        let m = dy / r;
        let n = dz / r;

        let xy = (dx * dx + dy * dy).sqrt();

        // Build orthonormal frame:
        // z' is parallel to internuclear vector (l, m, n)
        // x' and y' are chosen orthogonal to z'
        let (d1_row0, d1_row1, d1_row2) = if xy > 1e-10 {
            // Normal case: atom not aligned with global z axis
            let x_prime = [-dy / xy, dx / xy, 0.0];
            let z_prime = [l, m, n];
            // y' = z' x x'
            let y_prime = [
                z_prime[1] * x_prime[2] - z_prime[2] * x_prime[1],
                z_prime[2] * x_prime[0] - z_prime[0] * x_prime[2],
                z_prime[0] * x_prime[1] - z_prime[1] * x_prime[0],
            ];
            (x_prime, y_prime, z_prime)
        } else {
            // Collinear with global z-axis
            let sign = if dz >= 0.0 { 1.0 } else { -1.0 };
            ([1.0, 0.0, 0.0], [0.0, sign, 0.0], [0.0, 0.0, sign])
        };

        Self {
            r,
            l,
            m,
            n,
            d1: [d1_row0, d1_row1, d1_row2],
        }
    }

    /// Check if the 3x3 rotation matrix is orthonormal to double-precision tolerance.
    pub fn is_orthonormal(&self, tol: f64) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                let dot = self.d1[i][0] * self.d1[j][0]
                    + self.d1[i][1] * self.d1[j][1]
                    + self.d1[i][2] * self.d1[j][2];
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Determinant of the rotation matrix (must equal +1.0 for proper rotation).
    pub fn determinant(&self) -> f64 {
        let m = &self.d1;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}
