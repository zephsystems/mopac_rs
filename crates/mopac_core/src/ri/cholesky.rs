//! Cholesky Decomposition and Inverse Square Root of the Coulomb Metric.
//!
//! Computes the Cholesky factorization of the positive-definite auxiliary Coulomb metric
//! $V_{PQ} = L L^T$, and evaluates $V^{-1/2} = L^{-T}$ such that
//! $V^{-1/2} (V^{-1/2})^T = V^{-1}$.

use crate::types::AlignedMatrix;

/// Error returned when metric matrix is not positive-definite.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricDefinitenessError {
    pub failed_index: usize,
    pub pivot_value: f64,
}

impl std::fmt::Display for MetricDefinitenessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Metric matrix is not positive definite at index {}: pivot = {:e}",
            self.failed_index, self.pivot_value
        )
    }
}

impl std::error::Error for MetricDefinitenessError {}

/// Computes the in-place Cholesky factorization $A = L L^T$.
///
/// Overwrites the lower triangle of `a` with $L$. The strictly upper triangle is zeroed.
pub fn cholesky_decompose(a: &mut AlignedMatrix<f64>) -> Result<(), MetricDefinitenessError> {
    let n = a.rows;
    assert_eq!(a.cols, n, "Matrix must be square");

    for j in 0..n {
        let mut d = a.get(j, j);
        for k in 0..j {
            let ljk = a.get(j, k);
            d -= ljk * ljk;
        }

        if d <= 1e-14 {
            return Err(MetricDefinitenessError {
                failed_index: j,
                pivot_value: d,
            });
        }

        let ljj = d.sqrt();
        a.set(j, j, ljj);
        let inv_ljj = 1.0 / ljj;

        for i in (j + 1)..n {
            let mut s = a.get(i, j);
            for k in 0..j {
                s -= a.get(i, k) * a.get(j, k);
            }
            a.set(i, j, s * inv_ljj);
        }

        // Zero strictly upper triangle
        for i in 0..j {
            a.set(i, j, 0.0);
        }
    }

    Ok(())
}

/// Computes the inverse of a lower triangular matrix $L$ in-place.
pub fn invert_lower_triangular(l: &mut AlignedMatrix<f64>) {
    let n = l.rows;
    assert_eq!(l.cols, n);

    for j in 0..n {
        let ljj = l.get(j, j);
        assert!(ljj.abs() > 1e-15, "Diagonal of L cannot be zero");
        l.set(j, j, 1.0 / ljj);

        for i in (j + 1)..n {
            let mut sum = 0.0;
            for k in j..i {
                sum += l.get(i, k) * l.get(k, j);
            }
            let lii = l.get(i, i);
            l.set(i, j, -sum / lii);
        }
    }
}

/// Computes the inverse square root $V^{-1/2} = L^{-T}$ from a symmetric positive-definite matrix $V$.
///
/// Satisfies $V^{-1/2} (V^{-1/2})^T = V^{-1}$ to machine precision ($\le 10^{-14}$).
pub fn compute_inverse_square_root_metric(
    v: &AlignedMatrix<f64>,
    v_inv_sqrt: &mut AlignedMatrix<f64>,
) -> Result<(), MetricDefinitenessError> {
    let n = v.rows;
    assert_eq!(v.cols, n);
    assert_eq!(v_inv_sqrt.rows, n);
    assert_eq!(v_inv_sqrt.cols, n);

    let mut l = v.clone();
    cholesky_decompose(&mut l)?;
    invert_lower_triangular(&mut l);

    // v_inv_sqrt = L^{-T} (transpose of L^{-1})
    for i in 0..n {
        for j in 0..n {
            v_inv_sqrt.set(i, j, l.get(j, i));
        }
    }

    Ok(())
}
