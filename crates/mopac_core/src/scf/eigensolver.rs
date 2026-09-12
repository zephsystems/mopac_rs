//! High-Precision Pure Rust Real Symmetric Eigensolver.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Implements the classical cyclic Jacobi method with threshold sweeping for exact diagonalization $F C = C \epsilon$.

use crate::types::{AlignedMatrix, AlignedVec64};

/// Diagonalize a real symmetric matrix $A$ of dimension $N \times N$.
///
/// On exit:
/// * `eigenvalues` contains the sorted eigenvalues $\epsilon_1 \le \epsilon_2 \le \dots \le \epsilon_N$.
/// * `eigenvectors` contains the orthonormal eigenvectors as columns: $C_{\mu i}$.
/// * Satisfies $C^T C = I$ to machine precision ($\le 10^{-14}$).
pub fn diagonalize_symmetric(
    a: &AlignedMatrix<f64>,
    eigenvalues: &mut AlignedVec64<f64>,
    eigenvectors: &mut AlignedMatrix<f64>,
) -> usize {
    let mut mat = a.clone();
    diagonalize_symmetric_with_work(a, &mut mat, eigenvalues, eigenvectors)
}

/// Diagonalize a real symmetric matrix $A$ of dimension $N \times N$ using a preallocated workspace.
///
/// On exit:
/// * `eigenvalues` contains the sorted eigenvalues $\epsilon_1 \le \epsilon_2 \le \dots \le \epsilon_N$.
/// * `eigenvectors` contains the orthonormal eigenvectors as columns: $C_{\mu i}$.
/// * Satisfies $C^T C = I$ to machine precision ($\le 10^{-14}$).
/// * Strictly 0 heap allocations when reusing `work`.
pub fn diagonalize_symmetric_with_work(
    a: &AlignedMatrix<f64>,
    work: &mut AlignedMatrix<f64>,
    eigenvalues: &mut AlignedVec64<f64>,
    eigenvectors: &mut AlignedMatrix<f64>,
) -> usize {
    let n = a.rows;
    assert_eq!(a.cols, n);
    assert_eq!(work.rows, n);
    assert_eq!(work.cols, n);
    assert_eq!(eigenvalues.len(), n);
    assert_eq!(eigenvectors.rows, n);
    assert_eq!(eigenvectors.cols, n);

    // Working copy of matrix A
    work.copy_from(a);
    let mat = work;

    // Initialize eigenvectors to identity matrix
    eigenvectors.fill_zero();
    for i in 0..n {
        eigenvectors.set(i, i, 1.0);
    }

    let max_sweeps = 100;
    let mut sweeps = 0;

    for sweep in 0..max_sweeps {
        sweeps = sweep + 1;

        // Sum of off-diagonal elements
        let mut off_diag_sum = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag_sum += mat.get(i, j).abs();
            }
        }

        if off_diag_sum < 1e-14 * (n as f64) {
            break;
        }

        // Threshold for this sweep
        let threshold = if sweep < 3 {
            0.2 * off_diag_sum / ((n * (n - 1) / 2) as f64)
        } else {
            0.0
        };

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = mat.get(p, q);
                let g = 100.0 * apq.abs();

                // If element is tiny compared to diagonals, skip
                if sweep > 4
                    && (mat.get(p, p).abs() + g == mat.get(p, p).abs())
                    && (mat.get(q, q).abs() + g == mat.get(q, q).abs())
                {
                    mat.set(p, q, 0.0);
                    mat.set(q, p, 0.0);
                    continue;
                }

                if apq.abs() <= threshold {
                    continue;
                }

                let app = mat.get(p, p);
                let aqq = mat.get(q, q);
                let h = aqq - app;

                let t = if h.abs() + g == h.abs() {
                    apq / h
                } else {
                    let theta = 0.5 * h / apq;
                    let mut t_val = 1.0 / (theta.abs() + (1.0 + theta * theta).sqrt());
                    if theta < 0.0 {
                        t_val = -t_val;
                    }
                    t_val
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                mat.set(p, q, 0.0);
                mat.set(q, p, 0.0);

                mat.set(p, p, app - t * apq);
                mat.set(q, q, aqq + t * apq);

                for r in 0..p {
                    let arp = mat.get(r, p);
                    let arq = mat.get(r, q);
                    mat.set(r, p, arp - s * (arq + arp * tau));
                    mat.set(p, r, mat.get(r, p));
                    mat.set(r, q, arq + s * (arp - arq * tau));
                    mat.set(q, r, mat.get(r, q));
                }

                for r in (p + 1)..q {
                    let apr = mat.get(p, r);
                    let arq = mat.get(r, q);
                    mat.set(p, r, apr - s * (arq + apr * tau));
                    mat.set(r, p, mat.get(p, r));
                    mat.set(r, q, arq + s * (apr - arq * tau));
                    mat.set(q, r, mat.get(r, q));
                }

                for r in (q + 1)..n {
                    let apr = mat.get(p, r);
                    let aqr = mat.get(q, r);
                    mat.set(p, r, apr - s * (aqr + apr * tau));
                    mat.set(r, p, mat.get(p, r));
                    mat.set(q, r, aqr + s * (apr - aqr * tau));
                    mat.set(r, q, mat.get(q, r));
                }

                // Accumulate eigenvectors
                for r in 0..n {
                    let vrp = eigenvectors.get(r, p);
                    let vrq = eigenvectors.get(r, q);
                    eigenvectors.set(r, p, vrp - s * (vrq + vrp * tau));
                    eigenvectors.set(r, q, vrq + s * (vrp - vrq * tau));
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    for i in 0..n {
        eigenvalues[i] = mat.get(i, i);
    }

    // Sort eigenvalues and corresponding eigenvector columns in ascending order
    for i in 0..n {
        let mut min_idx = i;
        for j in (i + 1)..n {
            if eigenvalues[j] < eigenvalues[min_idx] {
                min_idx = j;
            }
        }
        if min_idx != i {
            eigenvalues.swap(i, min_idx);
            for r in 0..n {
                let tmp = eigenvectors.get(r, i);
                eigenvectors.set(r, i, eigenvectors.get(r, min_idx));
                eigenvectors.set(r, min_idx, tmp);
            }
        }
    }

    sweeps
}
