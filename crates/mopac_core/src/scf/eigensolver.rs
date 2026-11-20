//! High-Precision Pure Rust Real Symmetric Eigensolver.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Implements Householder tridiagonalization (`tred2`) with implicitly shifted QL (`tql2`)
//! for $O(N^3)$ symmetric eigendecomposition $F C = C \epsilon$, with cyclic Jacobi fallback
//! for tiny matrices ($N \le 12$).

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

    diagonalize_symmetric_submatrix_with_work(a, n, work, eigenvalues, eigenvectors)
}

/// Diagonalize a leading submatrix of dimension `dim x dim` of a real symmetric matrix $A$.
///
/// On exit:
/// * `eigenvalues[0..dim]` contains the sorted eigenvalues.
/// * `eigenvectors` contains the orthonormal eigenvectors in the leading `dim x dim` block.
/// * Strictly 0 heap allocations when reusing `work`.
pub fn diagonalize_symmetric_submatrix_with_work(
    a: &AlignedMatrix<f64>,
    dim: usize,
    work: &mut AlignedMatrix<f64>,
    eigenvalues: &mut [f64],
    eigenvectors: &mut AlignedMatrix<f64>,
) -> usize {
    let n = dim;
    assert!(a.rows >= n && a.cols >= n);
    assert!(work.rows >= n && work.cols >= n);
    assert!(eigenvalues.len() >= n);
    assert!(eigenvectors.rows >= n && eigenvectors.cols >= n);

    if n == 0 {
        return 0;
    }
    if n == 1 {
        eigenvalues[0] = a.get(0, 0);
        eigenvectors.set(0, 0, 1.0);
        return 0;
    }

    if n > 12 && diagonalize_tridiagonal_ql(a, n, work, &mut eigenvalues[..n], eigenvectors).is_ok()
    {
        sort_eigenpairs(n, &mut eigenvalues[..n], eigenvectors);
        return 1;
    }

    let sweeps = diagonalize_jacobi_submatrix(a, n, work, &mut eigenvalues[..n], eigenvectors);
    sort_eigenpairs(n, &mut eigenvalues[..n], eigenvectors);
    sweeps
}

/// Householder tridiagonalization (`tred2`) followed by implicit-shift QL (`tql2`).
/// Uses `eigenvectors` for accumulation and `work.row_mut(0)[..n]` for the subdiagonal vector `e`,
/// guaranteeing zero heap allocations.
fn diagonalize_tridiagonal_ql(
    a: &AlignedMatrix<f64>,
    n: usize,
    work: &mut AlignedMatrix<f64>,
    d: &mut [f64],
    z: &mut AlignedMatrix<f64>,
) -> Result<(), ()> {
    // Copy leading n x n block into z (row-major)
    for i in 0..n {
        let a_row = a.row(i);
        let z_row = z.row_mut(i);
        z_row[..n].copy_from_slice(&a_row[..n]);
    }

    // Use first row of `work` as subdiagonal vector `e` (size >= n)
    let e = &mut work.row_mut(0)[..n];
    e.fill(0.0);
    d.fill(0.0);

    // 1. Householder reduction to symmetric tridiagonal form (`tred2`)
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0f64;
        let mut scale = 0.0f64;

        if l > 0 {
            let z_i = z.row(i);
            for &val in &z_i[..=l] {
                scale += val.abs();
            }
            if scale == 0.0 {
                e[i] = z_i[l];
            } else {
                let inv_scale = 1.0 / scale;
                for k in 0..=l {
                    let v = z.get(i, k) * inv_scale;
                    z.set(i, k, v);
                    h += v * v;
                }
                let f = z.get(i, l);
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                z.set(i, l, f - g);

                let mut f_acc = 0.0f64;
                for (j, ej) in e.iter_mut().enumerate().take(l + 1) {
                    // Store u / H in column i of z
                    let z_ij = z.get(i, j);
                    z.set(j, i, z_ij / h);

                    let mut g_acc = 0.0f64;
                    let z_j = z.row(j);
                    let z_i = z.row(i);
                    for k in 0..=j {
                        g_acc += z_j[k] * z_i[k];
                    }
                    for k in (j + 1)..=l {
                        g_acc += z.get(k, j) * z.get(i, k);
                    }
                    *ej = g_acc / h;
                    f_acc += *ej * z.get(i, j);
                }

                let hh = f_acc / (h + h);
                for j in 0..=l {
                    let f_j = z.get(i, j);
                    let g_j = e[j] - hh * f_j;
                    e[j] = g_j;
                    for (k, &ek) in e.iter().enumerate().take(j + 1) {
                        let cur = z.get(j, k);
                        z.set(j, k, cur - (f_j * ek + g_j * z.get(i, k)));
                    }
                }
            }
        } else {
            e[i] = z.get(i, l);
        }
        d[i] = h;
    }

    d[0] = 0.0;
    e[0] = 0.0;

    // Accumulate transformation matrices Q in z
    for (i, di) in d.iter_mut().enumerate().take(n) {
        if *di != 0.0 {
            for j in 0..i {
                let mut g = 0.0f64;
                for k in 0..i {
                    g += z.get(i, k) * z.get(k, j);
                }
                for k in 0..i {
                    let cur = z.get(k, j);
                    z.set(k, j, cur - g * z.get(k, i));
                }
            }
        }
        *di = z.get(i, i);
        z.set(i, i, 1.0);
        for j in 0..i {
            z.set(j, i, 0.0);
            z.set(i, j, 0.0);
        }
    }

    // 2. Implicitly shifted QL algorithm (`tql2`)
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    let mut f = 0.0f64;
    let mut tst1 = 0.0f64;

    for l in 0..n {
        let mut iter = 0usize;
        let h_norm = d[l].abs() + e[l].abs();
        if tst1 < h_norm {
            tst1 = h_norm;
        }

        loop {
            let mut m = l;
            while m < n - 1 {
                let tst2 = tst1 + e[m].abs();
                if tst2 == tst1 {
                    break;
                }
                m += 1;
            }

            if m == l {
                break;
            }
            if iter >= 30 {
                return Err(());
            }
            iter += 1;

            // Wilkinson shift
            let g = d[l];
            let p_val = (d[l + 1] - g) / (2.0 * e[l]);
            let r_val = p_val.hypot(1.0);
            let signed_r = if p_val >= 0.0 { r_val } else { -r_val };
            let denom = p_val + signed_r;
            let shift = g - e[l] / denom;
            d[l] = e[l] / denom;
            d[l + 1] = e[l] * denom;
            let dl1 = d[l + 1];
            for item in d.iter_mut().take(n).skip(l + 2) {
                *item -= shift;
            }
            f += shift;

            // QL plane rotations
            let mut p = d[m];
            let mut c = 1.0f64;
            let mut c2 = 1.0f64;
            let mut c3 = 1.0f64;
            let el1 = e[l + 1];
            let mut s = 0.0f64;
            let mut s2 = 0.0f64;

            for i in (l..m).rev() {
                c3 = c2;
                c2 = c;
                s2 = s;
                let g_rot = c * e[i];
                let h_rot = c * p;
                let r_rot = p.hypot(e[i]);
                e[i + 1] = s * r_rot;
                s = e[i] / r_rot;
                c = p / r_rot;
                p = c * d[i] - s * g_rot;
                d[i + 1] = h_rot + s * (c * g_rot + s * d[i]);

                for k in 0..n {
                    let z_row = z.row_mut(k);
                    let h_z = z_row[i + 1];
                    let z_ki = z_row[i];
                    z_row[i + 1] = s * z_ki + c * h_z;
                    z_row[i] = c * z_ki - s * h_z;
                }
            }

            p = -s * s2 * c3 * el1 * e[l] / dl1;
            e[l] = s * p;
            d[l] = c * p;
        }

        d[l] += f;
    }

    Ok(())
}

/// Classical cyclic Jacobi symmetric eigensolver fallback.
fn diagonalize_jacobi_submatrix(
    a: &AlignedMatrix<f64>,
    n: usize,
    work: &mut AlignedMatrix<f64>,
    eigenvalues: &mut [f64],
    eigenvectors: &mut AlignedMatrix<f64>,
) -> usize {
    // Working copy of leading submatrix
    for i in 0..n {
        for j in 0..n {
            work.set(i, j, a.get(i, j));
        }
    }
    let mat = work;

    // Initialize eigenvectors to identity matrix
    for i in 0..n {
        for j in 0..n {
            eigenvectors.set(i, j, if i == j { 1.0 } else { 0.0 });
        }
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
    for (i, ev) in eigenvalues.iter_mut().enumerate().take(n) {
        *ev = mat.get(i, i);
    }

    sweeps
}

/// Sort eigenvalues ascending and permute eigenvector columns identically.
fn sort_eigenpairs(n: usize, eigenvalues: &mut [f64], eigenvectors: &mut AlignedMatrix<f64>) {
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
                let row = eigenvectors.row_mut(r);
                row.swap(i, min_idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_deterministic_symmetric(n: usize) -> AlignedMatrix<f64> {
        let mut a = AlignedMatrix::zeroed(n, n);
        for i in 0..n {
            for j in 0..=i {
                let val = if i == j {
                    (i as f64) * 0.13 + ((i % 7) as f64) * 0.41 - 1.5
                } else {
                    0.35 * ((i as f64 + 1.7) * (j as f64 + 0.9)).sin() / (1.0 + (i - j) as f64)
                };
                a.set(i, j, val);
                a.set(j, i, val);
            }
        }
        a
    }

    #[test]
    fn test_tred2_tql2_parity_and_invariants_up_to_150() {
        for &n in &[15usize, 64, 150] {
            let a = build_deterministic_symmetric(n);
            let mut evals = AlignedVec64::zeroed(n);
            let mut evecs = AlignedMatrix::zeroed(n, n);

            let ret = diagonalize_symmetric(&a, &mut evals, &mut evecs);
            assert_eq!(ret, 1);

            for i in 1..n {
                assert!(evals[i - 1] <= evals[i] + 1e-12);
            }

            // Orthonormality C^T * C = I and eigen-residual A * C = C * diag(E)
            let mut max_orth_err = 0.0f64;
            let mut max_res_err = 0.0f64;
            for j in 0..n {
                for i in 0..n {
                    let mut dot_orth = 0.0;
                    let mut ac_ij = 0.0;
                    for k in 0..n {
                        dot_orth += evecs.get(k, i) * evecs.get(k, j);
                        ac_ij += a.get(i, k) * evecs.get(k, j);
                    }
                    let target_orth = if i == j { 1.0 } else { 0.0 };
                    max_orth_err = max_orth_err.max((dot_orth - target_orth).abs());
                    let ce_ij = evecs.get(i, j) * evals[j];
                    max_res_err = max_res_err.max((ac_ij - ce_ij).abs());
                }
            }
            assert!(
                max_orth_err < 1e-11,
                "Orthonormality error too high for n={}: {:e}",
                n,
                max_orth_err
            );
            assert!(
                max_res_err < 1e-11,
                "Residual ||A C - C diag(E)|| too high for n={}: {:e}",
                n,
                max_res_err
            );

            if n <= 64 {
                let mut work = a.clone();
                let mut j_evals = vec![0.0f64; n];
                let mut j_evecs = AlignedMatrix::zeroed(n, n);
                diagonalize_jacobi_submatrix(&a, n, &mut work, &mut j_evals, &mut j_evecs);
                sort_eigenpairs(n, &mut j_evals, &mut j_evecs);

                let max_eval_diff = evals
                    .iter()
                    .zip(j_evals.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f64, f64::max);
                assert!(
                    max_eval_diff < 1e-11,
                    "tql2 vs Jacobi mismatch for n={}: {:e}",
                    n,
                    max_eval_diff
                );
            }
        }
    }
}
