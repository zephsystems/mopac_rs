//! High-Performance Tensorial Contractions for Density Fitting (RI-V).
//!
//! Replaces the $O(M^4)$ 4-center ERI evaluation with $O(M^2 N_{\text{aux}})$ and
//! $O(M^3 N_{\text{aux}})$ BLAS-3 / BLAS-2 tensor contractions for Coulomb ($J$)
//! and Exchange ($K$) operators.

use crate::ri::tensor_b::ThreeCenterTensorB;
use crate::types::{AlignedMatrix, AlignedVec64};

/// Evaluates the Coulomb matrix $J_{\mu\nu} = \sum_{\lambda\sigma} (\mu\nu|\lambda\sigma) P_{\lambda\sigma}$
/// using RI-V tensor contraction in $O(M^2 N_{\text{aux}})$ time.
///
/// Steps:
/// 1. $d_Q = \sum_{\lambda\sigma} B_{\lambda\sigma}^Q P_{\lambda\sigma}$
/// 2. $J_{\mu\nu} = \sum_Q B_{\mu\nu}^Q d_Q$
pub fn compute_coulomb_ri(
    b: &ThreeCenterTensorB,
    density: &AlignedMatrix<f64>,
    j_matrix: &mut AlignedMatrix<f64>,
    d_aux: &mut AlignedVec64<f64>,
) {
    let norbs = b.norbs;
    let naux = b.naux;
    assert_eq!(density.rows, norbs);
    assert_eq!(density.cols, norbs);
    assert_eq!(j_matrix.rows, norbs);
    assert_eq!(j_matrix.cols, norbs);
    assert_eq!(d_aux.len(), naux);

    // 1. Compute auxiliary density vector d_Q = Tr(B^Q * P)
    d_aux.fill(0.0);
    for lam in 0..norbs {
        for sig in 0..norbs {
            let p_val = density.get(lam, sig);
            if p_val.abs() > 1e-15 {
                let slice = b.pair_slice(lam, sig);
                for q in 0..naux {
                    d_aux[q] += slice[q] * p_val;
                }
            }
        }
    }

    // 2. Compute J_mu_nu = sum_Q B_mu_nu^Q d_Q
    for mu in 0..norbs {
        for nu in 0..norbs {
            let slice = b.pair_slice(mu, nu);
            let mut sum = 0.0;
            for q in 0..naux {
                sum += slice[q] * d_aux[q];
            }
            j_matrix.set(mu, nu, sum);
        }
    }
}

/// Evaluates the Exchange matrix $K_{\mu\nu} = \sum_{\lambda\sigma} (\mu\lambda|\nu\sigma) P_{\lambda\sigma}$
/// using RI-V tensor contraction in $O(M^3 N_{\text{aux}})$ time.
///
/// Steps for each auxiliary index $Q \in 1..N_{\text{aux}}$:
/// 1. $W^Q = B^Q P$ (Matrix Multiplication GEMM)
/// 2. $K = \sum_Q W^Q (B^Q)^T$ (Matrix Multiplication GEMM)
pub fn compute_exchange_ri(
    b: &ThreeCenterTensorB,
    density: &AlignedMatrix<f64>,
    k_matrix: &mut AlignedMatrix<f64>,
    w_mat: &mut AlignedMatrix<f64>,
) {
    let norbs = b.norbs;
    let naux = b.naux;
    assert_eq!(density.rows, norbs);
    assert_eq!(density.cols, norbs);
    assert_eq!(k_matrix.rows, norbs);
    assert_eq!(k_matrix.cols, norbs);
    assert_eq!(w_mat.rows, norbs);
    assert_eq!(w_mat.cols, norbs);

    k_matrix.fill_zero();

    for q in 0..naux {
        // Step 1: W^Q = B^Q * P
        for mu in 0..norbs {
            for sig in 0..norbs {
                let mut sum = 0.0;
                for lam in 0..norbs {
                    sum += b.get(mu, lam, q) * density.get(lam, sig);
                }
                w_mat.set(mu, sig, sum);
            }
        }

        // Step 2: Accumulate into K: K_mu_nu += sum_sigma W_mu_sigma^Q * B_nu_sigma^Q
        for mu in 0..norbs {
            for nu in 0..norbs {
                let mut sum = 0.0;
                for sig in 0..norbs {
                    sum += w_mat.get(mu, sig) * b.get(nu, sig, q);
                }
                k_matrix.set(mu, nu, k_matrix.get(mu, nu) + sum);
            }
        }
    }
}
