//! 3-Center Density Fitting Intermediate Tensor B.
//!
//! Represents $B_{\mu\nu}^Q = \sum_P (\mu \nu | P) [V^{-1/2}]_{PQ}$ in contiguous 64-byte
//! aligned memory, structured for direct SIMD dot products and BLAS-3 GEMM operations.

use crate::types::AlignedVec64;

/// 3-Center Intermediate Tensor $B_{\mu\nu}^Q$.
///
/// Flattened memory layout: $(\mu, \nu)$ row-major, $Q$ contiguous inner dimension.
/// This ensures that the auxiliary vector $B_{\mu\nu}^{\bullet}$ for any orbital pair
/// $(\mu, \nu)$ occupies a contiguous 64-byte aligned cache line.
#[derive(Debug, Clone)]
pub struct ThreeCenterTensorB {
    pub norbs: usize,
    pub naux: usize,
    pub data: AlignedVec64<f64>,
}

impl ThreeCenterTensorB {
    /// Allocate an intermediate tensor for `norbs` basis functions and `naux` auxiliary functions.
    pub fn allocate(norbs: usize, naux: usize) -> Self {
        let total_size = norbs * norbs * naux;
        Self {
            norbs,
            naux,
            data: AlignedVec64::zeroed(total_size),
        }
    }

    /// Linear memory offset for element $(\mu, \nu, Q)$.
    #[inline(always)]
    pub fn offset(&self, mu: usize, nu: usize, q: usize) -> usize {
        (mu * self.norbs + nu) * self.naux + q
    }

    /// Get tensor element $B_{\mu\nu}^Q$.
    #[inline(always)]
    pub fn get(&self, mu: usize, nu: usize, q: usize) -> f64 {
        self.data[self.offset(mu, nu, q)]
    }

    /// Set tensor element $B_{\mu\nu}^Q$.
    #[inline(always)]
    pub fn set(&mut self, mu: usize, nu: usize, q: usize, val: f64) {
        let idx = self.offset(mu, nu, q);
        self.data[idx] = val;
    }

    /// Get a contiguous slice of auxiliary coefficients $B_{\mu\nu}^{\bullet}$ for pair $(\mu, \nu)$.
    #[inline(always)]
    pub fn pair_slice(&self, mu: usize, nu: usize) -> &[f64] {
        let start = (mu * self.norbs + nu) * self.naux;
        &self.data[start..(start + self.naux)]
    }

    /// Reconstruct 4-center integral $(\mu \nu | \lambda \sigma) \approx \sum_{Q=1}^{N_{\text{aux}}} B_{\mu\nu}^Q B_{\lambda\sigma}^Q$.
    ///
    /// Evaluated as a contiguous vector dot product with zero memory allocation.
    #[inline(always)]
    pub fn reconstruct_4center(&self, mu: usize, nu: usize, lam: usize, sig: usize) -> f64 {
        let s1 = self.pair_slice(mu, nu);
        let s2 = self.pair_slice(lam, sig);
        let mut sum = 0.0;
        for q in 0..self.naux {
            sum += s1[q] * s2[q];
        }
        sum
    }
}
