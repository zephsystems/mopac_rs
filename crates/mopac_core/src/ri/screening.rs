//! Cauchy-Schwarz Integral Screening for 3-Center and 4-Center Tensors.
//!
//! Applies the rigorous Cauchy-Schwarz upper bound:
//! $|(\mu \nu | P)| \le \sqrt{(\mu \nu | \mu \nu) (P | P)}$
//! to eliminate negligible integral blocks prior to tensor contractions.

/// Cauchy-Schwarz screening table for basis function pairs and auxiliary functions.
#[derive(Debug, Clone)]
pub struct CauchySchwarzScreening {
    /// Upper bound estimates for orbital pairs: sqrt((mu nu | mu nu))
    pub pair_bounds: Vec<f64>,
    /// Upper bound estimates for auxiliary basis functions: sqrt((P | P))
    pub aux_bounds: Vec<f64>,
    /// Screening threshold below which integrals are skipped (e.g. 1e-10 eV)
    pub threshold: f64,
    pub norbs: usize,
    pub naux: usize,
}

impl CauchySchwarzScreening {
    /// Create a new screening table with specified threshold.
    pub fn new(norbs: usize, naux: usize, threshold: f64) -> Self {
        Self {
            pair_bounds: vec![0.0; norbs * norbs],
            aux_bounds: vec![0.0; naux],
            threshold,
            norbs,
            naux,
        }
    }

    /// Set the diagonal pair self-interaction estimate $\sqrt{(\mu \nu | \mu \nu)}$.
    #[inline(always)]
    pub fn set_pair_bound(&mut self, mu: usize, nu: usize, val: f64) {
        self.pair_bounds[mu * self.norbs + nu] = val;
    }

    /// Set the auxiliary diagonal metric estimate $\sqrt{(P | P)}$.
    #[inline(always)]
    pub fn set_aux_bound(&mut self, p: usize, val: f64) {
        self.aux_bounds[p] = val;
    }

    /// Returns true if the 3-center integral $(\mu \nu | P)$ is mathematically guaranteed
    /// to be smaller than the screening threshold.
    #[inline(always)]
    pub fn is_screened(&self, mu: usize, nu: usize, p: usize) -> bool {
        let max_estimate = self.pair_bounds[mu * self.norbs + nu] * self.aux_bounds[p];
        max_estimate < self.threshold
    }
}
