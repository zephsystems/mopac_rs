//! Direct Inversion in the Iterative Subspace (DIIS / Pulay Converger).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Implements Pulay's DIIS acceleration algorithm for Self-Consistent Field (SCF) iterations.
//! Reference:
//! - P. Pulay, "Convergence acceleration of iterative sequences. The case of SCF iteration",
//!   Chem. Phys. Lett. 73, 393-398 (1980).
//! - P. Pulay, "Improved SCF convergence acceleration", J. Comput. Chem. 3, 556-560 (1982).
//!
//! In semi-empirical quantum chemistry (orthogonal basis, $S = I$), the commutator error
//! between the Fock matrix $F$ and density matrix $P$ represents the orbital rotation gradient:
//! $$e = [F, P] = FP - PF$$
//! Since $F$ and $P$ are symmetric, $(FP)^T = PF$, meaning $e$ is anti-symmetric ($e^T = -e$)
//! with zero diagonal elements.
//!
//! Pulay DIIS constructs an optimal linear combination of past Fock matrices:
//! $$F^* = \sum_{k=1}^m c_k F_k$$
//! minimizing the Frobenius norm of the interpolated error $\|e^*\|^2$ subject to $\sum c_k = 1$.
//!
//! This module performs **zero heap allocations** during the SCF cycle by utilizing
//! pre-allocated 64-byte cache-line aligned matrix ring buffers.

use crate::types::AlignedMatrix;

/// Maximum number of past Fock/Error matrices retained in the DIIS subspace.
pub const MAX_DIIS_CAPACITY: usize = 8;

/// Default active subspace size for closed-shell SCF.
pub const DEFAULT_MAX_DIIS: usize = 6;

/// Result summary of a single DIIS step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiisStepResult {
    /// Whether the Fock matrix was extrapolated via DIIS
    pub extrapolated: bool,
    /// Number of error vectors in the current active subspace
    pub subspace_size: usize,
    /// Maximum absolute commutator error element $\max_{ij} |[F, P]_{ij}|$
    pub max_error: f64,
    /// Root-mean-square commutator error
    pub rms_error: f64,
}

/// Pre-allocated workspace for Pulay DIIS convergence acceleration.
#[derive(Debug, Clone)]
pub struct DiisWorkspace {
    pub norbs: usize,
    pub max_subspace: usize,
    pub num_stored: usize,
    pub active_slots: [usize; MAX_DIIS_CAPACITY],
    pub fock_history: Vec<AlignedMatrix<f64>>,
    pub error_history: Vec<AlignedMatrix<f64>>,
    pub b_mat: [[f64; MAX_DIIS_CAPACITY]; MAX_DIIS_CAPACITY],
}

impl DiisWorkspace {
    /// Allocate pre-sized DIIS history buffers for a system of `norbs` basis functions.
    pub fn allocate(norbs: usize, max_subspace: usize) -> Self {
        let cap = max_subspace.clamp(2, MAX_DIIS_CAPACITY);
        let mut fock_history = Vec::with_capacity(cap);
        let mut error_history = Vec::with_capacity(cap);
        let mut active_slots = [0usize; MAX_DIIS_CAPACITY];

        for (i, slot) in active_slots.iter_mut().enumerate().take(cap) {
            fock_history.push(AlignedMatrix::zeroed(norbs, norbs));
            error_history.push(AlignedMatrix::zeroed(norbs, norbs));
            *slot = i;
        }

        Self {
            norbs,
            max_subspace: cap,
            num_stored: 0,
            active_slots,
            fock_history,
            error_history,
            b_mat: [[0.0; MAX_DIIS_CAPACITY]; MAX_DIIS_CAPACITY],
        }
    }

    /// Reset all subspace counters and the $B$ matrix without reallocating buffers.
    pub fn reset(&mut self) {
        self.num_stored = 0;
        for i in 0..self.max_subspace {
            self.active_slots[i] = i;
            for j in 0..self.max_subspace {
                self.b_mat[i][j] = 0.0;
            }
        }
    }

    /// Drops the oldest vector from the active subspace to resolve linear dependence.
    pub fn drop_oldest(&mut self) {
        self.drop_vector(0);
    }

    /// Drops a specific vector by active index `idx` from the active subspace,
    /// shifting subsequent active slots and B matrix rows/columns in-place.
    pub fn drop_vector(&mut self, idx: usize) {
        if idx >= self.num_stored {
            return;
        }
        if self.num_stored <= 1 {
            self.reset();
            return;
        }
        let dropped_slot = self.active_slots[idx];
        let n = self.num_stored;
        // Shift active slots down
        for i in idx..(n - 1) {
            self.active_slots[i] = self.active_slots[i + 1];
        }
        self.active_slots[n - 1] = dropped_slot;

        // Shift B matrix rows
        for i in idx..(n - 1) {
            for j in 0..n {
                self.b_mat[i][j] = self.b_mat[i + 1][j];
            }
        }
        // Shift B matrix columns
        for j in idx..(n - 1) {
            for i in 0..(n - 1) {
                self.b_mat[i][j] = self.b_mat[i][j + 1];
            }
        }
        // Zero unused boundary row and column
        for k in 0..self.max_subspace {
            self.b_mat[n - 1][k] = 0.0;
            self.b_mat[k][n - 1] = 0.0;
        }
        self.num_stored -= 1;
    }

    /// Intelligently prunes the most linearly dependent (collinear) or worst outlier error vector.
    ///
    /// Rather than blindly discarding the oldest vector (which may carry critical low-frequency
    /// gradient information), this selective pruning computes the cross-correlation matrix
    /// $$\rho_{ij} = \frac{|\langle e_i, e_j \rangle|}{\sqrt{\langle e_i, e_i \rangle \langle e_j, e_j \rangle}}$$
    /// and drops the older vector of the most collinear pair ($\rho_{ij} > 0.98$).
    /// If no collinearity is found, it drops the vector with largest residual norm.
    pub fn prune_worst_vector(&mut self) {
        let m = self.num_stored;
        if m <= 1 {
            self.reset();
            return;
        }

        let mut max_corr = 0.0f64;
        let mut drop_idx = 0usize;

        for i in 0..m {
            let dii = self.b_mat[i][i].max(1e-30);
            for j in (i + 1)..m {
                let djj = self.b_mat[j][j].max(1e-30);
                let corr = self.b_mat[i][j].abs() / (dii * djj).sqrt();
                if corr > max_corr {
                    max_corr = corr;
                    drop_idx = i; // Drop the older vector of the collinear pair
                }
            }
        }

        if max_corr > 0.98 {
            self.drop_vector(drop_idx);
        } else {
            // Find vector with maximum norm excluding the newest vector (m - 1)
            let mut max_norm = 0.0f64;
            let mut worst_norm_idx = 0usize;
            for i in 0..(m - 1) {
                if self.b_mat[i][i] > max_norm {
                    max_norm = self.b_mat[i][i];
                    worst_norm_idx = i;
                }
            }
            if max_norm > 0.0 {
                self.drop_vector(worst_norm_idx);
            } else {
                self.drop_vector(0);
            }
        }
    }

    /// Add current Fock and Density matrices to DIIS history, compute commutator $[F, P]$,
    /// and extrapolate the Fock matrix if $m \ge 2$.
    ///
    /// Guaranteed **zero heap allocations**.
    pub fn push_and_extrapolate(
        &mut self,
        fock: &mut AlignedMatrix<f64>,
        density: &AlignedMatrix<f64>,
        tmp_mult: &mut AlignedMatrix<f64>,
    ) -> DiisStepResult {
        let norbs = self.norbs;

        // 1. Compute M = F * P into tmp_mult.
        // Contiguous row-slice traversal for auto-vectorization.
        for i in 0..norbs {
            let f_row = fock.row(i);
            let m_row = tmp_mult.row_mut(i);
            m_row.fill(0.0);
            for (k, &f_ik) in f_row.iter().enumerate().take(norbs) {
                let p_row = density.row(k);
                for j in 0..norbs {
                    m_row[j] += f_ik * p_row[j];
                }
            }
        }

        // 2. Determine target slot for the new history point
        let (new_idx, target_slot) = if self.num_stored < self.max_subspace {
            let idx = self.num_stored;
            let slot = self.active_slots[idx];
            self.num_stored += 1;
            (idx, slot)
        } else {
            // Ring buffer rotation: oldest (slot 0) shifted out, re-used at tail
            let oldest_slot = self.active_slots[0];
            let max_s = self.max_subspace;
            for i in 0..(max_s - 1) {
                self.active_slots[i] = self.active_slots[i + 1];
                for j in 0..(max_s - 1) {
                    self.b_mat[i][j] = self.b_mat[i + 1][j + 1];
                }
            }
            self.active_slots[max_s - 1] = oldest_slot;
            (max_s - 1, oldest_slot)
        };

        // 3. Compute commutator error e = M - M^T: e_ij = (FP)_ij - (PF)_ij = M_ij - M_ji
        // Anti-symmetric: e_ji = -e_ij, e_ii = 0.
        let mut max_err = 0.0f64;
        let mut sum_sq_err = 0.0f64;
        {
            let err_mat = &mut self.error_history[target_slot];
            for i in 0..norbs {
                for j in 0..norbs {
                    let val = tmp_mult.get(i, j) - tmp_mult.get(j, i);
                    err_mat.set(i, j, val);
                    let abs_val = val.abs();
                    if abs_val > max_err {
                        max_err = abs_val;
                    }
                    sum_sq_err += val * val;
                }
            }
        }

        let rms_err = (sum_sq_err / (norbs * norbs).max(1) as f64).sqrt();

        // 4. Store current unextrapolated Fock matrix into history
        self.fock_history[target_slot]
            .data
            .copy_from_slice(&fock.data);

        // 5. Update row and column new_idx of the B matrix: B_ij = <e_i, e_j>
        let m = self.num_stored;
        for j in 0..m {
            let other_slot = self.active_slots[j];
            let other_err = &self.error_history[other_slot];
            let new_err = &self.error_history[target_slot];
            let mut dot = 0.0f64;
            for idx in 0..(norbs * norbs) {
                dot += new_err.data[idx] * other_err.data[idx];
            }
            self.b_mat[new_idx][j] = dot;
            self.b_mat[j][new_idx] = dot;
        }

        // 6. If fewer than 2 vectors, extrapolation is not possible yet
        if m < 2 {
            return DiisStepResult {
                extrapolated: false,
                subspace_size: m,
                max_error: max_err,
                rms_error: rms_err,
            };
        }

        // 7. Solve Pulay linear system, dropping oldest vectors if ill-conditioned
        let mut current_m = m;
        let mut coeffs = [0.0f64; MAX_DIIS_CAPACITY];
        let mut solved = false;

        while current_m >= 2 {
            if solve_pulay_system(&self.b_mat, current_m, &mut coeffs) {
                solved = true;
                break;
            }
            // Linear dependency encountered: intelligently prune collinear/outlier vector
            self.prune_worst_vector();
            current_m = self.num_stored;
        }

        if !solved {
            // Cannot solve: reset DIIS buffer and keep only newest vector for future steps
            self.reset();
            self.active_slots[0] = target_slot;
            self.num_stored = 1;
            return DiisStepResult {
                extrapolated: false,
                subspace_size: 1,
                max_error: max_err,
                rms_error: rms_err,
            };
        }

        // 8. Extrapolate Fock matrix: F* = sum_{j=0}^{m-1} c_j * F_j
        fock.fill_zero();
        for (j, &slot_idx) in self.active_slots.iter().take(current_m).enumerate() {
            let c = coeffs[j];
            let past_fock = &self.fock_history[slot_idx];
            for idx in 0..(norbs * norbs) {
                fock.data[idx] += c * past_fock.data[idx];
            }
        }

        DiisStepResult {
            extrapolated: true,
            subspace_size: current_m,
            max_error: max_err,
            rms_error: rms_err,
        }
    }
}

/// Solves the augmented Pulay DIIS linear system using Gaussian elimination with partial pivoting.
///
/// System structure:
/// $$
/// \begin{pmatrix}
/// \tilde{B}_{0,0} & \dots & \tilde{B}_{0,m-1} & -1 \\
/// \vdots & \ddots & \vdots & \vdots \\
/// \tilde{B}_{m-1,0} & \dots & \tilde{B}_{m-1,m-1} & -1 \\
/// -1 & \dots & -1 & 0
/// \end{pmatrix}
/// \begin{pmatrix} c_0 \\ \vdots \\ c_{m-1} \\ \tilde{\lambda} \end{pmatrix}
/// =
/// \begin{pmatrix} 0 \\ \vdots \\ 0 \\ -1 \end{pmatrix}
/// $$
///
/// Returns `true` if a stable, uncorrupted solution with $|c_k| \le 50.0$ was obtained.
#[allow(clippy::needless_range_loop)]
pub fn solve_pulay_system(
    b_mat: &[[f64; MAX_DIIS_CAPACITY]; MAX_DIIS_CAPACITY],
    m: usize,
    coeffs: &mut [f64; MAX_DIIS_CAPACITY],
) -> bool {
    assert!((2..=MAX_DIIS_CAPACITY).contains(&m));
    let k_dim = m + 1;
    const MAX_K: usize = MAX_DIIS_CAPACITY + 1;

    // Determine max diagonal element for condition scaling
    let mut b_max = 0.0f64;
    for (i, row) in b_mat.iter().enumerate().take(m) {
        let d = row[i];
        if d > b_max {
            b_max = d;
        }
    }

    if b_max < 1e-16 {
        return false;
    }
    let scale = 1.0 / b_max;

    let mut a = [[0.0f64; MAX_K]; MAX_K];
    let mut b = [0.0f64; MAX_K];

    for i in 0..m {
        for j in 0..m {
            a[i][j] = b_mat[i][j] * scale;
        }
        a[i][m] = -1.0;
        a[m][i] = -1.0;
        b[i] = 0.0;
    }
    a[m][m] = 0.0;
    b[m] = -1.0;

    // Gaussian Elimination with Partial Pivoting
    for k in 0..k_dim {
        // Find pivot row
        let mut max_val = a[k][k].abs();
        let mut pivot_row = k;
        for (p, row) in a.iter().enumerate().take(k_dim).skip(k + 1) {
            let val = row[k].abs();
            if val > max_val {
                max_val = val;
                pivot_row = p;
            }
        }

        if max_val < 1e-12 {
            // Singular or near-singular system
            return false;
        }

        // Swap pivot row with current row k
        if pivot_row != k {
            for col in 0..k_dim {
                let tmp = a[k][col];
                a[k][col] = a[pivot_row][col];
                a[pivot_row][col] = tmp;
            }
            b.swap(k, pivot_row);
        }

        // Eliminate column k in rows below k
        let pivot = a[k][k];
        for row in (k + 1)..k_dim {
            let factor = a[row][k] / pivot;
            a[row][k] = 0.0;
            for col in (k + 1)..k_dim {
                let ak_col = a[k][col];
                a[row][col] -= factor * ak_col;
            }
            b[row] -= factor * b[k];
        }
    }

    // Back substitution
    let mut sol = [0.0f64; MAX_K];
    for row in (0..k_dim).rev() {
        let mut sum = b[row];
        for col in (row + 1)..k_dim {
            sum -= a[row][col] * sol[col];
        }
        sol[row] = sum / a[row][row];
    }

    // Sanity validation of coefficients
    let mut sum_c = 0.0f64;
    for i in 0..m {
        let c = sol[i];
        if c.is_nan() || c.is_infinite() || c.abs() > 50.0 {
            return false;
        }
        coeffs[i] = c;
        sum_c += c;
    }

    // Normalization constraint check: sum(c_i) == 1.0
    if (sum_c - 1.0).abs() > 1e-3 {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diis_selective_pruning_collinear() {
        let mut ws = DiisWorkspace::allocate(4, 6);
        ws.num_stored = 3;
        // Vector 0 and Vector 1 collinear: corr ~ 0.999
        ws.b_mat[0][0] = 1.0;
        ws.b_mat[1][1] = 1.0;
        ws.b_mat[2][2] = 2.0;

        ws.b_mat[0][1] = 0.999;
        ws.b_mat[1][0] = 0.999;

        ws.b_mat[0][2] = 0.1;
        ws.b_mat[2][0] = 0.1;

        ws.b_mat[1][2] = 0.1;
        ws.b_mat[2][1] = 0.1;

        ws.prune_worst_vector();

        assert_eq!(ws.num_stored, 2);
        // Collinear vector 0 should have been dropped, preserving vector 1 (now at index 0)
        assert_eq!(ws.b_mat[0][0], 1.0);
        assert_eq!(ws.b_mat[1][1], 2.0);
    }

    #[test]
    fn test_diis_drop_vector_middle() {
        let mut ws = DiisWorkspace::allocate(4, 6);
        ws.num_stored = 4;
        for i in 0..4 {
            ws.active_slots[i] = i;
            ws.b_mat[i][i] = (i + 1) as f64;
        }

        // Drop index 1
        ws.drop_vector(1);
        assert_eq!(ws.num_stored, 3);
        assert_eq!(ws.b_mat[0][0], 1.0);
        assert_eq!(ws.b_mat[1][1], 3.0);
        assert_eq!(ws.b_mat[2][2], 4.0);
    }
}
