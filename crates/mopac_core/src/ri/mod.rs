//! Resolution of the Identity / Density Fitting (RI-V) Engine.
//!
//! Replaces $O(M^4)$ 4-center electron repulsion integrals with 3-center tensor
//! factorizations $B_{\mu\nu}^Q = \sum_P (\mu \nu | P) [V^{-1/2}]_{PQ}$.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").

pub mod blas_contractions;
pub mod cholesky;
pub mod screening;
pub mod tensor_b;

pub use blas_contractions::*;
pub use cholesky::*;
pub use screening::*;
pub use tensor_b::*;
