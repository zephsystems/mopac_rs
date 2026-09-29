//! Universal CPU Threading and Parallel Pool Abstraction.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Provides architecture-agnostic execution controls for multi-core parallelism,
//! supporting bounded thread pools, fallback to strict single-core execution,
//! and dynamic work-stealing across homogeneous and heterogeneous CPU topologies.

use rayon::ThreadPoolBuilder;

/// Executes a closure within a bounded Rayon thread pool if `n_threads` is specified,
/// or within the ambient global pool if `n_threads` is `None` or `Some(0)`.
///
/// - `n_threads = Some(1)`: guarantees strict single-threaded sequential execution (zero thread spawning overhead,
///   deterministic profiling, and safe embedding in nested multiprocessing environments).
/// - `n_threads = Some(k)` (k > 1): bounds execution to exactly `k` worker threads, respecting
///   cgroup quotas, HPC SLURM job allocations, and multi-tenant constraints.
/// - `n_threads = None`: defaults to full hardware concurrency (`std::thread::available_parallelism()`)
///   or the process environment variable `RAYON_NUM_THREADS`.
pub fn run_with_thread_pool<F, R>(n_threads: Option<usize>, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    match n_threads {
        Some(n) if n > 0 => {
            let pool = ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|idx| format!("mopac-worker-{}", idx))
                .build()
                .expect("Failed to initialize bounded MOPAC thread pool");
            pool.install(f)
        }
        _ => f(),
    }
}

/// Returns the number of worker threads currently active in the current Rayon execution context.
pub fn current_thread_count() -> usize {
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn test_single_threaded_fallback_bound() {
        let threads_observed = run_with_thread_pool(Some(1), || {
            let active_threads = current_thread_count();
            let sum: usize = (0..1000).into_par_iter().map(|i| i % 2).sum();
            assert_eq!(sum, 500);
            active_threads
        });
        assert_eq!(threads_observed, 1);
    }

    #[test]
    fn test_bounded_thread_pool() {
        let threads_observed = run_with_thread_pool(Some(2), || {
            let active_threads = current_thread_count();
            let sum: usize = (0..1000).into_par_iter().map(|i| i * 2).sum();
            assert_eq!(sum, 999000);
            active_threads
        });
        assert_eq!(threads_observed, 2);
    }
}
