# Release Notes: MOPAC_RS v0.2.0

> **Status**: Official Release v0.2.0  
> **Date**: 2026-11-20  
> **Base Branch**: `main`  
> **Target Tag**: `v0.2.0`

---

## 1. Executive Summary

Version `0.2.0` of `mopac_rs` delivers Data-Oriented Programming (DOP) algorithmic speedups (`1.71x–42.87x`), zero-allocation geometry-cached parallel SCF assembly, compact linear-scaling COSMO reaction-field evaluation (`29.11x` less coupling memory), tiled >64-atom Vulkan volumetric density evaluation, charged RHF / open-shell UHF Python coverage, and single-SCF property pipelines (`QuantumSession`) while preserving canonical **OpenMOPAC v23.2.5** double-precision parity:

1. **$O(N^3)$ Tridiagonal Eigensolver (`mopac_core::scf::eigensolver`)**:
   - Replaced cyclic Jacobi sweeps for $N > 12$ with zero-allocation Householder tridiagonalization (`tred2`) and implicitly shifted QL (`tql2`), retaining Jacobi fallback for tiny matrices ($N \le 12$) and edge-case convergence guards.
   - Measured **`3.15x / 3.55x / 3.70x`** speedups on $64\times 64$, $150\times 150$, and $250\times 250$ symmetric matrices with `max|dLambda| = 0.00e+00`, orthonormality $\le 8.4 \times 10^{-15}$, and eigen-equation residual $\le 1.6 \times 10^{-13}$.
2. **Zero-Allocation Parallel NDDO Fock & Gradient Engines (`multipoles.rs`, `nuclear_gradients.rs`)**:
   - Introduced geometry-cached `NddoFockParallelPlan` in `ScfWorkspace` and `UhfWorkspace` with preallocated contiguous `diag_b_flat` arena (`0` heap allocations inside `execute()`) and work-weighted integral thresholds ($W = \sum |w_{AB}|$) spanning `s`, `sp`, and `d`-orbital regimes.
   - Eliminated the `v0.1.5` intermediate-pair regression valley (`C14H30` `0.41x -> 0.95x`, `C38H78` `1.68x`, `C48H98` `1.43x`; `S14` PM7 `d`-orbital parity `|dHOF| = 0.00007 kcal/mol` vs OpenMOPAC Fortran) with `|dFock_Fro| = 0.00e+00`.
   - Parallelized `precompute_diatomic_pairs` (`3.30x` on `C14H30`) and analytical RHF/UHF nuclear gradients via thread-local `DiatomicGradScratch` (`10.78x` on `C10H22`, `< 1e-12 eV/A` thread parity).
3. **Compact & Parallel COSMO Implicit Solvation (`mopac_core::solvation::cosmo`)**:
   - Compacted `CosmoState.b_matrix` from `(norbs^2 x nseg)` to intra-atom blocks (`29.11x` smaller footprint on `C14H30`, **`42.87x`** faster inner-SCF Fock step with `|dEdiel| = 0.00e+00 eV`).
   - Parallelized `CosmoState::initialize` boundary matrix $A$ rows and atom-outer compact $B$-matrix bands (`1.71x` init speedup on `C14H30`; end-to-end `C20H42` COSMO SCF+Grad **`8.28x`** faster with RSS reduced from `75 MiB` to `33 MiB`).
4. **Geometry Warm-Start, Charged RHF Optimization & Tightened Hessian Tolerances (`lbfgs.rs`, `hessian.rs`, `soscf.rs`)**:
   - Added `ScfWorkspace::reset_for_new_geometry(keep_density)`, `run_rhf_soscf_with_options`, and `OptimizationOptions::charge` enabling exact closed-shell ion optimizations (`NH4+` and `OH-` PM6 optimized to `|dHOF| = 0.00000 kcal/mol` vs OpenMOPAC Fortran; neutral `C6H14` **`7.54x`** faster, `RMSD = 4.86e-07 A`).
   - Parallelized Cartesian Hessian finite differences with tightened equilibrium/warm-start tolerances (`1e-9 / 1e-8`), achieving **`2.92x`** speedup on benzene PM6 with `0.000e+00 cm^-1` single- vs multi-thread drift and `|dZPVE| = 0.0006 kcal/mol` vs OpenMOPAC `FORCE`.
5. **Tiled Vulkan GPU Density Grid & Parallel Cube Export (`mopac_gpu`, `export/cube.rs`)**:
   - Replaced fixed 64-atom buffer truncation in `density_grid.comp` with tiled batch accumulation across arbitrary active-atom counts, reducing 72-atom dense cluster error from `3.88e-02` to **`4.00e-08`**.
   - Added `OnceLock`-cached `GpuCubeEvaluator` (`5.28x` faster repeated GPU density cubes) and slice-parallel `.cube` ASCII voxel formatting (`3.19x` CPU cube throughput).
6. **Python Single-SCF Pipelines, Open-Shell Properties & Correctness Fixes (`mopac_py`, `mopac_cli`)**:
   - Exposed `QuantumSession` (`5.97x` on 5-property workflow, `|dE_tot| = 1.82e-12 eV`), `compute_gradients=False` (`5.19x` on `C24H50`), open-shell UHF dispatch in `bond_orders`/cubes, and optional `charge`, `uhf`, and `multiplicity` parameters on `MopacCalculator`.
   - Fixed CLI `--gpu` silent NDDO disabling (`2.43914 -> 0.00000 kcal/mol` vs CPU), closed-shell charged RHF Mulliken occupation counts (`OH-` sum `+1.0 -> -1.0 e`), and `esp_charges` forwarding of `net_charge` (`OH-` panic -> `-1.0000 e`).

---

## 2. Crate-by-Crate Release Status (`v0.2.0`)

| Crate | Version | Key Updates in `v0.2.0` |
| :--- | :--- | :--- |
| **`mopac_core`** | `0.2.0` | `tred2`+`tql2` eigensolver, cached zero-malloc `NddoFockParallelPlan`, parallel gradients, compact/parallel COSMO, charged L-BFGS, tightened warm-start Hessian, `run_rhf_soscf_with_options`, `run_saddle_with_state`, parallel `.cube` formatter. |
| **`mopac_gpu`** | `0.2.0` | Tiled `density_grid.comp` / `.spv` removing the 64-atom active cluster truncation limit; parallel host voxel formatting. |
| **`mopac` (`mopac_cli`)** | `0.2.0` | Preserves NDDO under `--gpu` while routing GPU `gamma` strictly to non-NDDO runs; forwards parsed/CLI `charge` into `OptimizationOptions`. |
| **`mopac_py` (`mopac-py`)** | `0.2.0` | `QuantumSession`, `compute_gradients`, `OnceLock` GPU cube cache, open-shell UHF property/cube dispatch, `MopacCalculator` `charge`/`uhf`/`multiplicity` options, charged Mulliken/ESP/optimization fixes. |

---

## 3. Verification Summary

- `cargo fmt --all -- --check`: clean (0 diffs).
- `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets -- -D warnings`: clean (0 warnings).
- `RUSTFLAGS="-D warnings" cargo test --workspace`: all unit, integration, GPU parity, quantum theorem, and OpenMOPAC v23.2.5 differential tests passing.
- `python3 crates/mopac_py/tests/test_python_api.py`: 24/24 Python API tests passing against release build.
- `maturin` wheel & `benchmarks/provenance.py` stamp verification: `mopac_py.__version__ == "0.2.0"`.
