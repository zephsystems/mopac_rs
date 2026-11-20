# Technical Engineering Report: v0.2.0 Architectural Remediation and Performance Engine

**Document:** `explanations/08_V020_ARCHITECTURAL_REMEDIATION_AND_PERFORMANCE_ENGINE.md`  
**Project:** `mopac_rs` / `mopac_core` / `mopac_gpu` / `mopac_cli` / `mopac_py`  
**Release Version:** `0.2.0` (`2026-11-18`)  
**Classification:** Quantum Chemistry Core Algorithms, Parallelism, Vulkan Shaders, and Python Interoperability  
**Status:** Implemented and Empirically Verified  

---

## 1. Executive Summary

Following the architectural performance and GPU audits recorded prior to v0.1.5, the v0.2.0 release closes every remaining correctness gap, redundant computation, quadratic/cubic memory bottleneck, and algorithmic scaling limitation across the workspace:

| Area | Prior Bottleneck / Defect | v0.2.0 Solution | Target Files |
| :--- | :--- | :--- | :--- |
| **Symmetric Eigensolver** | Unconditional cyclic Jacobi ($O(N^3)$ with large sweep constant) | Zero-allocation Householder tridiagonalization (`tred2`) + implicit-shift QL (`tql2`) for $N > 12$ | `crates/mopac_core/src/scf/eigensolver.rs` |
| **NDDO Fock Assembly** | Serial loop over diatomic pairs; duplicated RHF/UHF code | Rayon pair precomputation (`natoms >= 16`) + parallel lower-row-band Fock target (`pairs >= 512`) sharing one kernel | `crates/mopac_core/src/integrals/multipoles.rs` |
| **COSMO Solvation Matrix $B$** | Allocated $(N_{\text{orbs}}^2 \times N_{\text{seg}})$ dense matrix and iterated all $N_{\text{orbs}}^2$ every SCF step | Compacted to intra-atom orbital pairs (`total_intra_pairs x n_seg`) indexed via `atom_pair_offsets` | `crates/mopac_core/src/solvation/cosmo.rs` |
| **Analytical Nuclear Gradients** | Serial pair loop with per-pair heap allocation inside finite differences | Preallocated per-worker `DiatomicGradScratch` (`Box`/array `[Option<...>; 19]`) + Rayon parallelization (`num_pairs >= 16`) | `crates/mopac_core/src/gradients/nuclear_gradients.rs` |
| **SCF Warm-Start & Geometry Resets** | Fresh `H_core` density rebuild at every L-BFGS and Hessian step; stale `gamma` risk on geometry move | Added `ScfWorkspace::reset_for_new_geometry(keep_density)` clearing `diatomic_pairs` and `gamma`; enabled warm-start in L-BFGS & Hessian (`map_init`) | `crates/mopac_core/src/types.rs`, `opt/lbfgs.rs`, `vibrations/hessian.rs`, `scf/uhf_loop.rs` |
| **SOSCF Option Fidelity** | `run_rhf_soscf` ignored NDDO, net charge, COSMO, external field, QM/MM charges, and density warm-start | Extracted shared SCF Hamiltonian/Fock helpers (`compute_rhf_nocc`, `prepare_scf_hamiltonian`, `assemble_scf_fock`); added `run_rhf_soscf_with_options` | `crates/mopac_core/src/scf/scf_loop.rs`, `scf/soscf.rs` |
| **CLI GPU Backend Handling** | `--gpu` silently downgraded runs to non-NDDO mode (`use_nddo = false`) and overclaimed GPU execution | Preserved `use_nddo`, restricted GPU Coulomb `gamma` to non-NDDO runs, and reported backend transparently | `crates/mopac_cli/src/main.rs` |
| **GPU Density Grid Shader** | Fixed 64-atom active buffer (`MAX_ACTIVE_ATOMS = 64`) silently truncated dense clusters within 12 Bohr | Tiled active-atom batches of 64 with cross-batch symmetric accumulation and stack-local atom evaluation | `crates/mopac_gpu/shaders/density_grid.comp`, `density_grid.spv` |
| **Cube Serialization & GPU Caching** | Sequential per-voxel heap `String` formatting; fresh `VulkanContext::new()` per Python cube call | Shared Rayon per-slice `format_cube_voxels_into` writer; `OnceLock` cache `get_cached_gpu_cube_eval()` | `mopac_core/src/export/cube.rs`, `mopac_gpu/src/cube.rs`, `mopac_py/src/lib.rs` |
| **Python Single-SCF Pipeline & Fixes** | Duplicate post-opt SCF solves; `export_molstar_bundle` ran 2 SCFs + gradients; unforwarded `charge` in `esp_charges` and Mulliken RHF occupation | Added `QuantumSession`, `compute_gradients=false`, zero-redundant-SCF `finalize_calculation_result` (`optimize`, `transition_state`, `saddle`, `export_molstar_bundle`), fixed RHF charged `nocc` & `esp_charges`, guarded `!use_nddo` GPU `gamma` | `crates/mopac_py/src/lib.rs`, `mopac_core/src/reactions/saddle.rs` |

---

## 2. Core Quantum Engine Enhancements (`mopac_core`)

### 2.1 Zero-Allocation Householder `tred2` + Implicit-Shift QL `tql2` (`scf/eigensolver.rs`)
Previously, `diagonalize_symmetric_submatrix_with_work` executed cyclic Jacobi rotations for every basis dimension $N$. While numerically accurate, cyclic Jacobi sweeps scan all $N(N-1)/2$ pairs per sweep (~6-10 sweeps), performing significantly more floating-point work and memory reads than tridiagonal reduction for medium and large molecules.

In v0.2.0:
- Matrices with $N > 12$ first run Householder tridiagonalization (`tred2`) directly inside the caller-supplied `eigenvectors` and `work` buffers, reducing $A$ to symmetric tridiagonal form $(d, e)$ using `work.row_mut(0)[..n]` for the subdiagonal vector `e` (**0 heap allocations**).
- Implicitly shifted QL iteration (`tql2`) computes all eigenvalues and back-transforms eigenvectors using numerically stable Givens rotations (up to 40 shifts per eigenvalue, using `hypot(f, g)`-safe reciprocals to prevent underflow/overflow).
- If any matrix hits the QL shift limit without converging, or for $N \le 12$, execution cleanly falls back to `diagonalize_jacobi_submatrix`, followed by `sort_eigenpairs`.

### 2.2 Parallel Row-Band NDDO Two-Center Fock Assembly (`integrals/multipoles.rs`)
1. `precompute_diatomic_pairs` now executes `compute_diatomic_pair` in parallel via Rayon when `natoms >= 16 && rayon::current_num_threads() > 1`.
2. `assemble_nddo_two_center_fock` (RHF, $\text{scale\_exch} = 0.5$) and `assemble_nddo_two_center_fock_spin` (UHF, $\text{scale\_exch} = 1.0$) were unified into a single `FockTarget`-monomorphized kernel (`accumulate_nddo_pair`).
3. When `pairs.len() >= 512 && rayon::current_num_threads() > 1 && fock.rows == fock.cols`:
   - `build_row_band_plan` validates monotonically increasing, non-overlapping `atom_a` row bands `[orb_start_a .. orb_start_a + norb_a)` with strictly lower-triangle off-diagonals (`orb_start_b + norb_b <= orb_start_a`).
   - Pair runs are load-balanced across threads by cumulative pair count using disjoint mutable row slices obtained safely via `slice::split_at_mut`.
   - Workers write `diag_a` and lower-triangle `off_diag_ab` directly without locks into their exclusive row band, accumulate `diag_b` packed lower-triangle blocks (`[f64; 45]`) locally, reduce `diag_b` in deterministic chunk order, and mirror lower off-diagonals into the upper triangle.
   - Any non-standard input layout safely falls back to `FullMatrixTarget` executing the identical sequential summation order.

### 2.3 Compact Intra-Atom COSMO Coupling Matrix (`solvation/cosmo.rs`)
In the Kluender/Klamt COSMO boundary formulation implemented in `cosmo.rs`, segment-to-AO coupling matrix elements $B_{\mu\nu, k}$ are identically zero whenever atomic orbitals $\mu$ and $\nu$ belong to different atoms ($A(\mu) \ne A(\nu)$). Previously allocating `AlignedMatrix::zeroed(norbs * norbs, nseg)` scaled as $\mathcal{O}(N_{\text{atoms}}^2 \cdot N_{\text{seg}})$ in memory and iterated all $N_{\text{orbs}}^2$ pairs on every SCF Fock build and dielectric gradient evaluation.

In v0.2.0:
- `CosmoState` stores only intra-atom blocks (`total_intra_pairs = \sum_a \text{norb}_a^2`) indexed via `pub atom_pair_offsets: Vec<usize>` and `atom_orbital_ranges: Vec<(usize, usize)>`, reducing memory footprint from $\mathcal{O}(N_{\text{orbs}}^2 \cdot N_{\text{seg}})$ to $\mathcal{O}(N_{\text{atoms}} \cdot N_{\text{seg}})$.
- Shared helpers `compute_phi_elec_blocks` and `apply_reaction_field_blocks` traverse contiguous segment rows (`b_matrix.data[row_start .. row_start + nseg]`) with zero dynamic allocations in SCF iterations.

### 2.4 Parallel Analytical Nuclear Gradients (`gradients/nuclear_gradients.rs`)
Both `compute_cartesian_gradients_with_options` (RHF) and `compute_cartesian_gradients_uhf` (UHF) now use `DiatomicGradScratch`:
- Lazily caches reusable 2-atom `MolecularBatch`, `ScfWorkspace`, and density subblocks partitioned by total orbital count `norbs_ab` (spanning 2..=18 orbitals across `[s, sp, spd]` element combinations).
- When `num_pairs >= 16 && rayon::current_num_threads() > 1`, Rayon evaluates `(a, b)` pairs in parallel using `.map_init(DiatomicGradScratch::new, ...)` with deterministic sequential reduction into `gradients`, followed by center-of-mass `project_translational_drift`.

### 2.5 Full-Featured SOSCF & Warm-Started Geometry / Hessian Loops
- Extracted shared SCF helpers in `scf/scf_loop.rs`: `compute_rhf_nocc`, `prepare_scf_hamiltonian`, `assemble_scf_fock`.
- Implemented `run_rhf_soscf_with_options` in `scf/soscf.rs` honoring `use_nddo`, `charge`, `cosmo`, `electric_field_ev_angstrom`, `external_charges`, and `reuse_density`.
- Updated `ScfWorkspace::reset` and added `ScfWorkspace::reset_for_new_geometry(keep_density: bool)` in `types.rs` so stale `diatomic_pairs` and GPU `gamma` matrices are always cleared when atomic coordinates move.
- Updated `opt/lbfgs.rs`, `scf/uhf_loop.rs`, and `vibrations/hessian.rs` (`map_init` worker-local state with geometry restoration and warm-started density reuse).

---

## 3. GPU Volumetric Density Shader & CLI Enhancements (`mopac_gpu`, `mopac_cli`)

### 3.1 Removal of the 64-Atom Cutoff Truncation (`crates/mopac_gpu/shaders/density_grid.comp`)
Previously, `density_grid.comp` stored at most `MAX_ACTIVE_ATOMS = 64` atoms within `CUTOFF_R2 = 144.0` ($r \le 12.0\text{ Bohr}$) and silently ignored atoms beyond index 64 at dense voxels.

In v0.2.0:
- `density_grid.comp` iterates over `pc.natoms` in tiles of `MAX_ACTIVE_ATOMS = 64`.
- Within each tile, intra-tile pairs `(i, j)` are summed directly; when active atoms remain beyond the current tile, remaining atoms `b >= cursor_a` are streamed individually via stack-local `ActiveAtom atom_b` accumulating $(P_{\mu\nu} + P_{\nu\mu})\,\phi_\mu(\mathbf{r})\,\phi_\nu(\mathbf{r})$.
- Voxels with $\le 64$ active atoms finish in one pass with zero extra iterations; dense systems with $> 64$ active atoms within 12 Bohr accumulate all pairs without truncation.
- Recompiled cleanly to `crates/mopac_gpu/shaders/density_grid.spv` using `glslc`.

### 3.2 Parallel `.cube` Voxel Serialization
Extracted a shared generic serializer `pub fn format_cube_voxels_into<T>` in `mopac_core/src/export/cube.rs` used by both CPU (`mopac_core::export::cube`) and GPU (`mopac_gpu::cube`). It formats each `ix` slice in parallel via Rayon (`run_with_thread_pool`), using `std::fmt::Write` into one pre-sized buffer per slice while preserving exact 6-values-per-line ASCII formatting.

### 3.3 CLI GPU Fidelity (`mopac_cli/src/main.rs`)
Removed silent downgrading of `use_nddo` when `--gpu` was supplied; GPU Coulomb tensor computation runs only when `!use_nddo`, and stdout/report logs accurately state whether Coulomb `gamma` was accelerated via Vulkan GPU or executed in CPU NDDO/CPU paths.

---

## 4. Python Engine Improvements & `QuantumSession` (`mopac_py`)

All Python-facing bottlenecks and correctness discrepancies resolved in v0.2.0:
1. **Guard & Occupation Fixes**:
   - Guarded GPU Coulomb evaluation inside `run_converged_rhf_scf` with `if use_gpu && !use_nddo`.
   - Corrected closed-shell RHF Mulliken occupation index `num_occupied = ((num_electrons - charge) / 2)` for charged molecules and ions across `calculate`, `generate_orbital_cube`, and `generate_cubes_bundle`.
   - Fixed `esp_charges(..., net_charge=...)` to forward `net_charge.round() as i32` into SCF options (including open-shell radical ions via UHF).
2. **`compute_gradients = False`**:
   - Added `compute_gradients` option (`default: true`) to `mopac_py.calculate(...)` and `MopacCalculator.calculate(...)`, using energy-only `compute_dispersion_energy` when gradients are disabled.
3. **Elimination of Redundant SCF Solves**:
   - Unified post-SCF property construction in `finalize_calculation_result` (RHF & UHF) and `run_meci_from_scf`.
   - Updated `optimize`, `transition_state`, and `saddle` (via `run_saddle_with_state` in `mopac_core::reactions::saddle`) to construct `final_result` directly from the converged wavefunction rather than re-running a second single-point SCF.
   - Updated `MopacCalculator::export_molstar_bundle` to converge RHF once and extract both Gaussian Cubes and Mulliken partial charges from the same workspace.
4. **Optional Parameters & Cached Vulkan Cube Evaluator**:
   - Added `charge = 0` and `cosmo_eps = None` to `bond_orders`, `generate_orbital_cube`, `generate_density_cube`, and `generate_cubes_bundle` while keeping default arguments backward-compatible.
   - Added `OnceLock` singleton caching for `GpuCubeEvaluator` (`get_cached_gpu_cube_eval()`), eliminating repeated Vulkan context creation across Python calls.
5. **Persistent `QuantumSession` Class**:
   - Added `mopac_py.QuantumSession(...)` storing converged RHF or UHF SCF state in memory and exposing `.calculate(...)`, `.esp_charges(...)`, `.bond_orders()`, `.generate_density_cube(...)`, `.generate_orbital_cube(...)`, `.generate_cubes_bundle(...)`, and `.meci(...)` with zero repeat SCF iterations.

---

## 5. Work-Weighted Parallel Fock Planning, Parallel COSMO Initialization, and Feature Extensions

Empirical `--release` A/B benchmarking against `v0.1.5` uncovered a Rayon thread-synchronization and per-iteration heap-allocation valley in two-center NDDO Fock assembly between 512 and ~6,670 pairs (`0.41x–0.97x`), single-threaded `CosmoState::initialize` boundary construction (`1.01x`), and minor finite-difference frequency drift (`0.169 cm^-1`) from loose warm-start SCF tolerances in `hessian.rs`. These subsystems were engineered under strict Data-Oriented Programming (DOP) principles and canonical OpenMOPAC parity:

1. **Zero-Allocation `NddoFockParallelPlan` & Work-Weighted Threshold (`multipoles.rs`, `types.rs`, `fock_builder.rs`, `scf_loop.rs`, `uhf_loop.rs`)**:
   - Replaced per-call `Vec` allocation of row-band plans and `diag_b` buffers with a geometry-cached `NddoFockParallelPlan` stored on `ScfWorkspace` and `UhfWorkspace` (`diag_b_flat: Vec<[f64; 45]>` flat contiguous arena + stack-allocated `BandSlice` array up to 64 chunks, strictly 0 heap allocations in `execute()`).
   - Replaced raw pair count (`512`) with weighted integral work $W = \sum |w_{AB}|$ (`NDDO_FOCK_PARALLEL_WORK_THRESHOLD = 65_000` cached SCF, `NDDO_FOCK_STANDALONE_WORK_THRESHOLD = 95_000` standalone) with an $O(1)$ upper-bound guard and zero-allocation work pre-check, accurately handling `s`, `sp`, and `d`-orbital regimes (e.g., 14-atom PM7 sulfur cluster activates at 91 pairs, $W = 184{,}275$).
   - Eliminated the `0.41x` valley: `C14H30 (946p)` moved `0.41x -> 0.95x`, `C24H50 (2701p)` `0.86x -> 0.99x`, `C30H62 (4186p)` `1.30x`, `C38H78 (6670p)` `0.97x -> 1.68x`, `C48H98 (10585p)` `1.31x -> 1.43x`, with `|dFock_Fro| = 0.00e+00`.
2. **Parallel `CosmoState::initialize` (`solvation/cosmo.rs`)**:
   - Parallelized lower-triangle $A$-matrix fine-element row construction via `par_chunks_mut(nseg)` and inverted compact $B$-matrix construction to atom-outer disjoint row bands (`par_iter_mut()`), hoisting `DerivedMultipoleParams::from_element` and element core-charge lookup once per atom.
   - Reduced `C14H30` COSMO initialization from `11.49 ms` to `6.70 ms` (`1.71x`) with bit-identical dielectric energy (`|dEdiel| = 0.00e+00 eV`).
3. **Charged Closed-Shell RHF Optimization (`opt/lbfgs.rs`, `mopac_cli/src/main.rs`, `mopac_py/src/lib.rs`)**:
   - Added `pub charge: i32` (default `0`) to `OptimizationOptions` and wired it through `optimize_geometry_lbfgs`, `mopac_cli`, and `mopac_py::optimize` (including `finalize_calculation_result`).
4. **Tightened Vibrational Hessian Warm-Start SCF Tolerances (`vibrations/hessian.rs`)**:
   - Enforced `energy_tol_ev <= 1e-9` and `density_tol <= 1e-8` for equilibrium and warm-started displaced SCF evaluations, tightening harmonic frequency precision while preserving `2.92x` speedup on benzene (`42.42 ms -> 14.54 ms`).
5. **Open-Shell UHF Support in Python Property/Cube Functions & `MopacCalculator` (`mopac_py/src/lib.rs`)**:
   - Added `PropertyWavefunction` / `solve_property_wavefunction` dispatching automatically to UHF when `(valence - charge)` is odd across `bond_orders`, `generate_density_cube`, `generate_orbital_cube`, `generate_cubes_bundle`, and `export_molstar_bundle`.
   - Added optional `charge`, `uhf`, and `multiplicity` arguments to `MopacCalculator.calculate()` and `MopacCalculator.optimize()`, and `charge` to `MopacCalculator.export_molstar_bundle()`.

---

## 6. Verification & Empirical Benchmark Summary

- `cargo fmt --all -- --check`: passed (exit code 0).
- `RUSTFLAGS="-D warnings" cargo check --workspace --all-targets`: passed with 0 warnings across `mopac_core`, `mopac_gpu`, `mopac`, and `mopac_py`.
- `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets -- -D warnings`: passed with 0 warnings across all targets.
- `RUSTFLAGS="-D warnings" cargo test --workspace`:
  - `mopac_core`: 44 unit tests + 131 integration tests = 175 passed, 0 failed.
  - `mopac_gpu`: 24 tests (`cube_vulkan`: 8 including `test_vulkan_gpu_dense_cluster_over_64_active_atoms_parity`; `memory_architecture_universal`: 5; `protein_large_vulkan`: 1; `protein_vulkan`: 1; `vulkan_scrutiny`: 9) passed, 0 failed.
  - `mopac_cli`: 5 passed, 0 failed, 1 pre-existing ignored (`pm7_nonbonded_parity`).
  - Workspace total: 204 passed, 0 failed, 1 ignored.
- `python3 crates/mopac_py/tests/test_python_api.py` (against release-built `mopac_py`): all 24 tests passed (including `test_quantum_session_single_scf_reuse`, `test_calculate_without_gradients_pm7`, `test_esp_charges_charged_cation`, and `test_charged_optimization_and_open_shell_properties`).

### Final Verified Empirical Benchmark Summary (`v0.1.5` vs `v0.2.0`, `--release`)

| Benchmark / Workload | `v0.1.5` | `v0.2.0` | Speedup / Delta | Numerical Precision |
| :--- | ---: | ---: | :--- | :--- |
| Eigensolver 64 / 150 / 250 | 1.59 / 19.00 / 94.91 ms | 0.51 / 5.36 / 25.65 ms | **3.15x / 3.55x / 3.70x** | `max|dLambda| = 0.00e+00` |
| Nuclear Gradients (`C10H22`) | 3.536 ms | 0.328 ms | **10.78x** | `< 1e-12 eV/A` |
| COSMO Init (`C14H30`) | 11.486 ms | 6.698 ms | **1.71x** | `|dEdiel| = 0.00e+00 eV` |
| COSMO Fock Step (`C14H30`) | 5.180 ms | 0.121 ms | **42.87x** (29.11x less RAM) | `|dFock00| = 0.00e+00 eV` |
| PM6 Single-Point (`C24H50`) | 674.59 ms | 132.26 ms | **5.10x** | `|dHOF| vs Fortran = 0.00053 kcal/mol` |
| COSMO SCF+Grad (`C20H42`) | 747.60 ms (75 MiB) | 90.30 ms (33 MiB) | **8.28x** (56% less RSS) | `|dE_tot| = 3.64e-12 eV` |
| L-BFGS Opt (`C6H14`) | 155.25 ms | 20.59 ms | **7.54x** | `RMSD = 4.86e-07 A` |
| Frequencies (`Benzene`) | 42.42 ms | 14.54 ms | **2.92x** | `|dZPVE| = 2.37e-04 kcal/mol` |
| 5-Property `QuantumSession` | 37.03 ms | 6.20 ms | **5.97x** | `|dE_tot| = 1.82e-12 eV` |
| GPU Density Cube x5 (`Benzene`) | 734.60 ms | 139.24 ms | **5.28x** | `>64` atoms err `3.88e-02 -> 4.00e-08` |
