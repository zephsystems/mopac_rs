# Changelog

All notable changes to `mopac_rs` and `mopac-py` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [0.1.4] - 2026-10-17

### Added
- **Exact OpenMOPAC Fortran 90 Parity Verification Suite**:
  - Implemented dual-test methodology (Performance Throughput Benchmark + Mathematical & Algorithmic Parity Test) across all core physical modules.
  - Verified bit-exact and sub-millielectronvolt parity against OpenMOPAC v23.2.5 reference binaries.
- **Canonical Solrot Solid-State Engine (`solrot.rs`)**:
  - Direct space Born-von Kármán periodic boundary condition solver mirroring OpenMOPAC `solrot.F90` and `moldat.F90:1125-1150`.
  - Reciprocal fractional lattice projection determining exact cell bounds ($L_1 = 13$ for polyacetylene).
  - $C^1$-continuous quadratic Madelung truncation function `trunk(r, clower, cutofp)`.
  - Pairwise Coulomb lattice summation $W_J$ across 27 unit cells and decoupled nearest-image exchange $W_K$.
  - Exact trans-polyacetylene nuclear repulsion parity: $E_{\text{nuc}} = 2142.99190\text{ eV}$ verbatim matching OpenMOPAC to 5 decimal places.
- **Klamt COSMO Cavity Construction Engine (`cavity.rs`, `cosmo.rs`)**:
  - Canonical 107-element Van der Waals radii table from `cosini.F90`.
  - Analytical cap intersection area and distance derivative function `ansude` (`cosmo.F90:125-189`).
  - Local frame atomic coordinate transformations $T_M$ (`cosmo.F90:326-376`).
  - Klamt 2-pass iterative clustering with complete re-entrant crevice closure `surclo` (`cosmo.F90:1440-1690`).
  - Water segment count ($NPS = 49$, $O=33, H_1=8, H_2=8$) and area ($42.316\text{ \AA}^2$) matching OpenMOPAC verbatim.

- **Vulkan GPU & Subcrates Parity Propagation (`mopac_gpu`, `mopac_cli`, `mopac_py`)**:
  - **Vulkan Transition-Metal `pocord` Screening (`coulomb.rs`, `coulomb_fp32.rs`)**: Propagated canonical `get_pocord(Z)` core radius override ($\rho_A = a_0 \cdot \text{pocord}_A$) to FP64 and FP32 GPU Coulomb pipelines, achieving $< 10^{-12}\text{ eV}$ parity on transition metals at **755,461 pair-evals/s**.
  - **Vulkan Double-Precision Core Repulsion Pipeline (`core_repulsion.rs`, `core_repulsion.comp`)**: Modernized and compiled `core_repulsion.spv` with range-reduced Horner polynomial exponentiation (`d_exp`, accuracy $< 10^{-14}$), Voityuk diatomic parameters, and Lennard-Jones $R^{-12}$ repulsion via `GpuCoreRepulsionCalculator` (**145,627 pair-evals/s**, $< 10^{-10}\text{ eV}$ parity).
  - **9-Orbital $SPD$ Spherical Harmonics in GPU Volumetric Cubes (`cube.rs`, `orbital_grid.comp`, `density_grid.comp`)**: Extended `AtomDataGpu` and Vulkan grid shaders with $z_d$, $\text{norm}_d$, principal quantum numbers $n \in \{1..6\}$, and all 5 normalized $d$-orbitals ($d_{z^2}, d_{xz}, d_{yz}, d_{x^2-y^2}, d_{xy}$), achieving **1.28 MVox/s** with max voxel difference $2.56 \times 10^{-7}$ against CPU on $\text{H}_2\text{S}$ PM7.
  - **Solid-State `solrot` Routing in CLI & Python (`mopac_cli`, `mopac_py`)**: Added `--solrot` CLI flag and Python `pbc_solrot(...)` / `pbc(..., engine="solrot")` bindings (1.03 ms unit cell convergence).
  - **Python Vulkan GPU Diagnostics & Coulomb API (`mopac_py`)**: Exposed `is_gpu_available()`, `gpu_device_info()`, and `compute_coulomb_matrix_gpu(...)` with bit-exact ($< 10^{-12}\text{ eV}$) analytical parity.

### Changed
- **Core-Core Repulsion PM6/PM7 & Coordinate Preprocessing (`core_repulsion.rs`)**:
  - Canonical port of OpenMOPAC `pocord`, `ccrep.F90`, `mndod.F90`, and `compfg.F90`.
  - Water PM6 nuclear repulsion residual vs OpenMOPAC reduced to $< 5 \times 10^{-7}\text{ eV}$ ($< 10^{-5}\text{ kcal/mol}$).
- **PM7 $d$-Orbital Valence Gating (`d_orbitals.rs`, `multipoles.rs`)**:
  - Ported OpenMOPAC `iod(Z)` parameter logic (`parameters_C.F90:142-148`) and conditional gating (`mndod.F90:1271, 1309`).
  - Ensured main-group hypervalent elements (P, S, Cl, As, Se, Br, etc.) with empty ground-state valence $d$-orbitals bypass transition metal Coulomb corrections.
  - Eliminated the $2.40\text{ kcal/mol}$ over-repulsion anomaly on $\text{H}_2\text{S}$, achieving exact $0.0000\text{ kcal/mol}$ parity.
- **MOZYME $\mathcal{O}(N)$ Localized Orbital Engine (`solver.rs`, `lewis.rs`, `hybrid.rs`)**:
  - Calibrated multiple-bond threshold ($0.89$) in `lewis.rs`, ensuring peptide $C-N$ bonds are correctly classified as single bonds with lone-pair retention on amide nitrogens.
  - Replaced ad-hoc Jacobi similarity rotation in `lowdin_orthogonalize` with the proven `diagonalize_symmetric` eigensolver, eliminating 20 spurious virtual orbitals.
  - Ac-Ala-NMe total energy matched OpenMOPAC to $1 \times 10^{-5}\text{ eV}$ ($-1840.80951\text{ eV}$ vs $-1840.8095\text{ eV}$) and nuclear repulsion to $< 10^{-5}\text{ eV}$ ($7421.71619\text{ eV}$ vs $7421.7162\text{ eV}$).
  - Confirmed asymptotic linear scaling $\mathcal{O}(N)$ with constant marginal pair growth ($\Delta P \approx 160\text{ pairs/carbon}$) under $9.0\text{ \AA}$ cutoff.
- **Diatomic Cartesian Gradients $\mathcal{O}(N^2)$**:
  - Replaced $6N$ full-molecule finite difference evaluations ($\mathcal{O}(N^3)$) with exact pairwise diatomic energy decomposition ($\mathcal{O}(N^2)$) in `gradients/nuclear_gradients.rs`, mirroring OpenMOPAC `dcart.F90` / `dhc.F90`.
  - Exact translational invariance ($\sum_A \vec{g}_A \equiv 0$) guaranteed analytically by construction; gradient evaluation accelerated by >95%.

### Fixed
- **SCF Iterative Fock Construction Precedence (`scf_loop.rs`)**:
  - Fixed order of precedence in `run_rhf_scf_with_options` so `options.use_nddo` strictly takes priority over precomputed isotropic monopole `ws.gamma`, ensuring identical NDDO convergence across CPU and GPU runs.
- **Sanitization of Local Paths**:
  - Audited all repository files, test fixtures, documentation, and metadata to eliminate absolute machine and user paths.
  - Configured compiler path remapping flags (`--remap-path-prefix`) in CI/CD pipeline.
- **Tightened Integration Tolerances**:
  - Replaced legacy loose assertions ($25-50\text{ kcal/mol}$) with strict physical thresholds ($< 0.05-0.25\text{ kcal/mol}$).
- **Eradication of Mock Data & Heuristic Shortcuts**:
  - Pruned all placeholder arrays, synthetic constants, and ad-hoc fallback branches across MNDO, AM1, PM3, RM1, PM6, and PM7.
  - Transcribed canonical PM6 parameter tables (83 elements), AM1-BCC parameter tables, and canonical atomization energies directly from reference data.

## [0.1.3] - 2026-10-01

### Added
- **NIST CCCBDB SRD 101 Experimental Validation**:
  - Full thermochemistry and geometry verification suite evaluating 707 gas-phase molecules against laboratory measurements.
  - Bit-exact CPU vs GPU Vulkan Float64 parity ($0.00\text{ kcal/mol}$ residual across all supported Hamiltonians).
  - OpenMOPAC v23.2.5 reference oracle benchmark with sub-millihartree ($P_{50} < 0.001\text{ kcal/mol}$) parity.

### Changed
- **Repository Tree Sanitization**:
  - Pruned transient 3D Gaussian cube grids, test charges, and audit files from the Git index, reducing repository footprint by ~38% (from 6.34 MB to 3.96 MB).
  - Hardened `.gitignore` to prevent tracking of volumetric grids and local cache artifacts.

### Fixed
- **CI/CD & GitHub Actions Stability**:
  - Resolved schema validation failure in `release.yml` by isolating secrets context from `if:` conditionals.
  - Enforced 100% `cargo fmt` adherence and fixed Clippy linter warnings (`field_reassign_with_default`, `if_same_then_else`).
  - Added `skip-existing: true` in `release-pypi.yml` for idempotent distribution uploads.

## [0.1.2] - 2026-09-28

### Added
- **Systematic Macromolecular Interoperability Suite**:
  - `export_molstar_pdb`: Affine mapping of continuous partial charges $q \in [-q_{\max}, +q_{\max}]$ into crystallographic B-factor domain $[10.0, 90.0]$, ensuring high-contrast Diverging Blue-White-Red coloring in Mol* Uncertainty theme without atomic radius collapse.
  - `export_mmcif_with_charges`: Native PDBx/mmCIF generation with `_atom_site.partial_charge` and `_atom_site.B_iso_or_equiv` for native Mol* partial charge color theme.
  - `export_mvs_session`: Automated MolViewSpec (MVS v1.0) JSON scene generation for 1-click loading with covalent skeleton (Ball & Stick) and translucent Gaussian Surface (`opacity: 0.35`).
  - `MopacCalculator.export_molstar_bundle`: Single unified SCF pass generating PDB, mmCIF, MVS session, and volumetric Gaussian Cubes (HOMO, LUMO, Density).
- **Volumetric Gaussian Cube Performance & Compliance**:
  - Strict Gaussian format compliance with exactly 6 values per line in scientific notation and explicit newline delimiter at the end of each $Z$-ray.
  - Physical Slater-Type Orbital (STO) spatial cutoff ($r = 12.0\text{ Bohr} = 6.35\text{ \AA}$) combined with Rayon multithreading, delivering a $20\times - 50\times$ speedup on 3D grid evaluations.
  - `generate_cubes_bundle`: Shared SCF pass for multiple orbital and density cube generations, eliminating duplicate quantum calculations.
- **Quantum Solver & Method Enhancements**:
  - Warm restart SCF with density matrix reuse (`reuse_density: true`).
  - Second-Order Self-Consistent Field (SOSCF) orbital rotation solver for heavily conjugated aromatic systems.
  - Selective DIIS subspace pruning and Saunders-Hillier dynamic damping.
  - Analytical COSMO implicit solvation gradients.
- **Ecosystem Interoperability**:
  - Bidirectional bridges with RDKit (`from_rdkit`) and ASE (`from_ase`, `MopacASECalculator`).
  - Multi-model animated trajectories (`export_trajectory_xyz`) and MDL Molfile / SDF V2000 serializer with Mayer bond orders.
- **Automated CI & Packaging**:
  - Multi-platform wheel builds for Linux (x86_64), Windows (x64), and macOS (Apple Silicon aarch64 & Intel x86_64) with PyPI Trusted Publishing (OIDC).
  - Graceful degradation for Vulkan GPU compute tests in headless environments without physical Float64 hardware.

### Fixed
- Fixed surface puncture and polygonal tearing artifacts in Mol* when visualizing molecules with negative partial charges.
- Fixed 3D grid desynchronization in external volumetric parsers by standardizing $Z$-ray line breaks.
- Fixed Clippy linter warnings across workspace targets and enforced 100% `cargo fmt` compliance.

## [0.1.1] - 2026-09-25

### Added
- First public release of `mopac-py` on PyPI.
- Core semi-empirical quantum chemistry engines: MNDO, AM1, PM3, RM1, PM6, PM7.
- COSMO dielectric implicit solvation model.
- Grimme D3-BJ empirical dispersion and H4 hydrogen bonding corrections.
- Quasi-Newton L-BFGS Cartesian geometry optimization.
- Vulkan GPU compute shader backend for two-center two-electron Coulomb repulsion matrices.
