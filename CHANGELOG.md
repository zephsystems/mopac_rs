# Changelog

All notable changes to `mopac_rs` and `mopac-py` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
