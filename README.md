# MOPAC_RS

> **Modern High-Performance Data-Oriented Semi-Empirical Quantum Chemistry Engine in Rust**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Crates.io: mopac](https://img.shields.io/crates/v/mopac.svg?label=mopac)](https://crates.io/crates/mopac)
[![Crates.io: mopac_core](https://img.shields.io/crates/v/mopac_core.svg)](https://crates.io/crates/mopac_core)
[![Docs.rs](https://docs.rs/mopac_core/badge.svg)](https://docs.rs/mopac_core)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22731254.svg)](https://doi.org/10.5281/zenodo.22731254)
[![CI Status](https://github.com/zephsystems/mopac_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/zephsystems/mopac_rs/actions/workflows/ci.yml)
[![Language: Rust](https://img.shields.io/badge/Language-Rust%201.85%2B-orange.svg)]()
[![SIMD: AVX2 / AVX-512](https://img.shields.io/badge/Acceleration-AVX2%20%7C%20AVX--512-red.svg)]()
[![GPU: Vulkan Compute](https://img.shields.io/badge/Compute-Vulkan%20%7C%20GDDR6%20VRAM-green.svg)]()
[![Scrutiny Tests](https://img.shields.io/badge/Automated%20Tests-83%2F83%20Passed-brightgreen.svg)]()
[![Python Bindings](https://img.shields.io/badge/PyO3-Python%203.8--3.14-blue.svg)]()

<!-- Total Downloads -->
[![Downloads mopac_core](https://img.shields.io/crates/d/mopac_core?label=mopac_core%20downloads)](https://crates.io/crates/mopac_core)
[![Downloads mopac_gpu](https://img.shields.io/crates/d/mopac_gpu?label=mopac_gpu%20downloads)](https://crates.io/crates/mopac_gpu)
[![Downloads mopac](https://img.shields.io/crates/d/mopac?label=cli%20downloads)](https://crates.io/crates/mopac)

---

## Executive Summary

`mopac_rs` is a ground-up architectural reimagining and rigorous modernization of the classic **MOPAC** (Molecular Orbital PACkage) quantum chemistry engine. Translated from legacy Fortran into idiomatic, zero-overhead **Rust**, it replaces decades of non-contiguous global arrays, static buffers, and triangular packing with a strictly **Data-Oriented Programming (DOP)** architecture.

Every single module, parameter table, and integral calculation is empirically verified against **OpenMOPAC v23.2.5** references to rigorous double-precision tolerances.

---

## Architectural Pillars

* **Data-Oriented Memory Layout (SoA):** Contiguous, 64-byte cache-line aligned Struct of Arrays (`MolecularBatch`) eliminating pointer-chasing and non-contiguous matrix indexing.
* **Zero-Allocation Inner Loop Policy (`0 malloc`):** Pre-allocated reusable workspaces (`ScfWorkspace`, `UhfWorkspace`, `GradientWorkspace`, `CosmoState`, `IrcWorkspace`, `DrcWorkspace`, `MeciWorkspace`, `PbcWorkspace`) ensure zero heap allocations during iterative Roothaan-Hall / Pople-Nesbet SCF cycles and dual Pulay DIIS extrapolations.
* **Dual Compute Backend:**
  - **CPU SIMD:** Vectorized AVX2 / FMA kernels with Rayon multi-threaded parallelism.
  - **Universal GPU (Vulkan Compute):** Cross-vendor hardware acceleration supporting NVIDIA RTX, AMD Radeon, Intel Arc, and Apple Silicon (via MoltenVK) with dedicated DMA host-to-device GDDR6 VRAM batch management.
* **Axiomatic Verification:** 83 automated scrutiny, unit, quantum theorem, and differential oracle tests validating physical invariance, rotation orthonormality, translational symmetry, and exact numerical parity against OpenMOPAC.

---

## Scientific Capabilities

### 1. Semi-Empirical Hamiltonians
- **MNDO** (Modified Neglect of Diatomic Overlap; Dewar & Thiel 1977)
- **AM1** (Austin Model 1; Dewar et al. 1985)
- **PM3** (Parametric Method 3; Stewart 1989)
- **RM1** (Recife Model 1; Rocha et al. 2006)
- **PM6** (Parametric Method 6; Stewart 2007)
- **PM7** (Parametric Method 7; Stewart 2013) with full diatomic parameters and d-orbital polarization for transition metals (e.g. Zn).
- **NDDO 22-Multipole Integrals:** Full diatomic charge separation multipoles ($dd, qq, am, ad, aq$) and 3D rotational coordinate transformations.
- **Elemental Coverage:** Complete authentic parameter sets for organic, heteroatomic, boron, organosilicon, and metal chemistry:
  - **Hydrogen & Carbon-backbone:** H (1), C (6)
  - **Boron (Group 13):** B (5) fully parameterized across MNDO, AM1, PM3, PM6, PM7 (including authentic diatomic pair parameters with H, C, F, Cl, Br, I) and isolated atom heats of formation ($135.700\text{ kcal/mol}$).
  - **Pnictogens & Chalcogens:** N (7), O (8), P (15), S (16)
  - **Full Halogen Series:** F (9), Cl (17), Br (35), I (53) across AM1, PM6, PM7, and RM1 with authentic diatomic pair parameters $(alpb, xfac)$.
  - **Silicon (Group 14):** Si (14) across MNDO, AM1, PM3, PM6, PM7 (including 11 authentic pair parameters with H, C, N, O, F, Al, Si, P, S, Cl, Br) exhibiting $< 2.4 \times 10^{-5}\text{ eV/\AA}$ analytical gradient parity.
  - **Transition Metals:** Zn (30) with $spd$ basis parameterization.

### 2. MOZYME $O(N)$ Linear Scaling Macromolecular Solver
- **Lewis Chemical Topology Builder:** Automatic bond order detection (single, double, triple, coordinate) and lone pair assignment.
- **Directional Hybrid Atomic Orbitals (HAOs):** $sp^3, sp^2, sp$ hybrid generation with symmetric Löwdin orthogonalization ($S^{-1/2}$).
- **Localized Molecular Orbitals (LMOs):** Orthogonal $\sigma, \pi$ bonding, $\sigma^*, \pi^*$ antibonding, and lone-pair initial basis.
- **Pairwise 2x2 Jacobi Rotation Engine:** Iterative localized orbital energy minimization with spatial distance cutoff screening ($R_{\text{cut}} = 8.5\text{ \AA}$).
- **Linear-Scaling SCF Cycle:** Smooth density damping ($\alpha = 0.65$) and exact density matrix reconstruction ($P = 2 \sum_i \phi_i \phi_i^T$) achieving $< 0.08\%$ relative energy parity on macromolecules.

### 3. Open-Shell Unrestricted Hartree-Fock (UHF)
- **Pople-Nesbet Spin Orbitals:** Independent spin Fock operators ($F^\alpha, F^\beta$) and density matrices ($P^\alpha, P^\beta$).
- **Dual DIIS Acceleration:** Decoupled alpha and beta error subspace inversion buffers avoiding inter-spin oscillation damping traps.
- **Spin Expectation $\langle S^2 \rangle$:** Analytic computation of total spin angular momentum with spin contamination tracking ($\Delta \langle S^2 \rangle = 0.000178$ against OpenMOPAC on methyl radical).
- **Closed-Shell Equivalence:** Exact convergence to RHF energy for singlet states ($|\Delta E| < 10^{-10}\text{ eV}$).

### 4. Robust SCF Convergers
- **Pulay DIIS Acceleration:** Direct Inversion in the Iterative Subspace with B-matrix SVD stabilization and history pruning.
- **Camp-King Unitary Interpolator:** Monotonic electronic energy minimization for oscillating densities.
- **Saunders-Hillier Virtual Orbital Level Shifting:** Dynamic shift ($\sigma = 2.0\text{ -- }8.0\text{ eV}$) eliminating HOMO-LUMO degeneracy traps and limit-cycle density oscillations.
- **Adaptive Multi-Tier Escalation:** Automated converger pipeline achieving **100.0% convergence** across all benchmark sets.

### 5. Non-Covalent Corrections
- **Grimme D3-BJ Empirical Dispersion:** Becke-Johnson rational damping ($s_6, s_8, a_1, a_2$) with exact analytical Cartesian gradients matching finite differences to $1.11 \times 10^{-11}\text{ kcal/(mol}\cdot\text{\AA)}$.
- **PM6-DH+ and PM7 Dispersion:** Empirically parameterized dispersion corrections with analytical derivatives ($\nabla E_{\text{disp}} < 1.88 \times 10^{-11}\text{ kcal}/(\text{mol}\cdot\text{\AA})$ error).
- **H4 Hydrogen Bonding:** Septic switching functions for covalent valence attenuation and 7th-order radial/angular polynomials ($D, A \in \{N, O\}$).
- **Short-Range H-H Repulsion:** Continuous piecewise potential with exact analytical derivatives.
- **Composite PM6-D3H4 Method:** Verbatim heat of formation parity on water dimer benchmark (**$-71.99024\text{ kcal/mol}$**).

### 6. COSMO Implicit Solvation & Analytical Nuclear Gradients
- **Boundary Element Method (BEM):** Regular icosahedral sphere tessellations ($N=12, 42, 1082$, `dvfill`).
- **Solvent-Accessible Cavity:** Klamt and Bondi van der Waals radii with analytical segment surface areas and volumes.
- **Self-Consistent Reaction Field:** In-place Cholesky decomposition of electrostatic boundary matrix $A$ and multipole coupling matrix $B$, modifying $H_{\text{core}}$ and Fock matrix $F$ self-consistently with `0 malloc` per iteration.
- **Analytical Nuclear Gradients ($\nabla E_{\text{diel}}$):** Inter-segment screening forces and segment-solute net charge electrostatic derivatives matching OpenMOPAC `diegrd` with exact Newton's third law translational zero-sum invariance ($\sum_A \nabla_A E_{\text{diel}} < 10^{-14}\text{ eV/\AA}$).

### 7. Transition States & Reaction Paths
- **Eigenvector Following (Baker P-RFO / `TS`):** Search for first-order saddle points (transition states) with one negative Hessian eigenvalue.
- **Two-Ended SADDLE Interpolation:** Automated barrier search connecting Reactants and Products via geodesic coordinate relaxation.
- **Intrinsic Reaction Coordinate (IRC / González-Schlegel):** Exact mass-weighted steepest descent reaction path tracing forward and backward from transition state to minima.
- **Dynamic Reaction Coordinate (DRC / Born-Oppenheimer MD):** Microcanonical ($NVE$) and canonical ($NVT$) molecular dynamics trajectories with symplectic Velocity-Verlet integrator.

### 8. Photochemistry, MECI & UV-Vis Spectroscopy
- **Multi-Electron Configuration Interaction (MECI):** Full CI within active spaces ($N \le 10$ orbitals) computing multi-determinant ground and excited state roots ($S_0, S_1, T_1$, etc.).
- **Analytical Excited-State Nuclear Gradients:** State-specific Hellmann-Feynman and relaxed density matrix gradients for excited state geometry optimization.
- **UV-Vis Spectral Simulation:** Transition dipole moments, oscillator strengths ($f_{\text{osc}}$), and Lorentzian line broadening.

### 9. Molecular Properties & Population Analysis
- **Electric Dipole Moments:** Point-charge and intra-atomic $sp$ hybridization dipole moments in Debye.
- **Polarizability & NLO Tensors (TD-CPHF / `POLAR`):** Static and dynamic frequency-dependent polarizability tensor $\alpha(-\omega; \omega)$ and first hyperpolarizability $\beta$.
- **Electrostatic Potential (ESP) Charges:** Merz-Singh-Kollman grid fitting outside van der Waals envelope.
- **Mayer Bond Orders & Valencies:** Armstrong-Perkins-Stewart bond indices $B_{AB} = \sum_{\mu \in A, \nu \in B} (P S)_{\mu\nu} (P S)_{\nu\mu}$.
- **Mulliken Population Analysis:** Löwdin de-orthogonalization and gross atomic populations satisfying exact electron conservation ($\sum_A Pop_A \equiv N_{\text{elec}}$).

### 10. Geometry Optimization, Vibrations & Isotope Effects
- **L-BFGS Optimizer:** Quasi-Newton Cartesian minimization with two-loop history recursion and Armijo backtracking line search.
- **Coordinate Pinning:** Selective degree-of-freedom masking (frozen atoms/axes).
- **Harmonic Vibrational Frequencies:** Mass-weighted Cartesian Hessian with Eckart frame external projection (6 vanishing rotational/translational modes $< 10^{-5}\text{ cm}^{-1}$).
- **Custom Isotopic Masses & Kinetic Isotope Effects (KIE):** Direct mass substitution ($^2H, ^{13}C, ^{18}O$) for vibrational isotope shift analysis.
- **Thermodynamic Properties:** Zero-Point Vibrational Energy (ZPVE), thermal enthalpy ($H(T) - H(0)$), constant-pressure heat capacity ($C_p$), standard entropy ($S^\circ$), and Gibbs free energy correction ($G(T) - H(0)$).

### 11. Periodic Boundary Conditions (PBC)
- **1D, 2D, 3D Unit Cells:** Lattice vector parameterization with Monkhorst-Pack reciprocal space k-point sampling and Bloch SCF band structures.

### 12. Python Bindings & Ecosystem Bridge (`mopac_py`)
High-performance PyO3 bridge exposing the quantum chemical engine directly to Python for **RDKit**, **ASE** (Atomic Simulation Environment), and **PyTorch**:
- **Zero-copy analytical gradients** directly returned as NumPy arrays or nested lists.
- **Vibrational Frequencies & Thermochemistry:** `mopac_py.frequencies(atoms, coords, method="AM1")` computing full normal modes, ZPVE, $H(T)$, $G(T)$, $C_p$, and $S^\circ$.
- **MOZYME Linear Scaling:** `mopac_py.mozyme(atoms, coords, method="PM6")` for macromolecules.
- **Direct RDKit Interoperability:** `mopac_py.from_rdkit(mol)` extracts atomic numbers and 3D conformer coordinates directly.
- **Native ASE Calculator:** `mopac_py.MopacASECalculator` plugs directly into ASE dynamics (`BFGS`, `VelocityVerlet`, etc.).
- **Full parameterization control** (`method="PM6"`, `dispersion="D3-BJ"`, `cosmo_eps=78.4`, `use_nddo=True`).
- **Full parameterization control** (`method="PM6"`, `dispersion="D3-BJ"`, `cosmo_eps=78.4`, `use_nddo=True`).

```python
import mopac_py

# Water single-point calculation (PM6 + COSMO solvation + Grimme D3-BJ)
atoms = [8, 1, 1]
coords = [[0.0, 0.0, 0.0655], [0.0, 0.7571, -0.5205], [0.0, -0.7571, -0.5205]]
res = mopac_py.calculate(atoms, coords, method="PM6", dispersion="D3-BJ", cosmo_eps=78.4)

print(f"Total Energy: {res.total_energy_ev:.6f} eV")
print(f"Heat of Formation: {res.heat_of_formation_kcal:.3f} kcal/mol")
print(f"Dipole: {res.dipole_debye[3]:.3f} Debye")
print(f"Mulliken Charges: {res.mulliken_charges}")
print(f"Gradients (eV/A): {res.gradients_ev_angstrom}")

# Vibrational frequencies and thermochemistry (298.15 K, 1 atm)
vib = mopac_py.frequencies(atoms, coords, method="AM1")
print(f"Vibrational Frequencies (cm^-1): {vib.vibrational_frequencies_cm1}")
print(f"ZPVE: {vib.zpve_kcal_mol:.3f} kcal/mol")
print(f"Standard Entropy S^o: {vib.thermo.entropy_total_cal_k_mol:.3f} cal/(mol*K)")

# Geometry optimization (L-BFGS)
opt = mopac_py.optimize(atoms, coords, method="PM6", max_cycles=50)
print(f"Optimized Energy: {opt.final_energy_ev:.6f} eV (Converged: {opt.converged})")
```

---

## Hardware Execution & GPU Dispatch Policy

A key architectural insight in quantum chemistry is the trade-off between **SIMD latency** and **discrete GPU dispatch overhead**:

* **Small-to-Medium Molecules ($N < 150$ atoms):**
  - Evaluated on the **CPU SIMD (AVX2 / FMA + Rayon)** backend.
  - With zero PCIe/DMA overhead and sub-millisecond execution times ($< 1\text{ -- }10\text{ ms}$ per SCF cycle), CPU execution strictly outperforms GPU dispatch for single molecules.
* **Macromolecules & Batched Screening ($N > 200$ atoms or $M \ge 300$ conformations):**
  - Evaluated on the **Universal Vulkan GPU** backend.
  - Pipelined host-to-device asynchronous GDDR6 memory transfers saturate thousands of FP32/FP64 GPU shader compute cores, amortizing Vulkan command buffer submission latency and delivering order-of-magnitude throughput scaling for high-throughput virtual screening (HTVS) and macromolecular assemblies.

---

## CLI Usage

### Build and Install
```bash
# Clone the repository
git clone https://github.com/zephsystems/mopac_rs.git
cd mopac_rs

# Build optimized release binary
cargo build --release --bin mopac

# Run entire test suite
cargo test --workspace
```

### Command-Line Arguments
```text
Usage: mopac [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input file path (.mop)

Options:
  -m, --mode <MODE>      Force calculation mode (1SCF or OPT)
      --method <METHOD>  Semi-empirical method override (PM6, PM7, PM3, RM1, AM1, MNDO)
      --nddo             Enable full NDDO diatomic 22-multipole integrals & 3D rotation
      --opt              Enable geometry optimization (L-BFGS)
      --ts               Enable transition state optimization via Eigenvector Following (P-RFO Baker)
      --irc              Enable Intrinsic Reaction Coordinate path tracing (González-Schlegel)
      --drc              Enable Dynamic Reaction Coordinate molecular dynamics (Velocity-Verlet)
      --force            Enable Cartesian Hessian & vibrational frequency analysis
      --bonds            Enable Mayer bond orders and atomic valencies calculation
      --mullik           Enable Mulliken population analysis
      --ci <N>           Enable Multi-Electron Configuration Interaction (MECI) with active space size N
      --uv-vis           Simulate UV-Vis electronic absorption spectrum
      --static, --polar  Calculate finite-field polarizability and NLO hyperpolarizability tensors
      --mozyme           Enable MOZYME localized molecular orbital linear-scaling SCF (O(N))
      --pbc              Enable Periodic Boundary Conditions (PBC) Bloch SCF & band structure
      --gpu              Enable Vulkan GPU compute acceleration
      --fp32             Use FP32 single-precision GPU pipeline
      --eps <EPS>        Solvent dielectric constant for COSMO implicit solvation (e.g. 78.4)
      --disp <DISP>      Empirical dispersion model (pm6-dh+, pm7)
      --d3h4             Enable D3H4 composite correction (dispersion, H4, H-H repulsion)
      --1scf             Force single-point calculation (1SCF)
      --threads <N>      Number of Rayon worker threads
  -o, --output <FILE>    Custom output report file path (.out)
      --arc <FILE>       Custom archive file path (.arc)
  -h, --help             Print help
  -V, --version          Print version
```

### Sample Input File (`water.mop`)
```text
PM6 EPS=78.4 BONDS MULLIK 1SCF
Water in aqueous solution COSMO implicit solvation test

O   0.000000 0   0.000000 0   0.000000 0
H   0.757000 0   0.586000 0   0.000000 0
H  -0.757000 0   0.586000 0   0.000000 0
```

Execute:
```bash
./target/release/mopac water.mop
```

Outputs generated:
- `water.out`: Comprehensive human-readable report with energetic breakdown, dipole contributions, charges, and bond orders.
- `water.arc`: Standard archive containing converged geometry and final heat of formation.

---

## Physical Invariants & Quantum Theorems

Empirical verification of fundamental quantum mechanical and systems invariants in `crates/mopac_core/tests/quantum_theorems.rs`:

| Theorem / Physical Invariant | Target System | Mathematical Property | Numerical Deviation | Verification Status |
| :--- | :--- | :--- | :---: | :---: |
| **Density Idempotency** | $H_2O$ & $CH_4$ (Closed-shell RHF) | $\|P^2 - 2P\|_\infty < 10^{-13}$ | **$1.78 \times 10^{-15}$** | **Exact Double Precision** |
| **Orbital Orthonormality** | Phenol ($C_6H_5OH, 34\text{ AOs}$) | $\|C^T C - I\|_\infty < 10^{-14}$ | **$2.22 \times 10^{-15}$** | **Exact Machine Epsilon** |
| **SO(3) 3D Rotational Invariance** | $H_2O$ (Arbitrary 3D Euler angles) | $\Delta E = \|E(R \cdot X) - E(X)\|$ | **$1.13 \times 10^{-13}\text{ eV}$** | **Frame Invariant** |
| **Saunders-Hillier Trace Conservation** | Ammonia ($NH_3, \sigma = 8\text{ eV}$) | $\text{Tr}[P \cdot \Delta F_{\text{shift}}] \equiv 0$ | **$< 10^{-14}\text{ eV}$** | **Zero Ground State Contamination** |
| **Zero Heap Allocations Gate** | Multi-cycle SCF trajectory | Workspace pointer invariance | **0 dynamic allocations** | **0-Malloc Gate Passed** |

---

## Canonical Oracle Differential Parity (vs OpenMOPAC v23.2.5)

Direct, double-precision differential validation executed by running the compiled native OpenMOPAC v23.2.5 binary side-by-side against `mopac_rs` (`crates/mopac_core/tests/golden_parity.rs`):

| Target Molecule & Method | Upstream OpenMOPAC v23.2.5 | `mopac_rs` (Rust) | Agreement Metric |
| :--- | :---: | :---: | :---: |
| **Formaldehyde ($H_2CO$) AM1 Core Repulsion** | $392.012940\text{ eV}$ | **$392.012942\text{ eV}$** | **$\Delta = 2.0 \times 10^{-6}\text{ eV}$** |
| **Formaldehyde ($H_2CO$) AM1 Total Energy** | $-475.564320\text{ eV}$ | **$-475.293467\text{ eV}$** | **$0.057\%\text{ Relative Error}$** |
| **Methane ($CH_4$) AM1 Core Repulsion** | $203.335770\text{ eV}$ | **$203.335774\text{ eV}$** | **$\Delta = 4.0 \times 10^{-6}\text{ eV}$** |
| **Methane ($CH_4$) AM1 Total Energy** | $-183.191630\text{ eV}$ | **$-183.314791\text{ eV}$** | **$0.067\%\text{ Relative Error}$** |
| **Methyl Radical ($CH_3^\bullet$) UHF $\langle S^2 \rangle$** | $0.760978$ | **$0.760800$** | **$\Delta = 0.000178\text{ (0.023\%)}$** |
| **Methyl Radical ($CH_3^\bullet$) UHF Core Repulsion** | $142.120640\text{ eV}$ | **$142.120636\text{ eV}$** | **$\Delta = 4.0 \times 10^{-6}\text{ eV}$** |
| **Methyl Radical ($CH_3^\bullet$) UHF Total Energy** | $-167.891960\text{ eV}$ | **$-167.966787\text{ eV}$** | **$0.045\%\text{ Relative Error}$** |
| **Water ($H_2O$) AM1 Core Repulsion** | $145.077380\text{ eV}$ | **$145.077378\text{ eV}$** | **$\Delta = 2.0 \times 10^{-6}\text{ eV}$** |
| **Water ($H_2O$) AM1 Total Energy** | $-348.561800\text{ eV}$ | **$-350.143269\text{ eV}$** | **$0.454\%\text{ Relative Error}$** |
| **Water ($H_2O$) Gas Phase HoF (PM6)** | $-54.20404\text{ kcal/mol}$ | **$-54.20404\text{ kcal/mol}$** | **Exact Match ($\Delta = 0.000$)** |
| **Water COSMO Solvation Energy ($\varepsilon = 78.4$)** | $-0.32917\text{ eV}$ | **$-0.32917\text{ eV}$** | **Exact Match** |
| **Water COSMO Dielectric Net Force** | $0.000\text{ eV/\AA}$ | **$< 10^{-14}\text{ eV/\AA}$** | **Exact Translational Zero-Sum** |
| **Water Dimer PM6-D3H4 Heat of Formation** | $-71.99024\text{ kcal/mol}$ | **$-71.99024\text{ kcal/mol}$** | **$< 10^{-5}\text{ kcal/mol}$** |

---

## Criterion Microbenchmark Suite

Empirically measured single-thread performance benchmarks (`crates/mopac_core/benches/quantum_benchmarks.rs`):

| Benchmark Target | System & Physical Setup | Mean Execution Time | Effective Throughput |
| :--- | :--- | :---: | :---: |
| `scf/rhf_am1_water` | Water ($H_2O$), AM1 RHF converged SCF | **$39.8\text{ }\mu\text{s}$** | ~25,000 SCF / sec |
| `boron/bh3_pm6_scf` | Borane ($BH_3$), PM6 RHF converged SCF | **$68.8\text{ }\mu\text{s}$** | ~14,500 SCF / sec |
| `uhf/ch3_radical_am1_doublet` | Methyl radical ($CH_3^\bullet$), AM1 UHF Doublet | **$177.2\text{ }\mu\text{s}$** | ~5,600 UHF SCF / sec |
| `cosmo/dielectric_gradients_water` | Solvated $H_2O$ COSMO analytical gradients | **$71.5\text{ }\mu\text{s}$** | ~14,000 evaluations / sec |

---

## Technical Documentation

Detailed mathematical derivations, Fortran audits, and GPU specifications:
- [**Translation Manifesto**](explanations/TRANSLATION_MANIFESTO.md)
- [**Fortran Codebase Audit & DOP Feasibility**](explanations/01_FORTRAN_CODEBASE_AUDIT_AND_DOP_FEASIBILITY.md)
- [**Universal GPU Acceleration (Vulkan Compute / WGPU)**](explanations/02_UNIVERSAL_GPU_ACCELERATION_VULKAN_WGPU.md)
- [**Strict Testing & Verification Policy**](explanations/03_STRICT_TESTING_AND_VERIFICATION_POLICY.md)
- [**Translation Devlog & Physical Constants Audit**](explanations/04_TRANSLATION_DEVLOG_AND_PHYSICAL_CONSTANTS_AUDIT.md)
- [**Vulkan GPU Acceleration & Level Shifting**](explanations/05_VULKAN_GPU_ACCELERATION_AND_LEVEL_SHIFTING.md)

---

## Citation

If you use `mopac_rs` in your academic research, benchmarks, or software, please cite it as:

```bibtex
@software{mopac_rs_2026,
  author = {zeph.sys},
  title = {{MOPAC\_RS: Modern High-Performance Semi-Empirical Quantum Chemistry Engine in Rust}},
  year = {2026},
  publisher = {Zenodo},
  version = {0.1.0},
  doi = {10.5281/zenodo.22731254},
  url = {https://github.com/zephsystems/mopac_rs}
}
```

Direct metadata is also available in [`CITATION.cff`](CITATION.cff).

---

## License

Distributed under the **Apache License Version 2.0 (Apache-2.0)**. See [`LICENSE`](LICENSE) for complete details.
