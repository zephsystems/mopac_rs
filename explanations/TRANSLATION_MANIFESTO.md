# 📜 MOPAC_RS: Translation & Modernization Manifesto
## Fortran to Rust Semi-Empirical Quantum Chemistry Engine

```text
  __  __  ___  ____   _    ____     ____  ____  
 |  \/  |/ _ \|  _ \ / \  / ___|   |  _ \/ ___| 
 | |\/| | | | | |_) / _ \| |   ____| |_) \___ \ 
 | |  | | |_| |  __/ ___ \ |__|____|  _ < ___) |
 |_|  |_|\___/|_| /_/   \_\____|   |_| \_\____/  
 Modern Data-Oriented Semi-Empirical Quantum Mechanics
```

* **Project Identifier:** `mopac_rs` (`05_mopac_rs`)
* **Upstream Ancestry:** OpenMOPAC (MOPAC v22 / v23 Fortran 77/90/2003 Engine)
* **License:** Apache License 2.0 (Apache-2.0)
* **Target Language & Toolchain:** Rust (Edition 2021 / 2024), Strict `no-std` mathematical core capability, explicit SIMD (AVX2/AVX-512)
* **Status:** Foundational Translation Manifesto & Architectural Directive

---

## 1. Executive Summary & Purpose

MOPAC (Molecular Orbital PACkage) is one of the most historically significant, battle-tested semi-empirical quantum chemistry programs in computational chemistry and molecular biology. Over more than four decades of evolution, it has enabled high-speed electronic structure calculations, geometry optimizations, and thermodynamic predictions for molecules containing thousands of atoms.

However, the legacy Fortran codebase suffers from structural limitations accumulated across generations:
* **Global Mutable State:** Extensive reliance on global modules and legacy `COMMON` blocks (`molkst_C.F90`, `Common_arrays_C.F90`), causing thread-safety hazards, memory leak liabilities, and side-effect coupling.
* **Implicit Triangular Indexing:** Flat 1D array allocations simulating symmetric packed matrices ($N(N+1)/2$) with manual, error-prone index arithmetic scattered throughout arithmetic loops.
* **Tightly Coupled I/O and Computation:** File parsing, disk operations, error handling, and core quantum mathematical algorithms are deeply intertwined, making modular embedding, testing, and modern concurrency difficult.
* **Cache-Hostile Memory Access:** Fortran column-major layouts translated without alignment guarantees, hindering modern SIMD vectorization and data prefetching.

**The Mission of `mopac_rs`:**  
To rescue, preserve, and modernize the foundational mathematical pillars of MOPAC into an uncompromised, thread-safe, high-performance, and mathematically verifiable Rust implementation structured strictly under a **Data-Oriented Programming (DOP)** paradigm.

---

## 2. License & Intellectual Property Governance

* **License Declaration:** Distributed under the **Apache License, Version 2.0** (`Apache-2.0`).
* **Upstream Compatibility:** Upstream OpenMOPAC is licensed under Apache-2.0. All translated routines, parameter sets, and derived algorithms adhere to the requirements, attributions, and copyright notices of the Apache-2.0 license.
* **Documentation & Code Standard:** All code comments, docstrings, technical documentation, architectural manifests, and explanatory specifications must strictly be written in **English**.

---

## 3. Delimited Scope: Rescuing the Mathematical Pillars

The translation deliberately decouples legacy peripheral utilities (such as legacy terminal formatters, old deck readers, and obsolete interactive menus) from the **indispensable mathematical and physical core**. 

The core quantum-mechanical engine to be rescued and translated encompasses:

### 3.1. Semi-Empirical Hamiltonians & Parameterizations
* **Target Models:** Complete support for NDDO (Neglect of Diatomic Differential Overlap) Hamiltonians:
  * **MNDO** (Modified Neglect of Diatomic Overlap)
  * **AM1** (Austin Model 1)
  * **PM3** (Parameterized Model 3)
  * **PM6** (Parameterized Model 6) and its modern corrections (PM6-D3H4)
  * **PM7 / PM8** & **RM1** parameter models
* **Parameter Architecture:**
  * Clean extraction of empirical parameters ($U_{ss}, U_{pp}, U_{dd}, \zeta_s, \zeta_p, \zeta_d, \beta_s, \beta_p, \beta_d, \alpha$, core repulsion Gaussians $a_k, b_k, c_k$).
  * Static, immutable, type-safe parameter registries with zero runtime parsing overhead.

### 3.2. One-Center & Two-Center Integral Evaluation
* **One-Center Integrals:**
  * Core kinetic + nuclear attraction energies: $U_{ss}, U_{pp}, U_{dd}$.
  * One-center two-electron Coulomb and exchange integrals: $(ss|ss), (ss|pp), (pp|pp), (pp|p'p'), (sp|sp)$, etc.
* **Two-Center One-Electron Integrals (Resonance & Core Attraction):**
  * Diatomic overlap matrix elements $S_{\mu\nu}$ over Slater-type orbitals (STOs).
  * Resonance integrals:
    $$H_{\mu\nu} = \frac{1}{2} (\beta_\mu^A + \beta_\nu^B) S_{\mu\nu}$$
  * Core-electron attraction integrals ($V_{\mu\nu, B}$) using point-charge and multipole representations.
* **Two-Center Two-Electron Repulsion Integrals (TEIs):**
  * Dewar-Klopman / Ohno multipole expansion over local diatomic reference frames (`rotate.F90`, `solrot.F90`, `diat.F90`).
  * Transformation from diatomic local axes to the molecular Cartesian coordinate frame via 3D rotation matrices.

### 3.3. Core-Core Nuclear Repulsion Energy
* Calculation of $E_{\text{core-core}}$:
  $$E_{AB}^{\text{core-core}} = Z_A Z_B (s_A s_A | s_B s_B) \left[ 1 + e^{-\alpha_A R_{AB}} + e^{-\alpha_B R_{AB}} \right] + F_{AB}^{\text{Gaussian}}(R_{AB})$$
* Parameterized Gaussian screening terms for AM1, PM3, and PM6 to correct non-covalent and short-range repulsion artifacts.

### 3.4. Fock Matrix Construction & Density Assembly
* Efficient, cache-optimized assembly of the Fock operator $F = H^{\text{core}} + G(P)$:
  * Diagonal one-center blocks:
    $$F_{\mu\mu} = U_{\mu\mu} + \sum_{B \neq A} V_{\mu\mu, B} + \sum_{\lambda \in A} P_{\lambda\lambda} \left[ (\mu\mu|\lambda\lambda) - \frac{1}{2}(\mu\lambda|\mu\lambda) \right] + \sum_{B \neq A} \sum_{\lambda,\sigma \in B} P_{\lambda\sigma} (\mu\mu|\lambda\sigma)$$
  * Off-diagonal two-center blocks:
    $$F_{\mu\nu} = H_{\mu\nu}^{\text{core}} - \frac{1}{2} \sum_{\lambda \in A} \sum_{\sigma \in B} P_{\lambda\sigma} (\mu\lambda|\nu\sigma)$$

### 3.5. Roothaan-Hall SCF Solver & Orthogonalization
* Secular equation solution in the orthogonal NDDO basis:
  $$F C = C \epsilon$$
  via high-performance symmetric/Hermitian eigensolvers.
* Density matrix evaluation:
  $$P_{\mu\nu} = 2 \sum_{i=1}^{\text{occ}} C_{\mu i} C_{\nu i}$$
  (with extension to open-shell UHF / ROKS formulations).

### 3.6. Iterative Convergence Accelerators
* **Pulay DIIS (Direct Inversion in the Iterative Subspace):**
  * Construction of the error matrix $e_k = F_k P_k - P_k F_k$.
  * Minimal residual quadratic programming over historical subspaces.
* Dynamic level-shifting, Fermi smearing, and damping for narrow HOMO-LUMO gap configurations.

### 3.7. Analytical Nuclear Gradients & Energy Derivatives
* Exact analytical evaluation of $\nabla_A E_{\text{total}}$ with respect to atomic Cartesian coordinates:
  * Derivative of the core Hamiltonian ($\nabla H^{\text{core}}$).
  * Derivative of two-electron repulsion integrals ($\nabla G(P)$).
  * Derivative of the core-core repulsion ($\nabla E^{\text{core-core}}$).
  * Hellmann-Feynman and Pulay force corrections.
* Basis for geometry optimization (BFGS, L-BFGS, Eigenvector Following).

---

## 4. Architectural Dictum: Modern Data-Oriented Programming (DOP)

To eliminate the inefficiencies of legacy Fortran, `mopac_rs` adopts a rigorous **Data-Oriented Programming** architecture designed around hardware cache lines and register throughput:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           MOPAC_RS DATA-ORIENTED TOPOLOGY                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   [ATOMIC BATCH (SoA)]                   [PRE-ALLOCATED SCF WORKSPACE]                  │
│   • x: [f64; N_atoms] (align 64)         • fock_matrix:   ContiguousAligned<f64>        │
│   • y: [f64; N_atoms] (align 64)         • density_mat:   ContiguousAligned<f64>        │
│   • z: [f64; N_atoms] (align 64)         • h_core:        ContiguousAligned<f64>        │
│   • atomic_numbers: [u8; N_atoms]        • eigenvectors:  ContiguousAligned<f64>        │
│   • orbital_offsets: [u32; N_atoms]      • diis_subspace: Pre-allocated circular ring   │
│                                                                                         │
│   ▲                                      ▲                                              │
│   │                                      │                                              │
│   └───► PURE MATHEMATICAL OPERATORS ◄────┘                                              │
│         (Explicit SIMD AVX2/AVX-512, Zero Allocations in Iteration Loops, No Heap)      │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1. Core DOP Principles

1. **Struct of Arrays (SoA) for Molecular Topology:**
   * Coordinates and atomic attributes are stored in contiguous, 64-byte aligned SIMD-friendly arrays rather than arrays of fragmented pointers or structs with padding holes.
2. **Zero-Allocation Inner Loop Policy:**
   * The Self-Consistent Field (SCF) iterative loop and DIIS convergence steps must execute **with zero dynamic heap allocations (`0 malloc / free`)**. All intermediate buffers, secular matrices, and eigenvectors are pre-allocated into a unified, reusable `ScfWorkspace`.
3. **Contiguous Flat Matrix Representations:**
   * Elimination of arbitrary nested slices or pointer matrices (`Vec<Vec<f64>>`). All matrices are stored in single continuous memory buffers with explicit cache-aligned strides, supporting both upper-triangular packed storage (for memory-constrained scales) and unrolled full matrices (for vectorized BLAS/LAPACK / AVX kernels).
4. **Purity of Mathematical Functions:**
   * Core numerical routines are pure functions: they take immutable inputs and write into explicit mutable workspace slices. No hidden global variables, no thread-local singletons, and no ambient system state.
5. **Hardware Vectorization (SIMD):**
   * Two-electron diatomic multipole expansion loops are structured to allow auto-vectorization and explicit AVX2/FMA instructions, evaluating multiple orbital pairs simultaneously.

---

## 5. Verification Standard & Zero-Mock Policy

In strict alignment with the engineering integrity of our platform:
* **Zero Mock Data:** No dummy outputs, synthetic placeholders, or mocked test returns. Every calculation must compute genuine quantum-mechanical values.
* **Empirical Verification Against Upstream MOPAC:**
  * Benchmark test systems (ranging from diatomics like H₂, N₂, CO, small organic molecules like Methane, Benzene, and Pyridine, to standard test decks in the upstream MOPAC test suite) will be computed.
  * Numerical agreement with upstream MOPAC must satisfy:
    * Total Heat of Formation / Electronic Energy: $\Delta E \le 10^{-7} \text{ eV}$ ($< 10^{-6} \text{ kcal/mol}$).
    * Gradient RMS: $\Delta (\nabla E) \le 10^{-6} \text{ kcal/mol/\AA}$.
    * Orbital Eigenvalues: $\Delta \epsilon_i \le 10^{-6} \text{ eV}$.

---

## 6. Project Directory Layout

```text
05_mopac_rs/
├── LICENSE                        # Apache License 2.0
├── Cargo.toml                     # Workspace & build configuration
├── explanations/
│   ├── TRANSLATION_MANIFESTO.md   # This document
│   ├── 01_MATHEMATICAL_PILLARS.md # Deep mathematical derivations of NDDO/MOPAC
│   ├── 02_DATA_ORIENTED_DESIGN.md # SoA memory layouts and cache-alignment specs
│   └── 03_FORTRAN_TO_RUST_MAP.md  # Detailed Fortran subroutine mapping matrix
├── crates/
│   ├── mopac-core/                # Pure mathematical functions & integral kernels
│   │   ├── src/
│   │   │   ├── integrals/         # One-electron, two-electron, multipole rotations
│   │   │   ├── fock/              # Fock matrix assembly
│   │   │   ├── scf/               # Roothaan-Hall diagonalization & DIIS
│   │   │   ├── gradients/         # Analytical energy derivatives
│   │   │   └── parameters/        # Type-safe parameter registries (AM1, PM6, etc.)
│   │   └── Cargo.toml
│   └── mopac-cli/                 # Input parsing & execution driver
```

---

*This manifesto serves as the foundational contract for the translation and modernization of MOPAC into Rust.*
