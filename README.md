# ⚛️ MOPAC_RS (`05_mopac_rs`)
> **Modern High-Performance Data-Oriented Semi-Empirical Quantum Chemistry Engine in Rust**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Rust 2021/2024](https://img.shields.io/badge/Language-Rust%202021%2F2024-orange.svg)]()
[![SIMD: AVX2 / AVX-512](https://img.shields.io/badge/Acceleration-AVX2%20%7C%20AVX--512-red.svg)]()
[![GPU: Vulkan / WGPU](https://img.shields.io/badge/Compute-Vulkan%20%7C%20WGPU-green.svg)]()

---

## 🏛️ Project Overview

`mopac_rs` is a complete rewrite and modernization of the legendary **MOPAC** (Molecular Orbital PACkage) from legacy Fortran into **Rust**, built strictly around a **Data-Oriented Programming (DOP)** architecture.

### Key Architectural Highlights
* **Data-Oriented Memory Layout:** Struct of Arrays (SoA) aligned to 64-byte hardware cache lines, eliminating pointer-chasing and non-contiguous triangular index arithmetic.
* **Zero-Allocation Inner Loop Policy:** Pre-allocated reusable `ScfWorkspace` for self-consistent field (SCF) cycles and Pulay DIIS extrapolation (`0 malloc/free` inside hot loops).
* **Universal Hardware Portability:** Zero proprietary lock-in. Dual-backend architecture supporting CPU SIMD (AVX2/AVX-512) and Universal GPU Compute via **Vulkan Compute / WGPU** (NVIDIA, AMD Radeon, Intel Arc, and Apple Silicon Metal).
* **Zero-Mock Verification Standard:** Every module is verified against the official **MOPAC v23.2.5** engine to $< 10^{-6} \text{ kcal/mol}$ precision.

---

## 📚 Architectural Manifests & Specifications

Comprehensive architectural documents and engineering treatises are maintained in the [`explanations/`](explanations/) directory:

1. 📄 [**Translation & Modernization Manifesto**](explanations/TRANSLATION_MANIFESTO.md)  
   Foundational charter, legal governance, and delimitation of rescued mathematical pillars (NDDO, AM1, PM3, PM6, PM7, RM1).
2. 📄 [**Fortran Codebase Audit & DOP Feasibility**](explanations/01_FORTRAN_CODEBASE_AUDIT_AND_DOP_FEASIBILITY.md)  
   In-depth technical audit of upstream Fortran pathologies (`Common_arrays_C`, `save` statics, packed triangular indexing) and the DOP memory layout solution in Rust.
3. 📄 [**Universal GPU Acceleration (Vulkan / WGPU)**](explanations/02_UNIVERSAL_GPU_ACCELERATION_VULKAN_WGPU.md)  
   Specification for cross-vendor GPU computing across AMD, Intel, NVIDIA, and Apple Silicon, replacing defunct legacy CUDA implementations.
4. 📄 [**Strict Testing & Verification Policy**](explanations/03_STRICT_TESTING_AND_VERIFICATION_POLICY.md)  
   The three-tier testing pyramid, mathematical tolerances, and empirical parity validation with upstream MOPAC.

---

## 📜 License

Distributed under the **Apache License Version 2.0 (Apache-2.0)**. See [`LICENSE`](LICENSE) for details.
