# ⚡ Universal Cross-Platform GPU Acceleration for Semi-Empirical Quantum Mechanics: Vulkan Compute & WGPU Architecture

```text
 __     __ _   _ _     _  __    _    _   _    ____ ____  _   _ 
 \ \   / /| | | | |   | |/ /   / \  | \ | |  / ___|  _ \| | | |
  \ \ / / | | | | |   | ' /   / _ \ |  \| | | |  _| |_) | | | |
   \ V /  | |_| | |___| . \  / ___ \| |\  | | |_| |  __/| |_| |
    \_/    \___/|_____|_|\_\/_/   \_\_| \_|  \____|_|    \___/  
 Universal Cross-Vendor GPU Compute Specification (Vulkan / WGPU)
```

* **Document ID:** `EXPLANATION-002-UNIVERSAL-GPU-VULKAN-WGPU`
* **Target Architecture:** Universal GPU Compute Engine for `mopac_rs`
* **Target Hardware:** Any modern GPU (NVIDIA, AMD Radeon, Intel Arc/Xe, Apple Silicon M-series via Metal, Qualcomm Adreno, Mali)
* **License:** Apache License 2.0 (Apache-2.0)
* **Language:** English

---

## 1. Executive Summary & Problem Diagnosis

### 1.1. The Failure of Legacy MOPAC GPU Acceleration
Upstream MOPAC attempted GPU acceleration in the Fermi/Kepler era using proprietary NVIDIA CUDA (`diag_for_GPU.F90`, `density_for_GPU.F90`, `mod_calls_cublas.F90`). 

As documented in MOPAC's own internal documentation ([`matrix/README.md`](file:///tmp/mopac_ref/src/matrix/README.md)):
> *"the prior support for GPUs in MOPAC, which is not presently functioning, needs to be re-introduced"*

This legacy attempt failed for three structural reasons:
1. **Proprietary Lock-in:** By depending exclusively on CUDA and `cuBLAS`, users with AMD GPUs, Intel GPUs, Apple Silicon MacBooks, or open-source Linux drivers were completely locked out.
2. **Brittle C-Binding Glue:** Fragile Fortran `iso_c_binding` interfaces to external CUDA binaries created build-system friction and broke across compiler updates.
3. **Improper Memory Granularity:** Transferring small triangular matrices back and forth across the PCIe bus for small molecules caused memory-transfer latency to dwarf computation time (`run_mopac.F90:344`).

---

## 2. The Universal Solution: Vendor-Agnostic Compute in Rust

To enable **every user on any consumer or workstation hardware** to accelerate semi-empirical quantum calculations, `mopac_rs` adopts a **universal, vendor-agnostic compute architecture**:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        MOPAC_RS UNIVERSAL COMPUTE PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│                      [HIGH-LEVEL QUANTUM ALGORITHMS]                                    │
│                 Fock Build • Two-Electron TEIs • SCF • DIIS                             │
│                                       │                                                 │
│                    ┌──────────────────┴──────────────────┐                              │
│                    ▼                                     ▼                              │
│        [CPU Backend (AVX2 / AVX-512)]           [Universal GPU Backend]                 │
│        • Rayon multi-core thread pool           • wgpu / Vulkan Compute Engine          │
│        • Small systems (N_atoms < 80)           • Large systems (N_atoms >= 80)         │
│                    │                                     │                              │
│                    │                            ┌────────┴────────┐                     │
│                    │                            ▼                 ▼                     │
│                    │                     [SPIR-V Shaders]  [WGSL Shaders]               │
│                    │                            │                 │                     │
│                    │        ┌───────────────────┼─────────────────┴──────────────────┐  │
│                    │        │                   │                                    │  │
│                    ▼        ▼                   ▼                                    ▼  │
│               [x86_64 / ARM]  [Vulkan 1.2+]      [Metal 3]                       [DirectX 12]   │
│               Host CPU SIMD   NVIDIA, AMD, Intel  Apple Silicon (M1-M4)          Windows        │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1. Technology Selection: `wgpu` + SPIR-V / WGSL
Rather than tying the codebase to proprietary APIs, `mopac_rs` uses **`wgpu`** (the Rust native WebGPU implementation) and **direct Vulkan compute shaders**:
* **Universal Hardware Portability:**
  * **Linux / Windows / Android:** Dispatches natively through **Vulkan 1.2+** (running on NVIDIA GeForce/RTX, AMD Radeon/RDNA, Intel Arc/Iris, and Qualcomm GPUs).
  * **macOS / iOS:** Dispatches natively through **Apple Metal 3** (running on Apple Silicon M1/M2/M3/M4 unified memory architecture with zero-copy buffer sharing).
  * **Windows (Fallback):** Can target **DirectX 12** if Vulkan drivers are absent.
* **No Proprietary Toolchains:** Compiles entirely using standard `cargo build` without requiring the multi-gigabyte NVIDIA CUDA Toolkit.

---

## 3. Algorithmic GPU Workload Mapping for Semi-Empirical QM

Semi-empirical NDDO methods possess computational characteristics that map naturally onto GPU compute shader architectures:

### 3.1. Workload 1: Two-Center Two-Electron Integral (TEI) Batch Engine
* **Complexity:** $O(N_{\text{atoms}}^2)$ atom pairs $(A, B)$.
* **GPU Dispatch Pattern:**
  * Each GPU Workgroup processes one atom pair $(A, B)$.
  * Local invocation threads cooperatively evaluate:
    1. Interatomic distance vector $\vec{R}_{AB} = \vec{r}_B - \vec{r}_A$.
    2. 3D coordinate rotation matrix from Cartesian axes to diatomic local $z$-axis.
    3. Monopole, dipole, and quadrupole charge multipoles over Slater-type orbitals.
    4. Klopman-Ohno / Dewar electron repulsion integrals.
    5. Rotation back into molecular coordinates (`rotate` / `solrot` equivalent).
  * **Memory Benefit:** The entire $16 \times 16$ (or $81 \times 81$) diatomic block is synthesized inside GPU high-speed **Shared Memory (L1 cache)** without writing intermediate unrotated tensors to global VRAM.

### 3.2. Workload 2: Two-Electron Fock Matrix Contraction
* **Operation:**
  $$F_{\mu\nu}^{(2)} = \sum_{\lambda,\sigma} P_{\lambda\sigma} \left[ (\mu\nu|\lambda\sigma) - \frac{1}{2}(\mu\lambda|\nu\sigma) \right]$$
* **GPU Mapping:**
  * Parallel reduction across orbital pairs using compute shader workgroup barriers (`workgroupBarrier()`).
  * Avoids atomic collision by assigning distinct rows of $F$ to distinct compute warps/wavefronts.

### 3.3. Workload 3: Density Matrix Generation & DIIS Matrix Algebra
* **Operations:**
  * Density matrix accumulation: $P = 2 C_{\text{occ}} C_{\text{occ}}^T$ (General Symmetric Rank-$k$ Update / GEMM).
  * DIIS commutator error matrix: $e = F P - P F$ (Dense Matrix Multiplications).
* **GPU Mapping:**
  * Tiled matrix multiplication compute shaders achieving multi-TFLOPS performance across all modern GPUs.

### 3.4. Workload 4: Core-Core Nuclear Repulsion Gradients
* Pairwise additive force evaluation $\sum_{B \neq A} \nabla_A E_{AB}^{\text{core-core}}$ mapped to an embarrassingly parallel 1D workgroup grid.

---

## 4. Dynamic Hybrid CPU-GPU Dispatch Strategy

GPU acceleration is not advantageous for all problem sizes due to host-to-device PCIe bus latency and shader launch overhead. `mopac_rs` implements an **autonomous dynamic crossover dispatcher**:

```text
               ┌────────────────────────────────────────────────────────┐
               │         System Analysis (Atoms N, Orbitals M)          │
               └───────────────────────────┬────────────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                N_atoms < 80                          N_atoms >= 80
            (M_orbs < ~300)                       (M_orbs >= ~300)
                        │                                     │
                        ▼                                     ▼
            ┌───────────────────────┐             ┌───────────────────────┐
            │   CPU SIMD Pipeline   │             │ Universal GPU Compute │
            │  • AVX2/AVX-512 FMA   │             │  • Vulkan / wgpu      │
            │  • Zero-copy L1/L2    │             │  • Asynchronous VRAM  │
            │  • Rayon thread pool  │             │  • High-throughput    │
            └───────────────────────┘             └───────────────────────┘
```

1. **Small Molecules ($N_{\text{atoms}} < 80$):**
   * Handled 100% on CPU using vector SIMD (AVX2/AVX-512). Latency is sub-millisecond; zero VRAM transfer overhead.
2. **Macromolecules & Crystals ($N_{\text{atoms}} \ge 80$):**
   * Automatically dispatched to the Universal GPU Compute pipeline.
   * On Apple Silicon, unified memory architecture allows host pointers to be mapped directly into GPU memory buffers without PCIe copy overhead (`StorageBuffer` zero-copy).

---

## 5. Implementation Roadmap for Universal GPU Support

1. **Phase 1 (Foundational CPU Core):**
   * Complete pure Rust mathematical routines and SoA memory structures.
2. **Phase 2 (WGPU / Vulkan Compute Abstraction):**
   * Implement `GpuContext` and `ComputeBuffer` allocations in `mopac-core` using `wgpu`.
3. **Phase 3 (Compute Shaders):**
   * Write WGSL / SPIR-V compute kernels for:
     * `tei_batch_nddo.comp` (Two-electron integrals).
     * `fock_contraction.comp` (Fock build).
     * `gemm_density.comp` (Density update).
4. **Phase 4 (Zero-Mock Verification):**
   * Rigorous unit tests ensuring bit-identical numerical results between the CPU AVX2 reference and the Vulkan/Metal GPU pipelines ($\Delta E \le 10^{-7} \text{ eV}$).

---

*Universal GPU acceleration ensures that high-performance quantum chemistry is democratized across all hardware vendors without proprietary barriers.*
