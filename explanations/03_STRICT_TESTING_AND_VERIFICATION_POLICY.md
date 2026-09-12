# MOPAC_RS: Strict Testing, Verification & Mathematical Parity Policy

```text
  ____ _____ ____  ___ ____ _____   ____   ___  _     ___ ______   __
 / ___|_   _|  _ \|_ _/ ___|_   _| |  _ \ / _ \| |   |_ _/ ___\ \ / /
 \___ \ | | | |_) || | |     | |   | |_) | | | | |    | | |    \ V / 
  ___) || | |  _ < | | |___  | |   |  __/| |_| | |___ | | |___  | |  
 |____/ |_| |_| \_\___\____| |_|   |_|    \___/|_____|___\____| |_|  
 Mathematical Parity & Verification Specification
```

* **Document ID:** `EXPLANATION-003-STRICT-TESTING-POLICY`
* **Target Project:** `mopac_rs` (`05_mopac_rs`)
* **Reference Upstream Standard:** MOPAC v23.2.5 Official Fortran Engine
* **License:** Apache License 2.0 (Apache-2.0)
* **Language:** English

---

## 1. Axiomatic Principles: The Zero-Mock Standard

In computational quantum chemistry, numerical accuracy is paramount. A single error in orbital phase, two-electron integral screening, or core-core repulsion can compromise molecular geometries, vibrational spectra, and thermodynamic heats of formation.

To ensure total integrity, `mopac_rs` enforces the following axioms:

1. **Axiom of Zero Mock Data:**  
   No unit test, integration test, or benchmark may utilize synthetic, mocked, or placeholder return values. Every test assertion must execute the actual physical and mathematical implementation against genuine molecular inputs.
2. **Axiom of Empirical Parity:**  
   Every newly translated module must be verified against the official reference Fortran binary (`MOPAC v23.2.5`) on identical input geometries.
3. **Axiom of Invariant Preservation:**  
   Quantum mechanical invariants (orthonormality, density matrix idempotency, rotation equivariance, gradient-energy consistency) must hold to machine precision.

---

## 2. Quantitative Tolerance Thresholds

All tests in `mopac_rs` are evaluated against strict numerical tolerances:

| Physical Metric | Notation | Maximum Allowed Error | Benchmark Source |
| :--- | :--- | :--- | :--- |
| **Total Heat of Formation** | $\Delta (\Delta H_f)$ | $\le 1.0 \times 10^{-6} \text{ kcal/mol}$ | Upstream MOPAC v23.2.5 |
| **Electronic Energy** | $\Delta E_{\text{elec}}$ | $\le 1.0 \times 10^{-7} \text{ eV}$ | Upstream MOPAC v23.2.5 |
| **Orbital Eigenvalues (HOMO/LUMO)** | $\Delta \epsilon_i$ | $\le 1.0 \times 10^{-6} \text{ eV}$ | Upstream MOPAC v23.2.5 |
| **Atomic Partial Charges** | $\Delta q_A$ | $\le 1.0 \times 10^{-5} e$ | Mulliken / Lowdin analysis |
| **Secular Matrix Orthogonality** | $\|C^T C - I\|_\infty$ | $\le 1.0 \times 10^{-14}$ | Double-precision machine $\epsilon$ |
| **Density Matrix Idempotency** | $\|P S P - 2P\|_\infty$ | $\le 1.0 \times 10^{-10}$ | Closed-shell idempotency |
| **Analytical vs Finite-Diff Force** | $\|\nabla_A E_{\text{anal}} - \nabla_A E_{\text{fd}}\|$ | $\le 1.0 \times 10^{-6} \text{ kcal/mol/\AA}$ | Central difference $\delta = 10^{-4} \text{ \AA}$ |
| **Cross-Backend Invariance (GPU vs CPU)** | $\Delta E_{\text{GPU-CPU}}$ | $\le 1.0 \times 10^{-8} \text{ eV}$ | Vulkan/Metal vs CPU AVX2 |

---

## 3. The Three-Tier Verification Pyramid

```text
               ┌─────────────────────────────────────────────────┐
               │                     TIER 3                      │
               │         End-to-End MOPAC Regression             │
               │   (20+ Real Molecules vs MOPAC v23 Binary)      │
               └───────────────────────┬─────────────────────────┘
                                       │
                       ┌───────────────┴───────────────┐
                       │            TIER 2             │
                       │     Subsystem Parity Tests    │
                       │  (Fock Build, DIIS, SCF Loop) │
                       └───────────────┬───────────────┘
                                       │
                       ┌───────────────┴───────────────┐
                       │            TIER 1             │
                       │  Atomic Math Unit Invariants  │
                       │(Rotations, Integrals, Diatomics)
                       └───────────────────────────────┘
```

### 3.1. Tier 1: Atomic Mathematical Unit Tests
* **Diatomic Overlap ($S_{\mu\nu}$):**
  * Tested over standard Slater orbitals against closed-form analytical formulas.
  * Verified for radial distances $R \in [0.5 \text{ \AA}, 10.0 \text{ \AA}]$.
* **Rotation Matrices ($D^{(1)}, D^{(2)}$):**
  * Transformation of $p$ ($3\times 3$) and $d$ ($5\times 5$) orbital basis sets under arbitrary 3D Euler rotation angles $(\alpha, \beta, \gamma)$.
  * Test condition: $\|D D^T - I\| < 10^{-14}$ and $\det(D) = 1.0$.
* **Two-Electron Integral Radial Asymptotics:**
  * For atom separations $R_{AB} > 15 \text{ \AA}$, multipole expansions must asymptotically approach the classical electrostatic point-charge limit $\frac{e^2}{R_{AB}}$ to within $0.001\%$.
* **Parameter Registry Immutability:**
  * Automated checks verifying every parameter ($U_{ss}, \zeta, \beta, \alpha$, Gaussian parameters) matches the upstream OpenMOPAC source files (`parameters_for_*.F90`) bit-for-bit.

### 3.2. Tier 2: Subsystem Parity Tests
* **Fock Matrix Build Invariance:**
  * Given a reference fixed density matrix $P_{\text{ref}}$, verify that $F[P_{\text{ref}}]$ reproduces the upstream MOPAC Fock matrix identically ($< 10^{-10} \text{ eV}$).
* **Roothaan-Hall Eigensolver:**
  * Orthonormality of molecular orbitals $C$.
  * Correct sorting of eigenvalues $\epsilon_1 \le \epsilon_2 \le \dots \le \epsilon_M$.
  * Accurate identification of the Fermi level and HOMO/LUMO gap.
* **DIIS Subspace Robustness:**
  * Verify that Pulay DIIS achieves monotonic quadratic error decay $\|e_k\| \to 0$.
  * Test pathological near-degenerate configurations to ensure DIIS fallback mechanisms (damping / level shifting) trigger seamlessly without crash.
* **Energy-Gradient Consistency (Finite Differences):**
  * For any arbitrary geometry, compute the numerical gradient:
    $$g_{\text{num}, i} = \frac{E(x_i + \delta) - E(x_i - \delta)}{2\delta}$$
  * Compare against the analytical gradient vector $\nabla_i E$.
  * Parity condition: $\|g_{\text{anal}} - g_{\text{num}}\|_\infty < 10^{-6} \text{ kcal/mol/\AA}$.

### 3.3. Tier 3: End-to-End Parity Benchmarks
A dedicated test suite executing real molecules against the installed `/home/cyclop/.conda/envs/mopac/bin/mopac` reference binary:

1. **Diatomics:** $\text{H}_2, \text{N}_2, \text{CO}, \text{HF}, \text{Cl}_2$.
2. **Saturated Hydrocarbons:** Methane ($\text{CH}_4$), Ethane ($\text{C}_2\text{H}_6$), Propane ($\text{C}_3\text{H}_8$), Cyclohexane ($\text{C}_6\text{H}_{12}$).
3. **Aromatic & Conjugated Systems:** Benzene ($\text{C}_6\text{H}_6$), Pyridine ($\text{C}_5\text{H}_5\text{N}$), Azulene ($\text{C}_{10}\text{H}_8$).
4. **Biomolecular building blocks:** Glycine, Alanine, Water cluster $(H_2O)_6$.
5. **Heteroatomic & Second-Row Systems:** Methanol ($\text{CH}_3\text{OH}$), Acetic Acid ($\text{CH}_3\text{COOH}$), Dimethyl Sulfoxide ($\text{DMSO}$), Silane ($\text{SiH}_4$).
6. **Charged Species & Radicals:** Ammonium ($\text{NH}_4^+$), Hydroxide ($\text{OH}^-$), Methyl radical ($\text{CH}_3^\bullet$).

---

## 4. Cross-Backend Invariance (CPU vs Universal GPU)

When calculations are dispatched to the Universal GPU Compute backend (Vulkan / WGPU / Metal):
* The GPU pipeline must execute the exact same mathematical formulation as the CPU reference engine.
* Automated dual-execution tests run both backends simultaneously and assert:
  $$\max_{\mu, \nu} |F_{\mu\nu}^{\text{GPU}} - F_{\mu\nu}^{\text{CPU}}| \le 1.0 \times 10^{-10} \text{ eV}$$
  $$|E_{\text{total}}^{\text{GPU}} - E_{\text{total}}^{\text{CPU}}| \le 1.0 \times 10^{-8} \text{ eV}$$
* This guarantees that users on AMD, Intel, NVIDIA, or Apple Silicon obtain 100% reproducible science.

---

## 5. Continuous Integration (CI) Enforcement Pipeline

Every Pull Request and commit must pass the automated GitHub Actions CI pipeline:

```yaml
name: mopac_rs CI & Mathematical Parity Gate
on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust Toolchain
        run: rustup default stable && rustup component add clippy rustfmt
      - name: Formatting Check
        run: cargo fmt --all -- --check
      - name: Clippy Linter
        run: cargo clippy --all-targets -- -D warnings
      - name: Run Atomic Mathematical Tests
        run: cargo test --workspace --lib
      - name: Install Reference MOPAC
        run: conda create -n mopac -c conda-forge mopac
      - name: Execute End-to-End Parity Suite
        run: cargo test --test mopac_upstream_parity -- --nocapture
```

### Merge Blocker Rule:
A Pull Request **CANNOT be merged** if:
1. Any unit test fails.
2. Any parity test with upstream MOPAC exceeds the $1.0 \times 10^{-6} \text{ kcal/mol}$ threshold.
3. Any memory allocation occurs inside the inner SCF iteration loop.
4. Any non-English code comments or documentation are introduced.

---

*This policy establishes the uncompromising standard of engineering and mathematical rigor for the MOPAC_RS project.*
