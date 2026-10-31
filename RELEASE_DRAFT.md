# Release Notes: MOPAC_RS v0.1.5

> **Status**: Official Release v0.1.5  
> **Date**: 2026-10-31  
> **Base Branch**: `main`  
> **Target Tag**: `v0.1.5`

---

## 1. Executive Summary

Version `0.1.5` of `mopac_rs` delivers universal HPC GPU memory architecture support, targeted fidelity refinements, CLI flag parity, SCF convergence invariants, and end-to-end benchmark provenance tracking against canonical **OpenMOPAC v23.2.5**:

1. **Universal HPC GPU Memory Architecture (`mopac_gpu::context`, `coulomb_fp32`)**:
   - Introduced `GpuMemoryArchitecture` and `GpuMemoryInfo` providing runtime physical heap introspection across discrete VRAM (GDDR6/HBM), unified memory architectures (UMA on Grace Hopper / APU), and PCIe Resizable BAR (ReBAR).
   - Added zero-copy direct host mapping on UMA/ReBAR, bypassing PCIe DMA staging latency.
   - Upgraded to `GpuBatchDeviceMemoryManager` with backward-compatible aliases for legacy GDDR6 pipelines.
2. **PM6 Additive Trigonal-Planar Nitrogen Correction (`nsp2`)**:
   - Implemented canonical OpenMOPAC `nsp2_correction` and `nsp2_atom_correction` (`set_up_dentate.F90:139-210`, `compfg.F90:371`) for trigonal planar $sp^2$ nitrogen centers with $< 2$ hydrogen substituents.
   - Eliminates the systematic $+0.500\text{ kcal/mol}$ discrepancy on nitro/nitrate compounds ($\text{CH}_3\text{NO}_2$, $\text{CH}_3\text{NO}_3$, $\text{C}_2\text{H}_5\text{NO}_3$, $\text{FNO}_3$) and $+0.980\text{ kcal/mol}$ on $\text{N}_2\text{O}_5$, reaching $< 0.001\text{ kcal/mol}$ parity with OpenMOPAC.
3. **CLI Dispersion Flag Dispatch Parity (`--disp`)**:
   - Fixed CLI flag dispatch so that `--disp none` / `--disp off` reliably disable dispersion corrections, while `--disp d3` / `--disp d3bj` / `--disp pm7` / `--disp pm6-dh+` activate the intended dispersion models in exact correspondence with the Python API.
   - Preserves exact PM7 default dispersion parity against OpenMOPAC.
4. **SCF Level-Shift Safety & Root Invariance (`run_rhf_scf_shift_safe`)**:
   - Added `run_rhf_scf_shift_safe` in `mopac_core::scf::scf_loop` ensuring that level-shift assisted SCF convergence lands on the genuine ground-state stationary point rather than trapping the iteration in higher-energy solutions.
   - Enhanced convergence telemetry with explicit `density_change` tracking.
5. **UHF Open-Shell Multiplicity Fidelity**:
   - Validated explicit spin-multiplicity UHF calculations against OpenMOPAC to $\le 0.001\text{ kcal/mol}$ on benchmark open shells ($\text{O}_2\ ^3\Sigma_g^-$, $\text{NH}\ ^3\Sigma^-$, $\text{CH}_2\ ^3B_1$, $\text{CH}_3\ ^2A_2''$).
6. **Runtime Python Module Version & Benchmark Provenance**:
   - Added runtime `mopac_py.__version__` attribute synchronized with the Cargo workspace package version.
   - Added unified provenance stamping (`benchmarks/provenance.py`) recording workspace version, installed wheel version, execution timestamp, and reference oracle.

---

## 2. Crate-by-Crate Release Status (`v0.1.5`)

| Crate | Version | Key Updates in `v0.1.5` |
| :--- | :--- | :--- |
| **`mopac_core`** | `0.1.5` | Added `compute_nsp2_correction`, `run_rhf_scf_shift_safe`, and `density_change` field in `ScfResult`. |
| **`mopac_gpu`** | `0.1.5` | Universal HPC memory architecture (`GpuMemoryArchitecture`, UMA, ReBAR, `GpuBatchDeviceMemoryManager`). |
| **`mopac` (`mopac_cli`)** | `0.1.5` | Added `nsp2` correction to PM6 non-covalent breakdown; added CLI `--disp` flag parsing and dispatch. |
| **`mopac_py` (`mopac-py`)** | `0.1.5` | Added `mopac_py.__version__`; wired `nsp2` correction and `run_rhf_scf_shift_safe`. |

---

## 3. Test & Parity Verification Summary

- **Unit & Integration Tests**: 100% passing across workspace (`cargo test --workspace`).
- **Differential Oracle Parity**:
  - `pm6_nitro_nitrate_nsp2_correction_parity`: PASSED ($|d| < 0.0005\text{ kcal/mol}$).
  - `pm6_control_without_sp2_nitrogen_unchanged`: PASSED.
  - `disp_none_disables_dispersion`: PASSED.
  - `disp_d3_selects_d3bj`: PASSED.
  - `pm7_default_matches_oracle`: PASSED.
  - `convergence_escalation` (4 tests): PASSED.
  - `multiplicity_handling` (4 UHF tests): PASSED.
- **Linting & Formatting**:
  - `cargo fmt --all -- --check`: clean (0 diffs).
  - `cargo clippy --workspace --all-targets -- -D warnings`: clean (0 warnings).
