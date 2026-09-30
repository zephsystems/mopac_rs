#!/usr/bin/env python3
"""
NIST CCCBDB Comprehensive Gas-Phase Experimental Benchmark Suite.
Differential validation of mopac-rs CPU, mopac-rs GPU (Vulkan Compute),
and OpenMOPAC v23.2.5 (gfortran) against laboratory-measured physicochemical data.

Evaluated methods: PM7, PM6, AM1, PM3, RM1, MNDO.
Target dataset: NIST CCCBDB SRD 101 (Gas-Phase Heats of Formation and Stark Dipoles).
"""

import os
import sys
import time
import json
import argparse
import tempfile
import subprocess
import numpy as np
from pathlib import Path

# Ensure mopac_py is available
try:
    import mopac_py
except ImportError:
    # Try local venv
    venv_site = Path(__file__).resolve().parent.parent / ".venv" / "lib" / "python3.12" / "site-packages"
    if venv_site.exists():
        sys.path.insert(0, str(venv_site))
    import mopac_py

OPENMOPAC_BIN = os.environ.get("OPENMOPAC_BIN", os.path.expanduser("~/.local/bin/mopac"))
DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "nist_cccbdb" / "nist_cccbdb_curated.json"
RESULTS_PATH = Path(__file__).resolve().parent / "data" / "cccbdb_gpu_benchmark_results.json"

PERIODIC_TABLE = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe',
    27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se',
    35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
    43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 72: 'Hf',
    73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi'
}

def run_openmopac(method: str, atomic_numbers: list, coordinates: list, tmp_dir: str):
    """Execute OpenMOPAC v23.2.5 via input file and parse outputs."""
    if not os.path.exists(OPENMOPAC_BIN):
        return None

    mop_path = os.path.join(tmp_dir, "mol.mop")
    out_path = os.path.join(tmp_dir, "mol.out")

    lines = [f"{method} 1SCF DISP", "NIST Benchmark Mol", ""]
    for z, c in zip(atomic_numbers, coordinates):
        sym = PERIODIC_TABLE.get(z, f"X{z}")
        lines.append(f"{sym:2s} {c[0]:14.8f} 0 {c[1]:14.8f} 0 {c[2]:14.8f} 0")
    lines.append("")

    with open(mop_path, "w") as f:
        f.write("\n".join(lines))

    t0 = time.perf_counter()
    res = subprocess.run([OPENMOPAC_BIN, mop_path], cwd=tmp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t_om = (time.perf_counter() - t0) * 1000.0

    if res.returncode != 0 or not os.path.exists(out_path):
        return None

    hf = None
    etot = None
    enuc = None
    dipole = None

    try:
        with open(out_path, "r", errors="ignore") as f:
            for line in f:
                if "FINAL HEAT OF FORMATION =" in line:
                    parts = line.split("=")
                    if len(parts) >= 2:
                        hf = float(parts[1].split("KCAL")[0].strip())
                elif "TOTAL ENERGY            =" in line and "EV" in line:
                    parts = line.split("=")
                    if len(parts) >= 2:
                        etot = float(parts[1].split("EV")[0].strip())
                elif "CORE-CORE REPULSION     =" in line and "EV" in line:
                    parts = line.split("=")
                    if len(parts) >= 2:
                        enuc = float(parts[1].split("EV")[0].strip())
                elif "DIPOLE" in line and "SUM" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            dipole = float(parts[4])
                        except ValueError:
                            pass
                elif "TOTAL" in line and "POINT-CHG" not in line and "HYBRID" not in line:
                    if "DIPOLE" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                dipole = float(parts[-1])
                            except ValueError:
                                pass
    except Exception:
        return None

    if hf is None or etot is None:
        return None

    return {
        "hf_kcal": hf,
        "total_energy_ev": etot,
        "enuc_ev": enuc,
        "dipole_debye": dipole,
        "time_ms": t_om
    }

def main():
    parser = argparse.ArgumentParser(description="Run CCCBDB GPU Benchmark")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of molecules (0 = all)")
    parser.add_argument("--methods", type=str, default="PM7,PM6,AM1,PM3,RM1,MNDO", help="Comma-separated methods")
    args = parser.parse_args()

    methods = [m.strip().upper() for m in args.methods.split(",")]

    print("=" * 80)
    print("NIST CCCBDB EXPERIMENTAL GAS-PHASE BENCHMARK SUITE")
    print(f"Target Methods: {', '.join(methods)}")
    print(f"OpenMOPAC Binary: {OPENMOPAC_BIN}")
    print(f"Dataset Path: {DATASET_PATH}")
    print("=" * 80)

    if not DATASET_PATH.exists():
        print(f"[FATAL] Dataset file not found at {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    all_molecules = data.get("molecules", [])
    # Filter for molecules with valid, finite gas-phase experimental measurements
    molecules = [
        m for m in all_molecules
        if m.get("expt_dhf_298k_kcal_mol") is not None and np.isfinite(m.get("expt_dhf_298k_kcal_mol"))
    ]
    if args.limit > 0:
        molecules = molecules[:args.limit]

    print(f"Loaded {len(molecules)} gas-phase molecules with valid experimental Delta H_f(298 K).")

    # Warmup GPU
    try:
        mopac_py.calculate([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], "AM1", use_gpu=True)
        print("[GPU] Vulkan compute accelerator initialized successfully.")
    except Exception as e:
        print(f"[WARN] GPU initialization error: {e}")

    results_by_method = {}
    tmp_dir = tempfile.mkdtemp(prefix="cccbdb_bench_")

    for method in methods:
        print(f"\nEvaluating Method: {method} ...")
        t_start_method = time.time()
        method_records = []

        compat_count = 0
        cpu_time_total = 0.0
        gpu_time_total = 0.0
        om_time_total = 0.0

        for idx, mol in enumerate(molecules):
            z = mol["atomic_numbers"]
            coords = mol["coordinates"]
            expt_hf = mol["expt_dhf_298k_kcal_mol"]
            expt_dipole = mol.get("expt_dipole_debye")
            mol_id = mol.get("id", f"mol-{idx}")
            species = mol.get("species", "")

            # 1. mopac-rs CPU
            t0 = time.perf_counter()
            try:
                res_cpu = mopac_py.calculate(z, coords, method, use_gpu=False)
                t_cpu = (time.perf_counter() - t0) * 1000.0
            except Exception:
                continue

            if not res_cpu.converged:
                continue

            # 2. mopac-rs GPU
            t0 = time.perf_counter()
            try:
                res_gpu = mopac_py.calculate(z, coords, method, use_gpu=True)
                t_gpu = (time.perf_counter() - t0) * 1000.0
            except Exception:
                continue

            # 3. OpenMOPAC v23.2.5
            res_om = run_openmopac(method, z, coords, tmp_dir)
            if res_om is None:
                continue

            compat_count += 1
            cpu_time_total += t_cpu
            gpu_time_total += t_gpu
            om_time_total += res_om["time_ms"]

            # Compute deltas
            d_gpu_cpu_hf = abs(res_gpu.heat_of_formation_kcal - res_cpu.heat_of_formation_kcal)
            d_gpu_cpu_etot = abs(res_gpu.total_energy_ev - res_cpu.total_energy_ev)
            d_om_cpu_hf = abs(res_cpu.heat_of_formation_kcal - res_om["hf_kcal"])
            d_om_cpu_etot = abs(res_cpu.total_energy_ev - res_om["total_energy_ev"])

            err_exp_hf_cpu = abs(res_cpu.heat_of_formation_kcal - expt_hf)
            err_exp_hf_om = abs(res_om["hf_kcal"] - expt_hf)

            err_exp_dipole_cpu = None
            err_exp_dipole_om = None
            cpu_dipole = res_cpu.dipole_debye[3]
            om_dipole = res_om.get("dipole_debye")
            if expt_dipole is not None:
                err_exp_dipole_cpu = abs(cpu_dipole - expt_dipole)
                if om_dipole is not None:
                    err_exp_dipole_om = abs(om_dipole - expt_dipole)

            record = {
                "id": mol_id,
                "species": species,
                "natoms": len(z),
                "atomic_numbers": z,
                "expt_dhf_kcal": expt_hf,
                "expt_dipole_debye": expt_dipole,
                "mopac_rs_cpu": {
                    "hf_kcal": res_cpu.heat_of_formation_kcal,
                    "etot_ev": res_cpu.total_energy_ev,
                    "enuc_ev": res_cpu.nuclear_repulsion_ev,
                    "dipole_debye": cpu_dipole,
                    "time_ms": t_cpu
                },
                "mopac_rs_gpu": {
                    "hf_kcal": res_gpu.heat_of_formation_kcal,
                    "etot_ev": res_gpu.total_energy_ev,
                    "enuc_ev": res_gpu.nuclear_repulsion_ev,
                    "dipole_debye": res_gpu.dipole_debye[3],
                    "time_ms": t_gpu
                },
                "openmopac": res_om,
                "d_gpu_cpu_hf": d_gpu_cpu_hf,
                "d_gpu_cpu_etot": d_gpu_cpu_etot,
                "d_om_cpu_hf": d_om_cpu_hf,
                "d_om_cpu_etot": d_om_cpu_etot,
                "err_exp_hf_cpu": err_exp_hf_cpu,
                "err_exp_hf_om": err_exp_hf_om,
                "err_exp_dipole_cpu": err_exp_dipole_cpu,
                "err_exp_dipole_om": err_exp_dipole_om,
            }
            method_records.append(record)

            if compat_count % 50 == 0:
                print(f"  Processed {compat_count} valid molecules ...", flush=True)

        # Statistical analysis
        if not method_records:
            print(f"  [WARN] No completed evaluations for method {method}.", flush=True)
            continue

        n_eval = len(method_records)
        arr_d_gpu_hf = np.array([r["d_gpu_cpu_hf"] for r in method_records])
        arr_d_gpu_etot = np.array([r["d_gpu_cpu_etot"] for r in method_records])
        arr_d_om_hf = np.array([r["d_om_cpu_hf"] for r in method_records])
        arr_err_hf_cpu = np.array([r["err_exp_hf_cpu"] for r in method_records])
        arr_err_hf_om = np.array([r["err_exp_hf_om"] for r in method_records])

        # Ensure all arrays are strictly finite
        valid_om_mask = np.isfinite(arr_d_om_hf)
        clean_d_om_hf = arr_d_om_hf[valid_om_mask] if np.any(valid_om_mask) else arr_d_om_hf

        valid_hf_mask = np.isfinite(arr_err_hf_cpu) & np.isfinite(arr_err_hf_om)
        clean_err_hf_cpu = arr_err_hf_cpu[valid_hf_mask]
        clean_err_hf_om = arr_err_hf_om[valid_hf_mask]

        # Dipole errors
        dipole_records = [
            r for r in method_records
            if r["err_exp_dipole_cpu"] is not None and np.isfinite(r["err_exp_dipole_cpu"])
        ]
        arr_err_dipole_cpu = np.array([r["err_exp_dipole_cpu"] for r in dipole_records])
        arr_err_dipole_om = np.array([
            r["err_exp_dipole_om"] for r in dipole_records
            if r["err_exp_dipole_om"] is not None and np.isfinite(r["err_exp_dipole_om"])
        ])

        stats = {
            "evaluated_molecules": n_eval,
            "dipole_evaluated_molecules": len(dipole_records),
            "gpu_vs_cpu": {
                "max_delta_hf_kcal": float(np.max(arr_d_gpu_hf)),
                "mean_delta_hf_kcal": float(np.mean(arr_d_gpu_hf)),
                "max_delta_etot_ev": float(np.max(arr_d_gpu_etot)),
                "bit_exact_parity_ratio": float(np.mean(arr_d_gpu_hf < 1e-12)),
                "total_cpu_time_ms": cpu_time_total,
                "total_gpu_time_ms": gpu_time_total,
                "mean_cpu_latency_ms": cpu_time_total / n_eval,
                "mean_gpu_latency_ms": gpu_time_total / n_eval,
                "speedup_factor": (cpu_time_total / gpu_time_total) if gpu_time_total > 0 else 1.0,
            },
            "openmopac_parity": {
                "mae_hf_kcal": float(np.mean(clean_d_om_hf)),
                "rmse_hf_kcal": float(np.sqrt(np.mean(clean_d_om_hf**2))),
                "p50_hf_kcal": float(np.median(clean_d_om_hf)),
                "p90_hf_kcal": float(np.percentile(clean_d_om_hf, 90)),
                "p99_hf_kcal": float(np.percentile(clean_d_om_hf, 99)),
                "max_delta_hf_kcal": float(np.max(clean_d_om_hf)),
            },
            "physical_accuracy_mopacrs": {
                "mae_dhf_kcal": float(np.mean(clean_err_hf_cpu)) if len(clean_err_hf_cpu) > 0 else 0.0,
                "rmse_dhf_kcal": float(np.sqrt(np.mean(clean_err_hf_cpu**2))) if len(clean_err_hf_cpu) > 0 else 0.0,
                "p50_dhf_kcal": float(np.median(clean_err_hf_cpu)) if len(clean_err_hf_cpu) > 0 else 0.0,
                "p90_dhf_kcal": float(np.percentile(clean_err_hf_cpu, 90)) if len(clean_err_hf_cpu) > 0 else 0.0,
                "max_err_dhf_kcal": float(np.max(clean_err_hf_cpu)) if len(clean_err_hf_cpu) > 0 else 0.0,
                "mae_dipole_debye": float(np.mean(arr_err_dipole_cpu)) if len(arr_err_dipole_cpu) > 0 else None,
                "rmse_dipole_debye": float(np.sqrt(np.mean(arr_err_dipole_cpu**2))) if len(arr_err_dipole_cpu) > 0 else None,
            },
            "physical_accuracy_openmopac": {
                "mae_dhf_kcal": float(np.mean(clean_err_hf_om)) if len(clean_err_hf_om) > 0 else 0.0,
                "rmse_dhf_kcal": float(np.sqrt(np.mean(clean_err_hf_om**2))) if len(clean_err_hf_om) > 0 else 0.0,
                "p50_dhf_kcal": float(np.median(clean_err_hf_om)) if len(clean_err_hf_om) > 0 else 0.0,
                "p90_dhf_kcal": float(np.percentile(clean_err_hf_om, 90)) if len(clean_err_hf_om) > 0 else 0.0,
                "max_err_dhf_kcal": float(np.max(clean_err_hf_om)) if len(clean_err_hf_om) > 0 else 0.0,
                "mae_dipole_debye": float(np.mean(arr_err_dipole_om)) if len(arr_err_dipole_om) > 0 else None,
                "rmse_dipole_debye": float(np.sqrt(np.mean(arr_err_dipole_om**2))) if len(arr_err_dipole_om) > 0 else None,
            }
        }

        print(f"  Completed {n_eval} molecules in {time.time() - t_start_method:.2f}s:")
        print(f"    GPU vs CPU Max Delta: {stats['gpu_vs_cpu']['max_delta_hf_kcal']:.2e} kcal/mol ({stats['gpu_vs_cpu']['bit_exact_parity_ratio']*100:.1f}% bit-exact)")
        print(f"    mopac-rs vs OpenMOPAC MAE: {stats['openmopac_parity']['mae_hf_kcal']:.4f} kcal/mol | P50: {stats['openmopac_parity']['p50_hf_kcal']:.4f} | Max: {stats['openmopac_parity']['max_delta_hf_kcal']:.4f}")
        print(f"    mopac-rs vs NIST Exp MAE: {stats['physical_accuracy_mopacrs']['mae_dhf_kcal']:.2f} kcal/mol | RMSE: {stats['physical_accuracy_mopacrs']['rmse_dhf_kcal']:.2f} kcal/mol")
        print(f"    OpenMOPAC vs NIST Exp MAE: {stats['physical_accuracy_openmopac']['mae_dhf_kcal']:.2f} kcal/mol | RMSE: {stats['physical_accuracy_openmopac']['rmse_dhf_kcal']:.2f} kcal/mol")

        results_by_method[method] = {
            "statistics": stats,
            "sample_records": method_records[:50],  # Save representative sample to avoid huge json
            "top_discrepancies": sorted(method_records, key=lambda x: x["d_om_cpu_hf"], reverse=True)[:20]
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_by_method, f, indent=2)

    print(f"\nBenchmark completed successfully! Full results exported to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
