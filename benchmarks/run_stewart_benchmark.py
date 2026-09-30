#!/usr/bin/env python3
"""
Stewart Validation Benchmark Suite for mopac-rs.

Evaluates PM7 and PM6 on >= 500 real molecules from the canonical Stewart Accuracy Dataset
(Stewart, J. Mol. Model. 2013, 19, 1; openmopac/PM7_and_PM6-D3H4_accuracy).

Computes:
1. Parity against canonical OpenMOPAC v23.2.5.
2. Thermodynamic heats of formation, electronic energies, and dipole moments.
3. Statistical metrics: MAE, RMSE, P50, P90, P99, Max Discrepancy.
4. Failure modes and convergence analysis.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import mopac_py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")
STEWART_DIR = os.path.join(DATASETS_DIR, "stewart_pm7_accuracy", "data_molecules")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "data", "stewart_500_benchmark_results.json")

PERIODIC_TABLE = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
}


def resolve_openmopac_bin() -> Optional[str]:
    if "OPENMOPAC_BIN" in os.environ and os.path.exists(os.environ["OPENMOPAC_BIN"]):
        return os.environ["OPENMOPAC_BIN"]
    which = shutil.which("mopac")
    if which:
        return which
    home_bin = os.path.expanduser("~/.local/bin/mopac")
    if os.path.exists(home_bin):
        return home_bin
    return None


OPENMOPAC_BIN = resolve_openmopac_bin()


def parse_xyz_file(path: str) -> Optional[Tuple[str, List[int], List[List[float]]]]:
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 2:
        return None
    try:
        n_atoms = int(lines[0])
    except ValueError:
        return None

    atom_lines = lines[2:] if len(lines) > n_atoms + 1 else lines[1:]
    if len(atom_lines) < n_atoms:
        return None

    atomic_numbers = []
    coordinates = []
    for line in atom_lines[:n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0].capitalize()
        if sym not in PERIODIC_TABLE:
            return None
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            return None
        atomic_numbers.append(PERIODIC_TABLE[sym])
        coordinates.append([x, y, z])

    if len(atomic_numbers) != n_atoms:
        return None
    return name, atomic_numbers, coordinates


def run_openmopac_oracle(atomic_numbers: List[int], coordinates: List[List[float]], method: str) -> Optional[Dict[str, float]]:
    if not OPENMOPAC_BIN:
        return None
    inv_table = {v: k for k, v in PERIODIC_TABLE.items()}
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "job.mop")
        out_file = os.path.join(tmpdir, "job.out")

        lines = [f"{method} 1SCF XYZ DISP THREADS=1\n", "Stewart Accuracy Benchmark Single Point\n", "\n"]
        for z, (x, y, z_coord) in zip(atomic_numbers, coordinates):
            sym = inv_table.get(z, "C")
            lines.append(f" {sym:<2}  {x:14.8f} 0  {y:14.8f} 0  {z_coord:14.8f} 0\n")

        with open(input_file, "w") as f:
            f.writelines(lines)

        try:
            res = subprocess.run([OPENMOPAC_BIN, input_file], cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            if res.returncode != 0 and not os.path.exists(out_file):
                return None
        except Exception:
            return None

        hof = None
        etot = None
        dipole = None

        if os.path.exists(out_file):
            with open(out_file, "r", errors="ignore") as f:
                content = f.read()
                for line in content.splitlines():
                    if "FINAL HEAT OF FORMATION" in line:
                        parts = line.split("=")
                        if len(parts) > 1:
                            try:
                                hof = float(parts[1].split()[0])
                            except ValueError:
                                pass
                    elif "TOTAL ENERGY" in line and "EV" in line:
                        parts = line.split("=")
                        if len(parts) > 1:
                            try:
                                etot = float(parts[1].split()[0])
                            except ValueError:
                                pass
                    elif "DIPOLE" in line and "DEBYE" in line and "POINT-CHG." not in line:
                        parts = line.split()
                        for p in parts:
                            try:
                                val = float(p)
                                dipole = val
                            except ValueError:
                                pass

        if hof is not None:
            return {
                "heat_of_formation_kcal": hof,
                "total_energy_ev": etot,
                "dipole_debye": dipole,
            }
        return None


def calculate_stats(errors: List[float]) -> Dict[str, float]:
    arr = np.array(errors)
    abs_arr = np.abs(arr)
    return {
        "count": len(arr),
        "mean_signed_error": float(np.mean(arr)),
        "mae": float(np.mean(abs_arr)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "p50": float(np.percentile(abs_arr, 50)),
        "p90": float(np.percentile(abs_arr, 90)),
        "p99": float(np.percentile(abs_arr, 99)),
        "max_abs_error": float(np.max(abs_arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Stewart benchmark on mopac_rs.")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of molecules to evaluate (default: 500)")
    parser.add_argument("--methods", nargs="+", default=["PM7", "PM6"], help="Hamiltonians to benchmark")
    parser.add_argument("--oracle-samples", type=int, default=100, help="Number of oracle parity checks")
    args = parser.parse_args()

    print("=" * 80)
    print(f"MOPAC_RS QUANTUM BENCHMARK: STEWART VALIDATION DATASET ({args.sample_size} MOLECULES)")
    print("=" * 80)
    print(f"Dataset path: {STEWART_DIR}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Oracle binary: {OPENMOPAC_BIN or 'Not found'}")
    print()

    all_xyz = sorted(glob.glob(os.path.join(STEWART_DIR, "*.xyz")))
    print(f"Total .xyz files discovered: {len(all_xyz)}")

    parsed_molecules = []
    for f in all_xyz:
        mol = parse_xyz_file(f)
        if mol is not None:
            # Check if all atoms are supported in PM7 / PM6
            name, z, coords = mol
            if all(atom_z in PERIODIC_TABLE.values() for atom_z in z):
                parsed_molecules.append(mol)
                if len(parsed_molecules) >= args.sample_size:
                    break

    print(f"Successfully prepared {len(parsed_molecules)} valid molecular systems for benchmark.")

    overall_results: Dict[str, Any] = {
        "metadata": {
            "dataset": "Stewart Validation Set",
            "citation": "Stewart, J. Mol. Model. 19:1-32 (2013)",
            "sample_size": len(parsed_molecules),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "openmopac_oracle": OPENMOPAC_BIN,
        },
        "methods": {},
    }

    for method in args.methods:
        print("\n" + "-" * 80)
        print(f"BENCHMARKING HAMILTONIAN: {method}")
        print("-" * 80)

        t_start = time.perf_counter()
        converged_count = 0
        total_calc_time = 0.0

        oracle_hof_diffs = []
        oracle_etot_diffs = []
        oracle_dipole_diffs = []
        outliers = []

        for idx, (name, atoms, coords) in enumerate(parsed_molecules):
            t_mol0 = time.perf_counter()
            try:
                res = mopac_py.calculate(atoms, coords, method=method, max_iter=80)
            except Exception:
                continue

            calc_time_ms = (time.perf_counter() - t_mol0) * 1000.0
            total_calc_time += calc_time_ms

            if not res.converged:
                continue

            converged_count += 1
            hof = res.heat_of_formation_kcal
            dip = res.dipole_debye[3]

            if idx < args.oracle_samples and OPENMOPAC_BIN:
                oracle_res = run_openmopac_oracle(atoms, coords, method=method)
                if oracle_res and oracle_res.get("heat_of_formation_kcal") is not None:
                    d_hof = hof - oracle_res["heat_of_formation_kcal"]
                    oracle_hof_diffs.append(d_hof)

                    if oracle_res.get("total_energy_ev") is not None and res.total_energy_ev is not None:
                        oracle_etot_diffs.append(res.total_energy_ev - oracle_res["total_energy_ev"])
                    if oracle_res.get("dipole_debye") is not None and dip is not None:
                        oracle_dipole_diffs.append(dip - oracle_res["dipole_debye"])

                    if abs(d_hof) > 0.5:
                        outliers.append({
                            "name": name,
                            "natoms": len(atoms),
                            "mopacrs_hof": hof,
                            "oracle_hof": oracle_res["heat_of_formation_kcal"],
                            "diff_hof": d_hof,
                        })

        wall_time_s = time.perf_counter() - t_start
        throughput = converged_count / wall_time_s if wall_time_s > 0 else 0
        mean_mol_ms = total_calc_time / converged_count if converged_count > 0 else 0

        print(f"Convergence: {converged_count}/{len(parsed_molecules)} ({converged_count / len(parsed_molecules) * 100:.1f}%)")
        print(f"Wall Time:   {wall_time_s:.2f} s | Throughput: {throughput:.1f} mol/s | Mean Time/Mol: {mean_mol_ms:.2f} ms")

        method_summary: Dict[str, Any] = {
            "converged": converged_count,
            "total": len(parsed_molecules),
            "convergence_pct": converged_count / len(parsed_molecules) * 100,
            "wall_time_s": wall_time_s,
            "throughput_mol_per_s": throughput,
            "mean_calc_time_ms": mean_mol_ms,
        }

        if oracle_hof_diffs:
            hof_stats = calculate_stats(oracle_hof_diffs)
            print(f"PARITY vs OpenMOPAC ({len(oracle_hof_diffs)} molecules):")
            print(f"  Δ(ΔHf) MAE:  {hof_stats['mae']:.6f} kcal/mol | RMSE: {hof_stats['rmse']:.6f} | Max: {hof_stats['max_abs_error']:.6f}")
            print(f"  Percentiles: P50={hof_stats['p50']:.6f} | P90={hof_stats['p90']:.6f} | P99={hof_stats['p99']:.6f}")
            method_summary["oracle_parity_hof"] = hof_stats

            if oracle_etot_diffs:
                etot_stats = calculate_stats(oracle_etot_diffs)
                print(f"  Δ(Etot) MAE: {etot_stats['mae']:.6f} eV | Max: {etot_stats['max_abs_error']:.6f} eV")
                method_summary["oracle_parity_etot"] = etot_stats

            if oracle_dipole_diffs:
                dip_stats = calculate_stats(oracle_dipole_diffs)
                print(f"  Δ(Dipole) MAE: {dip_stats['mae']:.4f} D | Max: {dip_stats['max_abs_error']:.4f} D")
                method_summary["oracle_parity_dipole"] = dip_stats

        if outliers:
            print(f"  Outliers detected (>0.5 kcal/mol diff): {len(outliers)}")
            for out in outliers[:5]:
                print(f"    {out['name']} ({out['natoms']} atoms): mopac_rs={out['mopacrs_hof']:.4f}, oracle={out['oracle_hof']:.4f}, diff={out['diff_hof']:+.4f} kcal/mol")
            method_summary["outliers"] = outliers

        overall_results["methods"][method] = method_summary

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED. Results exported to:\n  {OUTPUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
