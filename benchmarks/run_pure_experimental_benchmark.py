#!/usr/bin/env python3
"""
Pure Experimental Benchmark Suite for mopac-rs.

Evaluates semiempirical Hamiltonians directly against EMPIRICAL LABORATORY MEASUREMENTS
(Experimental Heats of Formation, Dipoles, and Ionization Potentials compiled from
NIST Chemistry WebBook, Cox & Pilcher, and Pedley thermochemical tables).

Quantifies:
1. mopac_rs MAE/RMSE vs Physical Experiment (Ground Truth).
2. OpenMOPAC MAE/RMSE vs Physical Experiment.
3. Win/Loss/Tie rate: Which engine is closer to physical experiment?
4. Statistical error distributions (P50, P90, P99, Max Error).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import mopac_py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")
STEWART_DIR = os.path.join(DATASETS_DIR, "stewart_pm7_accuracy")
HEATS_HTML = os.path.join(STEWART_DIR, "table_of_heats.html")
XYZ_DIR = os.path.join(STEWART_DIR, "data_molecules")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "data", "pure_experimental_benchmark_results.json")

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


def parse_xyz_file(path: str) -> Optional[Tuple[List[int], List[List[float]]]]:
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
    return atomic_numbers, coordinates


def load_experimental_heats(html_path: str) -> List[Dict[str, Any]]:
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    rows = re.findall(
        r'<tr>\s*<td>\s*([A-Za-z0-9]+)\s*</td><td>\s*<a href=\"([^\"]+)\">([^<]+)</a></td><td><p align=\"right\">\s*([0-9\.\-\+]+)</p></td>',
        content,
    )

    records = []
    for r in rows:
        formula, link, name, expt_str = r
        try:
            expt_val = float(expt_str)
        except ValueError:
            continue

        base = os.path.basename(link)
        base_unescaped = urllib.parse.unquote(base)
        xyz_name = base_unescaped.replace("_jmol.html", ".xyz")
        xyz_path = os.path.join(XYZ_DIR, xyz_name)

        if os.path.exists(xyz_path):
            # Skip open-shell charged ions and isolated atoms without charge metadata
            name_lower = name.lower()
            if any(term in name_lower for term in ["cation", "anion", "atom", "2p(", "3p(", "radical"]):
                continue

            parsed = parse_xyz_file(xyz_path)
            if parsed is not None:
                atoms, coords = parsed
                # Ensure even number of valence electrons for stable closed-shell RHF
                total_val = sum(atoms)
                if total_val % 2 != 0:
                    continue
                records.append({
                    "formula": formula,
                    "name": name,
                    "expt_hof_kcal": expt_val,
                    "xyz_file": xyz_name,
                    "atomic_numbers": atoms,
                    "coordinates": coords,
                    "natoms": len(atoms),
                })

    return records


def run_openmopac(atomic_numbers: List[int], coordinates: List[List[float]], method: str) -> Optional[float]:
    if not OPENMOPAC_BIN:
        return None
    inv_table = {v: k for k, v in PERIODIC_TABLE.items()}
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "job.mop")
        out_file = os.path.join(tmpdir, "job.out")

        lines = [f"{method} 1SCF XYZ DISP THREADS=1\n", "Experimental Benchmark\n", "\n"]
        for z, (x, y, z_coord) in zip(atomic_numbers, coordinates):
            sym = inv_table.get(z, "C")
            lines.append(f" {sym:<2}  {x:14.8f} 0  {y:14.8f} 0  {z_coord:14.8f} 0\n")

        with open(input_file, "w") as f:
            f.writelines(lines)

        try:
            res = subprocess.run([OPENMOPAC_BIN, input_file], cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if res.returncode != 0 and not os.path.exists(out_file):
                return None
        except Exception:
            return None

        if os.path.exists(out_file):
            with open(out_file, "r", errors="ignore") as f:
                for line in f:
                    if "FINAL HEAT OF FORMATION" in line:
                        parts = line.split("=")
                        if len(parts) > 1:
                            try:
                                return float(parts[1].split()[0])
                            except ValueError:
                                pass
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
    parser = argparse.ArgumentParser(description="Pure experimental validation benchmark.")
    parser.add_argument("--max-molecules", type=int, default=600, help="Number of experimental molecules to benchmark (default: 600)")
    parser.add_argument("--methods", nargs="+", default=["PM7", "PM6", "AM1"], help="Hamiltonians to benchmark")
    parser.add_argument("--organic-only", action="store_true", help="Restrict to organic molecules (H, C, N, O, F, P, S, Cl, Br, I)")
    args = parser.parse_args()

    print("=" * 80)
    print("PURE EXPERIMENTAL BENCHMARK: SEMIEMPIRICAL HAMILTONIANS VS PHYSICAL LABORATORY DATA")
    print("=" * 80)
    print(f"Data source: {HEATS_HTML}")
    print(f"Reference: NIST Standard Reference Database / Cox & Pilcher / Pedley")
    print(f"Target count: >= {args.max_molecules} experimental molecules")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"OpenMOPAC binary: {OPENMOPAC_BIN or 'Not found'}")
    print()

    records = load_experimental_heats(HEATS_HTML)
    print(f"Total matched neutral closed-shell experimental systems: {len(records)}")

    if args.organic_only:
        organic_z = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
        records = [r for r in records if set(r["atomic_numbers"]).issubset(organic_z)]
        print(f"Filtered to standard organic systems: {len(records)} molecules")

    records = records[:args.max_molecules]
    print(f"Proceeding with benchmark on {len(records)} experimental systems.")

    overall_results: Dict[str, Any] = {
        "metadata": {
            "dataset": "Stewart Canonical Experimental Heat of Formation Database",
            "primary_sources": ["NIST Chemistry WebBook (SRD 69)", "Cox & Pilcher (1970)", "Pedley et al. (1986)"],
            "sample_size": len(records),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "openmopac_oracle": OPENMOPAC_BIN,
        },
        "methods": {},
    }

    for method in args.methods:
        print("\n" + "-" * 80)
        print(f"EVALUATING HAMILTONIAN AGAINST EXPERIMENT: {method}")
        print("-" * 80)

        t_start = time.perf_counter()
        converged = 0
        total_time_ms = 0.0

        errors_mopacrs = []
        errors_openmopac = []
        inter_engine_diffs = []

        mopacrs_closer = 0
        openmopac_closer = 0
        tied = 0

        for idx, r in enumerate(records):
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  [Progress] Processed {idx + 1}/{len(records)} molecules...", flush=True)

            atoms = r["atomic_numbers"]
            coords = r["coordinates"]
            expt = r["expt_hof_kcal"]

            t0 = time.perf_counter()
            try:
                res = mopac_py.calculate(atoms, coords, method=method, max_iter=80)
            except Exception:
                continue

            total_time_ms += (time.perf_counter() - t0) * 1000.0

            if not res.converged:
                continue

            converged += 1
            calc_rs = res.heat_of_formation_kcal
            err_rs = calc_rs - expt
            errors_mopacrs.append(err_rs)

            # Evaluate OpenMOPAC
            calc_open = run_openmopac(atoms, coords, method=method)
            if calc_open is not None:
                err_open = calc_open - expt
                errors_openmopac.append(err_open)
                inter_engine_diffs.append(calc_rs - calc_open)

                abs_rs = abs(err_rs)
                abs_open = abs(err_open)

                if abs(abs_rs - abs_open) < 0.01:
                    tied += 1
                elif abs_rs < abs_open:
                    mopacrs_closer += 1
                else:
                    openmopac_closer += 1

        wall_time_s = time.perf_counter() - t_start
        throughput = converged / wall_time_s if wall_time_s > 0 else 0
        mean_time_ms = total_time_ms / converged if converged > 0 else 0

        print(f"Convergence: {converged}/{len(records)} ({converged / len(records) * 100:.1f}%)")
        print(f"Wall Time:   {wall_time_s:.2f} s | Throughput: {throughput:.1f} mol/s | Mean: {mean_time_ms:.2f} ms/mol")

        method_summary: Dict[str, Any] = {
            "converged": converged,
            "total": len(records),
            "convergence_pct": converged / len(records) * 100,
            "throughput_mol_s": throughput,
            "mean_calc_time_ms": mean_time_ms,
        }

        if errors_mopacrs:
            stats_rs = calculate_stats(errors_mopacrs)
            print(f"MOPAC_RS VS EXPERIMENTAL GROUND TRUTH ({len(errors_mopacrs)} systems):")
            print(f"  MAE:         {stats_rs['mae']:.3f} kcal/mol | RMSE: {stats_rs['rmse']:.3f} | Mean Signed Error: {stats_rs['mean_signed_error']:+.3f}")
            print(f"  Percentiles: P50={stats_rs['p50']:.3f} | P90={stats_rs['p90']:.3f} | P99={stats_rs['p99']:.3f} | Max Error: {stats_rs['max_abs_error']:.3f}")
            method_summary["mopac_rs_vs_experiment"] = stats_rs

        if errors_openmopac:
            stats_open = calculate_stats(errors_openmopac)
            print(f"OPENMOPAC VS EXPERIMENTAL GROUND TRUTH ({len(errors_openmopac)} systems):")
            print(f"  MAE:         {stats_open['mae']:.3f} kcal/mol | RMSE: {stats_open['rmse']:.3f} | Mean Signed Error: {stats_open['mean_signed_error']:+.3f}")
            print(f"  Percentiles: P50={stats_open['p50']:.3f} | P90={stats_open['p90']:.3f} | P99={stats_open['p99']:.3f} | Max Error: {stats_open['max_abs_error']:.3f}")
            method_summary["openmopac_vs_experiment"] = stats_open

            stats_diff = calculate_stats(inter_engine_diffs)
            print(f"INTER-ENGINE DISCREPANCY (MOPAC_RS vs OpenMOPAC):")
            print(f"  Δ(ΔHf) MAE:  {stats_diff['mae']:.3f} kcal/mol | P50={stats_diff['p50']:.3f} | P90={stats_diff['p90']:.3f} | Max Diff: {stats_diff['max_abs_error']:.3f}")
            method_summary["inter_engine_parity"] = stats_diff

            total_comp = mopacrs_closer + openmopac_closer + tied
            print(f"HEAD-TO-HEAD ACCURACY VS EXPERIMENTAL REALITY ({total_comp} molecules):")
            print(f"  mopac_rs closer to Experiment:  {mopacrs_closer} ({mopacrs_closer / total_comp * 100:.1f}%)")
            print(f"  OpenMOPAC closer to Experiment: {openmopac_closer} ({openmopac_closer / total_comp * 100:.1f}%)")
            print(f"  Tied (within 0.01 kcal/mol):    {tied} ({tied / total_comp * 100:.1f}%)")
            method_summary["head_to_head"] = {
                "mopac_rs_closer": mopacrs_closer,
                "openmopac_closer": openmopac_closer,
                "tied": tied,
                "total": total_comp,
            }

        overall_results["methods"][method] = method_summary

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED. Results successfully exported to:\n  {OUTPUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
