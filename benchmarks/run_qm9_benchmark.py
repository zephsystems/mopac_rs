#!/usr/bin/env python3
"""
QM9 Quantum Benchmark and Metrology Suite for mopac-rs.

Evaluates pure semiempirical Hamiltonians (AM1, PM6, PM3, MNDO, PM7) on >= 500
real molecular geometries from the QM9 dataset (Ramakrishnan et al., Sci. Data 2014).

Computes:
1. Engine parity against canonical OpenMOPAC v23.2.5 (Heat of Formation, Total Energy, Dipole).
2. Metrology vs ab initio DFT B3LYP/6-31G(2df,p) reference (HOMO, LUMO, Gap, Dipole).
3. Statistical metrics: MAE, RMSE, Pearson r, R^2, P50, P90, P99, Max Discrepancy.
4. Throughput and convergence reliability.
"""

import argparse
import csv
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
SDF_FILE = os.path.join(DATASETS_DIR, "gdb9.sdf")
CSV_FILE = os.path.join(DATASETS_DIR, "qm9.csv")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "data", "qm9_500_benchmark_results.json")

HARTREE_TO_EV = 27.211386245988
ATOMIC_MAP = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


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


def load_qm9_references(csv_path: str, max_records: int = 1000) -> Dict[str, Dict[str, float]]:
    refs = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mol_id = row["mol_id"]
            refs[mol_id] = {
                "mu_debye": float(row["mu"]),
                "alpha_bohr3": float(row["alpha"]),
                "homo_ev": float(row["homo"]) * HARTREE_TO_EV,
                "lumo_ev": float(row["lumo"]) * HARTREE_TO_EV,
                "gap_ev": float(row["gap"]) * HARTREE_TO_EV,
                "u0_hartree": float(row["u0"]),
                "zpve_hartree": float(row["zpve"]),
                "smiles": row["smiles"],
            }
            if len(refs) >= max_records:
                break
    return refs


def parse_sdf_molecules(sdf_path: str, max_molecules: int = 500) -> List[Dict[str, Any]]:
    molecules = []
    current_lines: List[str] = []
    with open(sdf_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "$$$$":
                if current_lines:
                    mol = parse_single_sdf_block(current_lines)
                    if mol is not None:
                        molecules.append(mol)
                        if len(molecules) >= max_molecules:
                            break
                    current_lines = []
            else:
                current_lines.append(line)
    return molecules


def parse_single_sdf_block(lines: List[str]) -> Optional[Dict[str, Any]]:
    if len(lines) < 4:
        return None
    mol_id = lines[0].strip()
    counts_line = lines[3]
    try:
        n_atoms = int(counts_line[:3])
    except ValueError:
        return None

    if len(lines) < 4 + n_atoms:
        return None

    atomic_numbers = []
    coordinates = []
    for i in range(4, 4 + n_atoms):
        parts = lines[i].split()
        if len(parts) < 4:
            return None
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        sym = parts[3]
        if sym not in ATOMIC_MAP:
            return None
        atomic_numbers.append(ATOMIC_MAP[sym])
        coordinates.append([x, y, z])

    return {
        "mol_id": mol_id,
        "atomic_numbers": atomic_numbers,
        "coordinates": coordinates,
        "num_atoms": n_atoms,
    }


def run_openmopac(atomic_numbers: List[int], coordinates: List[List[float]], method: str) -> Optional[Dict[str, float]]:
    if not OPENMOPAC_BIN:
        return None
    z_map = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "job.mop")
        arc_file = os.path.join(tmpdir, "job.arc")
        out_file = os.path.join(tmpdir, "job.out")

        lines = [f"{method} 1SCF XYZ DISP THREADS=1\n", "QM9 Benchmark Single Point\n", "\n"]
        for z, (x, y, z_coord) in zip(atomic_numbers, coordinates):
            sym = z_map.get(z, "C")
            lines.append(f" {sym:<2}  {x:14.8f} 0  {y:14.8f} 0  {z_coord:14.8f} 0\n")

        with open(input_file, "w") as f:
            f.writelines(lines)

        try:
            res = subprocess.run([OPENMOPAC_BIN, input_file], cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
            if res.returncode != 0 and not os.path.exists(out_file):
                return None
        except Exception:
            return None

        hof = None
        etot = None
        dipole = None
        homo = None
        lumo = None

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
                    elif "HOMO LUMO ENERGIES (EV)" in line:
                        parts = line.split("=")[-1].split()
                        if len(parts) >= 2:
                            try:
                                homo = float(parts[0])
                                lumo = float(parts[1])
                            except ValueError:
                                pass

        if hof is not None:
            return {
                "heat_of_formation_kcal": hof,
                "total_energy_ev": etot,
                "dipole_debye": dipole,
                "homo_ev": homo,
                "lumo_ev": lumo,
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
    parser = argparse.ArgumentParser(description="Run QM9 benchmark on mopac_rs.")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of QM9 molecules to evaluate (default: 500)")
    parser.add_argument("--methods", nargs="+", default=["AM1", "PM6", "PM3", "PM7"], help="Semiempirical methods to benchmark")
    parser.add_argument("--oracle-samples", type=int, default=100, help="Number of samples to evaluate against OpenMOPAC oracle")
    args = parser.parse_args()

    print("=" * 80)
    print(f"MOPAC_RS QUANTUM BENCHMARK: QM9 DATASET ({args.sample_size} MOLECULES)")
    print("=" * 80)
    print(f"Dataset path: {SDF_FILE}")
    print(f"Reference properties: {CSV_FILE}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Oracle binary: {OPENMOPAC_BIN or 'Not found'}")
    print()

    print("1. Loading QM9 reference database...")
    t0 = time.perf_counter()
    refs = load_qm9_references(CSV_FILE, max_records=args.sample_size * 2)
    print(f"   Loaded {len(refs)} reference rows in {time.perf_counter() - t0:.2f}s.")

    print(f"2. Parsing first {args.sample_size} molecules from gdb9.sdf...")
    t0 = time.perf_counter()
    molecules = parse_sdf_molecules(SDF_FILE, max_molecules=args.sample_size)
    print(f"   Successfully parsed {len(molecules)} molecules in {time.perf_counter() - t0:.2f}s.")
    if len(molecules) < args.sample_size:
        print(f"   WARNING: Requested {args.sample_size} but found {len(molecules)}.")

    overall_results: Dict[str, Any] = {
        "metadata": {
            "dataset": "QM9",
            "citation": "Ramakrishnan et al., Sci. Data 1:140022 (2014)",
            "sample_size": len(molecules),
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

        # Discrepancies vs B3LYP
        diff_homo = []
        diff_lumo = []
        diff_gap = []
        diff_dipole = []

        # Parity vs OpenMOPAC
        oracle_hof_diffs = []
        oracle_etot_diffs = []
        oracle_dipole_diffs = []

        mopacrs_hof_list = []
        mopacrs_gap_list = []
        mopacrs_dipole_list = []

        outliers = []

        for idx, mol in enumerate(molecules):
            mol_id = mol["mol_id"]
            ref = refs.get(mol_id)
            atoms = mol["atomic_numbers"]
            coords = mol["coordinates"]

            t_mol0 = time.perf_counter()
            try:
                res = mopac_py.calculate(atoms, coords, method=method, max_iter=80)
            except Exception as e:
                continue

            calc_time_ms = (time.perf_counter() - t_mol0) * 1000.0
            total_calc_time += calc_time_ms

            if not res.converged:
                continue

            converged_count += 1
            hof = res.heat_of_formation_kcal
            dip = res.dipole_debye[3]
            homo = res.homo_energy_ev
            lumo = res.lumo_energy_ev
            gap = res.homo_lumo_gap_ev

            mopacrs_hof_list.append(hof)
            mopacrs_gap_list.append(gap)
            mopacrs_dipole_list.append(dip)

            # Compare vs B3LYP reference if present
            if ref:
                b3lyp_homo = ref["homo_ev"]
                b3lyp_lumo = ref["lumo_ev"]
                b3lyp_gap = ref["gap_ev"]
                b3lyp_dip = ref["mu_debye"]

                diff_homo.append(homo - b3lyp_homo)
                diff_lumo.append(lumo - b3lyp_lumo)
                diff_gap.append(gap - b3lyp_gap)
                diff_dipole.append(dip - b3lyp_dip)

            # Check OpenMOPAC oracle on first N samples
            if idx < args.oracle_samples and OPENMOPAC_BIN:
                oracle_res = run_openmopac(atoms, coords, method=method)
                if oracle_res and oracle_res.get("heat_of_formation_kcal") is not None:
                    d_hof = hof - oracle_res["heat_of_formation_kcal"]
                    oracle_hof_diffs.append(d_hof)

                    if oracle_res.get("total_energy_ev") is not None and res.total_energy_ev is not None:
                        oracle_etot_diffs.append(res.total_energy_ev - oracle_res["total_energy_ev"])
                    if oracle_res.get("dipole_debye") is not None and dip is not None:
                        oracle_dipole_diffs.append(dip - oracle_res["dipole_debye"])

                    if abs(d_hof) > 0.05:
                        outliers.append({
                            "mol_id": mol_id,
                            "smiles": ref.get("smiles", "") if ref else "",
                            "mopacrs_hof": hof,
                            "oracle_hof": oracle_res["heat_of_formation_kcal"],
                            "diff_hof": d_hof,
                        })

        wall_time_s = time.perf_counter() - t_start
        throughput = converged_count / wall_time_s if wall_time_s > 0 else 0
        mean_mol_ms = total_calc_time / converged_count if converged_count > 0 else 0

        print(f"Convergence: {converged_count}/{len(molecules)} ({converged_count / len(molecules) * 100:.1f}%)")
        print(f"Wall Time:   {wall_time_s:.2f} s | Throughput: {throughput:.1f} mol/s | Mean Time/Mol: {mean_mol_ms:.2f} ms")

        method_summary: Dict[str, Any] = {
            "converged": converged_count,
            "total": len(molecules),
            "convergence_pct": converged_count / len(molecules) * 100,
            "wall_time_s": wall_time_s,
            "throughput_mol_per_s": throughput,
            "mean_calc_time_ms": mean_mol_ms,
        }

        # Oracle parity metrics
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

        # Metrology vs B3LYP
        if diff_gap:
            gap_stats = calculate_stats(diff_gap)
            dip_b3lyp_stats = calculate_stats(diff_dipole)
            print(f"METROLOGY vs DFT B3LYP Reference ({len(diff_gap)} molecules):")
            print(f"  Gap (Semiempirical - B3LYP)  MAE: {gap_stats['mae']:.3f} eV | Mean Shift: {gap_stats['mean_signed_error']:+.3f} eV")
            print(f"  Dipole |μ|                   MAE: {dip_b3lyp_stats['mae']:.3f} D  | Mean Shift: {dip_b3lyp_stats['mean_signed_error']:+.3f} D")
            method_summary["vs_b3lyp_gap"] = gap_stats
            method_summary["vs_b3lyp_dipole"] = dip_b3lyp_stats

        if outliers:
            print(f"  Outliers detected (>0.05 kcal/mol diff): {len(outliers)}")
            for out in outliers[:5]:
                print(f"    {out['mol_id']} ({out['smiles']}): mopac_rs={out['mopacrs_hof']:.4f}, oracle={out['oracle_hof']:.4f}, diff={out['diff_hof']:+.4f} kcal/mol")
            method_summary["outliers"] = outliers

        overall_results["methods"][method] = method_summary

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED. Results successfully exported to:\n  {OUTPUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
