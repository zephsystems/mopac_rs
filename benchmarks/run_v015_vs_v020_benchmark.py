#!/usr/bin/env python3
"""
Side-by-side empirical A/B benchmark and correctness diff suite comparing
mopac_rs v0.1.5 (baseline git ref) against v0.2.0 (current working tree).

Measures:
  1. Isolated Rust kernel microbenchmarks (mopac_core + mopac_gpu) in --release mode:
     - Eigensolver (tred2+tql2 vs cyclic Jacobi)
     - NDDO diatomic pair precomputation & two-center Fock assembly (Rayon row-band vs serial)
     - Analytical nuclear gradients (Rayon + DiatomicGradScratch vs serial)
     - COSMO initialization, b_matrix memory footprint, and reaction-field Fock step
     - CPU Gaussian Cube density grid + ASCII voxel serialization
     - Vulkan GPU >64 active atom density grid error vs CPU reference
  2. End-to-end Python API workflows (mopac_py) in fresh isolated subprocesses
     measuring wall-clock latency AND Peak RSS memory (MiB):
     - Large SCF (C24H50 PM6, compute_gradients True/False)
     - Solvated COSMO SCF (C20H42 PM6, eps=78.4)
     - L-BFGS geometry optimization (distorted C6H14 PM6)
     - Harmonic vibrational frequencies / Cartesian Hessian (C6H6 PM6)
     - Mol* bundle export (C10H8 PM6)
     - Multi-property workflow: QuantumSession (1 SCF) vs 5 standalone SCF calls
     - Repeated GPU density cube generation (OnceLock cache vs cold VulkanContext)
  3. Behavioral & physical parity diffs:
     - Charged OH- anion Mulliken charge conservation & ESP net_charge forwarding
     - CLI `mopac --gpu` NDDO preservation on PM6 water
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command {cmd} failed with exit code {proc.returncode}:\n{proc.stdout}"
        )
    return proc.stdout


def prepare_microbench_crate(crate_dir: Path, src_repo: Path) -> None:
    crate_dir.mkdir(parents=True, exist_ok=True)
    src_dir = crate_dir / "src"
    src_dir.mkdir(exist_ok=True)

    mopac_core_path = (src_repo / "crates" / "mopac_core").resolve()
    mopac_gpu_path = (src_repo / "crates" / "mopac_gpu").resolve()
    microbench_src = (ROOT / "benchmarks" / "v015_vs_v020_microbench.rs").resolve()

    pkg_name = crate_dir.name.lstrip(".")
    shutil.copy2(microbench_src, src_dir / "main.rs")
    cargo_toml = (
        '[package]\n'
        f'name = "{pkg_name}"\n'
        'version = "0.1.0"\n'
        'edition = "2024"\n'
        '\n'
        '[workspace]\n'
        '\n'
        '[dependencies]\n'
        f'mopac_core = {{ path = "{mopac_core_path}" }}\n'
        f'mopac_gpu = {{ path = "{mopac_gpu_path}" }}\n'
    )
    (crate_dir / "Cargo.toml").write_text(cargo_toml, encoding="utf-8")


def build_version_artifacts(
    repo_root: Path, target_dir: Path, harness_dir: Path
) -> dict[str, Path]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir.resolve())

    print(f"[build] Building release mopac_py and mopac CLI for {repo_root.name} ...")
    run_cmd(
        ["cargo", "build", "--release", "-p", "mopac_py", "-p", "mopac"],
        cwd=repo_root,
        env=env,
    )

    so_src = target_dir / "release" / "libmopac_py.so"
    so_link = target_dir / "release" / "mopac_py.so"
    if so_src.exists():
        if so_link.exists() or so_link.is_symlink():
            so_link.unlink()
        so_link.symlink_to(so_src.name)

    prepare_microbench_crate(harness_dir, repo_root)
    pkg_name = harness_dir.name.lstrip(".")
    print(f"[build] Building release Rust microbenchmark for {repo_root.name} ...")
    run_cmd(["cargo", "build", "--release"], cwd=harness_dir, env=env)

    return {
        "python_path": target_dir / "release",
        "cli_bin": target_dir / "release" / "mopac",
        "microbench_bin": target_dir / "release" / pkg_name,
    }


def parse_microbench_output(output: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        if line.startswith("BENCH_JSON:"):
            rec = json.loads(line[len("BENCH_JSON:") :])
            results[rec["name"]] = rec
    return results


PY_WORKER_SCRIPT = r"""
import json
import math
import resource
import sys
import time

lib_dir = sys.argv[1]
mode = sys.argv[2]
sys.path.insert(0, lib_dir)
import mopac_py

def make_alkane(n_carbons):
    z = []
    coords = []
    for i in range(n_carbons):
        cx = i * 1.25
        sign = 1.0 if (i % 2 == 0) else -1.0
        cy = sign * 0.44
        z.append(6)
        coords.append([cx, cy, 0.0])
        z.append(1)
        coords.append([cx, cy + sign * 0.64, 0.89])
        z.append(1)
        coords.append([cx, cy + sign * 0.64, -0.89])
        if i == 0:
            z.append(1)
            coords.append([cx - 0.90, cy - sign * 0.63, 0.0])
        if i == n_carbons - 1:
            z.append(1)
            coords.append([cx + 0.90, cy - sign * 0.63, 0.0])
    return z, coords

def make_benzene():
    z = [6] * 6 + [1] * 6
    coords = []
    for i in range(6):
        ang = i * math.pi / 3.0
        coords.append([1.395 * math.cos(ang), 1.395 * math.sin(ang), 0.0])
    for i in range(6):
        ang = i * math.pi / 3.0
        coords.append([2.485 * math.cos(ang), 2.485 * math.sin(ang), 0.0])
    return z, coords

def make_naphthalene():
    z = [6] * 10 + [1] * 8
    coords = [
        [0.000,  0.715, 0.0],
        [0.000, -0.715, 0.0],
        [1.230,  1.400, 0.0],
        [2.430,  0.700, 0.0],
        [2.430, -0.700, 0.0],
        [1.230, -1.400, 0.0],
        [-1.230, 1.400, 0.0],
        [-2.430, 0.700, 0.0],
        [-2.430,-0.700, 0.0],
        [-1.230,-1.400, 0.0],
        [1.230,  2.490, 0.0],
        [3.370,  1.240, 0.0],
        [3.370, -1.240, 0.0],
        [1.230, -2.490, 0.0],
        [-1.230, 2.490, 0.0],
        [-3.370, 1.240, 0.0],
        [-3.370,-1.240, 0.0],
        [-1.230,-2.490, 0.0],
    ]
    return z, coords

def make_phenol():
    z = [6]*6 + [8] + [1]*6
    coords = [
        [ 1.395,  0.000, 0.0],
        [ 0.698,  1.208, 0.0],
        [-0.698,  1.208, 0.0],
        [-1.395,  0.000, 0.0],
        [-0.698, -1.208, 0.0],
        [ 0.698, -1.208, 0.0],
        [ 2.765,  0.000, 0.0],
        [ 3.120,  0.900, 0.0],
        [ 1.240,  2.148, 0.0],
        [-1.240,  2.148, 0.0],
        [-2.485,  0.000, 0.0],
        [-1.240, -2.148, 0.0],
        [ 1.240, -2.148, 0.0],
    ]
    return z, coords

def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

payload = {}

if mode in ("scf_c24h50_grad", "scf_c24h50_nograd"):
    z, c = make_alkane(24)
    kwargs = {"method": "PM6"}
    if mode == "scf_c24h50_nograd":
        kwargs["compute_gradients"] = False
    times = []
    res = None
    for i in range(4):
        t0 = time.perf_counter()
        res = mopac_py.calculate(z, c, **kwargs)
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt * 1000.0)
    payload = {
        "median_ms": sorted(times)[len(times)//2],
        "peak_rss_mb": peak_rss_mb(),
        "total_energy_ev": res.total_energy_ev,
        "hof_kcal": res.heat_of_formation_kcal,
        "homo_ev": res.homo_energy_ev,
        "lumo_ev": res.lumo_energy_ev,
        "dipole_total": res.dipole_debye[3],
        "gradients": res.gradients_ev_angstrom,
        "mulliken_charges": res.mulliken_charges,
    }
elif mode == "cosmo_c20h42":
    z, c = make_alkane(20)
    times = []
    res = None
    for i in range(3):
        t0 = time.perf_counter()
        res = mopac_py.calculate(z, c, method="PM6", cosmo_eps=78.4)
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt * 1000.0)
    payload = {
        "median_ms": sorted(times)[len(times)//2],
        "peak_rss_mb": peak_rss_mb(),
        "total_energy_ev": res.total_energy_ev,
        "hof_kcal": res.heat_of_formation_kcal,
        "dipole_total": res.dipole_debye[3],
        "gradients": res.gradients_ev_angstrom,
    }
elif mode == "opt_c6h14":
    z, c = make_alkane(6)
    # Slightly perturb bond lengths to force L-BFGS steps
    for idx in range(len(c)):
        c[idx][0] *= 1.04
        c[idx][1] *= 0.97
    times = []
    res = None
    for i in range(3):
        t0 = time.perf_counter()
        res = mopac_py.optimize(z, c, method="PM6", max_cycles=40)
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt * 1000.0)
    payload = {
        "median_ms": sorted(times)[len(times)//2],
        "cycles": res.cycles,
        "final_energy_ev": res.final_energy_ev,
        "final_grad_rms": res.final_grad_rms,
        "coordinates": res.coordinates,
        "peak_rss_mb": peak_rss_mb(),
    }
elif mode == "freq_benzene":
    z, c = make_benzene()
    times = []
    res = None
    for i in range(3):
        t0 = time.perf_counter()
        res = mopac_py.frequencies(z, c, method="PM6")
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt * 1000.0)
    payload = {
        "median_ms": sorted(times)[len(times)//2],
        "zpe_kcal": res.zpve_kcal_mol,
        "freqs": res.vibrational_frequencies_cm1,
        "peak_rss_mb": peak_rss_mb(),
    }
elif mode == "molstar_naphthalene":
    z, c = make_naphthalene()
    calc = mopac_py.MopacCalculator(method="PM6")
    times = []
    bundle = None
    for i in range(3):
        t0 = time.perf_counter()
        bundle = calc.export_molstar_bundle(z, c, resolution_angstrom=0.25)
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt * 1000.0)
    payload = {
        "median_ms": sorted(times)[len(times)//2],
        "keys": sorted(bundle.keys()),
        "peak_rss_mb": peak_rss_mb(),
    }
elif mode == "multi_prop_standalone":
    z, c = make_phenol()
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        r_calc = mopac_py.calculate(z, c, method="PM6")
        r_esp = mopac_py.esp_charges(z, c, method="PM6")
        r_bo = mopac_py.bond_orders(z, c, method="PM6")
        r_cubes = mopac_py.generate_cubes_bundle(z, c, method="PM6", orbitals=["HOMO"], resolution_angstrom=0.35)
        r_meci = mopac_py.meci(z, c, method="PM6", active_orbitals=2)
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt * 1000.0)
    payload = {
        "median_ms": sorted(times)[len(times)//2],
        "energy_ev": r_calc.total_energy_ev,
        "meci_ev": r_meci.total_energy_ev,
        "esp_charges": r_esp.charges,
        "bond_orders": r_bo.bond_orders,
        "peak_rss_mb": peak_rss_mb(),
    }
elif mode == "multi_prop_session":
    z, c = make_phenol()
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        sess = mopac_py.QuantumSession(z, c, method="PM6")
        r_calc = sess.calculate(compute_gradients=True)
        r_esp = sess.esp_charges()
        r_bo = sess.bond_orders()
        r_cubes = sess.generate_cubes_bundle(orbitals=["HOMO"], resolution_angstrom=0.35)
        r_meci = sess.meci(active_orbitals=2)
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt * 1000.0)
    payload = {
        "median_ms": sorted(times)[len(times)//2],
        "energy_ev": r_calc.total_energy_ev,
        "meci_ev": r_meci.total_energy_ev,
        "esp_charges": r_esp.charges,
        "bond_orders": r_bo.bond_orders,
        "peak_rss_mb": peak_rss_mb(),
    }
elif mode == "gpu_cube_5x":
    if not mopac_py.is_gpu_available():
        payload = {"skipped": True}
    else:
        z, c = make_benzene()
        # Cold + 5 calls total
        t0 = time.perf_counter()
        for _ in range(5):
            s = mopac_py.generate_density_cube(z, c, method="PM6", padding_angstrom=3.0, resolution_angstrom=0.25, use_gpu=True)
        total_ms = (time.perf_counter() - t0) * 1000.0
        payload = {
            "skipped": False,
            "total_ms": total_ms,
            "per_call_ms": total_ms / 5.0,
            "cube_bytes": len(s),
            "peak_rss_mb": peak_rss_mb(),
        }
elif mode == "diff_oh_anion":
    z = [8, 1]
    c = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]
    calc_res = mopac_py.calculate(z, c, method="PM6", charge=-1, uhf=False)
    esp_ok = True
    esp_charges = []
    esp_sum = 0.0
    esp_rms_ev = 0.0
    esp_err = ""
    try:
        esp_res = mopac_py.esp_charges(z, c, method="PM6", net_charge=-1.0)
        esp_charges = esp_res.charges
        esp_sum = sum(esp_charges)
        esp_rms_ev = esp_res.rms_error_ev
    except BaseException as exc:
        esp_ok = False
        esp_err = str(exc).splitlines()[0]
    payload = {
        "mulliken_charges": calc_res.mulliken_charges,
        "mulliken_sum": sum(calc_res.mulliken_charges),
        "esp_ok": esp_ok,
        "esp_err": esp_err,
        "esp_charges": esp_charges,
        "esp_sum": esp_sum,
        "esp_rms_ev": esp_rms_ev,
    }
else:
    raise ValueError(f"Unknown mode: {mode}")

print("PY_JSON:" + json.dumps(payload))
"""


def run_py_worker(python_path: Path, mode: str) -> dict[str, Any]:
    out = run_cmd([sys.executable, "-c", PY_WORKER_SCRIPT, str(python_path), mode])
    for line in out.splitlines():
        if line.startswith("PY_JSON:"):
            return json.loads(line[len("PY_JSON:") :])
    raise RuntimeError(f"Missing PY_JSON in output for mode={mode}:\n{out}")


def measure_cli_gpu_hof(cli_bin: Path, scratch_dir: Path) -> dict[str, float]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    mop_path = scratch_dir / "water_pm6.mop"
    mop_path.write_text(
        "PM6 1SCF\n"
        "Water PM6 test\n"
        "GPU vs CPU NDDO parity check\n"
        "O   0.000000  1   0.000000  1   0.119000  1\n"
        "H   0.000000  1   0.763000  1  -0.477000  1\n"
        "H   0.000000  1  -0.763000  1  -0.477000  1\n",
        encoding="utf-8",
    )

    def extract_hof(cmd: list[str]) -> float:
        out = run_cmd(cmd, cwd=scratch_dir)
        m = re.search(r"Final Heat of Formation\s*:\s*([+-]?\d+\.\d+)\s*kcal/mol", out)
        if not m:
            out_file = scratch_dir / "water_pm6.out"
            if out_file.exists():
                m = re.search(
                    r"FINAL HEAT OF FORMATION\s*=\s*([+-]?\d+\.\d+)",
                    out_file.read_text(encoding="utf-8"),
                    re.IGNORECASE,
                )
        if not m:
            raise RuntimeError(f"Could not parse HOF from CLI output:\n{out}")
        return float(m.group(1))

    hof_cpu = extract_hof([str(cli_bin), str(mop_path)])
    hof_gpu = extract_hof([str(cli_bin), str(mop_path), "--gpu"])
    return {"hof_cpu_kcal": hof_cpu, "hof_gpu_kcal": hof_gpu, "abs_diff_kcal": abs(hof_gpu - hof_cpu)}


def find_openmopac_binary() -> Path | None:
    env_bin = os.environ.get("OPENMOPAC_BIN")
    if env_bin and env_bin.lower() not in ("skip", "none", "disabled"):
        p = Path(env_bin)
        if p.exists():
            return p
    local_bin = Path.home() / ".local" / "bin" / "mopac"
    if local_bin.exists():
        return local_bin
    which = shutil.which("mopac")
    return Path(which) if which else None


def py_make_alkane(n_carbons: int) -> tuple[list[int], list[list[float]]]:
    z: list[int] = []
    coords: list[list[float]] = []
    for i in range(n_carbons):
        cx = i * 1.25
        sign = 1.0 if (i % 2 == 0) else -1.0
        cy = sign * 0.44
        z.append(6)
        coords.append([cx, cy, 0.0])
        z.append(1)
        coords.append([cx, cy + sign * 0.64, 0.89])
        z.append(1)
        coords.append([cx, cy + sign * 0.64, -0.89])
        if i == 0:
            z.append(1)
            coords.append([cx - 0.90, cy - sign * 0.63, 0.0])
        if i == n_carbons - 1:
            z.append(1)
            coords.append([cx + 0.90, cy - sign * 0.63, 0.0])
    return z, coords


def run_openmopac_oracle(
    mopac_bin: Path,
    scratch_dir: Path,
    job_name: str,
    keywords: str,
    atomic_numbers: list[int],
    coords: list[list[float]],
) -> dict[str, Any]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    sym_map = {1: "H", 6: "C", 7: "N", 8: "O"}
    mop_file = scratch_dir / f"{job_name}.mop"
    out_file = scratch_dir / f"{job_name}.out"
    lines = [f"{keywords} XYZ", f"OpenMOPAC parity check: {job_name}", ""]
    for z, xyz in zip(atomic_numbers, coords):
        lines.append(f"{sym_map[z]:<2} {xyz[0]:14.8f} 0 {xyz[1]:14.8f} 0 {xyz[2]:14.8f} 0")
    mop_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_cmd([str(mopac_bin), str(mop_file)], cwd=scratch_dir)
    text = out_file.read_text(encoding="utf-8")
    hof = None
    tot_e = None
    zpve = None
    for ln in text.splitlines():
        if "FINAL HEAT OF FORMATION =" in ln:
            hof = float(ln.split("=")[1].split()[0])
        if "TOTAL ENERGY            =" in ln and ln.strip().endswith("EV"):
            tot_e = float(ln.split("=")[1].split()[0])
        if "ZERO POINT ENERGY" in ln:
            m = re.search(r"([+-]?\d+\.\d+)\s+KCAL/MOL", ln)
            if m:
                zpve = float(m.group(1))
    return {"hof_kcal": hof, "total_energy_ev": tot_e, "zpe_kcal": zpve}


def fmt_ratio(old_v: float, new_v: float) -> str:
    if new_v <= 0.0:
        return "inf"
    return f"{old_v / new_v:.2f}x"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B benchmark comparing baseline git ref (v0.1.5) vs current tree (v0.2.0)"
    )
    parser.add_argument("--baseline-ref", default="v0.1.5", help="Baseline git tag or commit")
    parser.add_argument("--json-out", default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    worktree_dir = ROOT / "target" / ".bench_v015_src"
    v015_target_dir = ROOT / "target" / ".bench_v015_target"
    v020_target_dir = ROOT / "target"
    v015_harness_dir = ROOT / "target" / ".bench_harness_v015"
    v020_harness_dir = ROOT / "target" / ".bench_harness_v020"
    cli_scratch_v015 = ROOT / "target" / ".bench_cli_v015"
    cli_scratch_v020 = ROOT / "target" / ".bench_cli_v020"

    if worktree_dir.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_dir)],
            cwd=ROOT,
            check=False,
        )
        shutil.rmtree(worktree_dir, ignore_errors=True)

    report: dict[str, Any] = {"baseline_ref": args.baseline_ref}

    try:
        print(f"[setup] Creating detached git worktree for {args.baseline_ref} ...")
        run_cmd(["git", "worktree", "add", "--detach", str(worktree_dir), args.baseline_ref], cwd=ROOT)

        art_015 = build_version_artifacts(worktree_dir, v015_target_dir, v015_harness_dir)
        art_020 = build_version_artifacts(ROOT, v020_target_dir, v020_harness_dir)

        print("[bench] Running Rust kernel microbenchmarks for v0.1.5 ...")
        micro_015 = parse_microbench_output(run_cmd([str(art_015["microbench_bin"])], cwd=ROOT))
        print("[bench] Running Rust kernel microbenchmarks for v0.2.0 ...")
        micro_020 = parse_microbench_output(run_cmd([str(art_020["microbench_bin"])], cwd=ROOT))
        report["rust_microbench"] = {"v0.1.5": micro_015, "v0.2.0": micro_020}

        print("[bench] Running isolated Python API workflows (v0.1.5 vs v0.2.0) ...")
        py_bench = {
            "scf_c24h50_grad": (
                run_py_worker(art_015["python_path"], "scf_c24h50_grad"),
                run_py_worker(art_020["python_path"], "scf_c24h50_grad"),
            ),
            "scf_c24h50_nograd_vs_015": (
                run_py_worker(art_015["python_path"], "scf_c24h50_grad"),
                run_py_worker(art_020["python_path"], "scf_c24h50_nograd"),
            ),
            "cosmo_c20h42": (
                run_py_worker(art_015["python_path"], "cosmo_c20h42"),
                run_py_worker(art_020["python_path"], "cosmo_c20h42"),
            ),
            "opt_c6h14": (
                run_py_worker(art_015["python_path"], "opt_c6h14"),
                run_py_worker(art_020["python_path"], "opt_c6h14"),
            ),
            "freq_benzene": (
                run_py_worker(art_015["python_path"], "freq_benzene"),
                run_py_worker(art_020["python_path"], "freq_benzene"),
            ),
            "molstar_naphthalene": (
                run_py_worker(art_015["python_path"], "molstar_naphthalene"),
                run_py_worker(art_020["python_path"], "molstar_naphthalene"),
            ),
            "multi_prop_session": (
                run_py_worker(art_015["python_path"], "multi_prop_standalone"),
                run_py_worker(art_020["python_path"], "multi_prop_session"),
            ),
            "gpu_cube_5x": (
                run_py_worker(art_015["python_path"], "gpu_cube_5x"),
                run_py_worker(art_020["python_path"], "gpu_cube_5x"),
            ),
        }
        report["python_workflows"] = {
            k: {"v0.1.5": v[0], "v0.2.0": v[1]} for k, v in py_bench.items()
        }

        print("[bench] Running correctness & behavioral diffs (v0.1.5 vs v0.2.0) ...")
        diff_oh_015 = run_py_worker(art_015["python_path"], "diff_oh_anion")
        diff_oh_020 = run_py_worker(art_020["python_path"], "diff_oh_anion")
        cli_015 = measure_cli_gpu_hof(art_015["cli_bin"], cli_scratch_v015)
        cli_020 = measure_cli_gpu_hof(art_020["cli_bin"], cli_scratch_v020)
        report["correctness_diffs"] = {
            "oh_anion": {"v0.1.5": diff_oh_015, "v0.2.0": diff_oh_020},
            "cli_gpu_nddo": {"v0.1.5": cli_015, "v0.2.0": cli_020},
        }

        print("\n" + "=" * 90)
        print("MOPAC_RS v0.1.5 vs v0.2.0 EMPIRICAL BENCHMARK REPORT (--release)")
        print("=" * 90)

        print("\n[1] Isolated Rust Core & GPU Microbenchmarks")
        print(f"{'Kernel / Workload':<36} | {'v0.1.5':>14} | {'v0.2.0':>14} | {'Speedup / Delta':>16}")
        print("-" * 90)
        for name, label in [
            ("eigensolver_n64", "Eigensolver 64x64"),
            ("eigensolver_n150", "Eigensolver 150x150"),
            ("eigensolver_n250", "Eigensolver 250x250"),
            ("nddo_precompute_c14h30", "NDDO Precompute (C14H30, 946p)"),
            ("nddo_fock_c14h30", "NDDO Fock Build (C14H30, 946p)"),
            ("nuclear_gradients_c10h22", "Nuclear Gradients (C10H22, 496p)"),
            ("cube_cpu_benzene_fine", "CPU Cube Gen+ASCII (Benzene 0.14A)"),
        ]:
            t0 = micro_015[name]["value"] / 1000.0
            t1 = micro_020[name]["value"] / 1000.0
            print(f"{label:<36} | {t0:11.3f} ms | {t1:11.3f} ms | {fmt_ratio(t0, t1):>16}")

        c0, c1 = micro_015["cosmo_c14h30"], micro_020["cosmo_c14h30"]
        print(
            f"{'COSMO Init (C14H30)':<36} | {c0['init_us']/1000:11.3f} ms | {c1['init_us']/1000:11.3f} ms | {fmt_ratio(c0['init_us'], c1['init_us']):>16}"
        )
        print(
            f"{'COSMO Fock Step (C14H30)':<36} | {c0['fock_us']/1000:11.3f} ms | {c1['fock_us']/1000:11.3f} ms | {fmt_ratio(c0['fock_us'], c1['fock_us']):>16}"
        )
        print(
            f"{'COSMO b_matrix Size (C14H30)':<36} | {c0['b_mb']:10.2f} MiB | {c1['b_mb']:10.2f} MiB | {fmt_ratio(c0['b_mb'], c1['b_mb']) + ' smaller':>16}"
        )
        print(
            f"{'COSMO b_matrix Rows (C14H30)':<36} | {c0['b_rows']:>14} | {c1['b_rows']:>14} | {fmt_ratio(c0['b_rows'], c1['b_rows']):>16}"
        )

        print("\n[2] Python API End-to-End Workflows (Wall-Clock & Peak RSS)")
        print(f"{'Workflow':<36} | {'v0.1.5':>14} | {'v0.2.0':>14} | {'Speedup / RSS':>18}")
        print("-" * 92)
        for key, label in [
            ("scf_c24h50_grad", "PM6 Single-Point + Grad (C24H50)"),
            ("scf_c24h50_nograd_vs_015", "PM6 Energy Only vs 0.1.5 (C24H50)"),
            ("cosmo_c20h42", "COSMO eps=78.4 SCF+Grad (C20H42)"),
            ("opt_c6h14", "L-BFGS Geometry Opt (C6H14)"),
            ("freq_benzene", "Harmonic Frequencies (Benzene)"),
            ("molstar_naphthalene", "Mol* Export Bundle (Naphthalene)"),
            ("multi_prop_session", "5-Property Pipeline (Session)"),
        ]:
            r0, r1 = py_bench[key]
            m0, m1 = r0["median_ms"], r1["median_ms"]
            extra = fmt_ratio(m0, m1)
            if key == "cosmo_c20h42":
                extra = f"{extra} | {r0['peak_rss_mb']:.0f}->{r1['peak_rss_mb']:.0f}M"
            print(f"{label:<36} | {m0:11.2f} ms | {m1:11.2f} ms | {extra:>18}")

        g0, g1 = py_bench["gpu_cube_5x"]
        if not g0.get("skipped") and not g1.get("skipped"):
            print(
                f"{'GPU Density Cube x5 (Benzene)':<36} | {g0['total_ms']:11.2f} ms | {g1['total_ms']:11.2f} ms | {fmt_ratio(g0['total_ms'], g1['total_ms']):>18}"
            )

        print("\n[3] Physical & Numerical Correctness Deltas")
        print("-" * 92)
        print(
            f"  OH- Mulliken charge sum (target -1.000): v0.1.5 = {diff_oh_015['mulliken_sum']:+.6f} e | "
            f"v0.2.0 = {diff_oh_020['mulliken_sum']:+.6f} e"
        )
        esp_015_str = (
            f"rms={diff_oh_015['esp_rms_ev']:.6f} eV"
            if diff_oh_015["esp_ok"]
            else f"PANIC ({diff_oh_015['esp_err'][:42]})"
        )
        esp_020_str = (
            f"sum={diff_oh_020['esp_sum']:+.4f} e, rms={diff_oh_020['esp_rms_ev']:.6f} eV"
            if diff_oh_020["esp_ok"]
            else f"PANIC ({diff_oh_020['esp_err'][:42]})"
        )
        print(
            f"  OH- ESP charges (net_charge=-1.0)    : v0.1.5 = {esp_015_str} | "
            f"v0.2.0 = {esp_020_str}"
        )
        print(
            f"  CLI --gpu vs CPU PM6 HOF (H2O) diff  : v0.1.5 = {cli_015['abs_diff_kcal']:.5f} kcal/mol "
            f"| v0.2.0 = {cli_020['abs_diff_kcal']:.5f} kcal/mol"
        )
        if "gpu_dense_cluster_72_atoms" in micro_015 and "gpu_dense_cluster_72_atoms" in micro_020:
            e0 = micro_015["gpu_dense_cluster_72_atoms"]["max_abs_err_vs_cpu"]
            e1 = micro_020["gpu_dense_cluster_72_atoms"]["max_abs_err_vs_cpu"]
            print(
                f"  GPU Density Cube (>64 active atoms)  : v0.1.5 max err = {e0:.6e} | "
                f"v0.2.0 max err = {e1:.6e}"
            )

        print("\n[4] Numerical Precision Audit (v0.1.5 vs v0.2.0 & OpenMOPAC Fortran)")
        print("-" * 92)
        for n in (64, 150, 250):
            m0 = micro_015[f"eigensolver_n{n}"]
            m1 = micro_020[f"eigensolver_n{n}"]
            de = max(
                abs(m1["e0"] - m0["e0"]),
                abs(m1["emid"] - m0["emid"]),
                abs(m1["elast"] - m0["elast"]),
            )
            print(
                f"  Eigensolver {n}x{n:<3} max|dLambda|={de:.2e} | "
                f"orth(v0.1.5/v0.2.0)={m0['orth_err']:.1e}/{m1['orth_err']:.1e} | "
                f"res={m0['res_err']:.1e}/{m1['res_err']:.1e}"
            )

        f0 = micro_015["nddo_fock_c14h30"]
        f1 = micro_020["nddo_fock_c14h30"]
        print(
            f"  NDDO Fock (C14H30) |dFock_Fro|={abs(f1['fock_fro'] - f0['fock_fro']):.2e} | "
            f"|dFock_Sum|={abs(f1['fock_sum'] - f0['fock_sum']):.2e}"
        )

        c0_e, c1_e = micro_015["cosmo_c14h30"], micro_020["cosmo_c14h30"]
        print(
            f"  COSMO C14H30       |dEdiel|   ={abs(c1_e['ediel_ev'] - c0_e['ediel_ev']):.2e} eV | "
            f"|dFock00|={abs(c1_e['fock00'] - c0_e['fock00']):.2e} eV"
        )

        def max_grad_diff(g_a: list[list[float]], g_b: list[list[float]]) -> float:
            return max(abs(x - y) for ra, rb in zip(g_a, g_b) for x, y in zip(ra, rb))

        s0, s1 = py_bench["scf_c24h50_grad"]
        print(
            f"  PM6 C24H50         |dE_tot|   ={abs(s1['total_energy_ev'] - s0['total_energy_ev']):.2e} eV | "
            f"|dHOF|={abs(s1['hof_kcal'] - s0['hof_kcal']):.2e} kcal/mol | "
            f"max|dGrad|={max_grad_diff(s0['gradients'], s1['gradients']):.2e} eV/A | "
            f"|dHOMO|={abs(s1['homo_ev'] - s0['homo_ev']):.2e} eV"
        )

        co0, co1 = py_bench["cosmo_c20h42"]
        print(
            f"  COSMO C20H42       |dE_tot|   ={abs(co1['total_energy_ev'] - co0['total_energy_ev']):.2e} eV | "
            f"|dHOF|={abs(co1['hof_kcal'] - co0['hof_kcal']):.2e} kcal/mol | "
            f"max|dGrad|={max_grad_diff(co0['gradients'], co1['gradients']):.2e} eV/A"
        )

        op0, op1 = py_bench["opt_c6h14"]
        rmsd_opt = math.sqrt(
            sum(
                (a - b) ** 2
                for ca, cb in zip(op0["coordinates"], op1["coordinates"])
                for a, b in zip(ca, cb)
            )
            / len(op0["coordinates"])
        )
        print(
            f"  L-BFGS C6H14 Opt   |dE_final| ={abs(op1['final_energy_ev'] - op0['final_energy_ev']):.2e} eV | "
            f"cycles={op0['cycles']}->{op1['cycles']} | geom_RMSD={rmsd_opt:.2e} A"
        )

        fr0, fr1 = py_bench["freq_benzene"]
        max_dfreq = max(abs(a - b) for a, b in zip(fr0["freqs"], fr1["freqs"]))
        print(
            f"  Frequencies Benz.  |dZPVE|    ={abs(fr1['zpe_kcal'] - fr0['zpe_kcal']):.2e} kcal/mol | "
            f"max|dFreq|={max_dfreq:.2e} cm^-1"
        )

        mp0, mp1 = py_bench["multi_prop_session"]
        max_desp = max(abs(a - b) for a, b in zip(mp0["esp_charges"], mp1["esp_charges"]))
        max_dbo = max(
            abs(a - b)
            for ra, rb in zip(mp0["bond_orders"], mp1["bond_orders"])
            for a, b in zip(ra, rb)
        )
        print(
            f"  QuantumSession     |dE_tot|   ={abs(mp1['energy_ev'] - mp0['energy_ev']):.2e} eV | "
            f"max|dESP|={max_desp:.2e} e | max|dBO|={max_dbo:.2e} | |dMECI|={abs(mp1['meci_ev'] - mp0['meci_ev']):.2e} eV"
        )

        openmopac_bin = find_openmopac_binary()
        if openmopac_bin:
            z24, c24 = py_make_alkane(24)
            z20, c20 = py_make_alkane(20)
            zbz = [6] * 6 + [1] * 6
            cbz = [
                [1.395 * math.cos(i * math.pi / 3.0), 1.395 * math.sin(i * math.pi / 3.0), 0.0]
                for i in range(6)
            ] + [
                [2.485 * math.cos(i * math.pi / 3.0), 2.485 * math.sin(i * math.pi / 3.0), 0.0]
                for i in range(6)
            ]
            om_c24 = run_openmopac_oracle(
                openmopac_bin, cli_scratch_v020, "om_c24h50", "PM6 1SCF PRECISE", z24, c24
            )
            om_c20 = run_openmopac_oracle(
                openmopac_bin, cli_scratch_v020, "om_c20h42", "PM6 1SCF PRECISE EPS=78.4", z20, c20
            )
            om_bz = run_openmopac_oracle(
                openmopac_bin, cli_scratch_v020, "om_benzene", "PM6 FORCE PRECISE", zbz, cbz
            )
            print("  -- Canonical OpenMOPAC v23.2.5 Comparison (v0.2.0 vs Fortran / v0.1.5 vs Fortran) --")
            if om_c24["hof_kcal"] is not None:
                print(
                    f"  C24H50 HOF vs Fortran : v0.1.5 |dHOF|={abs(s0['hof_kcal'] - om_c24['hof_kcal']):.5f} kcal/mol | "
                    f"v0.2.0 |dHOF|={abs(s1['hof_kcal'] - om_c24['hof_kcal']):.5f} kcal/mol "
                    f"(Fortran={om_c24['hof_kcal']:.5f})"
                )
            if om_c20["hof_kcal"] is not None:
                print(
                    f"  C20H42 COSMO vs Fortr.: v0.1.5 |dHOF|={abs(co0['hof_kcal'] - om_c20['hof_kcal']):.5f} kcal/mol | "
                    f"v0.2.0 |dHOF|={abs(co1['hof_kcal'] - om_c20['hof_kcal']):.5f} kcal/mol "
                    f"(Fortran={om_c20['hof_kcal']:.5f})"
                )
            if om_bz["zpe_kcal"] is not None:
                print(
                    f"  Benzene ZPVE vs Fortr.: v0.1.5 |dZPVE|={abs(fr0['zpe_kcal'] - om_bz['zpe_kcal']):.5f} kcal/mol | "
                    f"v0.2.0 |dZPVE|={abs(fr1['zpe_kcal'] - om_bz['zpe_kcal']):.5f} kcal/mol "
                    f"(Fortran={om_bz['zpe_kcal']:.5f})"
                )

        print("\n[5] Empirical NDDO Fock Crossover Sweep (v0.1.5 Serial vs v0.2.0 Parallel)")
        print(f"{'Alkane':<12} | {'Atoms':>6} | {'Pairs':>7} | {'v0.1.5 (us)':>12} | {'v0.2.0 (us)':>12} | {'Speedup':>10}")
        print("-" * 74)
        for nc in (6, 10, 14, 19, 24, 30, 38, 48):
            k = f"fock_sweep_c{nc}"
            r0, r1 = micro_015[k], micro_020[k]
            print(
                f"{f'C{nc}H{2*nc+2}':<12} | {r0['natoms']:>6} | {r0['pairs']:>7} | "
                f"{r0['value']:>12.2f} | {r1['value']:>12.2f} | {fmt_ratio(r0['value'], r1['value']):>10}"
            )

        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nSaved full JSON benchmark report to: {out_path}")

    finally:
        if worktree_dir.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)],
                cwd=ROOT,
                check=False,
            )
            shutil.rmtree(worktree_dir, ignore_errors=True)
        for p in (v015_harness_dir, v020_harness_dir, cli_scratch_v015, cli_scratch_v020):
            shutil.rmtree(p, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
