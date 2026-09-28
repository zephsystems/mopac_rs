#!/usr/bin/env python3
"""
Comprehensive Quantum Chemotheque Benchmark for mopac-py v0.1.1
Dataset: 1,000 PubChem Aspirin (CID 2244) structural analogs.
Testing:
  - High-throughput AM1-BCC charges (AMBER / GAFF2 parameterization)
  - High-throughput PM6 + D3-BJ + COSMO (aqueous thermochemistry & orbital gap)
  - Multi-Hamiltonian comparative benchmark (PM6, AM1, RM1, PM3, MNDO)
  - Quasi-Newton L-BFGS quantum geometry optimization
  - Vibrational frequencies, normal modes & statistical thermochemistry (H, S, G, ZPVE)
  - Merz-Singh-Kollman Electrostatic Potential (ESP) grid-fitted charges
  - Static & dynamic polarizability tensor and hyperpolarizability beta
  - Multi-Electron Configuration Interaction (MECI) & simulated UV-Vis spectrum
"""

import concurrent.futures
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import mopac_py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(SCRIPT_DIR, "data", "aspirin_1000.json")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "data", "benchmark_results.json")

ORGANIC_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}

def prepare_3d_molecule(smiles: str, seed: int = 42) -> Optional[Tuple[List[int], List[List[float]]]]:
    """Converts SMILES to 3D Cartesian coordinates with RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        # Check if neutral and standard organic
        nums = {a.GetAtomicNum() for a in mol.GetAtoms()}
        if nums - ORGANIC_ATOMS:
            return None
        if Chem.GetFormalCharge(mol) != 0:
            return None
        
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            return None
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        atoms, coords = mopac_py.mopac_py.from_rdkit(mol)
        return atoms, coords
    except Exception:
        return None

def worker_am1_bcc(item: Tuple[int, str]) -> Dict[str, Any]:
    """Computes AM1-BCC charges for a molecule."""
    cid, smi = item
    t0 = time.perf_counter()
    prep = prepare_3d_molecule(smi)
    if prep is None:
        return {"cid": cid, "status": "skipped", "reason": "non_organic_or_embedding_failure"}
    
    atoms, coords = prep
    try:
        res = mopac_py.am1_bcc(atoms, coords)
        t1 = time.perf_counter()
        return {
            "cid": cid,
            "status": "success",
            "time_ms": (t1 - t0) * 1000.0,
            "num_atoms": len(atoms),
            "total_charge": res.total_charge,
            "min_charge": float(np.min(res.bcc_charges)),
            "max_charge": float(np.max(res.bcc_charges)),
            "mean_abs_charge": float(np.mean(np.abs(res.bcc_charges))),
        }
    except Exception as e:
        return {"cid": cid, "status": "failed", "error": str(e)}

def worker_pm6_cosmo(item: Tuple[int, str]) -> Dict[str, Any]:
    """Computes PM6 + COSMO water solvation + electronic properties."""
    cid, smi = item
    t0 = time.perf_counter()
    prep = prepare_3d_molecule(smi)
    if prep is None:
        return {"cid": cid, "status": "skipped"}
    
    atoms, coords = prep
    try:
        res = mopac_py.calculate(
            atoms, coords,
            method="PM6",
            cosmo_eps=78.4,
            use_nddo=True,
            max_iter=80,
            damping=0.5
        )
        t1 = time.perf_counter()
        return {
            "cid": cid,
            "status": "success",
            "time_ms": (t1 - t0) * 1000.0,
            "converged": res.converged,
            "scf_iterations": res.scf_iterations,
            "heat_of_formation_kcal": res.heat_of_formation_kcal,
            "total_energy_ev": res.total_energy_ev,
            "dipole_debye": res.dipole_debye,
            "homo_ev": res.homo_energy_ev,
            "lumo_ev": res.lumo_energy_ev,
            "gap_ev": res.homo_lumo_gap_ev,
        }
    except Exception as e:
        return {"cid": cid, "status": "failed", "error": str(e)}

def run_benchmarks():
    print("=" * 80)
    print("MOPAC_PY v0.1.1 COMPREHENSIVE CHEMOTEQUE BENCHMARK")
    print("=" * 80)

    with open(DATASET_FILE, "r") as f:
        compounds = json.load(f)
    print(f"Loaded {len(compounds)} compounds from {DATASET_FILE}.")

    bench_output = {}

    # -------------------------------------------------------------------------
    # TIER 1: HIGH-THROUGHPUT AM1-BCC (AMBER GAFF2 PARAMETERIZATION)
    # -------------------------------------------------------------------------
    print("\n[TIER 1A] Executing High-Throughput AM1-BCC Charges on full library...")
    work_items = [(c["CID"], c["SMILES"]) for c in compounds if "SMILES" in c]
    
    t_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        am1_results = list(executor.map(worker_am1_bcc, work_items))
    t_total_am1 = time.perf_counter() - t_start

    am1_ok = [r for r in am1_results if r["status"] == "success"]
    am1_failed = [r for r in am1_results if r["status"] == "failed"]
    am1_skipped = [r for r in am1_results if r["status"] == "skipped"]
    
    avg_am1_ms = np.mean([r["time_ms"] for r in am1_ok]) if am1_ok else 0.0
    throughput_am1 = len(am1_ok) / t_total_am1 if t_total_am1 > 0 else 0.0

    print(f"  Total Processed: {len(work_items)}")
    print(f"  Succeeded:       {len(am1_ok)} ({len(am1_ok)/len(work_items)*100:.1f}%)")
    print(f"  Skipped (salts): {len(am1_skipped)}")
    print(f"  SCF Failed:      {len(am1_failed)}")
    print(f"  Wall Time:       {t_total_am1:.2f} s")
    print(f"  Throughput:      {throughput_am1:.1f} molecules/s")
    print(f"  Mean Time/Mol:   {avg_am1_ms:.2f} ms")
    
    bench_output["tier1_am1_bcc"] = {
        "total": len(work_items),
        "succeeded": len(am1_ok),
        "skipped": len(am1_skipped),
        "failed": len(am1_failed),
        "wall_time_s": t_total_am1,
        "throughput_mol_per_s": throughput_am1,
        "mean_time_ms": avg_am1_ms,
    }

    # -------------------------------------------------------------------------
    # TIER 1B: HIGH-THROUGHPUT PM6 + COSMO SOLVATION SCREENING
    # -------------------------------------------------------------------------
    print("\n[TIER 1B] Executing High-Throughput PM6 + COSMO Solvation (Water eps=78.4)...")
    # Run across the first 250 diverse compounds for fast wall-clock turn-around
    sample_items = work_items[:250]
    t_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        pm6_results = list(executor.map(worker_pm6_cosmo, sample_items))
    t_total_pm6 = time.perf_counter() - t_start

    pm6_ok = [r for r in pm6_results if r["status"] == "success" and r.get("converged")]
    avg_pm6_ms = np.mean([r["time_ms"] for r in pm6_ok]) if pm6_ok else 0.0
    throughput_pm6 = len(pm6_ok) / t_total_pm6 if t_total_pm6 > 0 else 0.0

    gaps = [r["gap_ev"] for r in pm6_ok]
    dipoles = [r["dipole_debye"] for r in pm6_ok]
    hofs = [r["heat_of_formation_kcal"] for r in pm6_ok]

    print(f"  Sample Tested:   {len(sample_items)} molecules")
    print(f"  Succeeded:       {len(pm6_ok)} ({len(pm6_ok)/len(sample_items)*100:.1f}%)")
    print(f"  Wall Time:       {t_total_pm6:.2f} s")
    print(f"  Throughput:      {throughput_pm6:.1f} molecules/s")
    print(f"  Mean Time/Mol:   {avg_pm6_ms:.2f} ms")
    print(f"  Mean Delta Hf:   {np.mean(hofs):.2f} kcal/mol (min: {np.min(hofs):.2f}, max: {np.max(hofs):.2f})")
    print(f"  Mean Dipole:     {np.mean(dipoles):.2f} Debye (max: {np.max(dipoles):.2f})")
    print(f"  Mean HOMO-LUMO:  {np.mean(gaps):.2f} eV (min: {np.min(gaps):.2f}, max: {np.max(gaps):.2f})")

    bench_output["tier1_pm6_cosmo"] = {
        "sample_size": len(sample_items),
        "succeeded": len(pm6_ok),
        "wall_time_s": t_total_pm6,
        "throughput_mol_per_s": throughput_pm6,
        "mean_time_ms": avg_pm6_ms,
        "mean_heat_of_formation_kcal": float(np.mean(hofs)) if hofs else None,
        "mean_dipole_debye": float(np.mean(dipoles)) if dipoles else None,
        "mean_gap_ev": float(np.mean(gaps)) if gaps else None,
    }

    # -------------------------------------------------------------------------
    # TIER 2: MULTI-HAMILTONIAN COMPARISON (ASPIRIN CID 2244)
    # -------------------------------------------------------------------------
    print("\n[TIER 2] Multi-Hamiltonian Quantum Comparison on Aspirin (CID 2244)...")
    aspirin_smi = "CC(=O)Oc1ccccc1C(=O)O"
    prep = prepare_3d_molecule(aspirin_smi)
    assert prep is not None, "Failed to embed Aspirin"
    atoms, coords = prep

    hamiltonians = ["PM6", "AM1", "RM1", "PM3", "MNDO"]
    ham_results = {}
    for h in hamiltonians:
        t0 = time.perf_counter()
        res = mopac_py.calculate(atoms, coords, method=h)
        t1 = time.perf_counter()
        ham_results[h] = {
            "time_ms": (t1 - t0) * 1000.0,
            "heat_of_formation_kcal": res.heat_of_formation_kcal,
            "total_energy_ev": res.total_energy_ev,
            "dipole_debye": res.dipole_debye,
            "homo_ev": res.homo_energy_ev,
            "lumo_ev": res.lumo_energy_ev,
            "gap_ev": res.homo_lumo_gap_ev,
            "scf_iterations": res.scf_iterations,
        }
        print(f"  {h:6s} | dHf: {res.heat_of_formation_kcal:8.2f} kcal/mol | Gap: {res.homo_lumo_gap_ev:6.2f} eV | Dipole: {res.dipole_debye:5.2f} D | SCF: {res.scf_iterations:2d} iter | Time: {(t1-t0)*1000:6.1f} ms")

    bench_output["tier2_hamiltonian_comparison"] = ham_results

    # -------------------------------------------------------------------------
    # TIER 3: QUANTUM GEOMETRY OPTIMIZATION (L-BFGS)
    # -------------------------------------------------------------------------
    print("\n[TIER 3] Quasi-Newton L-BFGS Geometry Optimization (PM6)...")
    t0 = time.perf_counter()
    opt_res = mopac_py.optimize(atoms, coords, method="PM6", max_cycles=60, grad_rms_tol=1.0)
    t1 = time.perf_counter()
    opt_coords = opt_res.coordinates
    print(f"  Converged:       {opt_res.converged}")
    print(f"  Cycles:          {opt_res.cycles}")
    print(f"  Initial Energy:  {opt_res.initial_energy_ev:.4f} eV")
    print(f"  Final Energy:    {opt_res.final_energy_ev:.4f} eV (Delta: {(opt_res.final_energy_ev - opt_res.initial_energy_ev):.4f} eV)")
    print(f"  Final Delta Hf:  {opt_res.final_heat_of_formation_kcal:.2f} kcal/mol")
    print(f"  Final Grad RMS:  {opt_res.final_grad_rms:.4f} kcal/(mol*A)")
    print(f"  Optimization Time: {(t1 - t0)*1000:.1f} ms")

    bench_output["tier3_optimization"] = {
        "converged": opt_res.converged,
        "cycles": opt_res.cycles,
        "initial_energy_ev": opt_res.initial_energy_ev,
        "final_energy_ev": opt_res.final_energy_ev,
        "final_heat_of_formation_kcal": opt_res.final_heat_of_formation_kcal,
        "final_grad_rms": opt_res.final_grad_rms,
        "time_ms": (t1 - t0) * 1000.0,
    }

    # -------------------------------------------------------------------------
    # TIER 4: VIBRATIONAL FREQUENCIES & STATISTICAL THERMOCHEMISTRY
    # -------------------------------------------------------------------------
    print("\n[TIER 4A] Harmonic Vibrational Frequencies & Thermodynamics (298.15 K)...")
    t0 = time.perf_counter()
    vib_res = mopac_py.frequencies(atoms, opt_coords, method="PM6")
    t1 = time.perf_counter()
    freqs = vib_res.vibrational_frequencies_cm1
    thermo = vib_res.thermo
    print(f"  Vibrational Modes: {len(freqs)}")
    print(f"  Top 3 Frequencies: {freqs[-3]:.1f}, {freqs[-2]:.1f}, {freqs[-1]:.1f} cm^-1")
    print(f"  Lowest Frequency:  {freqs[0]:.1f} cm^-1 (Transition state: {vib_res.is_transition_state})")
    print(f"  ZPVE:              {vib_res.zpve_kcal_mol:.2f} kcal/mol")
    print(f"  Enthalpy H(298K):  {thermo.enthalpy_thermal_cal_mol/1000.0:.2f} kcal/mol")
    print(f"  Entropy S(298K):   {thermo.entropy_total_cal_k_mol:.2f} cal/(mol*K)")
    print(f"  Gibbs Corr G(298): {thermo.gibbs_correction_kcal_mol:.2f} kcal/mol")
    print(f"  Hessian Time:      {(t1 - t0)*1000:.1f} ms")

    bench_output["tier4_vibrations_and_thermo"] = {
        "modes_count": len(freqs),
        "zpve_kcal_mol": vib_res.zpve_kcal_mol,
        "lowest_freq_cm1": freqs[0] if freqs else None,
        "highest_freq_cm1": freqs[-1] if freqs else None,
        "entropy_cal_mol_k": thermo.entropy_total_cal_k_mol,
        "enthalpy_thermal_kcal_mol": thermo.enthalpy_thermal_cal_mol / 1000.0,
        "gibbs_correction_kcal_mol": thermo.gibbs_correction_kcal_mol,
        "time_ms": (t1 - t0) * 1000.0,
    }

    # -------------------------------------------------------------------------
    # TIER 4B: MERZ-SINGH-KOLLMAN ESP CHARGES
    # -------------------------------------------------------------------------
    print("\n[TIER 4B] Merz-Singh-Kollman ESP Fitted Charges on Optimized Aspirin...")
    t0 = time.perf_counter()
    esp_res = mopac_py.esp_charges(atoms, opt_coords, method="AM1")
    t1 = time.perf_counter()
    esp_q = esp_res.charges
    print(f"  ESP Atoms fitted: {len(esp_q)}")
    print(f"  Sum of Charges:   {sum(esp_q):.6f}")
    print(f"  ESP Range:        [{min(esp_q):.3f}, {max(esp_q):.3f}]")
    print(f"  ESP Time:         {(t1 - t0)*1000:.1f} ms")

    bench_output["tier4_esp_charges"] = {
        "atom_count": len(esp_q),
        "sum_charges": sum(esp_q),
        "min_charge": min(esp_q),
        "max_charge": max(esp_q),
        "time_ms": (t1 - t0) * 1000.0,
    }

    # -------------------------------------------------------------------------
    # TIER 4C: POLARIZABILITY & HYPERPOLARIZABILITY (NLO)
    # -------------------------------------------------------------------------
    print("\n[TIER 4C] Polarizability Tensor & First Hyperpolarizability beta...")
    t0 = time.perf_counter()
    pol_res = mopac_py.polarizability(atoms, opt_coords, method="AM1")
    t1 = time.perf_counter()
    print(f"  Isotropic Polarizability alpha: {pol_res.alpha_isotropic_angstrom3:.3f} A^3 ({pol_res.alpha_isotropic_au:.3f} a.u.)")
    print(f"  Anisotropy Delta alpha:        {pol_res.alpha_anisotropy_au:.3f} a.u.")
    print(f"  Hyperpolarizability beta_tot:  {pol_res.beta_total_esu:.4e} esu ({pol_res.beta_total_au:.2f} a.u.)")
    print(f"  Dipole Moment:                 {pol_res.dipole_debye:.3f} Debye")
    print(f"  Polarizability Time:           {(t1 - t0)*1000:.1f} ms")

    bench_output["tier4_polarizability"] = {
        "alpha_isotropic_angstrom3": pol_res.alpha_isotropic_angstrom3,
        "alpha_isotropic_au": pol_res.alpha_isotropic_au,
        "alpha_anisotropy_au": pol_res.alpha_anisotropy_au,
        "beta_total_esu": pol_res.beta_total_esu,
        "beta_total_au": pol_res.beta_total_au,
        "dipole_debye": pol_res.dipole_debye,
        "time_ms": (t1 - t0) * 1000.0,
    }

    # -------------------------------------------------------------------------
    # TIER 5: MECI & UV-VIS ABSORPTION SPECTRUM
    # -------------------------------------------------------------------------
    print("\n[TIER 5] Multi-Electron Configuration Interaction (MECI) & UV-Vis Spectrum...")
    t0 = time.perf_counter()
    meci_res = mopac_py.meci(atoms, opt_coords, method="AM1", active_orbitals=2)
    t1 = time.perf_counter()
    print(f"  MECI States: {len(meci_res.states)} (Total CI E = {meci_res.total_energy_ev:.2f} eV)")
    for s in meci_res.states:
        print(f"    Root {s.root} ({s.spin_label:7s}): deltaE = {s.excitation_energy_ev:5.2f} eV | lambda = {s.wavelength_nm:6.1f} nm | f = {s.oscillator_strength:6.4f}")

    uv_res = mopac_py.uv_vis_spectrum(atoms, opt_coords, method="AM1", active_orbitals=2)
    print(f"  UV-Vis Lambda Max: {uv_res.lambda_max_nm:.1f} nm (Epsilon Max = {uv_res.epsilon_max:.1f} L/(mol*cm))")

    bench_output["tier5_meci_and_uv_vis"] = {
        "states_count": len(meci_res.states),
        "lambda_max_nm": uv_res.lambda_max_nm,
        "epsilon_max": uv_res.epsilon_max,
        "states": [
            {
                "root": s.root,
                "spin": s.spin_label,
                "delta_e_ev": s.excitation_energy_ev,
                "lambda_nm": s.wavelength_nm,
                "oscillator_strength": s.oscillator_strength,
            }
            for s in meci_res.states
        ],
        "meci_time_ms": (t1 - t0) * 1000.0,
    }

    # Save complete benchmark results
    with open(RESULTS_FILE, "w") as f:
        json.dump(bench_output, f, indent=2)
    print(f"\nAll benchmark results successfully saved to {RESULTS_FILE}!")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmarks()
