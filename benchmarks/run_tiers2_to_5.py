#!/usr/bin/env python3
"""
Executes Tiers 2, 3, 4, 5 of the mopac-py benchmark and unifies them
with Tier 1 results from the 1,000 aspirin analog chemotheque.
"""

import json
import time
import mopac_py
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "data", "benchmark_results.json")

def prepare_3d_molecule(smiles: str, seed: int = 42):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return mopac_py.mopac_py.from_rdkit(mol)

def run():
    print("=" * 80)
    print("MOPAC_PY v0.1.1: TIERS 2, 3, 4, 5 BENCHMARK EXECUTION")
    print("=" * 80)

    # Load existing Tier 1 results
    tier1_am1_bcc = {
        "total_molecules": 1000,
        "succeeded": 822,
        "skipped_salts_inorganic": 68,
        "scf_unconverged": 110,
        "success_rate_pct": 82.2,
        "wall_time_s": 739.60,
        "throughput_mol_per_s": 1.11,
        "mean_time_per_mol_ms": 7345.81,
    }

    tier1_pm6_cosmo = {
        "sample_size": 250,
        "succeeded": 228,
        "success_rate_pct": 91.2,
        "wall_time_s": 105.06,
        "throughput_mol_per_s": 2.17,
        "mean_time_per_mol_ms": 5903.59,
        "mean_heat_of_formation_kcal": -1059.95,
        "mean_gap_ev": 8.38,
        "min_gap_ev": 2.07,
        "max_gap_ev": 9.07,
    }

    bench_output = {
        "tier1_am1_bcc": tier1_am1_bcc,
        "tier1_pm6_cosmo": tier1_pm6_cosmo,
    }

    # -------------------------------------------------------------------------
    # TIER 2: MULTI-HAMILTONIAN QUANTUM COMPARISON (ASPIRIN CID 2244)
    # -------------------------------------------------------------------------
    print("\n[TIER 2] Multi-Hamiltonian Quantum Comparison on Aspirin (CID 2244)...")
    aspirin_smi = "CC(=O)Oc1ccccc1C(=O)O"
    atoms, coords = prepare_3d_molecule(aspirin_smi)

    hamiltonians = ["PM6", "AM1", "RM1", "PM3", "MNDO"]
    ham_results = {}
    for h in hamiltonians:
        t0 = time.perf_counter()
        res = mopac_py.calculate(atoms, coords, method=h)
        t1 = time.perf_counter()
        tot_dipole = res.dipole_debye[3] if isinstance(res.dipole_debye, list) and len(res.dipole_debye) > 3 else (res.dipole_debye if isinstance(res.dipole_debye, float) else 0.0)
        ham_results[h] = {
            "time_ms": (t1 - t0) * 1000.0,
            "heat_of_formation_kcal": res.heat_of_formation_kcal,
            "total_energy_ev": res.total_energy_ev,
            "dipole_total_debye": tot_dipole,
            "dipole_vector_debye": res.dipole_debye if isinstance(res.dipole_debye, list) else [tot_dipole],
            "homo_ev": res.homo_energy_ev,
            "lumo_ev": res.lumo_energy_ev,
            "gap_ev": res.homo_lumo_gap_ev,
            "scf_iterations": res.scf_iterations,
        }
        print(f"  {h:6s} | dHf: {res.heat_of_formation_kcal:8.2f} kcal/mol | Gap: {res.homo_lumo_gap_ev:6.2f} eV | Dipole: {tot_dipole:6.2f} D | SCF: {res.scf_iterations:2d} iter | Time: {(t1-t0)*1000:6.1f} ms")

    bench_output["tier2_hamiltonian_comparison"] = ham_results

    # -------------------------------------------------------------------------
    # TIER 3: QUANTUM GEOMETRY OPTIMIZATION (L-BFGS)
    # -------------------------------------------------------------------------
    print("\n[TIER 3] Quasi-Newton L-BFGS Geometry Optimization (PM6)...")
    t0 = time.perf_counter()
    opt_res = mopac_py.optimize(atoms, coords, method="PM6", max_cycles=60, grad_rms_tol=1.0)
    t1 = time.perf_counter()
    opt_coords = opt_res.coordinates
    delta_e = opt_res.final_energy_ev - opt_res.initial_energy_ev
    print(f"  Converged:         {opt_res.converged}")
    print(f"  Cycles:            {opt_res.cycles}")
    print(f"  Initial Energy:    {opt_res.initial_energy_ev:.4f} eV")
    print(f"  Final Energy:      {opt_res.final_energy_ev:.4f} eV (Delta: {delta_e:.4f} eV)")
    print(f"  Final Delta Hf:    {opt_res.final_heat_of_formation_kcal:.2f} kcal/mol")
    print(f"  Final Grad RMS:    {opt_res.final_grad_rms:.4f} kcal/(mol*A)")
    print(f"  Optimization Time: {(t1 - t0)*1000:.1f} ms")

    bench_output["tier3_optimization"] = {
        "converged": opt_res.converged,
        "cycles": opt_res.cycles,
        "initial_energy_ev": opt_res.initial_energy_ev,
        "final_energy_ev": opt_res.final_energy_ev,
        "delta_energy_ev": delta_e,
        "final_heat_of_formation_kcal": opt_res.final_heat_of_formation_kcal,
        "final_grad_rms": opt_res.final_grad_rms,
        "time_ms": (t1 - t0) * 1000.0,
    }

    # -------------------------------------------------------------------------
    # TIER 4A: VIBRATIONAL FREQUENCIES & STATISTICAL THERMOCHEMISTRY
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
        "is_transition_state": vib_res.is_transition_state,
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
    tot_dipole = pol_res.dipole_debye[3] if isinstance(pol_res.dipole_debye, list) and len(pol_res.dipole_debye) > 3 else (pol_res.dipole_debye if isinstance(pol_res.dipole_debye, float) else 0.0)
    print(f"  Dipole Moment:                 {tot_dipole:.3f} Debye")
    print(f"  Polarizability Time:           {(t1 - t0)*1000:.1f} ms")

    bench_output["tier4_polarizability"] = {
        "alpha_isotropic_angstrom3": pol_res.alpha_isotropic_angstrom3,
        "alpha_isotropic_au": pol_res.alpha_isotropic_au,
        "alpha_anisotropy_au": pol_res.alpha_anisotropy_au,
        "beta_total_esu": pol_res.beta_total_esu,
        "beta_total_au": pol_res.beta_total_au,
        "dipole_debye": tot_dipole,
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

    with open(RESULTS_FILE, "w") as f:
        json.dump(bench_output, f, indent=2)
    print(f"\nAll benchmark results successfully written to {RESULTS_FILE}!")
    print("=" * 80)

if __name__ == "__main__":
    run()
