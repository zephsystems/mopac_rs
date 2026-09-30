#!/usr/bin/env python3
"""
NIST CCCBDB (Computational Chemistry Comparison and Benchmark Database) Fetcher & Curator.

Downloads and consolidates:
1. NIST CCCBDB Gas-Phase Experimental Benchmark Dataset (JARVIS-Tools / Choudhary et al., DOI: 10.6084/m9.figshare.26117998).
2. NIST CCCBDB Experimental Microwave & Stark Effect Dipole Moments (https://cccbdb.nist.gov/diplistx.asp).

Produces a unified, high-integrity JSON benchmark file with:
- Isolated gas-phase Cartesian coordinates (sub-angstrom microwave & electron diffraction).
- Laboratory-measured gas-phase heats of formation (kJ/mol and kcal/mol).
- Laboratory-measured experimental dipole moments (Debye).
"""

import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")
CCCBDB_DIR = os.path.join(DATASETS_DIR, "nist_cccbdb")
OUTPUT_JSON = os.path.join(CCCBDB_DIR, "nist_cccbdb_curated.json")

FIGSHARE_URL = "https://ndownloader.figshare.com/files/47283808"
NIST_DIPOLE_URL = "https://cccbdb.nist.gov/diplistx.asp"

PERIODIC_TABLE = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
}

KJ_TO_KCAL = 1.0 / 4.184


def fetch_experimental_dipoles() -> Dict[str, float]:
    """Fetch 574 experimental dipole moments from NIST CCCBDB diplistx.asp."""
    print("Fetching NIST CCCBDB experimental dipoles from diplistx.asp...")
    req = urllib.request.Request(NIST_DIPOLE_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Warning: Could not fetch diplistx.asp ({e}). Proceeding without dipoles.")
        return {}

    start = html.find("<table border=1>")
    if start == -1:
        start = html.find("<table")
    end = html.find("</table>", start) if start != -1 else len(html)
    table_html = html[start:end]

    rows = re.findall(r"<tr>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE)
    dipole_map: Dict[str, float] = {}

    for r in rows[1:]:
        cells = re.findall(r"<td.*?>(.*?)</td>", r, re.DOTALL | re.IGNORECASE)
        cleaned = [re.sub(r"<.*?>", "", c).strip() for c in cells]
        if len(cleaned) >= 7:
            formula = cleaned[0].replace(" ", "")
            name = cleaned[1].strip()
            tot_str = cleaned[6].strip()
            try:
                val = float(tot_str)
                # Normalize keys
                dipole_map[formula.upper()] = val
                if name:
                    dipole_map[name.lower()] = val
            except ValueError:
                continue

    print(f"Successfully extracted {len(dipole_map)} dipole lookup entries.")
    return dipole_map


def fetch_cccbdb_json() -> List[Dict[str, Any]]:
    """Fetch and extract cccbdb.json from Figshare."""
    print(f"Downloading NIST CCCBDB dataset from Figshare ({FIGSHARE_URL})...")
    req = urllib.request.Request(FIGSHARE_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    print(f"Downloaded {len(content)} bytes. Extracting ZIP archive...")
    z = zipfile.ZipFile(io.BytesIO(content))
    raw_data = json.loads(z.read("cccbdb.json").decode("utf-8"))
    print(f"Extracted {len(raw_data)} raw molecular entries from CCCBDB.")
    return raw_data


def curate_dataset() -> None:
    os.makedirs(CCCBDB_DIR, exist_ok=True)
    dipoles = fetch_experimental_dipoles()
    raw_entries = fetch_cccbdb_json()

    curated: List[Dict[str, Any]] = []
    with_dhf_count = 0
    with_dipole_count = 0

    for idx, entry in enumerate(raw_entries):
        jid = entry.get("jid", f"cc-{idx}")
        species = entry.get("species", "").strip()
        inchi = entry.get("inchi", "").strip()
        name = str(entry.get("name", "")).strip()

        atoms_dict = entry.get("atoms", {})
        elements = atoms_dict.get("elements", [])
        coords = atoms_dict.get("coords", [])

        if not elements or not coords or len(elements) != len(coords):
            continue

        # Map elements to atomic numbers
        atomic_numbers = []
        valid_elements = True
        for el in elements:
            el_clean = el.capitalize()
            if el_clean in PERIODIC_TABLE:
                atomic_numbers.append(PERIODIC_TABLE[el_clean])
            else:
                valid_elements = False
                break
        if not valid_elements:
            continue

        # Parse experimental heats of formation (kJ/mol -> kcal/mol)
        dhf_298_kj = entry.get("enthalpy_formation_298K")
        dhf_298_kcal: Optional[float] = None
        if isinstance(dhf_298_kj, (int, float)):
            dhf_298_kcal = float(dhf_298_kj) * KJ_TO_KCAL
        elif isinstance(dhf_298_kj, str):
            try:
                val = float(dhf_298_kj)
                dhf_298_kj = val
                dhf_298_kcal = val * KJ_TO_KCAL
            except ValueError:
                dhf_298_kj = None

        dhf_0_kj = entry.get("enthalpy_formation_0K")
        dhf_0_kcal: Optional[float] = None
        if isinstance(dhf_0_kj, (int, float)):
            dhf_0_kcal = float(dhf_0_kj) * KJ_TO_KCAL
        elif isinstance(dhf_0_kj, str):
            try:
                val = float(dhf_0_kj)
                dhf_0_kj = val
                dhf_0_kcal = val * KJ_TO_KCAL
            except ValueError:
                dhf_0_kj = None

        if dhf_298_kcal is not None:
            with_dhf_count += 1

        # Match experimental dipole
        dipole_val = None
        sp_upper = species.upper()
        if sp_upper in dipoles:
            dipole_val = dipoles[sp_upper]
        elif name and name.lower() in dipoles:
            dipole_val = dipoles[name.lower()]

        if dipole_val is not None:
            with_dipole_count += 1

        curated.append({
            "id": jid,
            "species": species,
            "name": name,
            "inchi": inchi,
            "natoms": len(atomic_numbers),
            "atomic_numbers": atomic_numbers,
            "elements": elements,
            "coordinates": coords,
            "expt_dhf_298k_kj_mol": dhf_298_kj,
            "expt_dhf_298k_kcal_mol": dhf_298_kcal,
            "expt_dhf_0k_kj_mol": dhf_0_kj,
            "expt_dhf_0k_kcal_mol": dhf_0_kcal,
            "expt_dipole_debye": dipole_val,
            "homo_au": entry.get("homo"),
            "lumo_au": entry.get("lumo"),
        })

    metadata = {
        "dataset_name": "NIST Computational Chemistry Comparison and Benchmark Database (CCCBDB)",
        "source": "NIST SRD 101 / JARVIS-Tools (Kamal Choudhary et al.)",
        "doi": "10.6084/m9.figshare.26117998",
        "total_systems": len(curated),
        "systems_with_experimental_dhf_298k": with_dhf_count,
        "systems_with_experimental_dipole": with_dipole_count,
    }

    final_obj = {
        "metadata": metadata,
        "molecules": curated,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_obj, f, indent=2)

    print(f"\n[OK] Curated NIST CCCBDB dataset written to: {OUTPUT_JSON}")
    print(f"Total valid molecular systems: {len(curated)}")
    print(f"Systems with experimental dHf(298K): {with_dhf_count}")
    print(f"Systems with experimental dipole moments: {with_dipole_count}")


if __name__ == "__main__":
    curate_dataset()
