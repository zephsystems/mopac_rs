#!/usr/bin/env python3
"""
Downloads and caches a chemically diverse molecular library from PubChem PUG REST.
Covers:
- Hydrocarbons (aliphatic, cyclic, aromatic, strained)
- Oxygenates (alcohols, ethers, carbonyls, esters, acids)
- Nitrogen compounds (amines, amides, nitriles, heterocycles)
- Halogenated compounds (fluorides, chlorides, bromides, iodides, mixed)
- Organosulfur and organophosphorus compounds
- Pharmaceuticals & biomolecules
- Inorganic small molecules
- Stress / boundary / open-shell cases
"""

import json
import os
import sys
import time
import urllib.request
import urllib.parse

COMPOUNDS_TO_FETCH = [
    # 1. Hydrocarbons
    ("methane", "Hydrocarbon", "methane"),
    ("ethane", "Hydrocarbon", "ethane"),
    ("ethylene", "Hydrocarbon", "ethylene"),
    ("acetylene", "Hydrocarbon", "acetylene"),
    ("propane", "Hydrocarbon", "propane"),
    ("cyclopropane", "Hydrocarbon (strained)", "cyclopropane"),
    ("cyclobutane", "Hydrocarbon (strained)", "cyclobutane"),
    ("cyclopentane", "Hydrocarbon", "cyclopentane"),
    ("cyclohexane", "Hydrocarbon", "cyclohexane"),
    ("benzene", "Aromatic", "benzene"),
    ("toluene", "Aromatic", "toluene"),
    ("naphthalene", "Polyaromatic", "naphthalene"),

    # 2. Oxygenates
    ("water", "Inorganic/O", "water"),
    ("methanol", "Alcohol", "methanol"),
    ("ethanol", "Alcohol", "ethanol"),
    ("diethyl_ether", "Ether", "diethyl ether"),
    ("tetrahydrofuran", "Ether (cyclic)", "tetrahydrofuran"),
    ("formaldehyde", "Carbonyl", "formaldehyde"),
    ("acetaldehyde", "Carbonyl", "acetaldehyde"),
    ("acetone", "Carbonyl", "acetone"),
    ("formic_acid", "Carboxylic acid", "formic acid"),
    ("acetic_acid", "Carboxylic acid", "acetic acid"),
    ("methyl_acetate", "Ester", "methyl acetate"),
    ("carbon_dioxide", "Inorganic/O", "carbon dioxide"),

    # 3. Nitrogen compounds
    ("ammonia", "Inorganic/N", "ammonia"),
    ("methylamine", "Amine (1st)", "methylamine"),
    ("dimethylamine", "Amine (2nd)", "dimethylamine"),
    ("trimethylamine", "Amine (3rd)", "trimethylamine"),
    ("formamide", "Amide", "formamide"),
    ("acetamide", "Amide", "acetamide"),
    ("urea", "Amide/Urea", "urea"),
    ("acetonitrile", "Nitrile", "acetonitrile"),
    ("pyridine", "Heterocycle (N)", "pyridine"),
    ("pyrimidine", "Heterocycle (N)", "pyrimidine"),
    ("pyrrole", "Heterocycle (N)", "pyrrole"),
    ("imidazole", "Heterocycle (N)", "imidazole"),
    ("aniline", "Aromatic amine", "aniline"),
    ("hydrogen_cyanide", "Nitrile/Inorganic", "hydrogen cyanide"),

    # 4. Halogenated
    ("fluoromethane", "Halide (F)", "fluoromethane"),
    ("chloromethane", "Halide (Cl)", "chloromethane"),
    ("bromomethane", "Halide (Br)", "bromomethane"),
    ("iodomethane", "Halide (I)", "iodomethane"),
    ("dichloromethane", "Halide (Cl)", "dichloromethane"),
    ("chloroform", "Halide (Cl)", "chloroform"),
    ("carbon_tetrachloride", "Halide (Cl)", "carbon tetrachloride"),
    ("fluorobenzene", "Aromatic halide (F)", "fluorobenzene"),
    ("chlorobenzene", "Aromatic halide (Cl)", "chlorobenzene"),
    ("bromobenzene", "Aromatic halide (Br)", "bromobenzene"),
    ("iodobenzene", "Aromatic halide (I)", "iodobenzene"),
    ("halothane", "Mixed polyhalide", "halothane"),

    # 5. Sulfur & Phosphorus
    ("hydrogen_sulfide", "Inorganic/S", "hydrogen sulfide"),
    ("methanethiol", "Thiol", "methanethiol"),
    ("dimethyl_sulfide", "Thioether", "dimethyl sulfide"),
    ("dimethyl_sulfoxide", "Sulfoxide", "DMSO"),
    ("thiophene", "Heterocycle (S)", "thiophene"),
    ("sulfur_dioxide", "Inorganic/S", "sulfur dioxide"),
    ("phosphine", "Inorganic/P", "phosphine"),
    ("trimethylphosphine", "Phosphine", "trimethylphosphine"),
    ("trimethyl_phosphate", "Phosphate", "trimethyl phosphate"),

    # 6. Pharmaceuticals & Biomolecules
    ("aspirin", "Pharmaceutical", "aspirin"),
    ("paracetamol", "Pharmaceutical", "paracetamol"),
    ("caffeine", "Pharmaceutical", "caffeine"),
    ("ibuprofen", "Pharmaceutical", "ibuprofen"),
    ("glycine", "Amino acid", "glycine"),
    ("alanine", "Amino acid", "alanine"),

    # 7. Stress & Boundary Cases
    ("nitric_oxide", "Open-shell radical", "nitric oxide"),
    ("nitrogen_dioxide", "Open-shell radical", "nitrogen dioxide"),
    ("silane", "Unsupported element (Si)", "silane"),
    ("ferrocene", "Unsupported metal (Fe)", "ferrocene"),
]

CACHE_FILE = os.path.join(os.path.dirname(__file__), "molecules_cache.json")


def fetch_pubchem_3d(query_name):
    encoded = urllib.parse.quote(query_name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/record/JSON/?record_type=3d"
    req = urllib.request.Request(url, headers={"User-Agent": "MopacRsBenchmark/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.loads(response.read().decode())
            pc = data["PC_Compounds"][0]
            atomic_numbers = pc["atoms"]["element"]
            coords_obj = pc["coords"][0]["conformers"][0]
            xs = coords_obj["x"]
            ys = coords_obj["y"]
            zs = coords_obj.get("z", [0.0] * len(xs))
            coordinates = [[x, y, z] for x, y, z in zip(xs, ys, zs)]
            return {
                "atomic_numbers": atomic_numbers,
                "coordinates": coordinates,
                "natoms": len(atomic_numbers),
            }
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"Fetching {len(COMPOUNDS_TO_FETCH)} compounds from PubChem 3D database...")
    library = {}

    success_count = 0
    fail_count = 0

    for key, category, pubchem_name in COMPOUNDS_TO_FETCH:
        print(f"  Fetching [{category}] {key} ('{pubchem_name}')...", end="", flush=True)
        res = fetch_pubchem_3d(pubchem_name)
        if "error" in res:
            print(f" ❌ Error: {res['error']}")
            fail_count += 1
            library[key] = {
                "name": key,
                "category": category,
                "status": "fetch_failed",
                "error": res["error"],
            }
        else:
            print(f" ✅ OK ({res['natoms']} atoms, elements={sorted(set(res['atomic_numbers']))})")
            success_count += 1
            library[key] = {
                "name": key,
                "category": category,
                "status": "ready",
                "atomic_numbers": res["atomic_numbers"],
                "coordinates": res["coordinates"],
                "natoms": res["natoms"],
            }
        time.sleep(0.2)  # Respect PubChem rate limit (max 5 req/sec)

    # Add synthetic boundary cases
    print("  Adding synthetic collision/steric stress case...")
    library["steric_collision_water"] = {
        "name": "steric_collision_water",
        "category": "Stress / Steric clash",
        "status": "ready",
        "atomic_numbers": [8, 1, 1],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.15], [0.0, 0.7, -0.5]],
        "natoms": 3,
    }

    print("  Adding non-covalent water dimer complex...")
    library["water_dimer"] = {
        "name": "water_dimer",
        "category": "Non-covalent complex",
        "status": "ready",
        "atomic_numbers": [8, 1, 1, 8, 1, 1],
        "coordinates": [
            [-1.464, 0.000, 0.000],
            [-1.850, 0.760, 0.520],
            [-0.510, 0.000, 0.000],
            [1.464, 0.000, 0.000],
            [1.850, -0.760, 0.520],
            [1.850, 0.760, -0.520],
        ],
        "natoms": 6,
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(library, f, indent=2)

    print(f"\nChemotheque download complete: {success_count} succeeded, {fail_count} failed.")
    print(f"Library saved to: {CACHE_FILE}")


if __name__ == "__main__":
    main()
