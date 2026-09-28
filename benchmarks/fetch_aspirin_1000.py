#!/usr/bin/env python3
"""
Script to fetch the top 1000 PubChem compounds similar to Aspirin (CID 2244).
Uses PubChem PUG REST API compliant with NCBI rate-limiting guidelines.
"""

import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBCHEM_API_SCRIPT = os.path.expanduser("~/.gemini/config/plugins/science/skills/pubchem_database/scripts/pubchem_api.py")
INPUT_SIM_FILE = "/tmp/aspirin_sim.json"
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "data", "aspirin_1000.json")

def main():
    if not os.path.exists(INPUT_SIM_FILE):
        print(f"Error: {INPUT_SIM_FILE} does not exist.")
        sys.exit(1)

    with open(INPUT_SIM_FILE, "r") as f:
        data = json.load(f)

    cids = data.get("IdentifierList", {}).get("CID", [])
    target_cids = cids[:1000]
    print(f"Total similarity CIDs found: {len(cids)}. Selected target: {len(target_cids)} CIDs.")

    batch_size = 100
    all_properties = []

    for i in range(0, len(target_cids), batch_size):
        chunk = target_cids[i:i + batch_size]
        cid_str = ",".join(map(str, chunk))
        tmp_out = f"/tmp/pubchem_chunk_{i}.json"
        path = f"pug/compound/cid/{cid_str}/property/SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,IUPACName/JSON"
        
        cmd = [
            "uv", "run", PUBCHEM_API_SCRIPT,
            "query", "--path", path,
            "--output", tmp_out
        ]
        print(f"Fetching batch {i // batch_size + 1}/{(len(target_cids) + batch_size - 1) // batch_size} (CIDs {chunk[0]}..{chunk[-1]})...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Batch failed: {res.stderr}")
            sys.exit(1)
        
        with open(tmp_out, "r") as tf:
            chunk_data = json.load(tf)
        
        props = chunk_data.get("PropertyTable", {}).get("Properties", [])
        all_properties.extend(props)
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        time.sleep(0.2)

    print(f"Successfully retrieved properties for {len(all_properties)} compounds.")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_properties, f, indent=2)

    print(f"Saved complete chemotheque dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
