#!/usr/bin/env python3
"""
Script to fetch the top 1000 PubChem compounds similar to Aspirin (CID 2244).
Uses PubChem PUG REST API directly with standard library urllib.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "data", "aspirin_1000.json")

def fetch_json(url: str, retries: int = 3, delay: float = 1.0) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "mopac-rs-benchmark/0.1.1 (scientific-research)"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay * (attempt + 2))
                continue
            raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
    return {}

def main():
    print("Querying PubChem PUG REST API for Aspirin (CID 2244) structural similarity...")
    sim_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/similarity/cid/2244/JSON?Threshold=90&MaxRecords=1000"

    try:
        sim_data = fetch_json(sim_url)
    except Exception as e:
        print(f"Error fetching similarity CIDs: {e}")
        sys.exit(1)

    cids = sim_data.get("IdentifierList", {}).get("CID", [])
    target_cids = cids[:1000]
    print(f"Total similarity CIDs found: {len(cids)}. Selected target: {len(target_cids)} CIDs.")

    batch_size = 100
    all_properties = []

    for i in range(0, len(target_cids), batch_size):
        chunk = target_cids[i:i + batch_size]
        cid_str = ",".join(map(str, chunk))
        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,IUPACName/JSON"

        batch_num = i // batch_size + 1
        total_batches = (len(target_cids) + batch_size - 1) // batch_size
        print(f"Fetching batch {batch_num}/{total_batches} (CIDs {chunk[0]}..{chunk[-1]})...")

        try:
            chunk_data = fetch_json(prop_url)
            props = chunk_data.get("PropertyTable", {}).get("Properties", [])
            all_properties.extend(props)
        except Exception as e:
            print(f"Warning: batch {batch_num} failed: {e}")

        time.sleep(0.3)

    print(f"Successfully retrieved properties for {len(all_properties)} compounds.")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_properties, f, indent=2)

    print(f"Saved complete chemotheque dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
