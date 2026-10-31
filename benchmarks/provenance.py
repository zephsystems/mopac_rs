#!/usr/bin/env python3
"""Provenance stamping shared by all mopac-rs benchmark runners.

Every benchmark results JSON carries a `metadata` block recording which
workspace version and which installed wheel produced the numbers, ensuring
reproducibility and provenance across benchmark executions.
"""

import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def workspace_version() -> str:
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())
    return cargo["workspace"]["package"]["version"]


def wheel_version():
    """Version of the installed mopac_py wheel, or None if not importable.

    None must propagate to the CI gate and fail it — a benchmark run without a
    known engine version has no provenance and must not be trusted silently.
    """
    try:
        import mopac_py

        return getattr(mopac_py, "__version__", None)
    except Exception:
        return None


def stamp_metadata(extra=None):
    md = {
        "workspace_version": workspace_version(),
        "wheel_version": wheel_version(),
        "oracle": {
            "name": "OpenMOPAC",
            "version": "23.2.5",
            "binary": os.environ.get("OPENMOPAC_BIN", "mopac"),
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": Path(sys.argv[0]).name,
    }
    if extra:
        md.update(extra)
    return md
