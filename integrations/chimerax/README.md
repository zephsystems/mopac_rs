# ChimeraX MOPAC_RS Extension

Interactive UCSF ChimeraX plugin for **MOPAC_RS**, enabling high-performance semi-empirical quantum chemical calculations, AM1-BCC partial atomic charge calculation, L-BFGS geometry optimization, and Molecular Electrostatic Potential (MEP) surface visualization directly within the ChimeraX viewport.

## Installation

Within the ChimeraX Python environment:

```bash
# Install mopac_py first
pip install mopac_py

# Install ChimeraX bundle
cd integrations/chimerax
pip install .
```

## Supported Commands

### 1. Single-Point Calculation
```text
mopac calculate #1 method PM6 dispersion D3-BJ eps 78.4
```
Calculates total energy, heat of formation, dipole moment, and assigns Mulliken charges directly to atoms.

### 2. AM1-BCC Charge Assignment & MEP Coloring
```text
mopac bcc #1
color byattribute charge #1
```
Computes AM1-BCC atomic partial charges and enables immediate surface electrostatic coloring.

### 3. Geometry Optimization
```text
mopac optimize #1 method PM6 max_cycles 100
```
Runs Cartesian L-BFGS geometry minimization and updates atom coordinates in the 3D viewport.

### 4. MOZYME Macromolecular SCF
```text
mopac mozyme #1 method PM6
```
Runs linear-scaling $O(N)$ localized molecular orbital calculation on large proteins or nucleic acids.

## License

Licensed under the Apache License, Version 2.0 (Apache-2.0).
