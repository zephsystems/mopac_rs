# Notas de Release: MOPAC_RS v0.1.3 (Oficial)

> **Estado**: Lanzamiento Oficial v0.1.3  
> **Fecha**: 2026-10-01  
> **Rama base**: `main`  
> **Tag**: `v0.1.3`

---

## 1. Resumen Ejecutivo de Actualizaciones

Este release consolida la validación físicoquímica experimental del motor frente al estándar de laboratorio del NIST (**CCCBDB SRD 101**), el saneamiento estructural del árbol Git con reducción del 38% del tamaño del repositorio, y la estabilización integral de la infraestructura de integración continua (CI/CD) y publicación multi-arquitectura.

---

## 2. Registro Detallado de Cambios (Changelog de Trabajo)

### Validación Experimental y Benchmarks Rigurosos
- **Benchmark Físicoquímico NIST CCCBDB SRD 101 (Fase Gaseosa)**:
  - Evaluación sistemática sobre **707 moléculas experimentales** de termoquímica y geometría en fase gaseosa.
  - Validación cruzada de los 6 métodos semiempíricos soportados: **PM7**, **PM6**, **AM1**, **PM3**, **RM1** y **MNDO**.
  - **Paridad Bit-Exacta CPU vs GPU Vulkan Float64**:
    - Discrepancia idéntica a **$0.00\text{ kcal/mol}$** en calor de formación ($H_f$) en el 100.0% de las moléculas evaluadas.
    - Eliminación de sesgos o divergencias numéricas entre la ruta vectorizada SIMD (AVX2) de CPU y el pipeline de cómputo en VRAM GDDR6.
  - **Alineación con el Oráculo OpenMOPAC v23.2.5**:
    - Medianas de error residual ($P_{50}$) menores a $0.001\text{ kcal/mol}$ ($0.0001\text{ kcal/mol}$ en AM1, $0.0002$ en RM1/PM3, $0.0003$ en MNDO y $0.0014$ en PM7).
    - Evidencia empírica consolidada en `benchmarks/data/cccbdb_gpu_benchmark_results.json` y reporte de auditoría.

### Saneamiento de Arquitectura y Limpieza del Repositorio
- **Poda de Artefactos Pesados del Índice Git**:
  - Eliminación del árbol rastreado de 7 archivos huérfanos de pruebas y auditorías masivas (`1ao6_mopac_charges.pdb`, `1ao6_mopac.pdbqt`, `2lzm_catalytic_pocket.cube`, `2lzm_mopac_charges.pdb`, `2lzm_mopac.pdbqt`, `audit_results.json`, `massive_audit_results.json`).
  - Reducción del tamaño empaquetado del repositorio de **6.34 MB a 3.96 MB** (~38% de optimización de ancho de banda y almacenamiento de clonación).
- **Blindaje de `.gitignore`**:
  - Exclusión automática de densidades cúbicas Gaussianas (`*.cube`), archivos temporales de visualización macroquímica (`benchmarks/data/*_mopac*`) y cachés de linters (`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`).
  - Preservación estricta de entradas estructurales esenciales (`1ao6.pdb`, `2lzm.pdb`, `aspirin_1000.json`, `molecules_cache.json`).

### Estabilidad de CI/CD y Calidad de Código
- **Corrección de Validación de Esquema en GitHub Actions**:
  - Subsanado el error de especificación en `.github/workflows/release.yml` donde se evaluaba el contexto `secrets` directamente en condiciones condicionales `if:`, lo cual invalidaba el análisis sintáctico de GitHub y generaba falsos fallos en cada push.
  - Manejo seguro y encapsulado de credenciales dentro del entorno Bash del paso de despliegue.
- **Cumplimiento Estricto de Linters y Formato**:
  - Formateo del 100% de la base de código Rust (`mopac_core`, `mopac_gpu`, CLI, tests y bindings de PyO3) con `cargo fmt`.
  - Resolución de advertencias de Clippy (`field_reassign_with_default` e `if_same_then_else` en pruebas de paridad matricial) bajo la bandera `-D warnings`.
- **Ruta Explícita y Renovación de Caché del Badge**:
  - Parametrización a `ci.yml/badge.svg?branch=main` en `README.md` para mitigar la persistencia de cachés residuales del CDN de GitHub Camo.

### Automatización de Despliegue (PyPI y Crates.io)
- **Despliegues Idempotentes en PyPI**:
  - Incorporación del atributo `skip-existing: true` en `release-pypi.yml` para prevenir interrupciones en builds multi-arquitectura cuando una rueda binaria ya ha sido indexada.

---

## 3. Checklist Previo al Corte de Release (v0.1.3)

Antes de crear el tag `v0.1.3` definitivo y publicar los binarios:
- [ ] Decidir si se incluyen optimizaciones adicionales de kernel GPU Vulkan en matrices de dos centros.
- [ ] Actualizar versiones en `Cargo.toml` raíz y subcrates (`version = "0.1.3"`).
- [ ] Actualizar versión en `pyproject.toml` (`version = "0.1.3"`).
- [ ] Mover el contenido de este borrador a la sección oficial `## [0.1.3]` en `CHANGELOG.md`.
- [ ] Ejecutar suite completa local: `cargo test --workspace` y verificar que los 89 tests pasen.
- [ ] Generar tag firmado: `git tag -a v0.1.3 -m "Release v0.1.3: NIST CCCBDB validation, repository sanitization, CI stabilization"`
- [ ] Realizar push del tag: `git push origin v0.1.3` (lo cual activará automáticamente `release.yml` y `release-pypi.yml`).

---

## 4. Anexo: Textos Curados para Actualizar Releases Anteriores en GitHub

Actualmente en la web de GitHub (https://github.com/zephsystems/mopac_rs/releases), las versiones `v0.1.2` y `v0.1.1` contienen solo el enlace comparativo automático por defecto. Pueden enriquecerse editándolas en la interfaz web de GitHub con las siguientes descripciones preparadas:

### Texto Sugerido para Release `v0.1.2` (Tag: `v0.1.2`)
```markdown
## MOPAC_RS v0.1.2: Mol* Macromolecular Interoperability & STO Spatial Cutoffs

### Key Highlights
- **Macromolecular Interoperability Suite**:
  - Affine partial charge mapping to crystallographic B-factor domain [10.0, 90.0] for high-contrast Diverging Blue-White-Red coloring in Mol*.
  - Native PDBx/mmCIF generation with `_atom_site.partial_charge` support.
  - Automated MolViewSpec (MVS v1.0) JSON scene generation for 1-click loading with covalent skeleton and translucent Gaussian Surface.
- **Volumetric Gaussian Cube Performance**:
  - Physical Slater-Type Orbital (STO) spatial cutoff ($r = 12.0\text{ Bohr}$) with Rayon multithreading delivering $20\times - 50\times$ speedup on 3D grid evaluations.
  - Strict Gaussian cube format compliance with explicit $Z$-ray line breaks.
- **Quantum Solver Enhancements**:
  - Warm restart SCF with density matrix reuse (`reuse_density: true`).
  - SOSCF orbital rotation solver for conjugated aromatic systems.
  - COSMO implicit solvation analytical gradients.
- **Ecosystem Integration**:
  - Bidirectional bridges with RDKit and ASE (`MopacASECalculator`).
  - Multi-platform PyPI wheels for Linux, Windows, and macOS (Apple Silicon + Intel).
```

### Texto Sugerido para Release `v0.1.1` (Tag: `v0.1.1`)
```markdown
## MOPAC_RS v0.1.1: Core Quantum Engines & Initial Python Bindings

### Key Highlights
- **Official PyPI Release**: First distribution of `mopac-py` Python bindings.
- **Supported Semi-Empirical Hamiltonians**: MNDO, AM1, PM3, RM1, PM6, PM7 with transition metal support.
- **Dispersion & Non-Covalent Interactions**: Grimme D3-BJ rational damping and H4 hydrogen bonding corrections.
- **Implicit Solvation**: COSMO Boundary Element Method (BEM) surface tessellation.
- **Geometry Optimization**: Quasi-Newton L-BFGS Cartesian coordinate optimizer.
- **Hardware Acceleration**: Vulkan GPU compute backend for two-center two-electron Coulomb repulsion matrices.
```
