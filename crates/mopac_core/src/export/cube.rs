//! Gaussian Cube (.cube) Volumetric Serializer for Molstar Marching Cubes.
//!
//! Generates 3D regular scalar field grids of Molecular Orbitals (HOMO, LUMO)
//! and total electron density $\rho(\vec{r})$ evaluated from Slater-type atomic orbitals (STOs).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").

use crate::parameters::ParameterModel;
use crate::types::{AlignedMatrix, BasisType, MolecularBatch};
use std::f64::consts::PI;

/// Conversion factor: 1 Angstrom = 1.8897261246 Bohr (atomic units).
pub const ANGSTROM_TO_BOHR: f64 = 1.8897261246;
/// Conversion factor: 1 Bohr = 0.5291772109 Angstroms.
pub const BOHR_TO_ANGSTROM: f64 = 0.5291772109;

/// 3D Grid configuration for Gaussian Cube generation.
#[derive(Debug, Clone)]
pub struct CubeGridConfig {
    /// Spatial padding surrounding the molecule in Angstroms (default: 3.5 A)
    pub padding_angstrom: f64,
    /// Grid voxel spacing in Angstroms (default: 0.25 A)
    pub resolution_angstrom: f64,
}

impl Default for CubeGridConfig {
    fn default() -> Self {
        Self {
            padding_angstrom: 3.5,
            resolution_angstrom: 0.25,
        }
    }
}

/// Evaluates normalized Slater-Type Atomic Orbitals (STOs) at local displacement vector d = (dx, dy, dz).
///
/// Returns orbital amplitudes [s, px, py, pz] in atomic units.
#[inline(always)]
fn evaluate_sto_sp(z: u8, dx: f64, dy: f64, dz: f64, zs: f64, zp: f64) -> [f64; 4] {
    let r2 = dx * dx + dy * dy + dz * dz;
    let r = r2.sqrt();

    if z == 1 {
        // H: 1s orbital, n=1
        // N = sqrt(zeta^3 / pi)
        let norm_s = (zs * zs * zs / PI).sqrt();
        let val_s = norm_s * (-zs * r).exp();
        [val_s, 0.0, 0.0, 0.0]
    } else if z <= 10 {
        // Row 2 (He..Ne): 2s, 2p, n=2
        // N_s = sqrt(zeta^5 / (3 * pi)), chi_2s = N_s * r * exp(-zeta * r)
        // N_p = sqrt(zeta^5 / pi), chi_2p = N_p * coord * exp(-zeta * r)
        let exp_s = (-zs * r).exp();
        let exp_p = (-zp * r).exp();
        let norm_s = (zs.powi(5) / (3.0 * PI)).sqrt();
        let norm_p = (zp.powi(5) / PI).sqrt();

        let val_s = norm_s * r * exp_s;
        let val_px = norm_p * dx * exp_p;
        let val_py = norm_p * dy * exp_p;
        let val_pz = norm_p * dz * exp_p;
        [val_s, val_px, val_py, val_pz]
    } else {
        // Row 3 (Na..Ar): 3s, 3p, n=3
        // N_s = sqrt(2 * zeta^7 / (45 * pi)), chi_3s = N_s * r^2 * exp(-zeta * r)
        // N_p = sqrt(2 * zeta^7 / (15 * pi)), chi_3p = N_p * coord * r * exp(-zeta * r)
        let exp_s = (-zs * r).exp();
        let exp_p = (-zp * r).exp();
        let norm_s = (2.0 * zs.powi(7) / (45.0 * PI)).sqrt();
        let norm_p = (2.0 * zp.powi(7) / (15.0 * PI)).sqrt();

        let val_s = norm_s * r2 * exp_s;
        let val_px = norm_p * dx * r * exp_p;
        let val_py = norm_p * dy * r * exp_p;
        let val_pz = norm_p * dz * r * exp_p;
        [val_s, val_px, val_py, val_pz]
    }
}

/// Evaluates molecular orbital amplitude $\psi_i(\vec{r}) = \sum_\mu C_{\mu i} \phi_\mu(\vec{r})$.
pub fn evaluate_molecular_orbital_at_point(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    mo_coefficients: &[f64],
    point_angstrom: [f64; 3],
) -> f64 {
    let px = point_angstrom[0] * ANGSTROM_TO_BOHR;
    let py = point_angstrom[1] * ANGSTROM_TO_BOHR;
    let pz = point_angstrom[2] * ANGSTROM_TO_BOHR;

    let mut psi = 0.0;

    for a in 0..batch.natoms {
        let z = batch.atomic_numbers[a];
        let p = model.get_element(z).expect("Element params missing in model");
        let ax = batch.x[a] * ANGSTROM_TO_BOHR;
        let ay = batch.y[a] * ANGSTROM_TO_BOHR;
        let az = batch.z[a] * ANGSTROM_TO_BOHR;

        let dx = px - ax;
        let dy = py - ay;
        let dz = pz - az;

        let sto = evaluate_sto_sp(z, dx, dy, dz, p.zs, p.zp);
        let offset = batch.orbital_offsets[a];

        match batch.basis_types[a] {
            BasisType::S => {
                psi += mo_coefficients[offset] * sto[0];
            }
            BasisType::SP => {
                psi += mo_coefficients[offset] * sto[0];
                psi += mo_coefficients[offset + 1] * sto[1];
                psi += mo_coefficients[offset + 2] * sto[2];
                psi += mo_coefficients[offset + 3] * sto[3];
            }
            BasisType::SPD => {
                // S and P contributions
                psi += mo_coefficients[offset] * sto[0];
                psi += mo_coefficients[offset + 1] * sto[1];
                psi += mo_coefficients[offset + 2] * sto[2];
                psi += mo_coefficients[offset + 3] * sto[3];
            }
        }
    }

    psi
}

/// Evaluates total electron density $\rho(\vec{r}) = \sum_{\mu,\nu} P_{\mu\nu} \phi_\mu(\vec{r}) \phi_\nu(\vec{r})$.
pub fn evaluate_density_at_point(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    density_matrix: &AlignedMatrix<f64>,
    point_angstrom: [f64; 3],
) -> f64 {
    let px = point_angstrom[0] * ANGSTROM_TO_BOHR;
    let py = point_angstrom[1] * ANGSTROM_TO_BOHR;
    let pz = point_angstrom[2] * ANGSTROM_TO_BOHR;

    let mut basis_vals = vec![0.0f64; batch.norbs];

    for a in 0..batch.natoms {
        let z = batch.atomic_numbers[a];
        let p = model.get_element(z).expect("Element params missing in model");
        let ax = batch.x[a] * ANGSTROM_TO_BOHR;
        let ay = batch.y[a] * ANGSTROM_TO_BOHR;
        let az = batch.z[a] * ANGSTROM_TO_BOHR;

        let dx = px - ax;
        let dy = py - ay;
        let dz = pz - az;

        let sto = evaluate_sto_sp(z, dx, dy, dz, p.zs, p.zp);
        let offset = batch.orbital_offsets[a];

        match batch.basis_types[a] {
            BasisType::S => {
                basis_vals[offset] = sto[0];
            }
            BasisType::SP | BasisType::SPD => {
                basis_vals[offset] = sto[0];
                basis_vals[offset + 1] = sto[1];
                basis_vals[offset + 2] = sto[2];
                basis_vals[offset + 3] = sto[3];
            }
        }
    }

    let mut rho = 0.0;
    for mu in 0..batch.norbs {
        let v_mu = basis_vals[mu];
        if v_mu.abs() > 1e-12 {
            for nu in 0..batch.norbs {
                let v_nu = basis_vals[nu];
                if v_nu.abs() > 1e-12 {
                    rho += density_matrix.get(mu, nu) * v_mu * v_nu;
                }
            }
        }
    }

    rho
}

/// Generates a Gaussian Cube (.cube) formatted string for a specific molecular orbital.
///
/// Molstar directly reads this file and applies Marching Cubes to render
/// smooth 3D positive and negative orbital lobes.
pub fn generate_molecular_orbital_cube(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    mo_coefficients: &[f64],
    orbital_index: usize,
    orbital_energy_ev: f64,
    config: &CubeGridConfig,
) -> String {
    // 1. Calculate bounding box in Angstroms
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;

    for a in 0..batch.natoms {
        min_x = min_x.min(batch.x[a]);
        max_x = max_x.max(batch.x[a]);
        min_y = min_y.min(batch.y[a]);
        max_y = max_y.max(batch.y[a]);
        min_z = min_z.min(batch.z[a]);
        max_z = max_z.max(batch.z[a]);
    }

    let pad = config.padding_angstrom;
    let step_a = config.resolution_angstrom;

    min_x -= pad;
    max_x += pad;
    min_y -= pad;
    max_y += pad;
    min_z -= pad;
    max_z += pad;

    let nx = ((max_x - min_x) / step_a).ceil() as usize + 1;
    let ny = ((max_y - min_y) / step_a).ceil() as usize + 1;
    let nz = ((max_z - min_z) / step_a).ceil() as usize + 1;

    // Convert origin and step to Bohr (standard Gaussian Cube unit)
    let origin_bohr = [
        min_x * ANGSTROM_TO_BOHR,
        min_y * ANGSTROM_TO_BOHR,
        min_z * ANGSTROM_TO_BOHR,
    ];
    let step_bohr = step_a * ANGSTROM_TO_BOHR;

    let mut out = String::with_capacity(1024 + nx * ny * nz * 14);

    // Line 1 & 2: Header comments
    out.push_str("MOPAC_RS Molecular Orbital Cube File\n");
    out.push_str(&format!(
        "Orbital {} Energy = {:.4} eV\n",
        orbital_index, orbital_energy_ev
    ));

    // Line 3: natoms and origin in Bohr (signed natoms < 0 indicates extra MO record line)
    out.push_str(&format!(
        "{:5} {:12.6} {:12.6} {:12.6}\n",
        -(batch.natoms as i64),
        origin_bohr[0],
        origin_bohr[1],
        origin_bohr[2]
    ));

    // Lines 4-6: Voxel counts and step vectors in Bohr
    out.push_str(&format!("{:5} {:12.6} {:12.6} {:12.6}\n", nx, step_bohr, 0.0, 0.0));
    out.push_str(&format!("{:5} {:12.6} {:12.6} {:12.6}\n", ny, 0.0, step_bohr, 0.0));
    out.push_str(&format!("{:5} {:12.6} {:12.6} {:12.6}\n", nz, 0.0, 0.0, step_bohr));

    // Atom lines: Z, core charge, X, Y, Z in Bohr
    for a in 0..batch.natoms {
        let z = batch.atomic_numbers[a];
        let p = model.get_element(z).expect("Element params missing in model");
        let ax = batch.x[a] * ANGSTROM_TO_BOHR;
        let ay = batch.y[a] * ANGSTROM_TO_BOHR;
        let az = batch.z[a] * ANGSTROM_TO_BOHR;
        out.push_str(&format!(
            "{:5} {:12.6} {:12.6} {:12.6} {:12.6}\n",
            z, p.core_charge, ax, ay, az
        ));
    }

    // MO Header: number of orbitals and orbital index
    out.push_str(&format!("{:5} {:5}\n", 1, orbital_index));

    // Grid scalar values: loop X -> Y -> Z with 6 values per line
    let mut val_count = 0;
    for ix in 0..nx {
        let px = min_x + ix as f64 * step_a;
        for iy in 0..ny {
            let py = min_y + iy as f64 * step_a;
            for iz in 0..nz {
                let pz = min_z + iz as f64 * step_a;
                let val = evaluate_molecular_orbital_at_point(
                    batch,
                    model,
                    mo_coefficients,
                    [px, py, pz],
                );
                out.push_str(&format!(" {:12.5E}", val));
                val_count += 1;
                if val_count % 6 == 0 {
                    out.push('\n');
                }
            }
        }
    }
    if val_count % 6 != 0 {
        out.push('\n');
    }

    out
}

/// Generates a Gaussian Cube (.cube) formatted string for total electron density.
pub fn generate_density_cube(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    density_matrix: &AlignedMatrix<f64>,
    config: &CubeGridConfig,
) -> String {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;

    for a in 0..batch.natoms {
        min_x = min_x.min(batch.x[a]);
        max_x = max_x.max(batch.x[a]);
        min_y = min_y.min(batch.y[a]);
        max_y = max_y.max(batch.y[a]);
        min_z = min_z.min(batch.z[a]);
        max_z = max_z.max(batch.z[a]);
    }

    let pad = config.padding_angstrom;
    let step_a = config.resolution_angstrom;

    min_x -= pad;
    max_x += pad;
    min_y -= pad;
    max_y += pad;
    min_z -= pad;
    max_z += pad;

    let nx = ((max_x - min_x) / step_a).ceil() as usize + 1;
    let ny = ((max_y - min_y) / step_a).ceil() as usize + 1;
    let nz = ((max_z - min_z) / step_a).ceil() as usize + 1;

    let origin_bohr = [
        min_x * ANGSTROM_TO_BOHR,
        min_y * ANGSTROM_TO_BOHR,
        min_z * ANGSTROM_TO_BOHR,
    ];
    let step_bohr = step_a * ANGSTROM_TO_BOHR;

    let mut out = String::with_capacity(1024 + nx * ny * nz * 14);

    out.push_str("MOPAC_RS Total Electron Density Cube File\n");
    out.push_str("rho(r) Total SCF Valence Density\n");

    out.push_str(&format!(
        "{:5} {:12.6} {:12.6} {:12.6}\n",
        batch.natoms, origin_bohr[0], origin_bohr[1], origin_bohr[2]
    ));

    out.push_str(&format!("{:5} {:12.6} {:12.6} {:12.6}\n", nx, step_bohr, 0.0, 0.0));
    out.push_str(&format!("{:5} {:12.6} {:12.6} {:12.6}\n", ny, 0.0, step_bohr, 0.0));
    out.push_str(&format!("{:5} {:12.6} {:12.6} {:12.6}\n", nz, 0.0, 0.0, step_bohr));

    for a in 0..batch.natoms {
        let z = batch.atomic_numbers[a];
        let p = model.get_element(z).expect("Element params missing in model");
        let ax = batch.x[a] * ANGSTROM_TO_BOHR;
        let ay = batch.y[a] * ANGSTROM_TO_BOHR;
        let az = batch.z[a] * ANGSTROM_TO_BOHR;
        out.push_str(&format!(
            "{:5} {:12.6} {:12.6} {:12.6} {:12.6}\n",
            z, p.core_charge, ax, ay, az
        ));
    }

    let mut val_count = 0;
    for ix in 0..nx {
        let px = min_x + ix as f64 * step_a;
        for iy in 0..ny {
            let py = min_y + iy as f64 * step_a;
            for iz in 0..nz {
                let pz = min_z + iz as f64 * step_a;
                let val = evaluate_density_at_point(
                    batch,
                    model,
                    density_matrix,
                    [px, py, pz],
                );
                out.push_str(&format!(" {:12.5E}", val));
                val_count += 1;
                if val_count % 6 == 0 {
                    out.push('\n');
                }
            }
        }
    }
    if val_count % 6 != 0 {
        out.push('\n');
    }

    out
}
