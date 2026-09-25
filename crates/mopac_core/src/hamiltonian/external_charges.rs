//! Electrostatic Embedding and External Point Charge Coupling for QM/MM.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Provides coupling between quantum solute electrons/cores and classical external point charges
//! (e.g., surrounding protein residues and solvent water in hybrid QM/MM simulations).
//!
//! Mathematical Formulation:
//! 1. One-electron core Hamiltonian modification:
//!    $$H_{\mu\mu}^{\text{ext}} = H_{\mu\mu} - \sum_{k=1}^{N_{\text{ext}}} Q_k \frac{e^2 / 4\pi\varepsilon_0}{\sqrt{R_{Ak}^2 + \rho_A^2}}$$
//!    where \rho_A = (e^2 / 4\pi\varepsilon_0) / (2 \cdot g_{ss}^A) is the Dewar-Klopman charge radius.
//! 2. Core-external charge electrostatic repulsion energy:
//!    $$E_{\text{core-ext}} = \sum_{A=1}^{N_{\text{atoms}}} \sum_{k=1}^{N_{\text{ext}}} Z_A^{\text{core}} Q_k \frac{e^2 / 4\pi\varepsilon_0}{R_{Ak}}$$
//! 3. Analytical nuclear gradients on atom A:
//!    $$\nabla_A E_{\text{ext}} = \sum_{k=1}^{N_{\text{ext}}} Q_k (e^2 / 4\pi\varepsilon_0) (\vec{R}_A - \vec{R}_k) \left[ -\frac{Z_A^{\text{core}}}{R_{Ak}^3} + \frac{\text{Pop}_A}{(R_{Ak}^2 + \rho_A^2)^{3/2}} \right]$$

use crate::constants::codata2018::EV_ANGSTROM_FACTOR;
use crate::parameters::ParameterModel;
use crate::types::{AlignedMatrix, BasisType, MolecularBatch};

/// A classical external point charge for QM/MM electrostatic embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExternalCharge {
    /// Cartesian X coordinate in Angstroms
    pub x: f64,
    /// Cartesian Y coordinate in Angstroms
    pub y: f64,
    /// Cartesian Z coordinate in Angstroms
    pub z: f64,
    /// Partial atomic charge Q in elementary charge units (e)
    pub charge: f64,
}

impl ExternalCharge {
    /// Construct a new external point charge at (x, y, z) with charge `charge`.
    pub fn new(x: f64, y: f64, z: f64, charge: f64) -> Self {
        Self { x, y, z, charge }
    }
}

/// Compute the Dewar-Klopman monopole damping radius \rho_A in Angstroms for atom A.
#[inline(always)]
pub fn klopman_radius(gss: f64) -> f64 {
    if gss > 1.0e-6 {
        EV_ANGSTROM_FACTOR / (2.0 * gss)
    } else {
        0.50
    }
}

/// Apply external point charge electrostatic potential to the diagonal elements of H_core.
pub fn apply_external_charges_to_hcore(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    external_charges: &[ExternalCharge],
    h_core: &mut AlignedMatrix<f64>,
) {
    if external_charges.is_empty() {
        return;
    }

    let natoms = batch.natoms;
    for i in 0..natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };

        let rho = klopman_radius(p_a.gss);
        let rho_sq = rho * rho;

        let xi = batch.x[i];
        let yi = batch.y[i];
        let zi = batch.z[i];

        let mut v_ext = 0.0;
        for ext in external_charges {
            let dx = xi - ext.x;
            let dy = yi - ext.y;
            let dz = zi - ext.z;
            let r2 = dx * dx + dy * dy + dz * dz;
            let denom = (r2 + rho_sq).sqrt();
            v_ext -= ext.charge * EV_ANGSTROM_FACTOR / denom;
        }

        let orb_start = batch.orbital_offsets[i];
        let norbs_atom = match batch.basis_types[i] {
            BasisType::S => 1,
            BasisType::SP => 4,
            BasisType::SPD => 9,
        };

        for mu in 0..norbs_atom {
            let idx = orb_start + mu;
            let old_val = h_core.get(idx, idx);
            h_core.set(idx, idx, old_val + v_ext);
        }
    }
}

/// Compute the core-external charge electrostatic repulsion energy in eV.
pub fn compute_external_charges_core_energy(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    external_charges: &[ExternalCharge],
) -> f64 {
    if external_charges.is_empty() {
        return 0.0;
    }

    let natoms = batch.natoms;
    let mut e_core_ext = 0.0;

    for i in 0..natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };
        let core_charge = p_a.core_charge;

        let xi = batch.x[i];
        let yi = batch.y[i];
        let zi = batch.z[i];

        for ext in external_charges {
            let dx = xi - ext.x;
            let dy = yi - ext.y;
            let dz = zi - ext.z;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r > 1.0e-8 {
                e_core_ext += core_charge * ext.charge * EV_ANGSTROM_FACTOR / r;
            }
        }
    }

    e_core_ext
}

/// Compute analytical nuclear gradients on quantum solute atoms due to external point charges in eV/A.
#[allow(clippy::needless_range_loop)]
pub fn compute_external_charges_gradients(
    batch: &MolecularBatch,
    density: &AlignedMatrix<f64>,
    model: &dyn ParameterModel,
    external_charges: &[ExternalCharge],
    gradients: &mut [[f64; 3]],
) {
    if external_charges.is_empty() {
        return;
    }

    let natoms = batch.natoms;
    for i in 0..natoms {
        let za = batch.atomic_numbers[i];
        let p_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };
        let core_charge = p_a.core_charge;

        let rho = klopman_radius(p_a.gss);
        let rho_sq = rho * rho;

        let orb_start = batch.orbital_offsets[i];
        let norbs_atom = match batch.basis_types[i] {
            BasisType::S => 1,
            BasisType::SP => 4,
            BasisType::SPD => 9,
        };

        // Compute atomic population: Pop_A = sum_{\mu \in A} P_{\mu\mu}
        let mut pop_a = 0.0;
        for mu in 0..norbs_atom {
            let idx = orb_start + mu;
            pop_a += density.get(idx, idx);
        }

        let xi = batch.x[i];
        let yi = batch.y[i];
        let zi = batch.z[i];

        for ext in external_charges {
            let dx = xi - ext.x;
            let dy = yi - ext.y;
            let dz = zi - ext.z;
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            if r < 1.0e-8 {
                continue;
            }

            let denom_core = r2 * r; // r^3
            let denom_elec = (r2 + rho_sq).powf(1.5); // (r^2 + \rho^2)^{3/2}

            let factor =
                ext.charge * EV_ANGSTROM_FACTOR * (-core_charge / denom_core + pop_a / denom_elec);

            gradients[i][0] += factor * dx;
            gradients[i][1] += factor * dy;
            gradients[i][2] += factor * dz;
        }
    }
}
