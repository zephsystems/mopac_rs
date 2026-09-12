//! Electric Dipole Moment Analysis Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct mathematical translation of OpenMOPAC `dipole.F90`.

use crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS as A0_BOHR;
use crate::constants::standard_atomic_mass;
use crate::integrals::multipoles::DerivedMultipoleParams;
use crate::parameters::ParameterModel;
use crate::types::{AlignedMatrix, MolecularBatch};

/// Constant conversion factor from (elementary charge * Angstrom) to Debye.
/// Derived from speed of light and elementary charge: $c \cdot e \cdot 10^{-10} \times 10^{21} \approx 4.80320425$ D / (e * A).
pub const E_ANGSTROM_TO_DEBYE: f64 = 4.80320425;

/// Comprehensive dipole moment evaluation result.
#[derive(Debug, Clone, PartialEq)]
pub struct DipoleResult {
    /// Point charge contribution (Debye): `[mu_x, mu_y, mu_z, total_magnitude]`.
    pub point_charge: [f64; 4],
    /// Intra-atomic hybridization contribution (Debye): `[mu_x, mu_y, mu_z, total_magnitude]`.
    pub hybridization: [f64; 4],
    /// Total electric dipole moment (Debye): `[mu_x, mu_y, mu_z, total_magnitude]`.
    pub total: [f64; 4],
    /// Net molecular charge $\sum_A q_A$.
    pub net_charge: f64,
    /// Center of mass (Angstroms) used for origin displacement if net charge != 0.
    pub center_of_mass: [f64; 3],
    /// Net atomic charges $q_A = Z_{\text{core}, A} - \sum_{\mu \in A} P_{\mu\mu}$.
    pub atomic_charges: Vec<f64>,
}

/// Compute the canonical electric dipole moment vector and magnitude matching OpenMOPAC `dipole.F90`.
///
/// Under the Zero Differential Overlap (NDDO) approximation, the dipole moment decomposes into:
/// 1. Point charge contribution: $\vec{\mu}_{\text{point}} = \sum_A q_A (\vec{R}_A - \vec{R}_{\text{cm}}) \times 4.80320425$
/// 2. One-center hybridization contribution: arising from $\langle ns | \vec{r} | np \rangle = D_1(Z)$ matrix elements:
///    $\vec{\mu}_{\text{hyb}, \alpha} = - \sum_A 2 \cdot D_{1, A} \cdot a_0 \times 4.80320425 \cdot P_{s, p_\alpha}(A)$
#[allow(clippy::needless_range_loop)]
pub fn compute_dipole_moment<M: ?Sized + ParameterModel>(
    batch: &MolecularBatch,
    model: &M,
    density: &AlignedMatrix<f64>,
) -> DipoleResult {
    let natoms = batch.natoms;

    // 1. Calculate net atomic charges
    let mut charges = Vec::with_capacity(natoms);
    let mut net_charge = 0.0;

    for a in 0..natoms {
        let z = batch.atomic_numbers[a];
        let core_charge = model.get_element(z).map(|p| p.core_charge).unwrap_or(0.0);
        let orb_start = batch.orbital_offsets[a];
        let norbs = batch.basis_types[a].num_orbitals();

        let mut pop = 0.0;
        for o in 0..norbs {
            pop += density.get(orb_start + o, orb_start + o);
        }
        let q = core_charge - pop;
        charges.push(q);
        net_charge += q;
    }

    // 2. Center of mass calculation for translation invariance of ions
    let is_charged = net_charge.abs() > 0.5;
    let mut com = [0.0; 3];
    let mut total_mass = 0.0;

    if is_charged {
        for a in 0..natoms {
            let m = standard_atomic_mass(batch.atomic_numbers[a]);
            total_mass += m;
            com[0] += m * batch.x[a];
            com[1] += m * batch.y[a];
            com[2] += m * batch.z[a];
        }
        if total_mass > 0.0 {
            com[0] /= total_mass;
            com[1] /= total_mass;
            com[2] /= total_mass;
        }
    }

    // 3. Point charge dipole calculation
    let mut pt_dip = [0.0; 4];
    for a in 0..natoms {
        let q = charges[a];
        let rx = batch.x[a] - com[0];
        let ry = batch.y[a] - com[1];
        let rz = batch.z[a] - com[2];

        pt_dip[0] += q * rx * E_ANGSTROM_TO_DEBYE;
        pt_dip[1] += q * ry * E_ANGSTROM_TO_DEBYE;
        pt_dip[2] += q * rz * E_ANGSTROM_TO_DEBYE;
    }
    pt_dip[3] = (pt_dip[0] * pt_dip[0] + pt_dip[1] * pt_dip[1] + pt_dip[2] * pt_dip[2]).sqrt();

    // 4. Intra-atomic hybridization dipole calculation
    let mut hyb_dip = [0.0; 4];
    for a in 0..natoms {
        let z = batch.atomic_numbers[a];
        if z == 1 {
            // Hydrogen has only 1s orbital, no intra-atomic sp hybridization
            continue;
        }

        let p = model
            .get_element(z)
            .unwrap_or_else(|| panic!("Parameters missing for element Z={}", z));
        let mp = DerivedMultipoleParams::from_element(&p);
        let orb_start = batch.orbital_offsets[a];
        let norbs = batch.basis_types[a].num_orbitals();

        if norbs >= 4 {
            // sp hybridization factor: 2.0 * D1 * a0 * 4.80320425
            let hyfsp = 2.0 * mp.dd * A0_BOHR * E_ANGSTROM_TO_DEBYE;

            let s_idx = orb_start;
            let px_idx = orb_start + 1;
            let py_idx = orb_start + 2;
            let pz_idx = orb_start + 3;

            let p_spx = density.get(s_idx, px_idx);
            let p_spy = density.get(s_idx, py_idx);
            let p_spz = density.get(s_idx, pz_idx);

            hyb_dip[0] -= hyfsp * p_spx;
            hyb_dip[1] -= hyfsp * p_spy;
            hyb_dip[2] -= hyfsp * p_spz;
        }
    }
    hyb_dip[3] =
        (hyb_dip[0] * hyb_dip[0] + hyb_dip[1] * hyb_dip[1] + hyb_dip[2] * hyb_dip[2]).sqrt();

    // 5. Total dipole moment vector and magnitude
    let mut tot_dip = [0.0; 4];
    tot_dip[0] = pt_dip[0] + hyb_dip[0];
    tot_dip[1] = pt_dip[1] + hyb_dip[1];
    tot_dip[2] = pt_dip[2] + hyb_dip[2];
    tot_dip[3] =
        (tot_dip[0] * tot_dip[0] + tot_dip[1] * tot_dip[1] + tot_dip[2] * tot_dip[2]).sqrt();

    DipoleResult {
        point_charge: pt_dip,
        hybridization: hyb_dip,
        total: tot_dip,
        net_charge,
        center_of_mass: com,
        atomic_charges: charges,
    }
}
