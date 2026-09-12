//! Canonical Heat of Formation and Isolated Atomic Energy Calculation.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Exact mathematical formula and experimental isolated heats \Delta H_{f,\text{atom}}
//! matching OpenMOPAC v23.2.5 (`compfg.F90`, `iter.F90`, `parameters_C.F90`).

use crate::constants::codata2018::EV_TO_KCAL_MOL;
use crate::parameters::ParameterModel;

/// Compute isolated atomic energy $E_{\text{isol}}$ (eV) and isolated atom heat of formation (kcal/mol).
pub fn get_isolated_atom_energy_and_heat(z: u8, model: &dyn ParameterModel) -> (f64, f64) {
    let p = match model.get_element(z) {
        Some(param) => param,
        None => return (0.0, 0.0),
    };

    let (ios, iop, iod, eheat): (f64, f64, f64, f64) = match z {
        1 => (1.0, 0.0, 0.0, 52.102),
        2 => (2.0, 0.0, 0.0, 0.0),
        3 => (1.0, 0.0, 0.0, 38.410),
        4 => (2.0, 0.0, 0.0, 76.960),
        5 => (2.0, 1.0, 0.0, 135.700),
        6 => (2.0, 2.0, 0.0, 170.890),
        7 => (2.0, 3.0, 0.0, 113.000),
        8 => (2.0, 4.0, 0.0, 59.559),
        9 => (2.0, 5.0, 0.0, 18.890),
        10 => (2.0, 6.0, 0.0, 0.0),
        11 => (1.0, 0.0, 0.0, 25.650),
        12 => (2.0, 0.0, 0.0, 35.000),
        13 => (2.0, 1.0, 0.0, 79.490),
        14 => (2.0, 2.0, 0.0, 108.390),
        15 => (2.0, 3.0, 0.0, 75.570),
        16 => (2.0, 4.0, 0.0, 66.400),
        17 => (2.0, 5.0, 0.0, 28.990),
        18 => (2.0, 6.0, 0.0, 0.0),
        19 => (1.0, 0.0, 0.0, 21.420),
        20 => (2.0, 0.0, 0.0, 42.600),
        21 => (2.0, 0.0, 1.0, 90.300),
        22 => (2.0, 0.0, 2.0, 112.300),
        23 => (2.0, 0.0, 3.0, 122.900),
        24 => (1.0, 0.0, 5.0, 95.000),
        25 => (2.0, 0.0, 5.0, 67.700),
        26 => (2.0, 0.0, 6.0, 99.300),
        27 => (2.0, 0.0, 7.0, 102.400),
        28 => (2.0, 0.0, 8.0, 102.800),
        29 => (1.0, 0.0, 10.0, 80.700),
        30 => (2.0, 0.0, 0.0, 31.170),
        31 => (2.0, 1.0, 0.0, 65.400),
        32 => (2.0, 2.0, 0.0, 89.500),
        33 => (2.0, 3.0, 0.0, 72.300),
        34 => (2.0, 4.0, 0.0, 54.300),
        35 => (2.0, 5.0, 0.0, 26.740),
        36 => (2.0, 6.0, 0.0, 0.0),
        53 => (2.0, 5.0, 0.0, 25.517),
        _ => (1.0, 0.0, 0.0, 0.0),
    };

    if z == 1 {
        return (p.uss, eheat);
    }

    let k: f64 = iop;
    let l: f64 = k.min(6.0 - k);
    let gssc: f64 = (ios - 1.0).max(0.0);
    let gspc: f64 = ios * k;
    let gp2c: f64 = (k * (k - 1.0)) / 2.0 + 0.5 * (l * (l - 1.0)) / 2.0;
    let gppc: f64 = -0.5 * (l * (l - 1.0)) / 2.0;
    let hspc: f64 = -k * ios * 0.5;

    let eisol = p.uss * ios
        + p.upp * iop
        + p.udd * iod
        + p.gss * gssc
        + p.gpp * gppc
        + p.gsp * gspc
        + p.gp2 * gp2c
        + p.hsp * hspc;

    (eisol, eheat)
}

/// Compute molecular binding energy (in eV) and standard heat of formation $\Delta H_f^\circ$ (in kcal/mol).
///
/// $\Delta H_f = E_{\text{bind}} \times 23.060548 \text{ kcal/(mol}\cdot\text{eV)} + \sum_A \Delta H_{f,\text{atom}}(A) + E_{\text{non-covalent}}$
pub fn compute_heat_of_formation(
    total_energy_ev: f64,
    atomic_numbers: &[u8],
    model: &dyn ParameterModel,
    non_covalent_kcal: f64,
) -> (f64, f64) {
    let mut sum_eisol = 0.0;
    let mut sum_eheat = 0.0;

    for &z in atomic_numbers {
        let (eisol, eheat) = get_isolated_atom_energy_and_heat(z, model);
        sum_eisol += eisol;
        sum_eheat += eheat;
    }

    let binding_energy_ev = total_energy_ev - sum_eisol;
    let heat_of_formation_kcal = binding_energy_ev * EV_TO_KCAL_MOL + sum_eheat + non_covalent_kcal;

    (binding_energy_ev, heat_of_formation_kcal)
}

/// Empirical C#C triple bond heat of formation correction for PM6, PM7, PM8.
/// Matching OpenMOPAC `set_up_dentate.F90` lines 214-262.
pub fn compute_c_triple_bond_c_correction(batch: &crate::types::MolecularBatch) -> f64 {
    const R_MIN: f64 = 1.21;
    const R_MAX: f64 = 1.33;
    const PARAM1: f64 = -5.0;
    const PARAM2: f64 = 25.0;

    let mut sum = 0.0;
    for i in 0..batch.natoms {
        if batch.atomic_numbers[i] != 6 {
            continue;
        }
        for j in 0..i {
            if batch.atomic_numbers[j] != 6 {
                continue;
            }
            let r = batch.distance(i, j);
            if r < R_MIN {
                sum += 1.0;
            } else if r < R_MAX {
                let x = (r - R_MIN) / (R_MAX - R_MIN);
                let x3 = x * x * x;
                let x4 = x3 * x;
                let x5 = x4 * x;
                let x6 = x5 * x;
                sum += 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5
                    + (PARAM1 + x * PARAM2) * (x3 - 3.0 * x4 + 3.0 * x5 - x6);
            }
        }
    }
    sum * 12.0
}
