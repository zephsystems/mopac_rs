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

    let (ios, iop, eheat): (f64, f64, f64) = match z {
        1 => (1.0, 0.0, 52.102),
        5 => (2.0, 1.0, 135.700),
        6 => (2.0, 2.0, 170.890),
        7 => (2.0, 3.0, 113.000),
        8 => (2.0, 4.0, 59.559),
        9 => (2.0, 5.0, 18.890),
        14 => (2.0, 2.0, 108.390),
        15 => (2.0, 3.0, 75.570),
        16 => (2.0, 4.0, 66.400),
        17 => (2.0, 5.0, 28.990),
        35 => (2.0, 5.0, 26.740),
        53 => (2.0, 5.0, 25.517),
        _ => (1.0, 0.0, 0.0),
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
