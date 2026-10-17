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

    let (ios, iop, iod, eheat): (f64, f64, f64, f64) = if (1..=107).contains(&z) {
        let entry = ISOLATED_ATOM_CONFIG[(z - 1) as usize];
        (entry.1, entry.2, entry.3, entry.4)
    } else {
        (1.0, 0.0, 0.0, 0.0)
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

/// Authentic isolated atom configurations (ios, iop, iod, eheat) for Z=1..107
/// extracted from OpenMOPAC `parameters_C.F90`.
pub const ISOLATED_ATOM_CONFIG: &[(u8, f64, f64, f64, f64)] = &[
    (1, 1.0, 0.0, 0.0, 52.102),
    (2, 2.0, 0.0, 0.0, 0.000),
    (3, 1.0, 0.0, 0.0, 38.410),
    (4, 2.0, 0.0, 0.0, 76.960),
    (5, 2.0, 1.0, 0.0, 135.700),
    (6, 2.0, 2.0, 0.0, 170.890),
    (7, 2.0, 3.0, 0.0, 113.000),
    (8, 2.0, 4.0, 0.0, 59.559),
    (9, 2.0, 5.0, 0.0, 18.890),
    (10, 0.0, 6.0, 0.0, 0.000),
    (11, 1.0, 0.0, 0.0, 25.650),
    (12, 2.0, 0.0, 0.0, 35.000),
    (13, 2.0, 1.0, 0.0, 79.490),
    (14, 2.0, 2.0, 0.0, 108.390),
    (15, 2.0, 3.0, 0.0, 75.570),
    (16, 2.0, 4.0, 0.0, 66.400),
    (17, 2.0, 5.0, 0.0, 28.990),
    (18, 0.0, 6.0, 0.0, 0.000),
    (19, 1.0, 0.0, 0.0, 21.420),
    (20, 2.0, 0.0, 0.0, 42.600),
    (21, 2.0, 0.0, 1.0, 90.300),
    (22, 2.0, 0.0, 2.0, 112.300),
    (23, 2.0, 0.0, 3.0, 122.900),
    (24, 1.0, 0.0, 5.0, 95.000),
    (25, 2.0, 0.0, 5.0, 67.700),
    (26, 2.0, 0.0, 6.0, 99.300),
    (27, 2.0, 0.0, 7.0, 102.400),
    (28, 2.0, 0.0, 8.0, 102.800),
    (29, 1.0, 0.0, 10.0, 80.700),
    (30, 2.0, 0.0, 0.0, 31.170),
    (31, 2.0, 1.0, 0.0, 65.400),
    (32, 2.0, 2.0, 0.0, 89.500),
    (33, 2.0, 3.0, 0.0, 72.300),
    (34, 2.0, 4.0, 0.0, 54.300),
    (35, 2.0, 5.0, 0.0, 26.740),
    (36, 0.0, 6.0, 0.0, 0.000),
    (37, 1.0, 0.0, 0.0, 19.600),
    (38, 2.0, 0.0, 0.0, 39.100),
    (39, 2.0, 0.0, 1.0, 101.500),
    (40, 2.0, 0.0, 2.0, 145.500),
    (41, 1.0, 0.0, 4.0, 172.400),
    (42, 1.0, 0.0, 5.0, 157.300),
    (43, 2.0, 0.0, 5.0, 162.000),
    (44, 1.0, 0.0, 7.0, 155.500),
    (45, 1.0, 0.0, 8.0, 133.000),
    (46, 0.0, 0.0, 10.0, 90.000),
    (47, 1.0, 0.0, 10.0, 68.100),
    (48, 2.0, 0.0, 0.0, 26.720),
    (49, 2.0, 1.0, 0.0, 58.000),
    (50, 2.0, 2.0, 0.0, 72.200),
    (51, 2.0, 3.0, 0.0, 63.200),
    (52, 2.0, 4.0, 0.0, 47.000),
    (53, 2.0, 5.0, 0.0, 25.517),
    (54, 0.0, 6.0, 0.0, 0.000),
    (55, 1.0, 0.0, 0.0, 18.700),
    (56, 2.0, 0.0, 0.0, 42.500),
    (57, 2.0, 0.0, 1.0, 103.011),
    (58, 2.0, 0.0, 1.0, 101.004),
    (59, 2.0, 0.0, 1.0, 84.990),
    (60, 2.0, 0.0, 1.0, 78.298),
    (61, 2.0, 0.0, 1.0, 83.174),
    (62, 2.0, 0.0, 1.0, 49.402),
    (63, 2.0, 0.0, 1.0, 41.898),
    (64, 2.0, 0.0, 1.0, 95.007),
    (65, 2.0, 0.0, 1.0, 92.902),
    (66, 2.0, 0.0, 1.0, 69.407),
    (67, 2.0, 0.0, 1.0, 71.893),
    (68, 2.0, 0.0, 1.0, 75.791),
    (69, 2.0, 0.0, 1.0, 55.500),
    (70, 2.0, 0.0, 1.0, 36.358),
    (71, 2.0, 0.0, 1.0, 102.199),
    (72, 2.0, 0.0, 2.0, 148.000),
    (73, 2.0, 0.0, 3.0, 186.900),
    (74, 1.0, 0.0, 5.0, 203.100),
    (75, 2.0, 0.0, 5.0, 185.000),
    (76, 2.0, 0.0, 6.0, 188.000),
    (77, 2.0, 0.0, 7.0, 160.000),
    (78, 1.0, 0.0, 9.0, 135.200),
    (79, 1.0, 0.0, 10.0, 88.000),
    (80, 2.0, 0.0, 0.0, 14.690),
    (81, 2.0, 1.0, 0.0, 43.550),
    (82, 2.0, 2.0, 0.0, 46.620),
    (83, 2.0, 3.0, 0.0, 50.100),
    (84, 2.0, 4.0, 0.0, 0.000),
    (85, 2.0, 5.0, 0.0, 0.000),
    (86, 0.0, 6.0, 0.0, 0.000),
    (87, 1.0, 0.0, 0.0, 0.000),
    (88, 1.0, 0.0, 0.0, 0.000),
    (89, 2.0, 0.0, 1.0, 0.000),
    (90, 4.0, 0.0, 0.0, 1674.640),
    (91, 2.0, 0.0, 0.0, 0.000),
    (92, 2.0, 0.0, 0.0, 0.000),
    (93, 2.0, 0.0, 0.0, 0.000),
    (94, 2.0, 0.0, 0.0, 0.000),
    (95, 2.0, 0.0, 0.0, 0.000),
    (96, 2.0, 0.0, 0.0, 0.000),
    (97, 2.0, 0.0, 0.0, 0.000),
    (98, 1.0, 0.0, 0.0, 0.000),
    (99, 0.0, 0.0, 0.0, 0.000),
    (100, 3.0, 0.0, 0.0, 0.000),
    (101, -3.0, 0.0, 0.0, 0.000),
    (102, 1.0, 0.0, 0.0, 207.000),
    (103, 2.0, 0.0, 0.0, 0.000),
    (104, 1.0, 0.0, 0.0, 0.000),
    (105, -2.0, 0.0, 0.0, 0.000),
    (106, -1.0, 0.0, 0.0, 0.000),
    (107, 0.0, 0.0, 0.0, 0.000),
];
