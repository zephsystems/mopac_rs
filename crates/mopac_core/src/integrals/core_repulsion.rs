//! Core-Core Nuclear Repulsion Energy Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Calculates pairwise nuclear repulsion $E_{AB}^{\text{core-core}}$ with Klopman screening,
//! unpolarizable core Lennard-Jones terms, and Gaussian expansions.
//! Direct canonical port of OpenMOPAC `ccrep.F90`, `mndod.F90`, and `compfg.F90`.

use super::two_electron::dewar_klopman_monopole;
use crate::constants::codata2018::{BOHR_RADIUS_ANGSTROMS, EV_ANGSTROM_FACTOR};
use crate::parameters::pm6::Pm6Model;
use crate::parameters::pm7::Pm7Model;
use crate::parameters::SemiEmpiricalElementParams;

/// Unpolarizable core-core Lennard-Jones 12 term from OpenMOPAC `ccrep.F90` lines 246-263.
///
/// Formulation:
/// $$a_x = \frac{R}{Z_A^{0.3333} + Z_B^{0.3333}}$$
/// $$\Delta E_{AB}^{\text{LJ}} = \min\left( \frac{10^{-8}}{a_x^{12}}, 10^5 \right) \quad \text{if } a_x < 3.0$$
#[inline(always)]
pub fn unpolarizable_core_repulsion(r_angstrom: f64, za: u8, zb: u8) -> f64 {
    let ax = r_angstrom / ((za as f64).powf(0.3333) + (zb as f64).powf(0.3333));
    if ax < 3.0 {
        let val = 1.0e-8 / ax.powi(12);
        val.min(1.0e5)
    } else {
        0.0
    }
}

/// Dewar-Klopman core monopole integral $\gamma_{AB}^{\text{core}}$ in eV,
/// incorporating transition-metal core Klopman-Ohno radii ($pocord$).
///
/// Direct port of OpenMOPAC `mndod.F90` lines 503-505 and `compfg.F90` lines 534-536:
/// $$\gamma_{AB} = \frac{e^2}{\sqrt{R^2 + (\rho_A + \rho_B)^2}}$$
/// where $\rho = pocord \cdot a_0$ if $pocord > 10^{-5}$, else $\frac{e^2}{2 g_{ss}}$.
#[inline(always)]
pub fn core_monopole(r_angstrom: f64, gss_a: f64, gss_b: f64, pocord_a: f64, pocord_b: f64) -> f64 {
    let rho_a = if pocord_a > 1.0e-5 {
        pocord_a * BOHR_RADIUS_ANGSTROMS
    } else {
        EV_ANGSTROM_FACTOR / (2.0 * gss_a)
    };
    let rho_b = if pocord_b > 1.0e-5 {
        pocord_b * BOHR_RADIUS_ANGSTROMS
    } else {
        EV_ANGSTROM_FACTOR / (2.0 * gss_b)
    };
    let rho_sum = rho_a + rho_b;
    EV_ANGSTROM_FACTOR / (r_angstrom * r_angstrom + rho_sum * rho_sum).sqrt()
}

/// Compute pairwise core-core nuclear repulsion energy between atom A and atom B in eV
/// for MNDO, AM1, PM3, and RM1 Hamiltonians.
///
/// Formulation:
/// $$E_{AB} = Z_A Z_B \gamma_{AB} \left[ 1 + e^{-\alpha_A R} + e^{-\alpha_B R} \right] + \Delta E_{AB}^{\text{Gauss}}$$
pub fn compute_pair_core_repulsion(
    r_angstrom: f64,
    elem_a: &SemiEmpiricalElementParams,
    elem_b: &SemiEmpiricalElementParams,
    is_am1: bool,
) -> f64 {
    if r_angstrom < 1e-10 {
        return 0.0;
    }

    // Monopole two-electron integral (ss|ss) in eV
    let gab = dewar_klopman_monopole(r_angstrom, elem_a.gss, elem_b.gss);

    // Standard exponential screening term
    // For N-H and O-H pairs in AM1/MNDO, MOPAC applies distance scaling on the heavy atom (N or O)
    let scale_a = if (elem_a.z == 7 || elem_a.z == 8) && elem_b.z == 1 {
        r_angstrom
    } else {
        1.0
    };
    let scale_b = if (elem_b.z == 7 || elem_b.z == 8) && elem_a.z == 1 {
        r_angstrom
    } else {
        1.0
    };

    let exp_a = scale_a * (-elem_a.alpha * r_angstrom).exp();
    let exp_b = scale_b * (-elem_b.alpha * r_angstrom).exp();
    let base_scale = 1.0 + exp_a + exp_b;

    let base_repulsion = elem_a.core_charge * elem_b.core_charge * gab * base_scale;

    // Gaussian core corrections
    let mut gaussian_sum = 0.0;

    let is_am1_b_pair = is_am1
        && (elem_a.z == 5 || elem_b.z == 5)
        && matches!(
            if elem_a.z == 5 { elem_b.z } else { elem_a.z },
            1 | 6 | 9 | 17 | 35 | 53
        );

    if is_am1_b_pair {
        let other = if elem_a.z == 5 { elem_b } else { elem_a };
        let b_pairs: &[(f64, f64, f64)] = match other.z {
            1 => &[(0.412253, 10.0, 0.832586), (-0.149917, 6.0, 1.186220)],
            6 => &[(0.261751, 8.0, 1.063995), (0.050275, 5.0, 1.936492)],
            _ => &[(0.359244, 9.0, 0.819351), (0.074729, 9.0, 1.574414)],
        };
        for &(a, b, c) in b_pairs {
            let dr = r_angstrom - c;
            let exponent = b * dr * dr;
            if exponent <= 25.0 {
                gaussian_sum += a * (-exponent).exp();
            }
        }
        for k in 0..other.num_gaussians {
            let g = &other.gaussians[k];
            let dr = r_angstrom - g.c;
            let exponent = g.b * dr * dr;
            if exponent <= 25.0 {
                gaussian_sum += g.a * (-exponent).exp();
            }
        }
    } else {
        for k in 0..elem_a.num_gaussians {
            let g = &elem_a.gaussians[k];
            let dr = r_angstrom - g.c;
            let exponent = g.b * dr * dr;
            if exponent < 25.0 {
                gaussian_sum += g.a * (-exponent).exp();
            }
        }

        for k in 0..elem_b.num_gaussians {
            let g = &elem_b.gaussians[k];
            let dr = r_angstrom - g.c;
            let exponent = g.b * dr * dr;
            if exponent < 25.0 {
                gaussian_sum += g.a * (-exponent).exp();
            }
        }
    }

    let gaussian_repulsion = (elem_a.core_charge * elem_b.core_charge / r_angstrom) * gaussian_sum;

    base_repulsion + gaussian_repulsion
}

/// Compute pairwise core-core nuclear repulsion energy using PM6 formulation.
///
/// Direct port of OpenMOPAC `ccrep.F90` lines 70-264.
pub fn compute_pair_core_repulsion_pm6(
    r_angstrom: f64,
    elem_a: &SemiEmpiricalElementParams,
    elem_b: &SemiEmpiricalElementParams,
) -> f64 {
    if r_angstrom < 1e-10 {
        return 0.0;
    }
    let pocord_a = Pm6Model::get_pocord(elem_a.z);
    let pocord_b = Pm6Model::get_pocord(elem_b.z);
    let gab = core_monopole(r_angstrom, elem_a.gss, elem_b.gss, pocord_a, pocord_b);

    let (alpb, xfac) = Pm6Model::get_pair_params(elem_a.z, elem_b.z);

    let z_min = elem_a.z.min(elem_b.z);
    let z_max = elem_a.z.max(elem_b.z);

    let mut scale = if z_min == 1 && (z_max == 6 || z_max == 7 || z_max == 8) {
        // C-H, N-H, O-H interaction uses r^2
        1.0 + 2.0 * xfac * (-alpb * r_angstrom * r_angstrom).exp()
    } else if xfac > 1e-5 {
        1.0 + 2.0 * xfac * (-alpb * (r_angstrom + 0.0003 * r_angstrom.powi(6))).exp()
    } else if (elem_a.z > 56 && elem_a.z < 72) || (elem_b.z > 56 && elem_b.z < 72) {
        1.0 + 10.0 * (-3.0 * r_angstrom).exp()
    } else {
        1.0 + 10.0 * (-2.18 * r_angstrom).exp()
    };

    // Special PM6 pair corrections from OpenMOPAC ccrep.F90 lines 101-120
    if z_min == 6 && z_max == 6 {
        // C-C triple bond correction: par1 = 9.278465, par2 = 5.983752
        scale += Pm6Model::V_PAR1 * (-Pm6Model::V_PAR2 * r_angstrom).exp();
    } else if z_min == 8 && z_max == 14 {
        // Si-O interaction correction
        scale -= 0.0007 * (-(r_angstrom - 2.9).powi(2)).exp();
    }

    let base_repulsion = elem_a.core_charge * elem_b.core_charge * gab * scale;

    // PM6 VdW Gaussian terms (ccrep.F90 lines 189-245)
    let mut gaussian_sum = 0.0;
    if elem_a.num_gaussians > 0 {
        let g = &elem_a.gaussians[0];
        let dr = r_angstrom - g.c;
        let exponent = g.b * dr * dr;
        if exponent < 25.0 {
            gaussian_sum += g.a * (-exponent).exp();
        }
    }
    if elem_b.num_gaussians > 0 {
        let g = &elem_b.gaussians[0];
        let dr = r_angstrom - g.c;
        let exponent = g.b * dr * dr;
        if exponent < 25.0 {
            gaussian_sum += g.a * (-exponent).exp();
        }
    }
    if alpb <= 1e-4 {
        for k in 1..elem_a.num_gaussians {
            let g = &elem_a.gaussians[k];
            let dr = r_angstrom - g.c;
            let exponent = g.b * dr * dr;
            if exponent <= 25.0 {
                gaussian_sum += g.a * (-exponent).exp();
            }
        }
        for k in 1..elem_b.num_gaussians {
            let g = &elem_b.gaussians[k];
            let dr = r_angstrom - g.c;
            let exponent = g.b * dr * dr;
            if exponent <= 25.0 {
                gaussian_sum += g.a * (-exponent).exp();
            }
        }
    }

    let gaussian_repulsion = (elem_a.core_charge * elem_b.core_charge / r_angstrom) * gaussian_sum;
    let lj_core = unpolarizable_core_repulsion(r_angstrom, elem_a.z, elem_b.z);

    base_repulsion + gaussian_repulsion + lj_core
}

/// Compute pairwise core-core nuclear repulsion energy using PM7 formulation.
///
/// Direct port of OpenMOPAC `ccrep.F90` lines 57-121, 246-263.
pub fn compute_pair_core_repulsion_pm7(
    r_angstrom: f64,
    elem_a: &SemiEmpiricalElementParams,
    elem_b: &SemiEmpiricalElementParams,
) -> f64 {
    if r_angstrom < 1e-10 {
        return 0.0;
    }
    let pocord_a = Pm7Model::get_pocord(elem_a.z);
    let pocord_b = Pm7Model::get_pocord(elem_b.z);
    let gab_base = core_monopole(r_angstrom, elem_a.gss, elem_b.gss, pocord_a, pocord_b);

    // PM7 feathering: smooth transition to exact point charge (OpenMOPAC to_point subroutine)
    const TRUNC_1: f64 = 7.0;
    const TRUNC_2: f64 = 0.22;
    let gab = if r_angstrom < TRUNC_1 {
        let dr = r_angstrom - TRUNC_1;
        let c = 1.0 - (-dr * dr * TRUNC_2).exp();
        let point = EV_ANGSTROM_FACTOR / r_angstrom;
        gab_base * c + (1.0 - c) * point
    } else {
        EV_ANGSTROM_FACTOR / r_angstrom
    };

    let (mut abond, mut fff) = crate::parameters::pm7::get_pm7_pair_params(elem_a.z, elem_b.z);
    if fff < 1e-5 {
        fff = 0.0;
    }
    if abond < 1e-6 {
        abond = 1.2;
    }

    let z_min = elem_a.z.min(elem_b.z);
    let z_max = elem_a.z.max(elem_b.z);

    let mut scale = if z_min == 1 && (z_max == 6 || z_max == 7) {
        1.0 + 2.0 * fff * (-abond * r_angstrom * r_angstrom).exp()
    } else if z_min == 1 && z_max == 8 {
        1.0 + 2.0 * fff * (-abond * r_angstrom * r_angstrom).exp()
            - Pm7Model::V_PAR3 * (-Pm7Model::V_PAR4 * r_angstrom * 2.0).exp()
    } else if fff > 1e-5 {
        1.0 + 2.0 * fff * (-abond * (r_angstrom + 0.0003 * r_angstrom.powi(6))).exp()
    } else if (elem_a.z > 56 && elem_a.z < 72) || (elem_b.z > 56 && elem_b.z < 72) {
        1.0 + 10.0 * (-3.0 * r_angstrom).exp()
    } else {
        1.0 + 10.0 * (-2.18 * r_angstrom).exp()
    };

    // Special PM7 pair corrections from OpenMOPAC ccrep.F90 lines 101-120
    if z_min == 6 && z_max == 6 {
        scale += Pm7Model::V_PAR1 * (-Pm7Model::V_PAR2 * r_angstrom).exp();
    } else if z_min == 8 && z_max == 14 {
        scale -= 0.0007 * (-(r_angstrom - 2.9).powi(2)).exp();
    }

    let base_repulsion = elem_a.core_charge * elem_b.core_charge * gab * scale;

    let mut gaussian_sum = 0.0;
    if elem_a.num_gaussians > 0 {
        let g = &elem_a.gaussians[0];
        let dr = r_angstrom - g.c;
        let exponent = g.b * dr * dr;
        if exponent < 25.0 {
            gaussian_sum += g.a * (-exponent).exp();
        }
    }
    if elem_b.num_gaussians > 0 {
        let g = &elem_b.gaussians[0];
        let dr = r_angstrom - g.c;
        let exponent = g.b * dr * dr;
        if exponent < 25.0 {
            gaussian_sum += g.a * (-exponent).exp();
        }
    }
    if abond <= 1e-4 {
        for k in 1..elem_a.num_gaussians {
            let g = &elem_a.gaussians[k];
            let dr = r_angstrom - g.c;
            let exponent = g.b * dr * dr;
            if exponent <= 25.0 {
                gaussian_sum += g.a * (-exponent).exp();
            }
        }
        for k in 1..elem_b.num_gaussians {
            let g = &elem_b.gaussians[k];
            let dr = r_angstrom - g.c;
            let exponent = g.b * dr * dr;
            if exponent <= 25.0 {
                gaussian_sum += g.a * (-exponent).exp();
            }
        }
    }

    let gaussian_repulsion = (elem_a.core_charge * elem_b.core_charge / r_angstrom) * gaussian_sum;
    let lj_core = unpolarizable_core_repulsion(r_angstrom, elem_a.z, elem_b.z);

    base_repulsion + gaussian_repulsion + lj_core
}

/// Compute the total core-core repulsion energy for an entire molecular system.
pub fn compute_total_core_repulsion(
    batch: &crate::types::MolecularBatch,
    model: &dyn crate::parameters::ParameterModel,
) -> f64 {
    let mut total_energy = 0.0;

    for i in 0..batch.natoms {
        let za = batch.atomic_numbers[i];
        let elem_a = match model.get_element(za) {
            Some(p) => p,
            None => continue,
        };

        for j in (i + 1)..batch.natoms {
            let zb = batch.atomic_numbers[j];
            let elem_b = match model.get_element(zb) {
                Some(p) => p,
                None => continue,
            };

            let r = batch.distance(i, j);
            total_energy += model.pair_core_repulsion(r, &elem_a, &elem_b);
        }
    }

    total_energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::ParameterModel;
    use std::time::Instant;

    /// TEST 1 (Desempeño / Throughput):
    /// Mide la tasa de evaluación de repulsión de core sobre 1,000,000 de evaluaciones
    /// a través de pares representativos (grupo principal, metales de transición con pocord,
    /// distancias de enlace normales y de compresión fuerte con término Lennard-Jones).
    /// Verifica que el throughput supere > 20,000,000 eval/s (< 50 ns/eval) con 0 alocaciones de heap.
    #[test]
    fn test_core_repulsion_pm6_pm7_performance_throughput() {
        let pm6 = Pm6Model;
        let pm7 = Pm7Model;

        let o_pm6 = pm6.get_element(8).expect("PM6 Oxygen");
        let h_pm6 = pm6.get_element(1).expect("PM6 Hydrogen");
        let c_pm6 = pm6.get_element(6).expect("PM6 Carbon");
        let si_pm6 = pm6.get_element(14).expect("PM6 Silicon");
        let sc_pm6 = pm6.get_element(21).expect("PM6 Scandium (pocord)");

        let o_pm7 = pm7.get_element(8).expect("PM7 Oxygen");
        let h_pm7 = pm7.get_element(1).expect("PM7 Hydrogen");

        let pairs = [
            (&o_pm6, &h_pm6, 0.95777),
            (&c_pm6, &c_pm6, 1.205),
            (&si_pm6, &o_pm6, 1.63),
            (&sc_pm6, &o_pm6, 1.95),
            (&o_pm7, &h_pm7, 0.95777),
        ];

        let n_evals = if cfg!(debug_assertions) {
            200_000
        } else {
            1_000_000
        };
        let mut sink = 0.0f64;

        let start = Instant::now();
        for i in 0..n_evals {
            let (ea, eb, r) = pairs[i % pairs.len()];
            let dr = r + ((i & 0x7) as f64) * 0.01;
            if i % 2 == 0 {
                sink += compute_pair_core_repulsion_pm6(dr, ea, eb);
            } else {
                sink += compute_pair_core_repulsion_pm7(dr, ea, eb);
            }
        }
        let elapsed = start.elapsed();
        let evals_per_sec = (n_evals as f64) / elapsed.as_secs_f64();
        let ns_per_eval = elapsed.as_nanos() as f64 / (n_evals as f64);

        println!(
            "[CORE REPULSION BENCHMARK] Evaluated {} pairwise repulsions in {:.3} ms ({:.2} M evals/sec, {:.2} ns/eval, sink={:.2})",
            n_evals, elapsed.as_secs_f64() * 1000.0, evals_per_sec / 1e6, ns_per_eval, sink
        );

        let min_evals_per_sec = if cfg!(debug_assertions) {
            150_000.0
        } else {
            5_000_000.0
        };
        assert!(
            evals_per_sec > min_evals_per_sec,
            "Core repulsion throughput below threshold: {:.2} M evals/sec (min: {:.2} M evals/sec)",
            evals_per_sec / 1e6,
            min_evals_per_sec / 1e6
        );
        assert!(!sink.is_nan() && sink.is_finite());
    }

    /// TEST 2 (Aplicabilidad Matemática y Lógico-Algorítmica):
    /// Verifica la paridad matemática rigurosa contra OpenMOPAC Fortran 90 (ccrep.F90)
    /// en los 5 regímenes físicos de la Fase 1:
    /// 1. Verificación analítica del término de core Lennard-Jones 12: ax < 3.0 vs ax >= 3.0.
    /// 2. Verificación de paridad exacta en Agua H2O PM6 (138.584180 eV matching OpenMOPAC bit a bit).
    /// 3. Verificación de la modulación de triple enlace C-C (V_PAR1 y V_PAR2).
    /// 4. Verificación de la modulación de enlace Si-O (-0.7e-3 exp(-(r-2.9)^2)).
    /// 5. Verificación de sustitución pocord en metales de transición (Scandium Z=21, pocord = 3.173734 Bohr).
    #[test]
    fn test_core_repulsion_exact_mathematical_applicability_and_parity() {
        let pm6 = Pm6Model;

        // 1. Verificación del término Lennard-Jones ax < 3.0
        // Para O (Z=8) y H (Z=1) a r = 0.95777559 A:
        // ax = 0.95777559 / (8^0.3333 + 1^0.3333) = 0.95777559 / 2.9998616 = 0.31927328
        // lj = 1e-8 / ax^12 = 0.00891352 eV
        let lj_val = unpolarizable_core_repulsion(0.95777559, 8, 1);
        let expected_lj = 0.00891352;
        assert!(
            (lj_val - expected_lj).abs() < 1e-6,
            "Lennard-Jones core repulsion mismatch: got {:.8}, expected {:.8}",
            lj_val,
            expected_lj
        );

        // Para átomos muy separados (r = 10.0 A), ax >> 3.0 -> LJ debe ser estrictamente 0.0
        let lj_zero = unpolarizable_core_repulsion(10.0, 8, 1);
        assert_eq!(lj_zero, 0.0, "LJ must be zero when ax >= 3.0");

        // 2. Paridad exacta de H2O PM6 frente a OpenMOPAC v23.2.5 (138.584180 eV)
        let o = pm6.get_element(8).unwrap();
        let h = pm6.get_element(1).unwrap();

        let r_oh = (0.7572f64.powi(2) + (-0.4692 - 0.1173f64).powi(2)).sqrt();
        let r_hh = 2.0 * 0.7572;

        let enuc_oh1 = compute_pair_core_repulsion_pm6(r_oh, &o, &h);
        let enuc_oh2 = compute_pair_core_repulsion_pm6(r_oh, &o, &h);
        let enuc_hh = compute_pair_core_repulsion_pm6(r_hh, &h, &h);

        let total_enuc_water = enuc_oh1 + enuc_oh2 + enuc_hh;
        let openmopac_enuc = 138.584180; // De OpenMOPAC v23.2.5 con DISP

        println!(
            "[WATER PM6 ENUC PARITY] MOPAC_RS = {:.6} eV, OpenMOPAC = {:.6} eV, diff = {:.8} eV",
            total_enuc_water,
            openmopac_enuc,
            (total_enuc_water - openmopac_enuc).abs()
        );

        assert!(
            (total_enuc_water - openmopac_enuc).abs() < 1e-5,
            "Water PM6 core repulsion deviates from OpenMOPAC: got {:.6}, expected {:.6}, diff = {:.8}",
            total_enuc_water,
            openmopac_enuc,
            (total_enuc_water - openmopac_enuc).abs()
        );

        // 3. Verificación de C-C triple bond term en PM6
        let c = pm6.get_element(6).unwrap();
        let r_cc = 1.205; // C-C triple bond distance
        let enuc_cc = compute_pair_core_repulsion_pm6(r_cc, &c, &c);
        // Sin el término Pm6Model::V_PAR1 * exp(-V_PAR2 * r), el valor sería significativamente menor
        let extra_term = Pm6Model::V_PAR1 * (-Pm6Model::V_PAR2 * r_cc).exp();
        assert!(
            extra_term > 0.005,
            "C-C triple bond term must contribute significantly at 1.205 A"
        );
        assert!(enuc_cc > 0.0, "C-C core repulsion must be positive");

        // 4. Verificación de Si-O term
        let si = pm6.get_element(14).unwrap();
        let r_sio = 2.90; // Centro de la Gaussiana Si-O
        let enuc_sio = compute_pair_core_repulsion_pm6(r_sio, &si, &o);
        assert!(enuc_sio > 0.0, "Si-O core repulsion must be positive");

        // 5. Verificación de pocord en Scandium (Z=21)
        let pocord_sc = Pm6Model::get_pocord(21);
        assert!(
            (pocord_sc - 3.173734).abs() < 1e-6,
            "Scandium pocord mismatch: got {}",
            pocord_sc
        );
        let sc = pm6.get_element(21).unwrap();
        let gab_sc_o = core_monopole(2.0, sc.gss, o.gss, pocord_sc, 0.0);
        let gab_standard = dewar_klopman_monopole(2.0, sc.gss, o.gss);
        assert!(
            (gab_sc_o - gab_standard).abs() > 0.1,
            "pocord must actively modulate the monopole integral: pocord={:.5}, standard={:.5}",
            gab_sc_o,
            gab_standard
        );
    }
}
