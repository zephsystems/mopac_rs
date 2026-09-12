//! UV-Vis Electronic Absorption Spectroscopy & Transition Dipoles.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Implements canonical transition dipole moment evaluation $\vec{\mu}_{0 \to n} = \langle \Psi_0 | \hat{\vec{\mu}} | \Psi_n \rangle$,
//! oscillator strengths $f_{\text{osc}}$, and UV-Vis absorption spectrum generation matching OpenMOPAC `ciosci.F90`.

use crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS as A0_BOHR;
use crate::integrals::multipoles::DerivedMultipoleParams;
use crate::parameters::ParameterModel;
use crate::properties::dipole::E_ANGSTROM_TO_DEBYE;
use crate::types::{AlignedMatrix, MolecularBatch};

use super::meci::{CiState, MeciResult};

/// Atomic units conversion factor for oscillator strength:
/// $f = \frac{2}{3} \Delta E (\text{a.u.}) |\mu(\text{a.u.})|^2$
/// $k = \frac{2}{3} \times \frac{1}{27.211386245988} \times \left(\frac{1}{2.541746473}\right)^2 \approx 0.0037922207906861114\text{ eV}^{-1}\text{ Debye}^{-2}$.
pub const OSCILLATOR_STRENGTH_FACTOR: f64 = 0.0037922207906861114;

/// Molar absorption coefficient peak conversion constant in $\text{L} \cdot \text{mol}^{-1} \cdot \text{cm}^{-1} \cdot \text{nm}$:
/// $\int \epsilon(\tilde{\nu}) d\tilde{\nu} = \frac{\pi e^2}{4 \pi \epsilon_0 m_e c^2 \ln(10)} f \approx 2.315 \times 10^8 f$.
pub const EPSILON_PEAK_FACTOR: f64 = 1.3062974e8;

/// Compute transition dipole moments $\vec{\mu}_{0 \to n}$ and oscillator strengths $f_{\text{osc}}$
/// between the ground state root and all excited states.
///
/// Direct port of OpenMOPAC `ciosci.F90`.
pub fn compute_transition_dipoles_and_oscillator_strengths(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    eigenvectors: &AlignedMatrix<f64>,
    active_mo_indices: &[usize],
    meci_result: &mut MeciResult,
) {
    let m = active_mo_indices.len();
    let natoms = batch.natoms;
    let lab = meci_result.microstates.len();

    // 1. Construct the Molecular Orbital coordinate operator matrix T2(c, i, j) = <psi_i | r_c | psi_j>
    // for each Cartesian axis c in {0: x, 1: y, 2: z}
    let mut t2 = vec![vec![vec![0.0f64; m]; m]; 3];

    for (c, t2_c) in t2.iter_mut().enumerate().take(3) {
        for i in 0..m {
            let mo_i = active_mo_indices[i];
            for j in 0..m {
                let mo_j = active_mo_indices[j];

                let mut val = 0.0f64;

                for a in 0..natoms {
                    let r_a = match c {
                        0 => batch.x[a],
                        1 => batch.y[a],
                        _ => batch.z[a],
                    };
                    let orb_start = batch.orbital_offsets[a];
                    let norbs = batch.basis_types[a].num_orbitals();

                    // Point charge / atomic position contribution: R_A * sum_mu C_{mu i} C_{mu j}
                    let mut q_a_ij = 0.0;
                    for o in 0..norbs {
                        q_a_ij += eigenvectors.get(orb_start + o, mo_i)
                            * eigenvectors.get(orb_start + o, mo_j);
                    }
                    val += r_a * q_a_ij;

                    // Intra-atomic hybridization dipole contribution: <ns | r_c | np_c> = D1 * a0
                    let z = batch.atomic_numbers[a];
                    if z != 1 && norbs >= 4 {
                        if let Some(p) = model.get_element(z) {
                            let mp = DerivedMultipoleParams::from_element(&p);
                            let d1_angstrom = mp.dd * A0_BOHR;
                            let s_idx = orb_start;
                            let p_idx = orb_start + 1 + c;

                            let c_si = eigenvectors.get(s_idx, mo_i);
                            let c_pi = eigenvectors.get(p_idx, mo_i);
                            let c_sj = eigenvectors.get(s_idx, mo_j);
                            let c_pj = eigenvectors.get(p_idx, mo_j);

                            val += d1_angstrom * (c_si * c_pj + c_pi * c_sj);
                        }
                    }
                }

                t2_c[i][j] = val;
            }
        }
    }

    // 2. Build microstate transition dipole matrix T4(c, K, L) = <K | r_c | L>
    // T4 is non-zero only for single excitations
    let mut t4 = vec![AlignedMatrix::zeroed(lab, lab); 3];

    for c in 0..3 {
        for i in 0..lab {
            let m_i = &meci_result.microstates[i];
            for j in 0..i {
                let m_j = &meci_result.microstates[j];

                let mut diff_a = Vec::new();
                let mut diff_b = Vec::new();

                for k in 0..m {
                    if m_i.alpha[k] != m_j.alpha[k] {
                        diff_a.push(k);
                    }
                    if m_i.beta[k] != m_j.beta[k] {
                        diff_b.push(k);
                    }
                }

                let l_a = diff_a.len();
                let l_b = diff_b.len();

                if (l_a == 2 && l_b == 0) || (l_b == 2 && l_a == 0) {
                    let (p, q) = if l_a == 2 {
                        (diff_a[0], diff_a[1])
                    } else {
                        (diff_b[0], diff_b[1])
                    };

                    let dip_elem = t2[c][p][q];

                    // Parity of permutation matching OpenMOPAC ciosci.F90
                    let ij = if l_a == 2 {
                        let mut sum = (m_i.beta[p] + m_i.alpha[p]) as usize;
                        for idx in (p + 1)..q {
                            sum += (m_i.alpha[idx] + m_i.beta[idx]) as usize;
                        }
                        sum
                    } else {
                        let mut sum = m_i.beta[p] as usize;
                        for idx in (p + 1)..q {
                            sum += (m_i.alpha[idx] + m_i.beta[idx]) as usize;
                        }
                        sum + m_i.alpha[q] as usize
                    };

                    let sign = if ij % 2 == 1 { -1.0 } else { 1.0 };
                    let val = dip_elem * sign;

                    t4[c].set(i, j, val);
                    t4[c].set(j, i, -val);
                }
            }
        }
    }

    // 3. Compute transition dipoles and oscillator strengths from root 0 (ground state)
    // <Psi_0 | r_c | Psi_n> = sum_{K, L} C_{0, K} C_{n, L} T4(c, K, L)
    for state_idx in 0..meci_result.states.len() {
        if state_idx == 0 {
            // Ground state to ground state: transition dipole is 0 by definition
            meci_result.states[0].transition_dipole_debye = [0.0; 3];
            meci_result.states[0].dipole_strength_debye = 0.0;
            meci_result.states[0].polarization_angstrom2 = [0.0; 3];
            meci_result.states[0].oscillator_strength = 0.0;
            continue;
        }

        let mut dip_debye = [0.0f64; 3];
        let mut pol_a2 = [0.0f64; 3];

        for c in 0..3 {
            let mut val_angstrom = 0.0f64;
            for k in 0..lab {
                let c_0k = meci_result.states[0].eigenvector[k];
                for l in 0..lab {
                    let c_nl = meci_result.states[state_idx].eigenvector[l];
                    val_angstrom += c_0k * t4[c].get(k, l) * c_nl;
                }
            }
            pol_a2[c] = val_angstrom * val_angstrom;
            dip_debye[c] = (val_angstrom * E_ANGSTROM_TO_DEBYE).abs();
        }

        let dip_sq =
            dip_debye[0] * dip_debye[0] + dip_debye[1] * dip_debye[1] + dip_debye[2] * dip_debye[2];
        let dip_mag = dip_sq.sqrt();

        let d_e = meci_result.states[state_idx].excitation_energy_ev;
        let f_osc = if d_e > 1e-4 {
            OSCILLATOR_STRENGTH_FACTOR * d_e * dip_sq
        } else {
            0.0
        };

        meci_result.states[state_idx].transition_dipole_debye = dip_debye;
        meci_result.states[state_idx].dipole_strength_debye = dip_mag;
        meci_result.states[state_idx].polarization_angstrom2 = pol_a2;
        meci_result.states[state_idx].oscillator_strength = f_osc;
    }
}

/// Simulated UV-Vis absorption spectrum.
#[derive(Debug, Clone)]
pub struct UvVisSpectrum {
    /// Wavelength grid points in nanometers ($\text{nm}$)
    pub wavelengths_nm: Vec<f64>,
    /// Molar extinction coefficients $\epsilon(\lambda)$ in $\text{L} \cdot \text{mol}^{-1} \cdot \text{cm}^{-1}$
    pub extinction_coefficients: Vec<f64>,
    /// Peak absorption wavelength $\lambda_{\max}$ in $\text{nm}$
    pub lambda_max_nm: f64,
    /// Maximum molar extinction coefficient $\epsilon_{\max}$
    pub epsilon_max: f64,
}

/// Simulate a continuous UV-Vis spectrum from discrete CI states with Gaussian band broadening.
///
/// # Arguments
/// * `states`: List of computed CI states from MECI
/// * `min_wl_nm`: Minimum wavelength of spectrum grid in nm (e.g. 150.0 nm)
/// * `max_wl_nm`: Maximum wavelength of spectrum grid in nm (e.g. 800.0 nm)
/// * `step_nm`: Grid sampling step size in nm (e.g. 1.0 nm)
/// * `fwhm_nm`: Full Width at Half Maximum band broadening in nm (e.g. 20.0 nm)
pub fn simulate_uv_vis_spectrum(
    states: &[CiState],
    min_wl_nm: f64,
    max_wl_nm: f64,
    step_nm: f64,
    fwhm_nm: f64,
) -> UvVisSpectrum {
    let n_points = (((max_wl_nm - min_wl_nm) / step_nm).floor() as usize) + 1;
    let mut wavelengths_nm = Vec::with_capacity(n_points);
    let mut extinction_coefficients = vec![0.0f64; n_points];

    for idx in 0..n_points {
        wavelengths_nm.push(min_wl_nm + (idx as f64) * step_nm);
    }

    // Standard deviation sigma = FWHM / (2 * sqrt(2 * ln(2)))
    let sigma = fwhm_nm / (2.0 * (2.0f64.ln() * 2.0).sqrt());
    let two_sigma_sq = 2.0 * sigma * sigma;
    let norm_factor = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());

    for state in states {
        if state.excitation_energy_ev <= 1e-4 || state.wavelength_nm <= 0.0 {
            continue;
        }
        let wl_0 = state.wavelength_nm;
        let f = state.oscillator_strength;

        if f <= 1e-6 {
            continue;
        }

        // Peak extinction coefficient contribution
        let peak_eps = f * EPSILON_PEAK_FACTOR * norm_factor;

        for (idx, &wl) in wavelengths_nm.iter().enumerate() {
            let diff = wl - wl_0;
            if diff.abs() > 4.0 * fwhm_nm {
                continue;
            }
            let gauss = (-diff * diff / two_sigma_sq).exp();
            extinction_coefficients[idx] += peak_eps * gauss;
        }
    }

    let mut lambda_max_nm = min_wl_nm;
    let mut epsilon_max = 0.0f64;
    for (idx, &eps) in extinction_coefficients.iter().enumerate() {
        if eps > epsilon_max {
            epsilon_max = eps;
            lambda_max_nm = wavelengths_nm[idx];
        }
    }

    UvVisSpectrum {
        wavelengths_nm,
        extinction_coefficients,
        lambda_max_nm,
        epsilon_max,
    }
}
