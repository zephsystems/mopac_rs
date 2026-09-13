//! Harmonic Vibrational Frequency & Hessian Analysis.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Evaluates the second Cartesian energy derivatives (Hessian matrix $\mathcal{H}$),
//! performs mass-weighting, projects external translation/rotation via the Eckart frame,
//! diagonalizes the secular vibrational equation to extract harmonic wavenumbers $\tilde{\nu}$ ($\text{cm}^{-1}$),
//! and computes complete statistical thermodynamic partition functions and properties.

use crate::constants::codata2018::{
    BOLTZMANN_CONSTANT_J_K, EV_TO_KCAL_MOL, GAS_CONSTANT_CAL, PLANCK_CONSTANT_ERG_S,
};
use crate::constants::standard_atomic_mass;
use crate::gradients::nuclear_gradients::{
    compute_cartesian_gradients_with_options, GradientWorkspace,
};
use crate::parameters::ParameterModel;
use crate::scf::eigensolver::diagonalize_symmetric;
use crate::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch, ScfWorkspace};

/// Conversion factor from $\sqrt{\text{kcal} / (\text{mol} \cdot \text{\AA}^2 \cdot \text{amu})}$ to $\text{cm}^{-1}$.
///
/// Derived from fundamental SI constants:
/// $$\text{factor} = \frac{\sqrt{4.184 \times 10^{26}}}{2\pi c} = 108.59135859237747 \text{ cm}^{-1}$$
pub const KCAL_MOL_A2_AMU_TO_CM1: f64 = 108.59135859237747;

/// Conversion factor from wavenumber ($\text{cm}^{-1}$) to ZPVE ($\text{kcal/mol}$):
/// $$\frac{h \cdot c \cdot N_A}{4184 \text{ J/kcal}} = 0.0028591437069 \text{ kcal}/(\text{mol}\cdot\text{cm}^{-1})$$
pub const CM1_TO_KCAL_MOL: f64 = 0.00285914370690045;

/// Conversion factor from wavenumber ($\text{cm}^{-1}$) to vibrational temperature $\Theta_{\text{vib}}$ (Kelvin):
/// $$\Theta_{\text{vib}} = \frac{h c \tilde{\nu}}{k_B} = 1.4387768775 \times \tilde{\nu} \text{ K}$$
pub const CM1_TO_KELVIN: f64 = 1.43877687750393;

/// Options for numerical Hessian and frequency evaluation.
#[derive(Debug, Clone)]
pub struct HessianOptions {
    /// Finite difference step size $\delta$ in Ångströms (default: $1.0 \times 10^{-3} \text{ \AA}$)
    pub delta: f64,
    /// Whether to recompute SCF density at each displaced geometry (default: true)
    pub recompute_scf: bool,
    /// Whether to evaluate gradients with full NDDO 22-multipole integrals (default: true)
    pub use_nddo: bool,
    /// Whether to project out 6 rigid translations and rotations (default: true)
    pub project_external: bool,
    /// Thermochemistry temperature in Kelvin (default: 298.15 K)
    pub temperature_k: f64,
    /// Thermochemistry pressure in atmospheres (default: 1.0 atm)
    pub pressure_atm: f64,
    /// Molecular rotational symmetry number $\sigma$ (default: 1.0)
    pub rotational_symmetry_number: f64,
    /// Optional custom atomic masses in amu (length = natoms) for isotope substitution / KIE
    pub custom_masses: Option<Vec<f64>>,
}

impl Default for HessianOptions {
    fn default() -> Self {
        Self {
            delta: 1.0e-3,
            recompute_scf: true,
            use_nddo: true,
            project_external: true,
            temperature_k: 298.15,
            pressure_atm: 1.0,
            rotational_symmetry_number: 1.0,
            custom_masses: None,
        }
    }
}

/// Harmonic normal vibrational mode.
#[derive(Debug, Clone)]
pub struct NormalMode {
    /// Harmonic frequency $\tilde{\nu}$ in $\text{cm}^{-1}$ (negative if imaginary/transition state)
    pub frequency_cm1: f64,
    /// Effective reduced mass in amu
    pub reduced_mass_amu: f64,
    /// Force constant in millidynes/Å ($\text{mdyne/\AA}$)
    pub force_constant_mdyne_a: f64,
    /// Normalized Cartesian displacements $(\delta x, \delta y, \delta z)$ for each atom
    pub displacements: Vec<[f64; 3]>,
}

/// Statistical thermodynamic properties computed at temperature $T$ and pressure $P$.
#[derive(Debug, Clone, Default)]
pub struct ThermodynamicProperties {
    pub temperature_k: f64,
    pub pressure_atm: f64,
    /// Zero-Point Vibrational Energy in kcal/mol
    pub zpve_kcal_mol: f64,
    /// Thermal vibrational energy $E_{\text{vib}}(T)$ in cal/mol
    pub e_vib_cal_mol: f64,
    /// Thermal rotational energy $E_{\text{rot}}(T)$ in cal/mol
    pub e_rot_cal_mol: f64,
    /// Thermal translational energy $E_{\text{trans}}(T)$ in cal/mol
    pub e_trans_cal_mol: f64,
    /// Total thermal enthalpy correction $H(T) - H(0) = E_{\text{vib}} + E_{\text{rot}} + E_{\text{trans}} + RT$ in cal/mol
    pub enthalpy_thermal_cal_mol: f64,
    /// Constant volume vibrational heat capacity $C_{v,\text{vib}}$ in cal/(mol·K)
    pub cv_vib_cal_k_mol: f64,
    /// Constant volume rotational heat capacity $C_{v,\text{rot}}$ in cal/(mol·K)
    pub cv_rot_cal_k_mol: f64,
    /// Constant pressure translational heat capacity $C_{p,\text{trans}} = \frac{5}{2}R$ in cal/(mol·K)
    pub cp_trans_cal_k_mol: f64,
    /// Total heat capacity $C_p(T) = C_v + R$ in cal/(mol·K)
    pub cp_total_cal_k_mol: f64,
    /// Vibrational entropy $S_{\text{vib}}$ in cal/(mol·K)
    pub entropy_vib_cal_k_mol: f64,
    /// Rotational entropy $S_{\text{rot}}$ in cal/(mol·K)
    pub entropy_rot_cal_k_mol: f64,
    /// Translational entropy $S_{\text{trans}}$ (Sackur-Tetrode) in cal/(mol·K)
    pub entropy_trans_cal_k_mol: f64,
    /// Total standard entropy $S^\circ(T)$ in cal/(mol·K)
    pub entropy_total_cal_k_mol: f64,
    /// Gibbs free energy thermal correction $G_{\text{corr}} = H_{\text{thermal}} - T \cdot S^\circ$ in kcal/mol
    pub gibbs_correction_kcal_mol: f64,
}

/// Comprehensive results of Hessian and normal coordinate analysis.
#[derive(Debug, Clone)]
pub struct HessianResult {
    /// Cartesian Hessian matrix $\mathcal{H}_{3N \times 3N}$ in $\text{kcal}/(\text{mol}\cdot\text{\AA}^2)$
    pub cartesian_hessian: AlignedMatrix<f64>,
    /// Mass-weighted Hessian $\tilde{\mathcal{H}}_{3N \times 3N}$ in $\text{kcal}/(\text{mol}\cdot\text{\AA}^2\cdot\text{amu})$
    pub mass_weighted_hessian: AlignedMatrix<f64>,
    /// All $3N$ harmonic frequencies in $\text{cm}^{-1}$ (sorted ascending)
    pub all_frequencies_cm1: Vec<f64>,
    /// Genuine internal vibrational frequencies ($3N-6$ or $3N-5$) in $\text{cm}^{-1}$
    pub vibrational_frequencies_cm1: Vec<f64>,
    /// Normal modes with atom displacement vectors
    pub normal_modes: Vec<NormalMode>,
    /// Zero-Point Vibrational Energy (ZPVE) in kcal/mol
    pub zpve_kcal_mol: f64,
    /// Statistical thermodynamic properties
    pub thermo: ThermodynamicProperties,
}

/// Evaluates Cartesian Hessian matrix $\mathcal{H}_{3N \times 3N}$, harmonic frequencies, and thermodynamics.
pub fn compute_hessian_and_frequencies(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    ws: &mut ScfWorkspace,
    scf_opts: &ScfOptions,
    hess_opts: &HessianOptions,
) -> HessianResult {
    let natoms = batch.natoms;
    let n3 = 3 * natoms;
    assert!(natoms >= 1, "MolecularBatch must contain at least 1 atom");

    // 1. Fetch authentic IUPAC standard atomic masses or custom isotopic masses
    let masses: Vec<f64> = if let Some(ref cm) = hess_opts.custom_masses {
        if cm.len() == natoms {
            cm.clone()
        } else {
            batch
                .atomic_numbers
                .iter()
                .map(|&z| standard_atomic_mass(z))
                .collect()
        }
    } else {
        batch
            .atomic_numbers
            .iter()
            .map(|&z| standard_atomic_mass(z))
            .collect()
    };

    // 2. Ensure initial SCF convergence
    let base_scf = run_rhf_scf_with_options(batch, model, ws, scf_opts);
    assert!(
        base_scf.converged,
        "Base SCF must converge before computing Hessian"
    );

    // Save initial converged density for warm-starts and frozen evaluations
    let init_density = ws.density.clone();

    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut g_plus = vec![[0.0; 3]; natoms];
    let mut g_minus = vec![[0.0; 3]; natoms];

    let mut cartesian_hessian = AlignedMatrix::zeroed(n3, n3);
    let delta = hess_opts.delta;
    let inv_2delta = 1.0 / (2.0 * delta);

    // 3. Central difference numerical gradient evaluation
    for a in 0..natoms {
        for alpha in 0..3 {
            let col = 3 * a + alpha;

            // --- Coordinate +delta ---
            displace_coord(batch, a, alpha, delta);
            if hess_opts.recompute_scf {
                ws.density.clone_from(&init_density);
                run_rhf_scf_with_options(batch, model, ws, scf_opts);
            }
            compute_cartesian_gradients_with_options(
                batch,
                model,
                &ws.density,
                &mut grad_ws,
                &mut g_plus,
                hess_opts.use_nddo,
            );

            // --- Coordinate -delta ---
            displace_coord(batch, a, alpha, -2.0 * delta);
            if hess_opts.recompute_scf {
                ws.density.clone_from(&init_density);
                run_rhf_scf_with_options(batch, model, ws, scf_opts);
            }
            compute_cartesian_gradients_with_options(
                batch,
                model,
                &ws.density,
                &mut grad_ws,
                &mut g_minus,
                hess_opts.use_nddo,
            );

            // Restore coordinate
            displace_coord(batch, a, alpha, delta);

            // Populate column of Hessian: d(grad_l)/d(coord_col)
            // Note: g is in eV/Angstrom. Convert to kcal/(mol*Angstrom^2)
            for b in 0..natoms {
                for beta in 0..3 {
                    let row = 3 * b + beta;
                    let dg = (g_plus[b][beta] - g_minus[b][beta]) * inv_2delta * EV_TO_KCAL_MOL;
                    cartesian_hessian.set(row, col, dg);
                }
            }
        }
    }

    // Restore density to initial converged state
    ws.density.clone_from(&init_density);

    // 4. Exact matrix symmetrization: H = 0.5 * (H + H^T)
    for i in 0..n3 {
        for j in (i + 1)..n3 {
            let val = 0.5 * (cartesian_hessian.get(i, j) + cartesian_hessian.get(j, i));
            cartesian_hessian.set(i, j, val);
            cartesian_hessian.set(j, i, val);
        }
    }

    // 5. Construct mass-weighted Hessian: H_mw(i, j) = H(i, j) / sqrt(m_i * m_j)
    let mut mass_weighted_hessian = AlignedMatrix::zeroed(n3, n3);
    for a in 0..natoms {
        let ma = masses[a];
        for alpha in 0..3 {
            let i = 3 * a + alpha;
            for (b, &mb) in masses.iter().enumerate().take(natoms) {
                let inv_sqrt_m = 1.0 / (ma * mb).sqrt();
                for beta in 0..3 {
                    let j = 3 * b + beta;
                    let h_val = cartesian_hessian.get(i, j);
                    mass_weighted_hessian.set(i, j, h_val * inv_sqrt_m);
                }
            }
        }
    }

    // 6. Project external translations and rotations via Eckart projector if requested
    let mut mat_to_diag = mass_weighted_hessian.clone();
    if hess_opts.project_external && natoms > 1 {
        project_out_translations_and_rotations(batch, &masses, &mut mat_to_diag);
    }

    // 7. Diagonalize mass-weighted Hessian
    let mut eigenvalues = AlignedVec64::zeroed(n3);
    let mut eigenvectors = AlignedMatrix::zeroed(n3, n3);
    diagonalize_symmetric(&mat_to_diag, &mut eigenvalues, &mut eigenvectors);

    // 8. Convert eigenvalues to harmonic frequencies (cm^-1)
    let mut all_frequencies_cm1 = Vec::with_capacity(n3);
    for &lam in eigenvalues.iter() {
        let freq = if lam >= 0.0 {
            lam.sqrt() * KCAL_MOL_A2_AMU_TO_CM1
        } else {
            -(-lam).sqrt() * KCAL_MOL_A2_AMU_TO_CM1
        };
        all_frequencies_cm1.push(freq);
    }

    // 9. Separate internal vibrations from external translations/rotations
    // For non-linear molecules, 6 zero modes; for linear molecules, 5 zero modes.
    let n_ext = if natoms == 1 {
        3
    } else if is_linear_molecule(batch) {
        5
    } else {
        6
    };

    let n_vib = n3.saturating_sub(n_ext);
    let mut indexed_modes: Vec<(usize, f64)> = eigenvalues
        .iter()
        .enumerate()
        .map(|(idx, &lam)| (idx, lam.abs()))
        .collect();
    // Sort by absolute eigenvalue ascending: smallest |lambda| first
    indexed_modes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // The first n_ext indices with smallest |lambda| are external (translations/rotations)
    let mut is_ext = vec![false; n3];
    for &(idx, _) in indexed_modes.iter().take(n_ext) {
        is_ext[idx] = true;
    }

    let mut vibrational_frequencies_cm1 = Vec::with_capacity(n_vib);
    for i in 0..n3 {
        if !is_ext[i] {
            vibrational_frequencies_cm1.push(all_frequencies_cm1[i]);
        }
    }

    // 10. Extract Normal Modes & atom displacement vectors
    let mut normal_modes = Vec::with_capacity(n3);
    for (k, &freq) in all_frequencies_cm1.iter().enumerate().take(n3) {
        let mut displacements = Vec::with_capacity(natoms);
        let mut sum_disp_sq = 0.0;

        for (a, &ma) in masses.iter().enumerate().take(natoms) {
            let inv_sqrt_m = 1.0 / ma.sqrt();
            let dx = eigenvectors.get(3 * a, k) * inv_sqrt_m;
            let dy = eigenvectors.get(3 * a + 1, k) * inv_sqrt_m;
            let dz = eigenvectors.get(3 * a + 2, k) * inv_sqrt_m;
            sum_disp_sq += dx * dx + dy * dy + dz * dz;
            displacements.push([dx, dy, dz]);
        }

        // Normalize displacement vector
        let norm = sum_disp_sq.sqrt();
        if norm > 1e-12 {
            for d in displacements.iter_mut() {
                d[0] /= norm;
                d[1] /= norm;
                d[2] /= norm;
            }
        }

        // Reduced mass and force constant
        // 1 mdyne/A = 100 N/m. Force constant k = 4 * pi^2 * c^2 * nu^2 * mu
        let eff_mass = if norm > 1e-12 {
            1.0 / (norm * norm)
        } else {
            1.0
        };
        // Force constant in mdyne/Angstrom:
        // k [mdyne/A] = (nu / 1302.7937)^2 * mu [amu]
        let k_mdyne = (freq.abs() / 1302.7937).powi(2) * eff_mass;

        normal_modes.push(NormalMode {
            frequency_cm1: freq,
            reduced_mass_amu: eff_mass,
            force_constant_mdyne_a: k_mdyne,
            displacements,
        });
    }

    // 11. Zero-Point Vibrational Energy (ZPVE)
    let zpve_kcal_mol = vibrational_frequencies_cm1
        .iter()
        .filter(|&&f| f > 0.0)
        .map(|&f| 0.5 * f * CM1_TO_KCAL_MOL)
        .sum();

    // 12. Statistical Thermodynamics
    let thermo = compute_thermodynamics(
        batch,
        &masses,
        &vibrational_frequencies_cm1,
        zpve_kcal_mol,
        hess_opts.temperature_k,
        hess_opts.pressure_atm,
        hess_opts.rotational_symmetry_number,
    );

    HessianResult {
        cartesian_hessian,
        mass_weighted_hessian,
        all_frequencies_cm1,
        vibrational_frequencies_cm1,
        normal_modes,
        zpve_kcal_mol,
        thermo,
    }
}

/// Displace coordinate of atom $a$, component $\alpha \in \{0, 1, 2\}$ by $\Delta$.
#[inline(always)]
fn displace_coord(batch: &mut MolecularBatch, a: usize, alpha: usize, delta: f64) {
    match alpha {
        0 => batch.x[a] += delta,
        1 => batch.y[a] += delta,
        2 => batch.z[a] += delta,
        _ => unreachable!(),
    }
}

/// Check if molecular coordinates are collinear.
fn is_linear_molecule(batch: &MolecularBatch) -> bool {
    let natoms = batch.natoms;
    if natoms <= 2 {
        return true;
    }
    // Check collinearity with first two distinct atoms
    let v0 = [
        batch.x[1] - batch.x[0],
        batch.y[1] - batch.y[0],
        batch.z[1] - batch.z[0],
    ];
    let n0 = (v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]).sqrt();
    if n0 < 1e-6 {
        return true;
    }
    for i in 2..natoms {
        let vi = [
            batch.x[i] - batch.x[0],
            batch.y[i] - batch.y[0],
            batch.z[i] - batch.z[0],
        ];
        // Cross product v0 x vi
        let cx = v0[1] * vi[2] - v0[2] * vi[1];
        let cy = v0[2] * vi[0] - v0[0] * vi[2];
        let cz = v0[0] * vi[1] - v0[1] * vi[0];
        let cross_norm = (cx * cx + cy * cy + cz * cz).sqrt();
        if cross_norm > 1e-4 {
            return false;
        }
    }
    true
}

/// Projects out rigid translations and rotations via the mass-weighted Eckart projector:
/// $$P_{\text{vib}} = I - \sum_{k=1}^6 \vec{u}_k \vec{u}_k^T$$
/// $$\tilde{\mathcal{H}}_{\text{proj}} = P_{\text{vib}} \tilde{\mathcal{H}} P_{\text{vib}}$$
fn project_out_translations_and_rotations(
    batch: &MolecularBatch,
    masses: &[f64],
    h_mw: &mut AlignedMatrix<f64>,
) {
    let natoms = batch.natoms;
    let n3 = 3 * natoms;

    // 1. Center of mass
    let mut total_mass = 0.0;
    let mut com = [0.0; 3];
    for (i, &m) in masses.iter().enumerate().take(natoms) {
        total_mass += m;
        com[0] += m * batch.x[i];
        com[1] += m * batch.y[i];
        com[2] += m * batch.z[i];
    }
    com[0] /= total_mass;
    com[1] /= total_mass;
    com[2] /= total_mass;

    // 2. Build 6 un-normalized mass-weighted vectors
    let mut tr_vecs = vec![vec![0.0; n3]; 6];
    for i in 0..natoms {
        let sqrt_m = masses[i].sqrt();
        let rx = batch.x[i] - com[0];
        let ry = batch.y[i] - com[1];
        let rz = batch.z[i] - com[2];

        // T_x, T_y, T_z
        tr_vecs[0][3 * i] = sqrt_m;
        tr_vecs[1][3 * i + 1] = sqrt_m;
        tr_vecs[2][3 * i + 2] = sqrt_m;

        // R_x: sqrt(m) * (x_hat x r) = sqrt(m) * [0, -rz, ry]
        tr_vecs[3][3 * i + 1] = -sqrt_m * rz;
        tr_vecs[3][3 * i + 2] = sqrt_m * ry;

        // R_y: sqrt(m) * (y_hat x r) = sqrt(m) * [rz, 0, -rx]
        tr_vecs[4][3 * i] = sqrt_m * rz;
        tr_vecs[4][3 * i + 2] = -sqrt_m * rx;

        // R_z: sqrt(m) * (z_hat x r) = sqrt(m) * [-ry, rx, 0]
        tr_vecs[5][3 * i] = -sqrt_m * ry;
        tr_vecs[5][3 * i + 1] = sqrt_m * rx;
    }

    // 3. Orthonormalize vectors via modified Gram-Schmidt
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(6);
    for v in tr_vecs.iter_mut() {
        for u in &basis {
            let dot: f64 = v.iter().zip(u.iter()).map(|(&a, &b)| a * b).sum();
            for (vi, &ui) in v.iter_mut().zip(u.iter()) {
                *vi -= dot * ui;
            }
        }
        let norm_sq: f64 = v.iter().map(|&a| a * a).sum();
        if norm_sq > 1e-10 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for vi in v.iter_mut() {
                *vi *= inv_norm;
            }
            basis.push(v.clone());
        }
    }

    // 4. Construct projection matrix: P = I - sum(u * u^T)
    let mut p = AlignedMatrix::zeroed(n3, n3);
    for i in 0..n3 {
        p.set(i, i, 1.0);
    }
    for u in &basis {
        for i in 0..n3 {
            let ui = u[i];
            if ui.abs() < 1e-15 {
                continue;
            }
            for (j, &uj) in u.iter().enumerate().take(n3) {
                let current = p.get(i, j);
                p.set(i, j, current - ui * uj);
            }
        }
    }

    // 5. Project Hessian: H_proj = P * H * P
    // Step a: Tmp = H * P
    let mut tmp = AlignedMatrix::zeroed(n3, n3);
    for i in 0..n3 {
        for j in 0..n3 {
            let mut sum = 0.0;
            for k in 0..n3 {
                sum += h_mw.get(i, k) * p.get(k, j);
            }
            tmp.set(i, j, sum);
        }
    }

    // Step b: H_proj = P * Tmp
    for i in 0..n3 {
        for j in 0..n3 {
            let mut sum = 0.0;
            for k in 0..n3 {
                sum += p.get(i, k) * tmp.get(k, j);
            }
            h_mw.set(i, j, sum);
        }
    }
}

/// Compute statistical thermodynamic properties from vibrational wavenumbers and molecular geometry.
fn compute_thermodynamics(
    batch: &MolecularBatch,
    masses: &[f64],
    vib_frequencies: &[f64],
    zpve_kcal_mol: f64,
    t: f64,
    p_atm: f64,
    sigma: f64,
) -> ThermodynamicProperties {
    assert!(t > 0.0, "Temperature must be strictly positive");
    let r = GAS_CONSTANT_CAL; // cal/(mol K)
    let rt = r * t; // cal/mol

    // --- 1. Vibrational Contributions ---
    let mut e_vib = 0.0;
    let mut cv_vib = 0.0;
    let mut s_vib = 0.0;

    for &nu in vib_frequencies {
        if nu <= 10.0 {
            // Ignore near-zero or imaginary modes
            continue;
        }
        let theta = CM1_TO_KELVIN * nu;
        let x = theta / t;
        if x > 100.0 {
            // High frequency mode: thermal population is zero
            continue;
        }
        let exp_x = x.exp();
        let exp_m1 = exp_x - 1.0;

        // E_vib = R * theta / (exp(x) - 1)
        e_vib += r * theta / exp_m1;

        // Cv_vib = R * x^2 * exp(x) / (exp(x) - 1)^2
        cv_vib += r * (x * x * exp_x) / (exp_m1 * exp_m1);

        // S_vib = R * [ x / (exp(x) - 1) - ln(1 - exp(-x)) ]
        let exp_neg_x = (-x).exp();
        s_vib += r * (x / exp_m1 - (1.0 - exp_neg_x).ln());
    }

    // --- 2. Rotational Contributions ---
    let natoms = batch.natoms;
    let (e_rot, cv_rot, s_rot) = if natoms == 1 {
        (0.0, 0.0, 0.0)
    } else if is_linear_molecule(batch) {
        // Linear molecule: 2 rotational degrees of freedom
        let e = rt;
        let cv = r;
        // Compute moment of inertia I about center of mass
        let mut total_mass = 0.0;
        let mut com = [0.0; 3];
        for (i, &m) in masses.iter().enumerate().take(natoms) {
            total_mass += m;
            com[0] += m * batch.x[i];
            com[1] += m * batch.y[i];
            com[2] += m * batch.z[i];
        }
        com[0] /= total_mass;
        com[1] /= total_mass;
        com[2] /= total_mass;

        let mut i_rot = 0.0;
        for (i, &m) in masses.iter().enumerate().take(natoms) {
            let dx = batch.x[i] - com[0];
            let dy = batch.y[i] - com[1];
            let dz = batch.z[i] - com[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            i_rot += m * r2;
        }
        // Conversion from amu * A^2 to kg * m^2:
        // 1 amu = 1.66053906660e-27 kg, 1 A = 1e-10 m -> 1.66053906660e-47 kg m^2
        let i_kg_m2 = i_rot * 1.66053906660e-47;
        let theta_rot = (PLANCK_CONSTANT_ERG_S * 1e-7).powi(2)
            / (8.0 * std::f64::consts::PI.powi(2) * i_kg_m2 * BOLTZMANN_CONSTANT_J_K);
        let s = r * (1.0 + (t / (sigma * theta_rot)).ln());
        (e, cv, s.max(0.0))
    } else {
        // Non-linear molecule: 3 rotational degrees of freedom
        let e = 1.5 * rt;
        let cv = 1.5 * r;

        // Principal moments of inertia
        let moments = compute_principal_moments_of_inertia(batch, masses);
        // theta_A = h^2 / (8 pi^2 I_A k_B)
        let conv = 1.66053906660e-47;
        let h_si = PLANCK_CONSTANT_ERG_S * 1e-7;
        let kb_si = BOLTZMANN_CONSTANT_J_K;
        let pre = h_si * h_si / (8.0 * std::f64::consts::PI.powi(2) * kb_si);

        let t_a = pre / (moments[0].max(1e-6) * conv);
        let t_b = pre / (moments[1].max(1e-6) * conv);
        let t_c = pre / (moments[2].max(1e-6) * conv);

        let q_rot = (std::f64::consts::PI.sqrt() / sigma) * (t.powi(3) / (t_a * t_b * t_c)).sqrt();
        let s = r * (1.5 + q_rot.max(1e-10).ln());
        (e, cv, s.max(0.0))
    };

    // --- 3. Translational Contributions (Sackur-Tetrode) ---
    let e_trans = 1.5 * rt;
    let cp_trans = 2.5 * r;

    let total_mass_amu: f64 = masses.iter().sum();
    let m_kg = total_mass_amu * 1.66053906660e-27;
    let p_pa = p_atm * 101325.0;
    let h_si = PLANCK_CONSTANT_ERG_S * 1e-7;
    let kb_si = BOLTZMANN_CONSTANT_J_K;

    // Thermal de Broglie wavelength lambda = h / sqrt(2 pi m k T)
    let lambda = h_si / (2.0 * std::f64::consts::PI * m_kg * kb_si * t).sqrt();
    let v_per_mol = kb_si * t / p_pa; // m^3 per molecule
    let q_trans = v_per_mol / lambda.powi(3);
    let s_trans = r * (2.5 + q_trans.max(1e-10).ln());

    // --- 4. Total Totals and Gibbs Free Energy ---
    let enthalpy_thermal_cal_mol = e_vib + e_rot + e_trans + rt;
    let cp_total_cal_k_mol = cv_vib + cv_rot + cp_trans;
    let entropy_total_cal_k_mol = s_vib + s_rot + s_trans;

    // Gibbs thermal correction: H_corr - T * S_corr (in kcal/mol)
    let h_total_kcal = (zpve_kcal_mol * 1000.0 + enthalpy_thermal_cal_mol) / 1000.0;
    let ts_kcal = (t * entropy_total_cal_k_mol) / 1000.0;
    let gibbs_correction_kcal_mol = h_total_kcal - ts_kcal;

    ThermodynamicProperties {
        temperature_k: t,
        pressure_atm: p_atm,
        zpve_kcal_mol,
        e_vib_cal_mol: e_vib,
        e_rot_cal_mol: e_rot,
        e_trans_cal_mol: e_trans,
        enthalpy_thermal_cal_mol,
        cv_vib_cal_k_mol: cv_vib,
        cv_rot_cal_k_mol: cv_rot,
        cp_trans_cal_k_mol: cp_trans,
        cp_total_cal_k_mol,
        entropy_vib_cal_k_mol: s_vib,
        entropy_rot_cal_k_mol: s_rot,
        entropy_trans_cal_k_mol: s_trans,
        entropy_total_cal_k_mol,
        gibbs_correction_kcal_mol,
    }
}

/// Computes principal moments of inertia $I_A, I_B, I_C$ in $\text{amu} \cdot \text{\AA}^2$.
fn compute_principal_moments_of_inertia(batch: &MolecularBatch, masses: &[f64]) -> [f64; 3] {
    let natoms = batch.natoms;
    let mut total_mass = 0.0;
    let mut com = [0.0; 3];
    for (i, &m) in masses.iter().enumerate().take(natoms) {
        total_mass += m;
        com[0] += m * batch.x[i];
        com[1] += m * batch.y[i];
        com[2] += m * batch.z[i];
    }
    com[0] /= total_mass;
    com[1] /= total_mass;
    com[2] /= total_mass;

    let mut i_mat = AlignedMatrix::zeroed(3, 3);
    for (i, &m) in masses.iter().enumerate().take(natoms) {
        let x = batch.x[i] - com[0];
        let y = batch.y[i] - com[1];
        let z = batch.z[i] - com[2];

        let i_xx = i_mat.get(0, 0) + m * (y * y + z * z);
        let i_yy = i_mat.get(1, 1) + m * (x * x + z * z);
        let i_zz = i_mat.get(2, 2) + m * (x * x + y * y);
        let i_xy = i_mat.get(0, 1) - m * x * y;
        let i_xz = i_mat.get(0, 2) - m * x * z;
        let i_yz = i_mat.get(1, 2) - m * y * z;

        i_mat.set(0, 0, i_xx);
        i_mat.set(1, 1, i_yy);
        i_mat.set(2, 2, i_zz);
        i_mat.set(0, 1, i_xy);
        i_mat.set(1, 0, i_xy);
        i_mat.set(0, 2, i_xz);
        i_mat.set(2, 0, i_xz);
        i_mat.set(1, 2, i_yz);
        i_mat.set(2, 1, i_yz);
    }

    let mut eigs = AlignedVec64::zeroed(3);
    let mut evecs = AlignedMatrix::zeroed(3, 3);
    diagonalize_symmetric(&i_mat, &mut eigs, &mut evecs);

    [eigs[0], eigs[1], eigs[2]]
}
