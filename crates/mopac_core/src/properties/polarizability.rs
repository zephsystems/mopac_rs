//! Polarizability and Non-Linear Optics (NLO) Tensor Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Direct mathematical translation and modern modernization of OpenMOPAC `static_polarizability.F90`
//! and `polar.F90`.
//!
//! # Methodological Details
//! Evaluates the linear dipole polarizability tensor $\alpha_{ij}$, the first hyperpolarizability
//! tensor $\beta_{ijk}$, and second hyperpolarizability $\gamma_{ijkl}$ via the finite electric
//! field method (Coupled-Perturbed self-consistent field response):
//!
//! 1. Perturbation Hamiltonian:
//!    $$H_{\mu\mu} \gets H_{\mu\mu} - \vec{\mathcal{E}} \cdot \vec{R}_A$$
//!    $$H_{s, p_\alpha} \gets H_{s, p_\alpha} - \mathcal{E}_\alpha D_{1, A}$$
//!    $$E_{\text{nuc}} \gets E_{\text{nuc}} + \sum_A Z_A \vec{\mathcal{E}} \cdot \vec{R}_A$$
//!
//! 2. Polarizability Tensor from dipole response:
//!    $$\alpha_{ij} = \frac{8 [\mu_i(+\mathcal{E}_j) - \mu_i(-\mathcal{E}_j)] - [\mu_i(+2\mathcal{E}_j) - \mu_i(-2\mathcal{E}_j)]}{12 \mathcal{E}_j}$$
//!
//! 3. First Hyperpolarizability $\beta_{ijk}$ (Second Harmonic Generation / Pockels response):
//!    $$\beta_{iii} = \frac{16[\mu_i(+\mathcal{E}_i) + \mu_i(-\mathcal{E}_i)] - [\mu_i(+2\mathcal{E}_i) + \mu_i(-2\mathcal{E}_i)] - 30\mu_{0,i}}{12 \mathcal{E}_i^2}$$
//!    $$\beta_{ijj} = \frac{\mu_i(+\mathcal{E}_j) + \mu_i(-\mathcal{E}_j) - 2\mu_{0,i}}{\mathcal{E}_j^2}$$
//!
//! 4. Second Hyperpolarizability $\gamma_{ijkl}$ (Third Harmonic Generation / Kerr response):
//!    $$\gamma_{iiii} = \frac{[\mu_i(+2\mathcal{E}_i) - \mu_i(-2\mathcal{E}_i)] - 2[\mu_i(+\mathcal{E}_i) - \mu_i(-\mathcal{E}_i)]}{2 \mathcal{E}_i^3}$$
//!
//! 5. Empirical Polarizability Correction (`pol_vol` in OpenMOPAC):
//!    Semi-empirical minimal basis sets underestimate atomic polarizabilities due to the lack of diffuse
//!    functions. OpenMOPAC applies an empirical parameter-dependent volume correction:
//!    $$P = P_{\text{raw}} C_1 + C_2 + \sum_A \text{polvol}(Z_A)$$

use crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS as A0_BOHR;
use crate::parameters::ParameterModel;
use crate::properties::dipole::{compute_dipole_moment, DipoleResult};
use crate::scf::eigensolver::diagonalize_symmetric;
use crate::scf::scf_loop::run_rhf_scf_adaptive_with_field;
use crate::types::{AlignedMatrix, AlignedVec64, MolecularBatch, ScfWorkspace};

/// Electric field conversion factor: 1 a.u. = 51.4220674763 eV / Angstrom (or V / Angstrom)
pub const AU_TO_EV_PER_ANGSTROM: f64 = 27.211386245988 / 0.529177210903;

/// Conversion factor from atomic units of energy (Hartree) to eV
pub const AU_TO_EV: f64 = 27.211386245988;

/// Conversion factor from Debye to atomic units of dipole moment: 1 Debye = 0.393430307 a.u.
pub const DEBYE_TO_AU_DIPOLE: f64 = 0.393430307;

/// Conversion factor from atomic units of polarizability to cubic Angstroms: 1 a.u. = a0^3 = 0.148184711472 A^3
pub const AU_TO_ANGSTROM3: f64 = A0_BOHR * A0_BOHR * A0_BOHR;

/// Conversion factor from atomic units of first hyperpolarizability beta to esu (cm^5 / statC): 1 a.u. = 8.639418e-33 esu
pub const AU_TO_ESU_BETA: f64 = 8.639418e-33;

/// Conversion factor from atomic units of second hyperpolarizability gamma to esu: 1 a.u. = 5.0367e-40 esu
pub const AU_TO_ESU_GAMMA: f64 = 5.0367e-40;

/// Options controlling polarizability and hyperpolarizability calculations.
#[derive(Debug, Clone)]
pub struct PolarizabilityOptions {
    /// Electric field displacement step in atomic units (default: 0.002 a.u. = 0.102844 eV / A)
    pub field_step_au: f64,
    /// Maximum SCF iterations for perturbed calculations (default: 80)
    pub scf_max_iter: usize,
    /// Energy convergence tolerance in eV (default: 1e-9)
    pub energy_tol_ev: f64,
    /// Density matrix convergence tolerance (default: 1e-8)
    pub density_tol: f64,
    /// Whether to evaluate full NDDO diatomic multipoles (default: true)
    pub use_nddo: bool,
    /// Whether to reorient coordinates to principal axes of inertia matching OpenMOPAC STATIC (default: true)
    pub reorient: bool,
    /// Whether to apply empirical pol_vol scaling matching OpenMOPAC (default: true)
    pub apply_empirical_scaling: bool,
    /// Whether to evaluate hyperpolarizabilities beta and gamma (default: true)
    pub compute_hyperpolarizabilities: bool,
}

impl Default for PolarizabilityOptions {
    fn default() -> Self {
        Self {
            field_step_au: 0.002,
            scf_max_iter: 80,
            energy_tol_ev: 1e-9,
            density_tol: 1e-8,
            use_nddo: true,
            reorient: true,
            apply_empirical_scaling: true,
            compute_hyperpolarizabilities: true,
        }
    }
}

/// Comprehensive polarizability and non-linear optical tensor results.
#[derive(Debug, Clone)]
pub struct PolarizabilityResult {
    /// 3x3 polarizability tensor $\alpha_{ij}$ in atomic units ($a_0^3$).
    /// Scaled by `pol_vol` if `apply_empirical_scaling` is enabled, otherwise raw.
    pub alpha_tensor: [[f64; 3]; 3],
    /// Isotropic average polarizability $\bar{\alpha} = \frac{1}{3}\text{Tr}(\alpha)$ in atomic units
    pub alpha_isotropic_au: f64,
    /// Isotropic average polarizability in $\text{\AA}^3$
    pub alpha_isotropic_angstrom3: f64,
    /// Polarizability anisotropy in a.u.
    pub alpha_anisotropy_au: f64,
    /// Principal polarizability eigenvalues in ascending order in atomic units
    pub alpha_eigenvalues: [f64; 3],
    /// Principal polarizability axes (eigenvectors as rows)
    pub alpha_eigenvectors: [[f64; 3]; 3],
    /// Raw uncorrected CP-SCF polarizability tensor in atomic units
    pub alpha_raw_tensor: [[f64; 3]; 3],
    /// Raw uncorrected isotropic polarizability in atomic units
    pub alpha_raw_isotropic_au: f64,
    /// Raw uncorrected isotropic polarizability in $\text{\AA}^3$
    pub alpha_raw_isotropic_angstrom3: f64,
    /// Polarizability tensor derived from total energies (Heat of Formation expansion)
    pub alpha_hof_tensor: [[f64; 3]; 3],
    /// Isotropic polarizability from total energies in atomic units
    pub alpha_hof_isotropic_au: f64,
    /// 3x3x3 first hyperpolarizability tensor $\beta_{ijk}$ in atomic units
    pub beta_tensor: [[[f64; 3]; 3]; 3],
    /// Vector components of first hyperpolarizability $\beta_i = \sum_j \beta_{ijj}$ in atomic units
    pub beta_vector: [f64; 3],
    /// Norm of first hyperpolarizability vector $\|\vec{\beta}\|$ in atomic units
    pub beta_total_au: f64,
    /// Norm of first hyperpolarizability vector in $10^{-33}\ \text{esu}$
    pub beta_total_esu: f64,
    /// Diagonal components of second hyperpolarizability $[\gamma_{xxxx}, \gamma_{yyyy}, \gamma_{zzzz}]$ in atomic units
    pub gamma_diag: [f64; 3],
    /// Average second hyperpolarizability in atomic units
    pub gamma_average_au: f64,
    /// Average second hyperpolarizability in $10^{-36}\ \text{esu}$
    pub gamma_average_esu: f64,
    /// Unperturbed permanent dipole moment in Debye `[x, y, z, total]`
    pub dipole_debye: [f64; 4],
    /// Unperturbed total energy in eV
    pub unperturbed_energy_ev: f64,
}

/// Align molecular batch with its principal axes of inertia.
///
/// Direct port of OpenMOPAC subroutine `axis` in `geometry/axis.F90`.
#[allow(clippy::needless_range_loop)]
pub fn align_principal_axes(batch: &MolecularBatch) -> (MolecularBatch, [[f64; 3]; 3]) {
    let n = batch.natoms;
    if n == 0 {
        return (
            batch.clone(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        );
    }

    // 1. Center of mass
    let mut total_mass = 0.0f64;
    let mut com = [0.0f64; 3];
    for i in 0..n {
        let m = crate::constants::standard_atomic_mass(batch.atomic_numbers[i]);
        total_mass += m;
        com[0] += m * batch.x[i];
        com[1] += m * batch.y[i];
        com[2] += m * batch.z[i];
    }
    if total_mass > 0.0 {
        com[0] /= total_mass;
        com[1] /= total_mass;
        com[2] /= total_mass;
    }

    let mut shifted_x = vec![0.0; n];
    let mut shifted_y = vec![0.0; n];
    let mut shifted_z = vec![0.0; n];
    for i in 0..n {
        shifted_x[i] = batch.x[i] - com[0];
        shifted_y[i] = batch.y[i] - com[1];
        shifted_z[i] = batch.z[i] - com[2];
    }

    // 2. Inertia tensor with OpenMOPAC tie-breaking shifts
    let mut i_mat = AlignedMatrix::zeroed(3, 3);
    let mut t = [0.0f64; 6];
    for i in 0..6 {
        t[i] = (i + 1) as f64 * 1.0e-10;
    }

    for i in 0..n {
        let m = crate::constants::standard_atomic_mass(batch.atomic_numbers[i]);
        let x = shifted_x[i];
        let y = shifted_y[i];
        let z = shifted_z[i];
        t[0] += m * (y * y + z * z);
        t[1] -= m * x * y;
        t[2] += m * (z * z + x * x);
        t[3] -= m * z * x;
        t[4] -= m * y * z;
        t[5] += m * (x * x + y * y);
    }

    i_mat.set(0, 0, t[0]);
    i_mat.set(1, 0, t[1]);
    i_mat.set(0, 1, t[1]);
    i_mat.set(1, 1, t[2]);
    i_mat.set(2, 0, t[3]);
    i_mat.set(0, 2, t[3]);
    i_mat.set(2, 1, t[4]);
    i_mat.set(1, 2, t[4]);
    i_mat.set(2, 2, t[5]);

    let mut eig_vals = AlignedVec64::zeroed(3);
    let mut eig_vecs = AlignedMatrix::zeroed(3, 3);
    diagonalize_symmetric(&i_mat, &mut eig_vals, &mut eig_vecs);

    let mut rotvec = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            rotvec[r][c] = eig_vecs.get(r, c);
        }
    }

    // 3. Make diagonal terms obligate positive (OpenMOPAC axis.F90 convention)
    for c in 0..3 {
        if rotvec[c][c] < 0.0 {
            for r in 0..3 {
                rotvec[r][c] = -rotvec[r][c];
            }
        }
    }

    // 4. Ensure right-handed coordinate system (det > 0)
    let det = rotvec[0][0] * (rotvec[1][1] * rotvec[2][2] - rotvec[2][1] * rotvec[1][2])
        + rotvec[0][1] * (rotvec[1][2] * rotvec[2][0] - rotvec[1][0] * rotvec[2][2])
        + rotvec[0][2] * (rotvec[1][0] * rotvec[2][1] - rotvec[1][1] * rotvec[2][0]);

    if det < 0.0 {
        let mut min_diag = rotvec[0][0];
        let mut min_col = 0;
        for c in 1..3 {
            if rotvec[c][c] < min_diag {
                min_diag = rotvec[c][c];
                min_col = c;
            }
        }
        for r in 0..3 {
            rotvec[r][min_col] = -rotvec[r][min_col];
        }
    }

    // 5. Rotate coordinates: new_coord[axis] = sum_k shifted[k] * rotvec[k][axis]
    let mut new_x = vec![0.0; n];
    let mut new_y = vec![0.0; n];
    let mut new_z = vec![0.0; n];
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_z = 0.0f64;

    for i in 0..n {
        let x = shifted_x[i];
        let y = shifted_y[i];
        let z = shifted_z[i];

        let nx = x * rotvec[0][0] + y * rotvec[1][0] + z * rotvec[2][0];
        let ny = x * rotvec[0][1] + y * rotvec[1][1] + z * rotvec[2][1];
        let nz = x * rotvec[0][2] + y * rotvec[1][2] + z * rotvec[2][2];

        new_x[i] = nx;
        new_y[i] = ny;
        new_z[i] = nz;

        sum_x += nx;
        sum_y += ny;
        sum_z += nz;
    }

    let inv_n = 1.0 / n as f64;
    let cx = sum_x * inv_n;
    let cy = sum_y * inv_n;
    let cz = sum_z * inv_n;

    let mut reoriented_coords = Vec::with_capacity(n);
    for i in 0..n {
        reoriented_coords.push([new_x[i] - cx, new_y[i] - cy, new_z[i] - cz]);
    }

    let new_batch = MolecularBatch::new_with_basis(
        batch.atomic_numbers.clone(),
        &reoriented_coords,
        batch.basis_types.clone(),
    );

    (new_batch, rotvec)
}

/// Empirical polarizability volume correction matching OpenMOPAC `pol_vol` in `polar.F90:4143`.
pub fn compute_pol_vol(model_name: &str, atomic_numbers: &[u8], alpha_au: f64) -> f64 {
    let (c1, c2) = match model_name.to_uppercase().as_str() {
        "MNDO" => (0.5651170, 0.2859620),
        "AM1" => (0.54687, 0.321973),
        "PM3" => (0.2465020, 0.0),
        "PM6" => (0.791396, 0.373638),
        "PM7" => (0.445109, 0.351109),
        _ => (0.0, 0.0),
    };

    if c1 <= 1.0e-4 {
        return alpha_au;
    }

    let a03 = AU_TO_ANGSTROM3;
    let mut polarizability = alpha_au * a03 * c1 + c2;

    for &z in atomic_numbers {
        let pv = match (model_name.to_uppercase().as_str(), z) {
            // PM6
            ("PM6", 1) => 0.262114,
            ("PM6", 6) => 0.485071,
            ("PM6", 7) => 0.204743,
            ("PM6", 8) => 0.154301,
            ("PM6", 9) => 0.199611,
            ("PM6", 14) => 1.886110,
            ("PM6", 15) => 2.314610,
            ("PM6", 16) => 1.453310,
            ("PM6", 17) => 1.236210,
            ("PM6", 35) => 2.142420,
            ("PM6", 53) => 3.823160,

            // AM1
            ("AM1", 1) => 0.1719890,
            ("AM1", 6) => 0.8472950,
            ("AM1", 7) => 0.6838130,
            ("AM1", 8) => 0.3245890,
            ("AM1", 9) => 0.1286740,
            ("AM1", 17) => 1.6635500,
            ("AM1", 35) => 2.5941900,
            ("AM1", 53) => 4.5523800,

            // PM7
            ("PM7", 1) => 0.229769,
            ("PM7", 6) => 0.935524,
            ("PM7", 7) => 0.551046,
            ("PM7", 8) => 0.312052,
            ("PM7", 9) => 0.121273,
            ("PM7", 14) => 3.453040,
            ("PM7", 15) => 2.858410,
            ("PM7", 16) => 1.726420,
            ("PM7", 17) => 1.617180,
            ("PM7", 35) => 2.645090,
            ("PM7", 53) => 4.400520,

            // PM3
            ("PM3", 1) => 0.1785810,
            ("PM3", 6) => 1.2427300,
            ("PM3", 7) => 0.9201900,
            ("PM3", 8) => 0.4346050,
            ("PM3", 9) => 0.2358930,
            ("PM3", 17) => 1.8556000,
            ("PM3", 35) => 2.8454200,
            ("PM3", 53) => 4.7997300,

            // MNDO
            ("MNDO", 1) => 0.1836440,
            ("MNDO", 6) => 0.8215720,
            ("MNDO", 7) => 0.7031620,
            ("MNDO", 8) => 0.3615210,
            ("MNDO", 9) => 0.2000100,
            ("MNDO", 17) => 1.6618700,
            ("MNDO", 35) => 2.6532400,
            ("MNDO", 53) => 4.5973300,

            _ => 0.0,
        };
        polarizability += pv;
    }

    polarizability / a03
}

/// Helper to run an SCF calculation under a specific static electric field.
fn run_field_scf(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    ws: &mut ScfWorkspace,
    efield_au: [f64; 3],
    options: &PolarizabilityOptions,
) -> (f64, DipoleResult) {
    let efield_ev_a = [
        efield_au[0] * AU_TO_EV_PER_ANGSTROM,
        efield_au[1] * AU_TO_EV_PER_ANGSTROM,
        efield_au[2] * AU_TO_EV_PER_ANGSTROM,
    ];

    ws.reset();
    let res = run_rhf_scf_adaptive_with_field(
        batch,
        model,
        ws,
        options.scf_max_iter,
        options.energy_tol_ev,
        options.density_tol,
        options.use_nddo,
        efield_ev_a,
    );

    let dip = compute_dipole_moment(batch, model, &ws.density);
    (res.total_energy_ev, dip)
}

/// Compute the molecular polarizability tensor $\alpha_{ij}$ and hyperpolarizabilities $\beta_{ijk}, \gamma_{ijkl}$.
///
/// Port of canonical OpenMOPAC `static_polarizability.F90`.
#[allow(clippy::needless_range_loop)]
pub fn compute_polarizability(
    batch_input: &MolecularBatch,
    model: &dyn ParameterModel,
    options: &PolarizabilityOptions,
) -> PolarizabilityResult {
    let (batch, _rotvec) = if options.reorient {
        align_principal_axes(batch_input)
    } else {
        (
            batch_input.clone(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
    };

    let mut ws = ScfWorkspace::allocate(batch.norbs);

    // 1. Unperturbed state
    let (e0, dip0) = run_field_scf(&batch, model, &mut ws, [0.0, 0.0, 0.0], options);

    let f = options.field_step_au;
    let f2 = 2.0 * f;

    // Convert Debye dipole components to a.u.
    let dip0_au = [
        dip0.total[0] * DEBYE_TO_AU_DIPOLE,
        dip0.total[1] * DEBYE_TO_AU_DIPOLE,
        dip0.total[2] * DEBYE_TO_AU_DIPOLE,
    ];

    // Matrices to store dipole response: dip_plus[axis][point], dip_minus[axis][point]
    // axis 0 = X, 1 = Y, 2 = Z
    // point 0 = 1*F, point 1 = 2*F
    let mut dip_plus = [[[0.0f64; 3]; 2]; 3];
    let mut dip_minus = [[[0.0f64; 3]; 2]; 3];
    let mut e_plus = [[0.0f64; 2]; 3];
    let mut e_minus = [[0.0f64; 2]; 3];

    for axis in 0..3 {
        // +1*F
        let mut f_vec = [0.0; 3];
        f_vec[axis] = f;
        let (e, dip) = run_field_scf(&batch, model, &mut ws, f_vec, options);
        e_plus[axis][0] = e;
        dip_plus[axis][0] = [
            dip.total[0] * DEBYE_TO_AU_DIPOLE,
            dip.total[1] * DEBYE_TO_AU_DIPOLE,
            dip.total[2] * DEBYE_TO_AU_DIPOLE,
        ];

        // -1*F
        f_vec[axis] = -f;
        let (e, dip) = run_field_scf(&batch, model, &mut ws, f_vec, options);
        e_minus[axis][0] = e;
        dip_minus[axis][0] = [
            dip.total[0] * DEBYE_TO_AU_DIPOLE,
            dip.total[1] * DEBYE_TO_AU_DIPOLE,
            dip.total[2] * DEBYE_TO_AU_DIPOLE,
        ];

        // +2*F
        f_vec[axis] = f2;
        let (e, dip) = run_field_scf(&batch, model, &mut ws, f_vec, options);
        e_plus[axis][1] = e;
        dip_plus[axis][1] = [
            dip.total[0] * DEBYE_TO_AU_DIPOLE,
            dip.total[1] * DEBYE_TO_AU_DIPOLE,
            dip.total[2] * DEBYE_TO_AU_DIPOLE,
        ];

        // -2*F
        f_vec[axis] = -f2;
        let (e, dip) = run_field_scf(&batch, model, &mut ws, f_vec, options);
        e_minus[axis][1] = e;
        dip_minus[axis][1] = [
            dip.total[0] * DEBYE_TO_AU_DIPOLE,
            dip.total[1] * DEBYE_TO_AU_DIPOLE,
            dip.total[2] * DEBYE_TO_AU_DIPOLE,
        ];
    }

    // 2. Compute raw polarizability tensor alpha_raw_ij
    // OpenMOPAC 4-point central difference formula on dipole response:
    // alpha_{ij} = (8 * (mu_i(+F_j) - mu_i(-F_j)) - (mu_i(+2F_j) - mu_i(-2F_j))) / (12 * F_j)
    let mut alpha_raw = [[0.0f64; 3]; 3];
    let inv_12f = 1.0 / (12.0 * f);

    for j in 0..3 {
        for i in 0..3 {
            let diff1 = dip_plus[j][0][i] - dip_minus[j][0][i];
            let diff2 = dip_plus[j][1][i] - dip_minus[j][1][i];
            alpha_raw[i][j] = (8.0 * diff1 - diff2) * inv_12f;
        }
    }

    // Symmetrize raw alpha tensor
    for i in 0..3 {
        for j in 0..i {
            let avg = 0.5 * (alpha_raw[i][j] + alpha_raw[j][i]);
            alpha_raw[i][j] = avg;
            alpha_raw[j][i] = avg;
        }
    }

    let alpha_raw_iso = (alpha_raw[0][0] + alpha_raw[1][1] + alpha_raw[2][2]) / 3.0;
    let alpha_raw_iso_a3 = alpha_raw_iso * AU_TO_ANGSTROM3;

    // 3. Compute energy-derived polarizability tensor (Heat of Formation expansion)
    let mut alpha_hof = [[0.0f64; 3]; 3];
    let inv_f2_hartree = 1.0 / (f * f * AU_TO_EV);
    for i in 0..3 {
        let eterm = 2.5 * e0 - (4.0 / 3.0) * (e_plus[i][0] + e_minus[i][0])
            + (1.0 / 12.0) * (e_plus[i][1] + e_minus[i][1]);
        alpha_hof[i][i] = eterm * inv_f2_hartree;
    }

    // 4. Apply empirical pol_vol scaling if enabled
    let mut alpha = alpha_raw;
    if options.apply_empirical_scaling {
        let model_name = model.name();
        let atomic_numbers = &batch.atomic_numbers;

        // If reoriented, diagonal elements are principal components
        if options.reorient {
            for i in 0..3 {
                alpha[i][i] = compute_pol_vol(model_name, atomic_numbers, alpha_raw[i][i]);
                alpha_hof[i][i] = compute_pol_vol(model_name, atomic_numbers, alpha_hof[i][i]);
            }
        } else {
            // In an arbitrary frame, scale principal eigenvalues to preserve rotational invariance
            let mut a_mat = AlignedMatrix::zeroed(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    a_mat.set(i, j, alpha_raw[i][j]);
                }
            }
            let mut eig_vals = AlignedVec64::zeroed(3);
            let mut eig_vecs = AlignedMatrix::zeroed(3, 3);
            diagonalize_symmetric(&a_mat, &mut eig_vals, &mut eig_vecs);

            let mut scaled_eigs = [0.0; 3];
            for k in 0..3 {
                scaled_eigs[k] = compute_pol_vol(model_name, atomic_numbers, eig_vals[k]);
            }

            // Reconstruct: alpha = V * diag(scaled_eigs) * V^T
            for i in 0..3 {
                for j in 0..3 {
                    let mut sum = 0.0;
                    for k in 0..3 {
                        sum += eig_vecs.get(i, k) * scaled_eigs[k] * eig_vecs.get(j, k);
                    }
                    alpha[i][j] = sum;
                }
            }
        }
    }

    // Diagonalize final alpha tensor
    let mut a_mat = AlignedMatrix::zeroed(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            a_mat.set(i, j, alpha[i][j]);
        }
    }
    let mut eig_vals = AlignedVec64::zeroed(3);
    let mut eig_vecs = AlignedMatrix::zeroed(3, 3);
    diagonalize_symmetric(&a_mat, &mut eig_vals, &mut eig_vecs);

    let alpha_eigenvalues = [eig_vals[0], eig_vals[1], eig_vals[2]];
    let mut alpha_eigenvectors = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            alpha_eigenvectors[i][j] = eig_vecs.get(i, j);
        }
    }

    let alpha_iso = (alpha[0][0] + alpha[1][1] + alpha[2][2]) / 3.0;
    let alpha_iso_a3 = alpha_iso * AU_TO_ANGSTROM3;

    let aniso_sq = 0.5
        * ((alpha[0][0] - alpha[1][1]).powi(2)
            + (alpha[1][1] - alpha[2][2]).powi(2)
            + (alpha[2][2] - alpha[0][0]).powi(2)
            + 6.0 * (alpha[0][1].powi(2) + alpha[0][2].powi(2) + alpha[1][2].powi(2)));
    let alpha_aniso = aniso_sq.max(0.0).sqrt();

    let alpha_hof_iso = (alpha_hof[0][0] + alpha_hof[1][1] + alpha_hof[2][2]) / 3.0;

    // 5. Hyperpolarizabilities beta_iii and gamma_iiii
    let mut beta = [[[0.0f64; 3]; 3]; 3];
    let mut gamma_diag = [0.0f64; 3];

    let inv_12f2 = 1.0 / (12.0 * f * f);
    let inv_2f3 = 1.0 / (2.0 * f * f * f);

    for i in 0..3 {
        // beta_{iii} with permanent dipole subtraction:
        // (16*(mu(+F) + mu(-F)) - (mu(+2F) + mu(-2F)) - 30*mu0) / (12 * F^2)
        let sum1 = dip_plus[i][0][i] + dip_minus[i][0][i];
        let sum2 = dip_plus[i][1][i] + dip_minus[i][1][i];
        let b_val = (16.0 * sum1 - sum2 - 30.0 * dip0_au[i]) * inv_12f2;
        beta[i][i][i] = b_val;

        // gamma_{iiii} = ( (mu_i(+2F_i) - mu_i(-2F_i)) - 2(mu_i(+F_i) - mu_i(-F_i)) ) / (2 * F^3)
        let g_val = ((dip_plus[i][1][i] - dip_minus[i][1][i])
            - 2.0 * (dip_plus[i][0][i] - dip_minus[i][0][i]))
            * inv_2f3;
        gamma_diag[i] = g_val;
    }

    // Cross components of beta: beta_{ijj} = (mu_i(+F_j) + mu_i(-F_j) - 2*mu0_i) / F^2
    let inv_f2 = 1.0 / (f * f);
    for j in 0..3 {
        for i in 0..3 {
            if i != j {
                let b_ijj = (dip_plus[j][0][i] + dip_minus[j][0][i] - 2.0 * dip0_au[i]) * inv_f2;
                beta[i][j][j] = b_ijj;
                beta[j][i][j] = b_ijj;
                beta[j][j][i] = b_ijj;
            }
        }
    }

    let mut beta_vec = [0.0f64; 3];
    for i in 0..3 {
        beta_vec[i] = beta[i][0][0] + beta[i][1][1] + beta[i][2][2];
    }
    let beta_tot_au = (beta_vec[0].powi(2) + beta_vec[1].powi(2) + beta_vec[2].powi(2)).sqrt();
    let beta_tot_esu = beta_tot_au * AU_TO_ESU_BETA;

    // Cross components of gamma from cross-field calculations
    let mut gamma_cross_sum = 0.0f64;
    let pairs = [(0, 1), (0, 2), (1, 2)];
    let inv_4f4_hartree = 1.0 / (4.0 * f * f * f * f * AU_TO_EV);

    if options.compute_hyperpolarizabilities {
        for &(j, k) in &pairs {
            let mut f_pp = [0.0; 3];
            f_pp[j] = f;
            f_pp[k] = f;
            let (e_pp, _) = run_field_scf(&batch, model, &mut ws, f_pp, options);

            let mut f_pm = [0.0; 3];
            f_pm[j] = f;
            f_pm[k] = -f;
            let (e_pm, _) = run_field_scf(&batch, model, &mut ws, f_pm, options);

            let mut f_mp = [0.0; 3];
            f_mp[j] = -f;
            f_mp[k] = f;
            let (e_mp, _) = run_field_scf(&batch, model, &mut ws, f_mp, options);

            let mut f_mm = [0.0; 3];
            f_mm[j] = -f;
            f_mm[k] = -f;
            let (e_mm, _) = run_field_scf(&batch, model, &mut ws, f_mm, options);

            // d^4 E / (dF_j^2 dF_k^2)
            let g_jjkk = (e_pp + e_mm - e_pm - e_mp) * inv_4f4_hartree;
            gamma_cross_sum += g_jjkk;
        }
    }

    let gamma_avg_au =
        (gamma_diag[0] + gamma_diag[1] + gamma_diag[2] + 2.0 * gamma_cross_sum) / 5.0;
    let gamma_avg_esu = gamma_avg_au * AU_TO_ESU_GAMMA;

    PolarizabilityResult {
        alpha_tensor: alpha,
        alpha_isotropic_au: alpha_iso,
        alpha_isotropic_angstrom3: alpha_iso_a3,
        alpha_anisotropy_au: alpha_aniso,
        alpha_eigenvalues,
        alpha_eigenvectors,
        alpha_raw_tensor: alpha_raw,
        alpha_raw_isotropic_au: alpha_raw_iso,
        alpha_raw_isotropic_angstrom3: alpha_raw_iso_a3,
        alpha_hof_tensor: alpha_hof,
        alpha_hof_isotropic_au: alpha_hof_iso,
        beta_tensor: beta,
        beta_vector: beta_vec,
        beta_total_au: beta_tot_au,
        beta_total_esu: beta_tot_esu,
        gamma_diag,
        gamma_average_au: gamma_avg_au,
        gamma_average_esu: gamma_avg_esu,
        dipole_debye: dip0.total,
        unperturbed_energy_ev: e0,
    }
}
