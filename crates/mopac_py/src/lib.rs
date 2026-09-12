//! PyO3 Python Bindings for MOPAC_RS Quantum Chemistry Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Exposes single-point SCF energy, heat of formation, electric dipole moment,
//! Mulliken partial charges, analytical Cartesian gradients, and geometry optimization
//! directly to Python for seamless integration with PyTorch, RDKit, and ASE.

#![allow(clippy::useless_conversion)]
#![allow(clippy::too_many_arguments)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopac_core::constants::codata2018::EV_TO_KCAL_MOL;
use mopac_core::corrections::dispersion::{
    compute_dispersion_energy_and_gradients, DispersionModel,
};
use mopac_core::corrections::h_bonds4::{
    compute_h4_energy, compute_hh_repulsion_energy_and_gradients, H4Parameters,
};
use mopac_core::gradients::nuclear_gradients::{
    compute_cartesian_gradients_with_options, GradientWorkspace,
};
use mopac_core::opt::eigenvector_following::{
    optimize_transition_state, EigenvectorFollowingWorkspace, TransitionStateOptions,
};
use mopac_core::opt::hessian_update::HessianUpdateScheme;
use mopac_core::opt::lbfgs::{optimize_geometry_lbfgs, OptimizationOptions};
use mopac_core::parameters::{
    Am1Model, MndoModel, ParameterModel, Pm3Model, Pm6Model, Pm7Model, Rm1Model,
};
use mopac_core::properties::{
    compute_dipole_moment, compute_heat_of_formation, compute_mulliken_population,
};
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::solvation::CosmoParams;
use mopac_core::types::{MolecularBatch, ScfWorkspace};
use mopac_core::vibrations::{compute_hessian_and_frequencies, HessianOptions};

/// Resolve model instance by string identifier.
fn get_model(method: &str) -> PyResult<Box<dyn ParameterModel>> {
    match method.to_uppercase().as_str() {
        "PM7" => Ok(Box::new(Pm7Model)),
        "PM6" => Ok(Box::new(Pm6Model)),
        "AM1" => Ok(Box::new(Am1Model)),
        "RM1" => Ok(Box::new(Rm1Model)),
        "PM3" => Ok(Box::new(Pm3Model)),
        "MNDO" => Ok(Box::new(MndoModel)),
        other => Err(PyValueError::new_err(format!(
            "Unsupported semi-empirical method: '{}'. Supported methods: PM7, PM6, AM1, RM1, PM3, MNDO",
            other
        ))),
    }
}

/// Calculation results containing quantum chemical properties and analytical gradients.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct CalculationResult {
    /// Total SCF energy in electron-volts (eV)
    pub total_energy_ev: f64,
    /// Electronic energy in electron-volts (eV)
    pub electronic_energy_ev: f64,
    /// Core-core nuclear repulsion energy in electron-volts (eV)
    pub nuclear_repulsion_ev: f64,
    /// Molecular binding energy relative to isolated atoms in electron-volts (eV)
    pub binding_energy_ev: f64,
    /// Standard heat of formation \Delta H_f^\circ in kcal/mol
    pub heat_of_formation_kcal: f64,
    /// Cartesian analytical energy gradients in eV / \AA (shape: [natoms, 3])
    pub gradients_ev_angstrom: Vec<[f64; 3]>,
    /// Cartesian analytical energy gradients in kcal / (mol * \AA) (shape: [natoms, 3])
    pub gradients_kcal_mol_angstrom: Vec<[f64; 3]>,
    /// Electric dipole moment components and magnitude [dx, dy, dz, total] in Debye
    pub dipole_debye: [f64; 4],
    /// Diagonal density atomic partial charges (shape: [natoms])
    pub atomic_charges: Vec<f64>,
    /// Mulliken net partial atomic charges (shape: [natoms])
    pub mulliken_charges: Vec<f64>,
    /// Whether SCF converged
    pub converged: bool,
    /// Number of SCF iterations executed
    pub scf_iterations: usize,
    /// Highest Occupied Molecular Orbital energy in eV
    pub homo_energy_ev: f64,
    /// Lowest Unoccupied Molecular Orbital energy in eV
    pub lumo_energy_ev: f64,
    /// Fundamental HOMO-LUMO gap in eV
    pub homo_lumo_gap_ev: f64,
}

#[pymethods]
impl CalculationResult {
    fn __repr__(&self) -> String {
        format!(
            "<CalculationResult E={:.6} eV, dHf={:.3} kcal/mol, dipole={:.3} D, converged={}>",
            self.total_energy_ev, self.heat_of_formation_kcal, self.dipole_debye[3], self.converged
        )
    }

    /// Convert calculation result to a standard Python dictionary.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("total_energy_ev", self.total_energy_ev)?;
        dict.set_item("electronic_energy_ev", self.electronic_energy_ev)?;
        dict.set_item("nuclear_repulsion_ev", self.nuclear_repulsion_ev)?;
        dict.set_item("binding_energy_ev", self.binding_energy_ev)?;
        dict.set_item("heat_of_formation_kcal", self.heat_of_formation_kcal)?;
        dict.set_item("gradients_ev_angstrom", &self.gradients_ev_angstrom)?;
        dict.set_item(
            "gradients_kcal_mol_angstrom",
            &self.gradients_kcal_mol_angstrom,
        )?;
        dict.set_item("dipole_debye", self.dipole_debye.to_vec())?;
        dict.set_item("atomic_charges", &self.atomic_charges)?;
        dict.set_item("mulliken_charges", &self.mulliken_charges)?;
        dict.set_item("converged", self.converged)?;
        dict.set_item("scf_iterations", self.scf_iterations)?;
        dict.set_item("homo_energy_ev", self.homo_energy_ev)?;
        dict.set_item("lumo_energy_ev", self.lumo_energy_ev)?;
        dict.set_item("homo_lumo_gap_ev", self.homo_lumo_gap_ev)?;
        Ok(dict)
    }
}

/// Result of molecular geometry optimization.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct OptimizationPyResult {
    /// Whether geometry optimization converged
    pub converged: bool,
    /// Number of quasi-Newton L-BFGS cycles executed
    pub cycles: usize,
    /// Initial total energy in eV
    pub initial_energy_ev: f64,
    /// Final total energy in eV
    pub final_energy_ev: f64,
    /// Final heat of formation in kcal/mol
    pub final_heat_of_formation_kcal: f64,
    /// Final RMS gradient in kcal / (mol * \AA)
    pub final_grad_rms: f64,
    /// Final maximum gradient component in kcal / (mol * \AA)
    pub final_grad_max: f64,
    /// Optimized Cartesian coordinates in \AA (shape: [natoms, 3])
    pub coordinates: Vec<[f64; 3]>,
    /// Final calculation result at optimized geometry
    pub final_result: CalculationResult,
}

#[pymethods]
impl OptimizationPyResult {
    fn __repr__(&self) -> String {
        format!(
            "<OptimizationPyResult converged={} cycles={} E_final={:.6} eV, dHf={:.3} kcal/mol, grad_rms={:.4}>",
            self.converged, self.cycles, self.final_energy_ev, self.final_heat_of_formation_kcal, self.final_grad_rms
        )
    }
}

/// Result of transition state search via Eigenvector Following.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct TransitionStatePyResult {
    /// Whether the transition state search converged
    pub converged: bool,
    /// Number of EF optimization cycles executed
    pub cycles: usize,
    /// Final total energy in eV
    pub final_energy_ev: f64,
    /// Final heat of formation in kcal/mol
    pub final_heat_of_formation_kcal: f64,
    /// Initial RMS gradient in kcal / (mol * \AA)
    pub initial_grad_rms: f64,
    /// Final RMS gradient in kcal / (mol * \AA)
    pub final_grad_rms: f64,
    /// Final maximum gradient component in kcal / (mol * \AA)
    pub final_grad_max: f64,
    /// Eigenvalue of the transition state mode along which energy is maximized (kcal / (mol * \AA^2))
    pub ts_mode_eigenvalue: f64,
    /// Transition state mode index (0-indexed)
    pub ts_mode_index: usize,
    /// Transition state Cartesian coordinates in \AA (shape: [natoms, 3])
    pub coordinates: Vec<[f64; 3]>,
    /// Final calculation result at the transition state
    pub final_result: CalculationResult,
}

#[pymethods]
impl TransitionStatePyResult {
    fn __repr__(&self) -> String {
        format!(
            "<TransitionStatePyResult converged={} cycles={} E_final={:.6} eV, dHf={:.3} kcal/mol, ts_mode_val={:.3}, grad_rms={:.4}>",
            self.converged, self.cycles, self.final_energy_ev, self.final_heat_of_formation_kcal, self.ts_mode_eigenvalue, self.final_grad_rms
        )
    }

    /// Convert transition state result to a standard Python dictionary.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("converged", self.converged)?;
        dict.set_item("cycles", self.cycles)?;
        dict.set_item("final_energy_ev", self.final_energy_ev)?;
        dict.set_item(
            "final_heat_of_formation_kcal",
            self.final_heat_of_formation_kcal,
        )?;
        dict.set_item("initial_grad_rms", self.initial_grad_rms)?;
        dict.set_item("final_grad_rms", self.final_grad_rms)?;
        dict.set_item("final_grad_max", self.final_grad_max)?;
        dict.set_item("ts_mode_eigenvalue", self.ts_mode_eigenvalue)?;
        dict.set_item("ts_mode_index", self.ts_mode_index)?;
        dict.set_item("coordinates", &self.coordinates)?;
        dict.set_item("final_result", self.final_result.to_dict(py)?)?;
        Ok(dict)
    }
}

/// Normal vibrational mode with harmonic frequency and Cartesian displacement vectors.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct NormalModePy {
    /// Harmonic frequency in cm^-1 (negative if imaginary)
    pub frequency_cm1: f64,
    /// Effective reduced mass in amu
    pub reduced_mass_amu: f64,
    /// Force constant in mdyne/Å
    pub force_constant_mdyne_a: f64,
    /// Normalized Cartesian displacements (delta x, delta y, delta z) for each atom
    pub displacements: Vec<[f64; 3]>,
}

#[pymethods]
impl NormalModePy {
    fn __repr__(&self) -> String {
        format!(
            "<NormalMode freq={:.1} cm^-1, mass={:.3} amu, k={:.3} mdyne/A>",
            self.frequency_cm1, self.reduced_mass_amu, self.force_constant_mdyne_a
        )
    }

    /// Convert normal mode to Python dictionary.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("frequency_cm1", self.frequency_cm1)?;
        dict.set_item("reduced_mass_amu", self.reduced_mass_amu)?;
        dict.set_item("force_constant_mdyne_a", self.force_constant_mdyne_a)?;
        dict.set_item("displacements", &self.displacements)?;
        Ok(dict)
    }
}

/// Statistical thermodynamic properties computed from vibrational, rotational, and translational partition functions.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct ThermodynamicsPy {
    pub temperature_k: f64,
    pub pressure_atm: f64,
    /// Zero-Point Vibrational Energy in kcal/mol
    pub zpve_kcal_mol: f64,
    /// Thermal vibrational energy E_vib(T) in cal/mol
    pub e_vib_cal_mol: f64,
    /// Thermal rotational energy E_rot(T) in cal/mol
    pub e_rot_cal_mol: f64,
    /// Thermal translational energy E_trans(T) in cal/mol
    pub e_trans_cal_mol: f64,
    /// Total thermal enthalpy correction H(T) - H(0) in cal/mol
    pub enthalpy_thermal_cal_mol: f64,
    /// Constant volume vibrational heat capacity Cv,vib in cal/(mol·K)
    pub cv_vib_cal_k_mol: f64,
    /// Constant volume rotational heat capacity Cv,rot in cal/(mol·K)
    pub cv_rot_cal_k_mol: f64,
    /// Constant pressure translational heat capacity Cp,trans = 5/2 R in cal/(mol·K)
    pub cp_trans_cal_k_mol: f64,
    /// Total constant pressure heat capacity Cp(T) = Cv + R in cal/(mol·K)
    pub cp_total_cal_k_mol: f64,
    /// Vibrational entropy S_vib in cal/(mol·K)
    pub entropy_vib_cal_k_mol: f64,
    /// Rotational entropy S_rot in cal/(mol·K)
    pub entropy_rot_cal_k_mol: f64,
    /// Translational entropy S_trans (Sackur-Tetrode) in cal/(mol·K)
    pub entropy_trans_cal_k_mol: f64,
    /// Total standard entropy S°(T) in cal/(mol·K)
    pub entropy_total_cal_k_mol: f64,
    /// Gibbs free energy thermal correction G_corr = H_thermal - T*S° in kcal/mol
    pub gibbs_correction_kcal_mol: f64,
}

#[pymethods]
impl ThermodynamicsPy {
    fn __repr__(&self) -> String {
        format!(
            "<Thermodynamics T={:.2} K, ZPVE={:.2} kcal/mol, S°={:.2} cal/(mol·K), H_thermal={:.2} cal/mol, G_corr={:.2} kcal/mol>",
            self.temperature_k, self.zpve_kcal_mol, self.entropy_total_cal_k_mol, self.enthalpy_thermal_cal_mol, self.gibbs_correction_kcal_mol
        )
    }

    /// Convert thermodynamics properties to Python dictionary.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("temperature_k", self.temperature_k)?;
        dict.set_item("pressure_atm", self.pressure_atm)?;
        dict.set_item("zpve_kcal_mol", self.zpve_kcal_mol)?;
        dict.set_item("e_vib_cal_mol", self.e_vib_cal_mol)?;
        dict.set_item("e_rot_cal_mol", self.e_rot_cal_mol)?;
        dict.set_item("e_trans_cal_mol", self.e_trans_cal_mol)?;
        dict.set_item("enthalpy_thermal_cal_mol", self.enthalpy_thermal_cal_mol)?;
        dict.set_item("cv_vib_cal_k_mol", self.cv_vib_cal_k_mol)?;
        dict.set_item("cv_rot_cal_k_mol", self.cv_rot_cal_k_mol)?;
        dict.set_item("cp_trans_cal_k_mol", self.cp_trans_cal_k_mol)?;
        dict.set_item("cp_total_cal_k_mol", self.cp_total_cal_k_mol)?;
        dict.set_item("entropy_vib_cal_k_mol", self.entropy_vib_cal_k_mol)?;
        dict.set_item("entropy_rot_cal_k_mol", self.entropy_rot_cal_k_mol)?;
        dict.set_item("entropy_trans_cal_k_mol", self.entropy_trans_cal_k_mol)?;
        dict.set_item("entropy_total_cal_k_mol", self.entropy_total_cal_k_mol)?;
        dict.set_item("gibbs_correction_kcal_mol", self.gibbs_correction_kcal_mol)?;
        Ok(dict)
    }
}

/// Comprehensive results of Hessian and normal coordinate analysis.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct VibrationalResultPy {
    /// All 3N harmonic frequencies in cm^-1 (sorted ascending)
    pub all_frequencies_cm1: Vec<f64>,
    /// Genuine internal vibrational frequencies (3N-6 or 3N-5) in cm^-1
    pub vibrational_frequencies_cm1: Vec<f64>,
    /// Normal modes with atom displacement vectors
    pub normal_modes: Vec<NormalModePy>,
    /// Zero-Point Vibrational Energy (ZPVE) in kcal/mol
    pub zpve_kcal_mol: f64,
    /// Statistical thermodynamic properties
    pub thermo: ThermodynamicsPy,
    /// Whether molecule is a transition state (first vibrational frequency < -10 cm^-1)
    pub is_transition_state: bool,
    /// Full Cartesian Hessian matrix [3N x 3N] in kcal / (mol * Å^2)
    pub cartesian_hessian: Vec<Vec<f64>>,
}

#[pymethods]
impl VibrationalResultPy {
    fn __repr__(&self) -> String {
        format!(
            "<VibrationalResult num_vib={}, ZPVE={:.2} kcal/mol, is_TS={}>",
            self.vibrational_frequencies_cm1.len(),
            self.zpve_kcal_mol,
            self.is_transition_state
        )
    }

    /// Convert vibrational result to Python dictionary.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("all_frequencies_cm1", &self.all_frequencies_cm1)?;
        dict.set_item(
            "vibrational_frequencies_cm1",
            &self.vibrational_frequencies_cm1,
        )?;
        dict.set_item("zpve_kcal_mol", self.zpve_kcal_mol)?;
        dict.set_item("is_transition_state", self.is_transition_state)?;
        dict.set_item("thermo", self.thermo.to_dict(py)?)?;
        dict.set_item("cartesian_hessian", &self.cartesian_hessian)?;
        let modes_list = pyo3::types::PyList::empty_bound(py);
        for m in &self.normal_modes {
            modes_list.append(m.to_dict(py)?)?;
        }
        dict.set_item("normal_modes", modes_list)?;
        Ok(dict)
    }
}

/// Internal engine execution for single-point calculation.
fn run_calculation_internal(
    atomic_numbers: &[u8],
    coordinates: &[[f64; 3]],
    method: &str,
    cosmo_eps: Option<f64>,
    dispersion: Option<&str>,
    h_bonds: bool,
    use_nddo: bool,
    max_iter: usize,
    energy_tol_ev: f64,
    density_tol: f64,
    level_shift_ev: f64,
    damping: f64,
) -> PyResult<CalculationResult> {
    let natoms = atomic_numbers.len();
    if natoms == 0 {
        return Err(PyValueError::new_err("atomic_numbers cannot be empty"));
    }
    if coordinates.len() != natoms {
        return Err(PyValueError::new_err(format!(
            "Length of atomic_numbers ({}) does not match coordinates ({})",
            natoms,
            coordinates.len()
        )));
    }

    let model = get_model(method)?;

    let mut total_valence_elecs = 0.0;
    for &z in atomic_numbers {
        if let Some(p) = model.get_element(z) {
            total_valence_elecs += p.core_charge;
        } else {
            return Err(PyValueError::new_err(format!(
                "Unsupported element Z={} for semi-empirical method '{}'",
                z, method
            )));
        }
    }

    let nelec = total_valence_elecs.round() as usize;
    if !nelec.is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "Open-shell radical detected ({} valence electrons). Closed-shell RHF requires an even number of valence electrons (requires UHF/ROHF).",
            nelec
        )));
    }

    let mut batch = MolecularBatch::new(atomic_numbers.to_vec(), coordinates);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);

    let cosmo = cosmo_eps.map(|eps| CosmoParams {
        epsilon: eps,
        ..Default::default()
    });

    let opts = ScfOptions {
        max_iter,
        energy_tol_ev,
        density_tol,
        level_shift_ev,
        damping,
        use_nddo,
        cosmo,
    };

    let scf_res = run_rhf_scf_with_options(&batch, model.as_ref(), &mut scf_ws, &opts);

    // Analytical nuclear gradients
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut grads_ev = vec![[0.0f64; 3]; natoms];
    compute_cartesian_gradients_with_options(
        &mut batch,
        model.as_ref(),
        &scf_ws.density,
        &mut grad_ws,
        &mut grads_ev,
        use_nddo,
    );

    let mut grads_kcal = vec![[0.0f64; 3]; natoms];
    for i in 0..natoms {
        for c in 0..3 {
            grads_kcal[i][c] = grads_ev[i][c] * EV_TO_KCAL_MOL;
        }
    }

    // Non-covalent corrections
    let mut non_cov_kcal = 0.0;
    if let Some(dm_name) = dispersion {
        let dm = match dm_name.to_uppercase().as_str() {
            "PM6-DH+" | "DH+" | "PM6_DH+" => DispersionModel::Pm6DhPlus,
            "PM7" => DispersionModel::Pm7,
            "D3" | "D3BJ" | "D3-BJ" | "PM6-D3" | "AM1-D3" => DispersionModel::D3Bj,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown dispersion model '{}'. Supported: 'PM6-DH+', 'PM7', 'D3-BJ'",
                    other
                )))
            }
        };
        let mut disp_grads = vec![[0.0f64; 3]; natoms];
        let e_disp = compute_dispersion_energy_and_gradients(&batch, dm, &mut disp_grads);
        non_cov_kcal += e_disp;
        for i in 0..natoms {
            for c in 0..3 {
                grads_kcal[i][c] += disp_grads[i][c];
                grads_ev[i][c] += disp_grads[i][c] / EV_TO_KCAL_MOL;
            }
        }
    }

    if h_bonds {
        let h4_params = H4Parameters::default();
        let e_h4 = compute_h4_energy(&batch, &h4_params);
        non_cov_kcal += e_h4;

        let (e_hh, hh_grads) = compute_hh_repulsion_energy_and_gradients(&batch);
        non_cov_kcal += e_hh;

        for i in 0..natoms {
            for c in 0..3 {
                grads_kcal[i][c] += hh_grads[i][c];
                grads_ev[i][c] += hh_grads[i][c] / EV_TO_KCAL_MOL;
            }
        }
    }

    let (binding_energy_ev, heat_of_formation_kcal) = compute_heat_of_formation(
        scf_res.total_energy_ev,
        atomic_numbers,
        model.as_ref(),
        non_cov_kcal,
    );

    let dipole_res = compute_dipole_moment(&batch, model.as_ref(), &scf_ws.density);

    let num_electrons: f64 = batch
        .atomic_numbers
        .iter()
        .filter_map(|&z| model.get_element(z))
        .map(|p| p.core_charge)
        .sum();
    let num_occupied = (num_electrons.round() as usize) / 2;

    let mulliken_res =
        compute_mulliken_population(&batch, model.as_ref(), &scf_ws.eigenvectors, num_occupied);

    let homo = scf_res.homo_energy_ev;
    let lumo = scf_res.lumo_energy_ev;

    Ok(CalculationResult {
        total_energy_ev: scf_res.total_energy_ev + (non_cov_kcal / EV_TO_KCAL_MOL),
        electronic_energy_ev: scf_res.electronic_energy_ev,
        nuclear_repulsion_ev: scf_res.nuclear_repulsion_ev,
        binding_energy_ev,
        heat_of_formation_kcal,
        gradients_ev_angstrom: grads_ev,
        gradients_kcal_mol_angstrom: grads_kcal,
        dipole_debye: dipole_res.total,
        atomic_charges: dipole_res.atomic_charges,
        mulliken_charges: mulliken_res.net_charges,
        converged: scf_res.converged,
        scf_iterations: scf_res.iterations,
        homo_energy_ev: homo,
        lumo_energy_ev: lumo,
        homo_lumo_gap_ev: lumo - homo,
    })
}

/// Compute single-point semi-empirical quantum chemical properties and analytical gradients.
///
/// Parameters:
/// - `atomic_numbers`: list of integer atomic numbers (e.g. `[6, 1, 1, 1, 35]`)
/// - `coordinates`: list of 3D coordinates in Ångströms (e.g. `[[0.0, 0.0, 0.0], ...]`)
/// - `method`: Semi-empirical Hamiltonian ("PM6", "AM1", "RM1", "PM3", "MNDO"; default: "PM6")
/// - `cosmo_eps`: Optional solvent dielectric constant for COSMO solvation (e.g. 78.4 for water)
/// - `dispersion`: Optional empirical dispersion ("PM6-DH+", "PM7")
/// - `h_bonds`: Enable H4 hydrogen bonding and H-H core repulsion corrections (default: false)
/// - `use_nddo`: Enable full NDDO 22 multipoles and rotation (default: true)
/// - `max_iter`: Maximum SCF iterations (default: 60)
/// - `energy_tol_ev`: Energy convergence threshold in eV (default: 1e-7)
/// - `density_tol`: Density matrix convergence threshold (default: 1e-6)
#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coordinates,
    method = "PM6",
    cosmo_eps = None,
    dispersion = None,
    h_bonds = false,
    use_nddo = true,
    max_iter = 60,
    energy_tol_ev = 1e-7,
    density_tol = 1e-6,
    level_shift_ev = 0.0,
    damping = 0.5
))]
pub fn calculate(
    atomic_numbers: Vec<u8>,
    coordinates: Vec<[f64; 3]>,
    method: Option<&str>,
    cosmo_eps: Option<f64>,
    dispersion: Option<&str>,
    h_bonds: Option<bool>,
    use_nddo: Option<bool>,
    max_iter: Option<usize>,
    energy_tol_ev: Option<f64>,
    density_tol: Option<f64>,
    level_shift_ev: Option<f64>,
    damping: Option<f64>,
) -> PyResult<CalculationResult> {
    run_calculation_internal(
        &atomic_numbers,
        &coordinates,
        method.unwrap_or("PM6"),
        cosmo_eps,
        dispersion,
        h_bonds.unwrap_or(false),
        use_nddo.unwrap_or(true),
        max_iter.unwrap_or(60),
        energy_tol_ev.unwrap_or(1e-7),
        density_tol.unwrap_or(1e-6),
        level_shift_ev.unwrap_or(0.0),
        damping.unwrap_or(0.5),
    )
}

/// Optimize molecular geometry using quasi-Newton L-BFGS optimizer.
///
/// Parameters:
/// - `atomic_numbers`: list of integer atomic numbers
/// - `coordinates`: initial 3D coordinates in Ångströms
/// - `method`: Semi-empirical Hamiltonian ("PM6", "AM1", "RM1", "PM3", "MNDO"; default: "PM6")
/// - `max_cycles`: Maximum L-BFGS optimization cycles (default: 100)
/// - `grad_rms_tol`: RMS gradient convergence threshold in kcal / (mol · Å) (default: 1.0)
/// - `grad_max_tol`: Maximum gradient norm threshold in kcal / (mol · Å) (default: 2.0)
/// - `use_nddo`: Enable full NDDO potential energy surface and gradients (default: false)
#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coordinates,
    method = "PM6",
    max_cycles = 100,
    grad_rms_tol = 1.0,
    grad_max_tol = 2.0,
    use_nddo = false
))]
pub fn optimize(
    atomic_numbers: Vec<u8>,
    coordinates: Vec<[f64; 3]>,
    method: Option<&str>,
    max_cycles: Option<usize>,
    grad_rms_tol: Option<f64>,
    grad_max_tol: Option<f64>,
    use_nddo: Option<bool>,
) -> PyResult<OptimizationPyResult> {
    let natoms = atomic_numbers.len();
    if natoms == 0 {
        return Err(PyValueError::new_err("atomic_numbers cannot be empty"));
    }
    if coordinates.len() != natoms {
        return Err(PyValueError::new_err("Coordinate count mismatch"));
    }

    let m_name = method.unwrap_or("PM6");
    let model = get_model(m_name)?;

    let mut total_valence_elecs = 0.0;
    for &z in &atomic_numbers {
        if let Some(p) = model.get_element(z) {
            total_valence_elecs += p.core_charge;
        } else {
            return Err(PyValueError::new_err(format!(
                "Unsupported element Z={} for semi-empirical method '{}'",
                z, m_name
            )));
        }
    }

    let nelec = total_valence_elecs.round() as usize;
    if !nelec.is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "Open-shell radical detected ({} valence electrons). Closed-shell RHF requires an even number of valence electrons (requires UHF/ROHF).",
            nelec
        )));
    }

    let mut batch = MolecularBatch::new(atomic_numbers.clone(), &coordinates);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);

    let opt_opts = OptimizationOptions {
        max_cycles: max_cycles.unwrap_or(100),
        grad_rms_tol: grad_rms_tol.unwrap_or(1.0),
        grad_max_tol: grad_max_tol.unwrap_or(2.0),
        use_nddo: use_nddo.unwrap_or(false),
        ..Default::default()
    };

    let opt_res = optimize_geometry_lbfgs(
        &mut batch,
        model.as_ref(),
        &mut scf_ws,
        &mut grad_ws,
        &opt_opts,
    );

    let mut final_coords = Vec::with_capacity(natoms);
    for i in 0..natoms {
        final_coords.push([batch.x[i], batch.y[i], batch.z[i]]);
    }

    let final_calc = run_calculation_internal(
        &atomic_numbers,
        &final_coords,
        m_name,
        None,
        None,
        false,
        use_nddo.unwrap_or(false),
        60,
        1e-7,
        1e-6,
        0.0,
        0.5,
    )?;

    Ok(OptimizationPyResult {
        converged: opt_res.converged,
        cycles: opt_res.cycles,
        initial_energy_ev: opt_res.initial_energy_ev,
        final_energy_ev: opt_res.final_energy_ev,
        final_heat_of_formation_kcal: final_calc.heat_of_formation_kcal,
        final_grad_rms: opt_res.final_grad_rms,
        final_grad_max: opt_res.final_grad_max,
        coordinates: final_coords,
        final_result: final_calc,
    })
}

/// Locate a transition state (first-order saddle point) using Eigenvector Following (P-RFO Baker).
///
/// Parameters:
/// - `atomic_numbers`: list of integer atomic numbers (e.g. `[7, 1, 1, 1]`)
/// - `coordinates`: 3D coordinates in Ångströms (shape: `[natoms, 3]`)
/// - `method`: Semi-empirical Hamiltonian ("PM7", "PM6", "AM1", "RM1", "PM3", "MNDO"; default: "PM6")
/// - `max_cycles`: Maximum EF optimization cycles (default: 100)
/// - `grad_rms_tol`: RMS gradient convergence threshold in kcal / (mol * Å) (default: 0.1)
/// - `grad_max_tol`: Maximum gradient component convergence threshold in kcal / (mol * Å) (default: 0.2)
/// - `trust_radius`: Initial P-RFO trust radius in Ångströms (default: 0.1)
/// - `target_mode`: Index of normal mode to follow (0-indexed, default: None -> lowest eigenvalue)
/// - `use_nddo`: Enable full NDDO 22-multipole integrals (default: false)
#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coordinates,
    method = "PM6",
    max_cycles = 100,
    grad_rms_tol = 0.1,
    grad_max_tol = 0.2,
    trust_radius = 0.1,
    target_mode = None,
    use_nddo = false,
))]
pub fn transition_state(
    atomic_numbers: Vec<u8>,
    coordinates: Vec<[f64; 3]>,
    method: Option<&str>,
    max_cycles: Option<usize>,
    grad_rms_tol: Option<f64>,
    grad_max_tol: Option<f64>,
    trust_radius: Option<f64>,
    target_mode: Option<usize>,
    use_nddo: Option<bool>,
) -> PyResult<TransitionStatePyResult> {
    let natoms = atomic_numbers.len();
    if natoms == 0 {
        return Err(PyValueError::new_err("atomic_numbers cannot be empty"));
    }
    if coordinates.len() != natoms {
        return Err(PyValueError::new_err("Coordinate count mismatch"));
    }

    let m_name = method.unwrap_or("PM6");
    let model = get_model(m_name)?;

    let mut total_valence_elecs = 0.0;
    for &z in &atomic_numbers {
        if let Some(p) = model.get_element(z) {
            total_valence_elecs += p.core_charge;
        } else {
            return Err(PyValueError::new_err(format!(
                "Unsupported element Z={} for semi-empirical method '{}'",
                z, m_name
            )));
        }
    }

    let nelec = total_valence_elecs.round() as usize;
    if !nelec.is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "Open-shell radical detected ({} valence electrons). Closed-shell RHF requires an even number of valence electrons (requires UHF/ROHF).",
            nelec
        )));
    }

    let mut batch = MolecularBatch::new(atomic_numbers.clone(), &coordinates);
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut ef_ws = EigenvectorFollowingWorkspace::allocate(batch.natoms);

    let ts_opts = TransitionStateOptions {
        max_cycles: max_cycles.unwrap_or(100),
        grad_rms_tol: grad_rms_tol.unwrap_or(0.1),
        grad_max_tol: grad_max_tol.unwrap_or(0.2),
        trust_radius: trust_radius.unwrap_or(0.1),
        min_trust_radius: 0.005,
        max_trust_radius: 0.3,
        update_scheme: HessianUpdateScheme::Bofill,
        mode_following: true,
        target_mode,
        opt_mask: None,
        use_nddo: use_nddo.unwrap_or(false),
        hessian_delta: 0.005,
        initial_hessian: None,
    };

    let ts_res = optimize_transition_state(
        &mut batch,
        model.as_ref(),
        &mut scf_ws,
        &mut grad_ws,
        &mut ef_ws,
        &ts_opts,
    );

    let mut final_coords = Vec::with_capacity(natoms);
    for i in 0..natoms {
        final_coords.push([batch.x[i], batch.y[i], batch.z[i]]);
    }

    let final_calc = run_calculation_internal(
        &atomic_numbers,
        &final_coords,
        m_name,
        None,
        None,
        false,
        use_nddo.unwrap_or(false),
        60,
        1e-7,
        1e-6,
        0.0,
        0.5,
    )?;

    Ok(TransitionStatePyResult {
        converged: ts_res.converged,
        cycles: ts_res.cycles,
        final_energy_ev: ts_res.final_energy_ev,
        final_heat_of_formation_kcal: final_calc.heat_of_formation_kcal,
        initial_grad_rms: ts_res.initial_grad_rms,
        final_grad_rms: ts_res.final_grad_rms,
        final_grad_max: ts_res.final_grad_max,
        ts_mode_eigenvalue: ts_res.ts_mode_eigenvalue,
        ts_mode_index: ts_res.ts_mode_index,
        coordinates: final_coords,
        final_result: final_calc,
    })
}

/// Compute Cartesian Hessian, harmonic vibrational frequencies, normal modes, and thermodynamics.
///
/// Parameters:
/// - `atomic_numbers`: list of integer atomic numbers (e.g. `[8, 1, 1]`)
/// - `coordinates`: 3D coordinates in Ångströms (e.g. `[[0.0, 0.0, 0.0], ...]`)
/// - `method`: Semi-empirical Hamiltonian ("PM6", "AM1", "RM1", "PM3", "MNDO"; default: "PM6")
/// - `temperature_k`: Temperature in Kelvin for thermodynamics (default: 298.15 K)
/// - `pressure_atm`: Pressure in standard atmospheres (default: 1.0 atm)
/// - `rotational_symmetry_number`: Rotational symmetry number sigma (e.g. 2 for C2v water; default: 1.0)
/// - `step_size_angstrom`: Finite difference displacement step size in Ångströms (default: 0.005 Å)
/// - `project_external`: Project out 6 translational/rotational external motions via Eckart frame (default: true)
/// - `use_nddo`: Enable full NDDO potential energy surface and second derivatives (default: false)
#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coordinates,
    method = "PM6",
    temperature_k = 298.15,
    pressure_atm = 1.0,
    rotational_symmetry_number = 1.0,
    step_size_angstrom = 0.005,
    project_external = true,
    use_nddo = false,
))]
pub fn frequencies(
    atomic_numbers: Vec<u8>,
    coordinates: Vec<[f64; 3]>,
    method: Option<&str>,
    temperature_k: Option<f64>,
    pressure_atm: Option<f64>,
    rotational_symmetry_number: Option<f64>,
    step_size_angstrom: Option<f64>,
    project_external: Option<bool>,
    use_nddo: Option<bool>,
) -> PyResult<VibrationalResultPy> {
    let method_str = method.unwrap_or("PM6");
    let model = get_model(method_str)?;

    let natoms = atomic_numbers.len();
    if natoms == 0 {
        return Err(PyValueError::new_err(
            "Molecule must have at least one atom",
        ));
    }
    if coordinates.len() != natoms {
        return Err(PyValueError::new_err(format!(
            "Mismatch between atomic_numbers ({}) and coordinates ({})",
            natoms,
            coordinates.len()
        )));
    }

    let mut total_valence_elecs = 0.0;
    for &z in &atomic_numbers {
        if let Some(p) = model.get_element(z) {
            total_valence_elecs += p.core_charge;
        } else {
            return Err(PyValueError::new_err(format!(
                "Unsupported element Z={} for semi-empirical method '{}'",
                z, method_str
            )));
        }
    }

    let nelec = total_valence_elecs.round() as usize;
    if !nelec.is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "Open-shell radical detected ({} valence electrons). Closed-shell RHF requires an even number of valence electrons (requires UHF/ROHF).",
            nelec
        )));
    }

    let mut batch = MolecularBatch::new(atomic_numbers, &coordinates);
    let mut ws = ScfWorkspace::allocate(batch.norbs);

    let scf_opts = ScfOptions {
        max_iter: 100,
        energy_tol_ev: 1e-8,
        density_tol: 1e-7,
        use_nddo: use_nddo.unwrap_or(false),
        ..Default::default()
    };

    let hess_opts = HessianOptions {
        delta: step_size_angstrom.unwrap_or(1.0e-3),
        recompute_scf: true,
        use_nddo: use_nddo.unwrap_or(false),
        project_external: project_external.unwrap_or(true),
        temperature_k: temperature_k.unwrap_or(298.15),
        pressure_atm: pressure_atm.unwrap_or(1.0),
        rotational_symmetry_number: rotational_symmetry_number.unwrap_or(1.0),
    };

    let res =
        compute_hessian_and_frequencies(&mut batch, model.as_ref(), &mut ws, &scf_opts, &hess_opts);

    let normal_modes: Vec<NormalModePy> = res
        .normal_modes
        .into_iter()
        .map(|m| NormalModePy {
            frequency_cm1: m.frequency_cm1,
            reduced_mass_amu: m.reduced_mass_amu,
            force_constant_mdyne_a: m.force_constant_mdyne_a,
            displacements: m.displacements,
        })
        .collect();

    let thermo = ThermodynamicsPy {
        temperature_k: res.thermo.temperature_k,
        pressure_atm: res.thermo.pressure_atm,
        zpve_kcal_mol: res.thermo.zpve_kcal_mol,
        e_vib_cal_mol: res.thermo.e_vib_cal_mol,
        e_rot_cal_mol: res.thermo.e_rot_cal_mol,
        e_trans_cal_mol: res.thermo.e_trans_cal_mol,
        enthalpy_thermal_cal_mol: res.thermo.enthalpy_thermal_cal_mol,
        cv_vib_cal_k_mol: res.thermo.cv_vib_cal_k_mol,
        cv_rot_cal_k_mol: res.thermo.cv_rot_cal_k_mol,
        cp_trans_cal_k_mol: res.thermo.cp_trans_cal_k_mol,
        cp_total_cal_k_mol: res.thermo.cp_total_cal_k_mol,
        entropy_vib_cal_k_mol: res.thermo.entropy_vib_cal_k_mol,
        entropy_rot_cal_k_mol: res.thermo.entropy_rot_cal_k_mol,
        entropy_trans_cal_k_mol: res.thermo.entropy_trans_cal_k_mol,
        entropy_total_cal_k_mol: res.thermo.entropy_total_cal_k_mol,
        gibbs_correction_kcal_mol: res.thermo.gibbs_correction_kcal_mol,
    };

    let is_transition_state = res
        .vibrational_frequencies_cm1
        .first()
        .is_some_and(|&f| f < -10.0);

    let n3 = 3 * natoms;
    let mut cartesian_hessian = vec![vec![0.0; n3]; n3];
    for (r, row) in cartesian_hessian.iter_mut().enumerate() {
        for (c, val) in row.iter_mut().enumerate() {
            *val = res.cartesian_hessian.get(r, c);
        }
    }

    Ok(VibrationalResultPy {
        all_frequencies_cm1: res.all_frequencies_cm1,
        vibrational_frequencies_cm1: res.vibrational_frequencies_cm1,
        normal_modes,
        zpve_kcal_mol: res.zpve_kcal_mol,
        thermo,
        is_transition_state,
        cartesian_hessian,
    })
}

/// Object-oriented MOPAC Calculator class compatible with PyTorch / RDKit / ASE workflows.
#[pyclass]
pub struct MopacCalculator {
    #[pyo3(get, set)]
    pub method: String,
    #[pyo3(get, set)]
    pub use_nddo: bool,
    #[pyo3(get, set)]
    pub cosmo_eps: Option<f64>,
    #[pyo3(get, set)]
    pub dispersion: Option<String>,
    #[pyo3(get, set)]
    pub h_bonds: bool,
    #[pyo3(get, set)]
    pub max_iter: usize,
}

#[pymethods]
impl MopacCalculator {
    #[new]
    #[pyo3(signature = (
        method = "PM6",
        use_nddo = true,
        cosmo_eps = None,
        dispersion = None,
        h_bonds = false,
        max_iter = 60
    ))]
    fn new(
        method: Option<&str>,
        use_nddo: Option<bool>,
        cosmo_eps: Option<f64>,
        dispersion: Option<&str>,
        h_bonds: Option<bool>,
        max_iter: Option<usize>,
    ) -> Self {
        Self {
            method: method.unwrap_or("PM6").to_string(),
            use_nddo: use_nddo.unwrap_or(true),
            cosmo_eps,
            dispersion: dispersion.map(|s| s.to_string()),
            h_bonds: h_bonds.unwrap_or(false),
            max_iter: max_iter.unwrap_or(60),
        }
    }

    /// Calculate energy, gradients, dipole, and charges for a given molecule.
    fn calculate(
        &self,
        atomic_numbers: Vec<u8>,
        coordinates: Vec<[f64; 3]>,
    ) -> PyResult<CalculationResult> {
        run_calculation_internal(
            &atomic_numbers,
            &coordinates,
            &self.method,
            self.cosmo_eps,
            self.dispersion.as_deref(),
            self.h_bonds,
            self.use_nddo,
            self.max_iter,
            1e-7,
            1e-6,
            0.0,
            0.5,
        )
    }

    /// Optimize geometry for a given molecule.
    #[pyo3(signature = (atomic_numbers, coordinates, max_cycles = 100))]
    fn optimize(
        &self,
        atomic_numbers: Vec<u8>,
        coordinates: Vec<[f64; 3]>,
        max_cycles: Option<usize>,
    ) -> PyResult<OptimizationPyResult> {
        optimize(
            atomic_numbers,
            coordinates,
            Some(&self.method),
            max_cycles,
            Some(1.0),
            Some(2.0),
            Some(self.use_nddo),
        )
    }

    /// Compute harmonic vibrational frequencies, normal modes, and thermodynamics.
    #[pyo3(signature = (atomic_numbers, coordinates, temperature_k = 298.15, pressure_atm = 1.0, rotational_symmetry_number = 1.0))]
    fn frequencies(
        &self,
        atomic_numbers: Vec<u8>,
        coordinates: Vec<[f64; 3]>,
        temperature_k: Option<f64>,
        pressure_atm: Option<f64>,
        rotational_symmetry_number: Option<f64>,
    ) -> PyResult<VibrationalResultPy> {
        frequencies(
            atomic_numbers,
            coordinates,
            Some(&self.method),
            temperature_k,
            pressure_atm,
            rotational_symmetry_number,
            Some(0.005),
            Some(true),
            Some(self.use_nddo),
        )
    }

    /// Locate transition state using Eigenvector Following (P-RFO Baker).
    #[pyo3(signature = (atomic_numbers, coordinates, max_cycles = 100, grad_rms_tol = 0.1, grad_max_tol = 0.2, trust_radius = 0.1, target_mode = None))]
    fn transition_state(
        &self,
        atomic_numbers: Vec<u8>,
        coordinates: Vec<[f64; 3]>,
        max_cycles: Option<usize>,
        grad_rms_tol: Option<f64>,
        grad_max_tol: Option<f64>,
        trust_radius: Option<f64>,
        target_mode: Option<usize>,
    ) -> PyResult<TransitionStatePyResult> {
        transition_state(
            atomic_numbers,
            coordinates,
            Some(&self.method),
            max_cycles,
            grad_rms_tol,
            grad_max_tol,
            trust_radius,
            target_mode,
            Some(self.use_nddo),
        )
    }
}

/// MOPAC_RS Python Module
#[pymodule]
fn mopac_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CalculationResult>()?;
    m.add_class::<OptimizationPyResult>()?;
    m.add_class::<TransitionStatePyResult>()?;
    m.add_class::<NormalModePy>()?;
    m.add_class::<ThermodynamicsPy>()?;
    m.add_class::<VibrationalResultPy>()?;
    m.add_class::<MopacCalculator>()?;
    m.add_function(wrap_pyfunction!(calculate, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(transition_state, m)?)?;
    m.add_function(wrap_pyfunction!(frequencies, m)?)?;

    let py = m.py();
    let py_code = r#"
def from_rdkit(mol, conf_id=-1):
    """Extract atomic numbers and 3D Cartesian coordinates from an RDKit Mol object."""
    conf = mol.GetConformer(conf_id)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    coords = [list(conf.GetAtomPosition(i)) for i in range(len(atomic_numbers))]
    return atomic_numbers, coords

def from_ase(atoms):
    """Extract atomic numbers and 3D Cartesian coordinates from an ASE Atoms object."""
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    coords = atoms.get_positions().tolist()
    return atomic_numbers, coords

try:
    from ase.calculators.calculator import Calculator, all_changes
    import numpy as np

    class MopacASECalculator(Calculator):
        """Atomic Simulation Environment (ASE) Calculator backed by MOPAC_RS."""
        implemented_properties = ['energy', 'forces', 'dipole', 'charges']

        def __init__(self, method='PM6', dispersion=None, cosmo_eps=None, use_nddo=True, **kwargs):
            super().__init__(**kwargs)
            self.method = method
            self.dispersion = dispersion
            self.cosmo_eps = cosmo_eps
            self.use_nddo = use_nddo

        def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            atomic_numbers = atoms.get_atomic_numbers().tolist()
            coordinates = atoms.get_positions().tolist()
            calc_res = calculate(
                atomic_numbers,
                coordinates,
                method=self.method,
                dispersion=self.dispersion,
                cosmo_eps=self.cosmo_eps,
                use_nddo=self.use_nddo
            )
            self.results['energy'] = calc_res.total_energy_ev
            self.results['forces'] = -np.array(calc_res.gradients_ev_angstrom)
            self.results['dipole'] = np.array(calc_res.dipole_debye[:3])
            self.results['charges'] = np.array(calc_res.mulliken_charges)
except ImportError:
    class MopacASECalculator:
        def __init__(self, *args, **kwargs):
            raise ImportError("ase and numpy are required to instantiate MopacASECalculator")
"#;

    let dict = m.dict();
    py.run_bound(py_code, Some(&dict), Some(&dict))?;

    Ok(())
}
