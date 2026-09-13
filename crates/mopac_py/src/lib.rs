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

use mopac_core::ci::meci::{run_meci, CiActiveSpace, MeciOptions, MeciWorkspace};
use mopac_core::ci::spectrum::{
    compute_transition_dipoles_and_oscillator_strengths, simulate_uv_vis_spectrum,
};
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
use mopac_core::pbc::{run_pbc_scf, PbcOptions, PbcWorkspace, UnitCell};
use mopac_core::properties::{
    compute_dipole_moment, compute_esp_charges, compute_heat_of_formation,
    compute_mulliken_population, EspOptions,
};
use mopac_core::reactions::{
    run_dynamic_reaction_coordinate, trace_intrinsic_reaction_coordinate, DrcEnsemble, DrcOptions,
    DrcWorkspace, InitialVelocities, IrcDirection, IrcOptions, IrcWorkspace,
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

/// Merz-Singh-Kollman Electrostatic Potential (ESP) fitting result.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct EspResultPy {
    /// Fitted atom-centered partial charges in atomic units (e)
    pub charges: Vec<f64>,
    /// Cartesian electric dipole moment computed from fitted ESP charges in Debye [x, y, z]
    pub dipole_debye: [f64; 3],
    /// Total dipole moment magnitude in Debye
    pub dipole_magnitude_debye: f64,
    /// Root-mean-square fitting error of the electrostatic potential in eV
    pub rms_error_ev: f64,
    /// Total number of grid points retained on the solvent-accessible envelope
    pub num_grid_points: usize,
}

#[pymethods]
impl EspResultPy {
    fn __repr__(&self) -> String {
        format!(
            "<EspResult charges={:?}, dipole={:.3} D, rms_err={:.4} eV, grid_pts={}>",
            self.charges, self.dipole_magnitude_debye, self.rms_error_ev, self.num_grid_points
        )
    }

    /// Convert ESP result to Python dictionary.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("charges", &self.charges)?;
        dict.set_item("dipole_debye", self.dipole_debye)?;
        dict.set_item("dipole_magnitude_debye", self.dipole_magnitude_debye)?;
        dict.set_item("rms_error_ev", self.rms_error_ev)?;
        dict.set_item("num_grid_points", self.num_grid_points)?;
        Ok(dict)
    }
}

/// Single reaction path point along an Intrinsic Reaction Coordinate.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct IrcPointPy {
    /// Integrated reaction coordinate s in amu^(1/2) * Å (0 at TS, >0 forward, <0 reverse)
    pub path_coordinate: f64,
    /// Total electronic energy in eV
    pub energy_ev: f64,
    /// Standard heat of formation in kcal/mol
    pub heat_of_formation_kcal: f64,
    /// Cartesian coordinates in Å (shape: [natoms, 3])
    pub coordinates: Vec<[f64; 3]>,
    /// Cartesian gradient RMS in kcal / (mol * Å)
    pub cartesian_gradient_rms: f64,
    /// Mass-weighted gradient RMS in kcal / (mol * Å * amu^(1/2))
    pub mass_weighted_gradient_rms: f64,
}

#[pymethods]
impl IrcPointPy {
    fn __repr__(&self) -> String {
        format!(
            "<IrcPoint s={:+.4} E={:.6} eV, dHf={:.3} kcal/mol, grad_rms={:.4}>",
            self.path_coordinate,
            self.energy_ev,
            self.heat_of_formation_kcal,
            self.cartesian_gradient_rms
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("path_coordinate", self.path_coordinate)?;
        dict.set_item("energy_ev", self.energy_ev)?;
        dict.set_item("heat_of_formation_kcal", self.heat_of_formation_kcal)?;
        dict.set_item("coordinates", &self.coordinates)?;
        dict.set_item("cartesian_gradient_rms", self.cartesian_gradient_rms)?;
        dict.set_item(
            "mass_weighted_gradient_rms",
            self.mass_weighted_gradient_rms,
        )?;
        Ok(dict)
    }
}

/// Comprehensive result of Intrinsic Reaction Coordinate path tracing.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct IrcPyResult {
    /// Sequence of reaction path points ordered monotonically along reaction coordinate s
    pub points: Vec<IrcPointPy>,
    /// Index of transition state point in the points list (s = 0.0)
    pub ts_point_index: usize,
    /// Whether forward reaction branch converged to a local minimum
    pub forward_converged: bool,
    /// Whether reverse reaction branch converged to a local minimum
    pub reverse_converged: bool,
}

#[pymethods]
impl IrcPyResult {
    fn __repr__(&self) -> String {
        format!(
            "<IrcPyResult points={} ts_idx={} fwd_conv={} rev_conv={}>",
            self.points.len(),
            self.ts_point_index,
            self.forward_converged,
            self.reverse_converged
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("ts_point_index", self.ts_point_index)?;
        dict.set_item("forward_converged", self.forward_converged)?;
        dict.set_item("reverse_converged", self.reverse_converged)?;
        let pts_list = pyo3::types::PyList::empty_bound(py);
        for p in &self.points {
            pts_list.append(p.to_dict(py)?)?;
        }
        dict.set_item("points", pts_list)?;
        Ok(dict)
    }
}

/// Recorded frame along Dynamic Reaction Coordinate Born-Oppenheimer Molecular Dynamics.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct DrcFramePy {
    /// Integration step index (0-indexed)
    pub step: usize,
    /// Elapsed simulation time in femtoseconds
    pub time_fs: f64,
    /// Potential energy in eV
    pub potential_energy_ev: f64,
    /// Classical kinetic energy in eV
    pub kinetic_energy_ev: f64,
    /// Total energy (potential + kinetic) in eV
    pub total_energy_ev: f64,
    /// Instantaneous kinetic temperature in Kelvin
    pub temperature_k: f64,
    /// Atomic Cartesian coordinates in Å (shape: [natoms, 3])
    pub coordinates: Vec<[f64; 3]>,
    /// Atomic velocities in Å / fs (shape: [natoms, 3])
    pub velocities: Vec<[f64; 3]>,
    /// Cartesian forces in eV / Å (shape: [natoms, 3])
    pub forces: Vec<[f64; 3]>,
}

#[pymethods]
impl DrcFramePy {
    fn __repr__(&self) -> String {
        format!(
            "<DrcFrame step={} t={:.2} fs, E_tot={:.6} eV, T={:.1} K>",
            self.step, self.time_fs, self.total_energy_ev, self.temperature_k
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("step", self.step)?;
        dict.set_item("time_fs", self.time_fs)?;
        dict.set_item("potential_energy_ev", self.potential_energy_ev)?;
        dict.set_item("kinetic_energy_ev", self.kinetic_energy_ev)?;
        dict.set_item("total_energy_ev", self.total_energy_ev)?;
        dict.set_item("temperature_k", self.temperature_k)?;
        dict.set_item("coordinates", &self.coordinates)?;
        dict.set_item("velocities", &self.velocities)?;
        dict.set_item("forces", &self.forces)?;
        Ok(dict)
    }
}

/// Comprehensive result of Dynamic Reaction Coordinate simulation.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct DrcPyResult {
    /// Recorded trajectory frames
    pub frames: Vec<DrcFramePy>,
    /// Initial total energy in eV
    pub initial_energy_ev: f64,
    /// Final total energy in eV
    pub final_energy_ev: f64,
    /// Linear energy drift in eV / ps
    pub energy_drift_ev_per_ps: f64,
    /// Maximum absolute energy drift in eV
    pub max_energy_drift_ev: f64,
    /// Average trajectory temperature in Kelvin
    pub average_temperature_k: f64,
}

#[pymethods]
impl DrcPyResult {
    fn __repr__(&self) -> String {
        format!(
            "<DrcPyResult frames={} drift={:.3e} eV/ps, max_dev={:.3e} eV, avg_T={:.1} K>",
            self.frames.len(),
            self.energy_drift_ev_per_ps,
            self.max_energy_drift_ev,
            self.average_temperature_k
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("initial_energy_ev", self.initial_energy_ev)?;
        dict.set_item("final_energy_ev", self.final_energy_ev)?;
        dict.set_item("energy_drift_ev_per_ps", self.energy_drift_ev_per_ps)?;
        dict.set_item("max_energy_drift_ev", self.max_energy_drift_ev)?;
        dict.set_item("average_temperature_k", self.average_temperature_k)?;
        let frames_list = pyo3::types::PyList::empty_bound(py);
        for f in &self.frames {
            frames_list.append(f.to_dict(py)?)?;
        }
        dict.set_item("frames", frames_list)?;
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

    let mut batch =
        MolecularBatch::new_for_model(atomic_numbers.to_vec(), coordinates, model.as_ref());
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

    let mut batch =
        MolecularBatch::new_for_model(atomic_numbers.clone(), &coordinates, model.as_ref());
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

    let mut batch =
        MolecularBatch::new_for_model(atomic_numbers.clone(), &coordinates, model.as_ref());
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
    custom_masses = None,
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
    custom_masses: Option<Vec<f64>>,
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

    let mut batch = MolecularBatch::new_for_model(atomic_numbers, &coordinates, model.as_ref());
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
        custom_masses,
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

/// Compute Merz-Singh-Kollman atom-centered Electrostatic Potential (ESP) partial charges.
///
/// Parameters:
/// - `atomic_numbers`: list of integer atomic numbers
/// - `coordinates`: 3D Cartesian coordinates in Angstroms
/// - `method`: Semi-empirical Hamiltonian (default: "PM6")
/// - `net_charge`: Molecular net charge constraint (default: 0.0)
/// - `points_per_shell`: Fibonacci sampling points per atom per radial shell (default: 64)
#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coordinates,
    method = "PM6",
    net_charge = 0.0,
    points_per_shell = 64
))]
pub fn esp_charges(
    atomic_numbers: Vec<u8>,
    coordinates: Vec<[f64; 3]>,
    method: Option<&str>,
    net_charge: Option<f64>,
    points_per_shell: Option<usize>,
) -> PyResult<EspResultPy> {
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

    let batch = MolecularBatch::new_for_model(atomic_numbers, &coordinates, model.as_ref());
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_opts = ScfOptions::default();

    let scf_res = run_rhf_scf_with_options(&batch, model.as_ref(), &mut ws, &scf_opts);
    if !scf_res.converged {
        return Err(PyValueError::new_err(
            "Base SCF failed to converge for ESP calculation",
        ));
    }

    let opts = EspOptions {
        shell_multipliers: vec![1.4, 1.6, 1.8, 2.0],
        points_per_shell: points_per_shell.unwrap_or(64),
        net_charge: net_charge.unwrap_or(0.0),
    };

    let res = compute_esp_charges(&batch, model.as_ref(), &ws.density, &opts)
        .map_err(PyValueError::new_err)?;

    Ok(EspResultPy {
        charges: res.charges,
        dipole_debye: res.dipole_debye,
        dipole_magnitude_debye: res.dipole_magnitude_debye,
        rms_error_ev: res.rms_error_ev,
        num_grid_points: res.num_grid_points,
    })
}

/// Trace Intrinsic Reaction Coordinate (IRC) path from a transition state.
///
/// Parameters:
/// - `atomic_numbers`: list of integer atomic numbers (e.g. `[7, 1, 1, 1]`)
/// - `coordinates`: 3D coordinates in Ångströms of the transition state
/// - `method`: Semi-empirical Hamiltonian ("PM6", "AM1", "RM1", "PM3", "MNDO"; default: "PM6")
/// - `step_size`: Path arc length step size in amu^(1/2) * Å (default: 0.1)
/// - `max_points`: Maximum number of reaction path points per direction (default: 50)
/// - `direction`: Path direction: "both", "forward", or "reverse" (default: "both")
/// - `use_nddo`: Enable full NDDO potential energy surface (default: false)
#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coordinates,
    method = None,
    step_size = None,
    max_points = None,
    direction = None,
    use_nddo = None,
))]
pub fn irc(
    atomic_numbers: Vec<u8>,
    coordinates: Vec<[f64; 3]>,
    method: Option<&str>,
    step_size: Option<f64>,
    max_points: Option<usize>,
    direction: Option<&str>,
    use_nddo: Option<bool>,
) -> PyResult<IrcPyResult> {
    let natoms = atomic_numbers.len();
    if natoms == 0 {
        return Err(PyValueError::new_err("atomic_numbers cannot be empty"));
    }
    if coordinates.len() != natoms {
        return Err(PyValueError::new_err(format!(
            "Mismatch between atomic_numbers ({}) and coordinates ({})",
            natoms,
            coordinates.len()
        )));
    }

    let m_name = method.unwrap_or("PM6");
    let model = get_model(m_name)?;

    let irc_dir = match direction.unwrap_or("both").to_lowercase().as_str() {
        "forward" | "fwd" => IrcDirection::Forward,
        "reverse" | "rev" => IrcDirection::Reverse,
        "both" => IrcDirection::Both,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown IRC direction '{}'. Supported: 'both', 'forward', 'reverse'",
                other
            )))
        }
    };

    let mut batch = MolecularBatch::new_for_model(atomic_numbers, &coordinates, model.as_ref());
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut irc_ws = IrcWorkspace::allocate(&batch);

    let opts = IrcOptions {
        step_size: step_size.unwrap_or(0.1),
        max_points: max_points.unwrap_or(50),
        corrector_max_iter: 25,
        corrector_tol: 1e-4,
        grad_rms_tol: 0.05,
        energy_increase_tol: 0.02,
        direction: irc_dir,
        use_nddo: use_nddo.unwrap_or(false),
        transition_vector: None,
    };

    let res = trace_intrinsic_reaction_coordinate(
        &mut batch,
        model.as_ref(),
        &mut scf_ws,
        &mut grad_ws,
        &mut irc_ws,
        &opts,
    );

    let points = res
        .points
        .into_iter()
        .map(|p| IrcPointPy {
            path_coordinate: p.path_coordinate,
            energy_ev: p.energy_ev,
            heat_of_formation_kcal: p.heat_of_formation_kcal,
            coordinates: p.coordinates,
            cartesian_gradient_rms: p.cartesian_gradient_rms,
            mass_weighted_gradient_rms: p.mass_weighted_gradient_rms,
        })
        .collect();

    Ok(IrcPyResult {
        points,
        ts_point_index: res.ts_point_index,
        forward_converged: res.forward_converged,
        reverse_converged: res.reverse_converged,
    })
}

/// Run Dynamic Reaction Coordinate (DRC) / Born-Oppenheimer Molecular Dynamics.
///
/// Parameters:
/// - `atomic_numbers`: list of integer atomic numbers (e.g. `[7, 7]`)
/// - `coordinates`: 3D coordinates in Ångströms
/// - `method`: Semi-empirical Hamiltonian ("PM6", "AM1", "RM1", "PM3", "MNDO"; default: "PM6")
/// - `time_step_fs`: Integrator time step in femtoseconds (default: 0.5 fs)
/// - `total_steps`: Number of MD integration steps (default: 500)
/// - `ensemble`: Thermodynamic ensemble: "nve" or "nvt" (default: "nve")
/// - `temperature_k`: Target temperature in Kelvin for NVT ensemble (default: 298.15 K)
/// - `berendsen_tau_fs`: Berendsen thermostat coupling constant in femtoseconds (default: 100.0 fs)
/// - `recording_interval`: Stride interval for saving trajectory frames (default: 1)
/// - `use_nddo`: Enable full NDDO potential energy surface (default: false)
#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coordinates,
    method = None,
    time_step_fs = None,
    total_steps = None,
    ensemble = None,
    temperature_k = None,
    berendsen_tau_fs = None,
    recording_interval = None,
    use_nddo = None,
))]
pub fn drc(
    atomic_numbers: Vec<u8>,
    coordinates: Vec<[f64; 3]>,
    method: Option<&str>,
    time_step_fs: Option<f64>,
    total_steps: Option<usize>,
    ensemble: Option<&str>,
    temperature_k: Option<f64>,
    berendsen_tau_fs: Option<f64>,
    recording_interval: Option<usize>,
    use_nddo: Option<bool>,
) -> PyResult<DrcPyResult> {
    let natoms = atomic_numbers.len();
    if natoms == 0 {
        return Err(PyValueError::new_err("atomic_numbers cannot be empty"));
    }
    if coordinates.len() != natoms {
        return Err(PyValueError::new_err(format!(
            "Mismatch between atomic_numbers ({}) and coordinates ({})",
            natoms,
            coordinates.len()
        )));
    }

    let m_name = method.unwrap_or("PM6");
    let model = get_model(m_name)?;

    let target_t = temperature_k.unwrap_or(298.15);
    let (drc_ensemble, init_vel) = match ensemble.unwrap_or("nve").to_lowercase().as_str() {
        "nve" => (DrcEnsemble::Nve, InitialVelocities::Zero),
        "nvt" => (
            DrcEnsemble::Nvt,
            InitialVelocities::MaxwellBoltzmann {
                temperature_k: target_t,
                seed: None,
            },
        ),
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown DRC ensemble '{}'. Supported: 'nve', 'nvt'",
                other
            )))
        }
    };

    let mut batch = MolecularBatch::new_for_model(atomic_numbers, &coordinates, model.as_ref());
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut drc_ws = DrcWorkspace::allocate(&batch);

    let opts = DrcOptions {
        time_step_fs: time_step_fs.unwrap_or(0.5),
        total_steps: total_steps.unwrap_or(500),
        ensemble: drc_ensemble,
        target_temperature_k: target_t,
        berendsen_tau_fs: berendsen_tau_fs.unwrap_or(100.0),
        recording_interval: recording_interval.unwrap_or(1),
        initial_velocities: init_vel,
        use_nddo: use_nddo.unwrap_or(false),
        scf_energy_tol: 1e-8,
        scf_density_tol: 1e-7,
    };

    let res = run_dynamic_reaction_coordinate(
        &mut batch,
        model.as_ref(),
        &mut scf_ws,
        &mut grad_ws,
        &mut drc_ws,
        &opts,
    );

    let frames = res
        .frames
        .into_iter()
        .map(|f| DrcFramePy {
            step: f.step,
            time_fs: f.time_fs,
            potential_energy_ev: f.potential_energy_ev,
            kinetic_energy_ev: f.kinetic_energy_ev,
            total_energy_ev: f.total_energy_ev,
            temperature_k: f.temperature_k,
            coordinates: f.coordinates,
            velocities: f.velocities,
            forces: f.forces,
        })
        .collect();

    Ok(DrcPyResult {
        frames,
        initial_energy_ev: res.initial_energy_ev,
        final_energy_ev: res.final_energy_ev,
        energy_drift_ev_per_ps: res.energy_drift_ev_per_ps,
        max_energy_drift_ev: res.max_energy_drift_ev,
        average_temperature_k: res.average_temperature_k,
    })
}

/// Single CI electronic state result exposed to Python.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct CiStatePy {
    /// State index (1-indexed)
    pub root: usize,
    /// Absolute CI energy eigenvalue in eV
    pub energy_ev: f64,
    /// Excitation energy relative to ground state in eV
    pub excitation_energy_ev: f64,
    /// Excitation energy in cm^-1
    pub excitation_energy_cm1: f64,
    /// Absorption wavelength in nm
    pub wavelength_nm: f64,
    /// Spin multiplicity (1=Singlet, 2=Doublet, 3=Triplet)
    pub multiplicity: usize,
    /// Spin label
    pub spin_label: String,
    /// S^2 expectation value
    pub s_squared: f64,
    /// Transition dipole moment [x, y, z] in Debye
    pub transition_dipole_debye: [f64; 3],
    /// Total transition dipole magnitude in Debye
    pub dipole_strength_debye: f64,
    /// Polarization in Angstroms^2 [x, y, z]
    pub polarization_angstrom2: [f64; 3],
    /// Dimensionless oscillator strength
    pub oscillator_strength: f64,
}

/// Complete MECI calculation result exposed to Python.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct MeciPyResult {
    /// All computed CI states
    pub states: Vec<CiStatePy>,
    /// Number of microstates
    pub num_microstates: usize,
    /// Target root index (1-indexed)
    pub target_root: usize,
    /// CI energy correction in eV
    pub ci_energy_correction_ev: f64,
    /// Total electronic energy in eV
    pub electronic_energy_ev: f64,
    /// Total energy in eV
    pub total_energy_ev: f64,
    /// Heat of formation in kcal/mol
    pub heat_of_formation_kcal: f64,
}

/// Simulated UV-Vis spectrum exposed to Python.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct UvVisSpectrumPy {
    /// Wavelength grid in nm
    pub wavelengths_nm: Vec<f64>,
    /// Molar extinction coefficients in L/(mol*cm)
    pub extinction_coefficients: Vec<f64>,
    /// Peak absorption wavelength in nm
    pub lambda_max_nm: f64,
    /// Maximum extinction coefficient
    pub epsilon_max: f64,
}

/// Run Multi-Electron Configuration Interaction (MECI) excited state calculation.
///
/// Parameters:
/// - `atomic_numbers`: list of integer atomic numbers
/// - `coordinates`: 3D coordinates in Angstroms
/// - `method`: Semi-empirical Hamiltonian (default: "PM6")
/// - `active_orbitals`: Number of active orbitals (default: 2)
/// - `target_root`: Target root state, 1-indexed (default: 1 = ground state)
/// - `use_nddo`: Enable full NDDO integrals (default: true)
#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coordinates,
    method = "PM6",
    active_orbitals = 2,
    target_root = 1,
    use_nddo = true
))]
pub fn meci(
    atomic_numbers: Vec<u8>,
    coordinates: Vec<[f64; 3]>,
    method: Option<&str>,
    active_orbitals: Option<usize>,
    target_root: Option<usize>,
    use_nddo: Option<bool>,
) -> PyResult<MeciPyResult> {
    let natoms = atomic_numbers.len();
    if natoms == 0 {
        return Err(PyValueError::new_err("atomic_numbers cannot be empty"));
    }
    if coordinates.len() != natoms {
        return Err(PyValueError::new_err("Coordinate count mismatch"));
    }

    let model = get_model(method.unwrap_or("PM6"))?;
    let batch = MolecularBatch::new_for_model(atomic_numbers, &coordinates, model.as_ref());
    let mut scf_ws = ScfWorkspace::allocate(batch.norbs);
    let nddo = use_nddo.unwrap_or(true);

    let scf_opts = ScfOptions {
        use_nddo: nddo,
        ..ScfOptions::default()
    };
    let scf_res = run_rhf_scf_with_options(&batch, model.as_ref(), &mut scf_ws, &scf_opts);
    if !scf_res.converged {
        return Err(PyValueError::new_err("SCF did not converge"));
    }

    let n_orbs = active_orbitals.unwrap_or(2);
    let active_space = CiActiveSpace::new(n_orbs, n_orbs);
    let options = MeciOptions {
        active_space,
        target_root: target_root.unwrap_or(1),
        spin_target: None,
        use_nddo: nddo,
    };

    // Compute active MO indices
    let mut total_valence_elecs = 0.0f64;
    for &z in &batch.atomic_numbers {
        if let Some(p) = model.get_element(z) {
            total_valence_elecs += p.core_charge;
        }
    }
    let n_occ = (total_valence_elecs.round() as usize) / 2;
    let n_occ_active = n_orbs.div_ceil(2);
    let start_mo = n_occ - n_occ_active;
    let active_mo_indices: Vec<usize> = (start_mo..start_mo + n_orbs).collect();

    let max_microstates = 100;
    let mut meci_ws = MeciWorkspace::allocate(n_orbs, max_microstates);

    let mut meci_res = run_meci(
        &batch,
        model.as_ref(),
        &scf_ws.eigenvectors,
        &scf_ws.eigenvalues,
        scf_res.electronic_energy_ev,
        scf_res.total_energy_ev,
        &options,
        &mut meci_ws,
    );

    compute_transition_dipoles_and_oscillator_strengths(
        &batch,
        model.as_ref(),
        &scf_ws.eigenvectors,
        &active_mo_indices,
        &mut meci_res,
    );

    let states: Vec<CiStatePy> = meci_res
        .states
        .iter()
        .map(|s| CiStatePy {
            root: s.root,
            energy_ev: s.energy_ev,
            excitation_energy_ev: s.excitation_energy_ev,
            excitation_energy_cm1: s.excitation_energy_cm1,
            wavelength_nm: s.wavelength_nm,
            multiplicity: s.spin.multiplicity,
            spin_label: s.spin.label.to_string(),
            s_squared: s.spin.s_squared,
            transition_dipole_debye: s.transition_dipole_debye,
            dipole_strength_debye: s.dipole_strength_debye,
            polarization_angstrom2: s.polarization_angstrom2,
            oscillator_strength: s.oscillator_strength,
        })
        .collect();

    Ok(MeciPyResult {
        states,
        num_microstates: meci_res.microstates.len(),
        target_root: meci_res.target_root,
        ci_energy_correction_ev: meci_res.ci_energy_correction_ev,
        electronic_energy_ev: meci_res.electronic_energy_ev,
        total_energy_ev: meci_res.total_energy_ev,
        heat_of_formation_kcal: meci_res.heat_of_formation_kcal,
    })
}

/// Simulate UV-Vis electronic absorption spectrum from CI states.
///
/// Parameters:
/// - `atomic_numbers`: list of integer atomic numbers
/// - `coordinates`: 3D coordinates in Angstroms
/// - `method`: Semi-empirical Hamiltonian (default: "PM6")
/// - `active_orbitals`: Number of active orbitals (default: 2)
/// - `min_wavelength_nm`: Minimum wavelength in nm (default: 100.0)
/// - `max_wavelength_nm`: Maximum wavelength in nm (default: 800.0)
/// - `step_nm`: Wavelength step size in nm (default: 1.0)
/// - `fwhm_nm`: Gaussian broadening FWHM in nm (default: 20.0)
/// - `use_nddo`: Enable full NDDO integrals (default: true)
#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coordinates,
    method = "PM6",
    active_orbitals = 2,
    min_wavelength_nm = 100.0,
    max_wavelength_nm = 800.0,
    step_nm = 1.0,
    fwhm_nm = 20.0,
    use_nddo = true
))]
pub fn uv_vis_spectrum(
    atomic_numbers: Vec<u8>,
    coordinates: Vec<[f64; 3]>,
    method: Option<&str>,
    active_orbitals: Option<usize>,
    min_wavelength_nm: Option<f64>,
    max_wavelength_nm: Option<f64>,
    step_nm: Option<f64>,
    fwhm_nm: Option<f64>,
    use_nddo: Option<bool>,
) -> PyResult<UvVisSpectrumPy> {
    // Run MECI first to get CI states
    let meci_res = meci(
        atomic_numbers,
        coordinates,
        method,
        active_orbitals,
        Some(1),
        use_nddo,
    )?;

    // Convert back to CiState-like data for spectrum simulation
    // We can directly use the oscillator strengths and wavelengths from CiStatePy
    let min_wl = min_wavelength_nm.unwrap_or(100.0);
    let max_wl = max_wavelength_nm.unwrap_or(800.0);
    let step = step_nm.unwrap_or(1.0);
    let fwhm = fwhm_nm.unwrap_or(20.0);

    // Build lightweight CiState vec for spectrum simulation
    use mopac_core::ci::meci::{CiState, StateSpin};
    let ci_states: Vec<CiState> = meci_res
        .states
        .iter()
        .map(|s| CiState {
            root: s.root,
            energy_ev: s.energy_ev,
            excitation_energy_ev: s.excitation_energy_ev,
            excitation_energy_cm1: s.excitation_energy_cm1,
            wavelength_nm: s.wavelength_nm,
            spin: StateSpin {
                s_squared: s.s_squared,
                s: ((1.0 + 4.0 * s.s_squared).sqrt() - 1.0) / 2.0,
                multiplicity: s.multiplicity,
                label: match s.multiplicity {
                    1 => "Singlet",
                    2 => "Doublet",
                    3 => "Triplet",
                    _ => "Unknown",
                },
            },
            transition_dipole_debye: s.transition_dipole_debye,
            dipole_strength_debye: s.dipole_strength_debye,
            polarization_angstrom2: s.polarization_angstrom2,
            oscillator_strength: s.oscillator_strength,
            eigenvector: Vec::new(),
        })
        .collect();

    let spectrum = simulate_uv_vis_spectrum(&ci_states, min_wl, max_wl, step, fwhm);

    Ok(UvVisSpectrumPy {
        wavelengths_nm: spectrum.wavelengths_nm,
        extinction_coefficients: spectrum.extinction_coefficients,
        lambda_max_nm: spectrum.lambda_max_nm,
        epsilon_max: spectrum.epsilon_max,
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
    #[pyo3(signature = (atomic_numbers, coordinates, temperature_k = 298.15, pressure_atm = 1.0, rotational_symmetry_number = 1.0, custom_masses = None))]
    fn frequencies(
        &self,
        atomic_numbers: Vec<u8>,
        coordinates: Vec<[f64; 3]>,
        temperature_k: Option<f64>,
        pressure_atm: Option<f64>,
        rotational_symmetry_number: Option<f64>,
        custom_masses: Option<Vec<f64>>,
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
            custom_masses,
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

    /// Trace Intrinsic Reaction Coordinate (IRC) path from a transition state.
    #[pyo3(signature = (atomic_numbers, coordinates, step_size = 0.1, max_points = 50, direction = "both"))]
    fn irc(
        &self,
        atomic_numbers: Vec<u8>,
        coordinates: Vec<[f64; 3]>,
        step_size: Option<f64>,
        max_points: Option<usize>,
        direction: Option<&str>,
    ) -> PyResult<IrcPyResult> {
        irc(
            atomic_numbers,
            coordinates,
            Some(&self.method),
            step_size,
            max_points,
            direction,
            Some(self.use_nddo),
        )
    }

    /// Propagate Dynamic Reaction Coordinate (DRC) / Born-Oppenheimer Molecular Dynamics.
    #[pyo3(signature = (atomic_numbers, coordinates, time_step_fs = 0.5, total_steps = 500, ensemble = "nve", temperature_k = 298.15, berendsen_tau_fs = 100.0, recording_interval = 1))]
    fn drc(
        &self,
        atomic_numbers: Vec<u8>,
        coordinates: Vec<[f64; 3]>,
        time_step_fs: Option<f64>,
        total_steps: Option<usize>,
        ensemble: Option<&str>,
        temperature_k: Option<f64>,
        berendsen_tau_fs: Option<f64>,
        recording_interval: Option<usize>,
    ) -> PyResult<DrcPyResult> {
        drc(
            atomic_numbers,
            coordinates,
            Some(&self.method),
            time_step_fs,
            total_steps,
            ensemble,
            temperature_k,
            berendsen_tau_fs,
            recording_interval,
            Some(self.use_nddo),
        )
    }

    /// Run MECI excited state calculation.
    #[pyo3(signature = (atomic_numbers, coordinates, active_orbitals = 2, target_root = 1))]
    fn meci(
        &self,
        atomic_numbers: Vec<u8>,
        coordinates: Vec<[f64; 3]>,
        active_orbitals: Option<usize>,
        target_root: Option<usize>,
    ) -> PyResult<MeciPyResult> {
        meci(
            atomic_numbers,
            coordinates,
            Some(&self.method),
            active_orbitals,
            target_root,
            Some(self.use_nddo),
        )
    }

    /// Simulate UV-Vis absorption spectrum.
    #[pyo3(signature = (atomic_numbers, coordinates, active_orbitals = 2, min_wavelength_nm = 100.0, max_wavelength_nm = 800.0, step_nm = 1.0, fwhm_nm = 20.0))]
    fn uv_vis_spectrum(
        &self,
        atomic_numbers: Vec<u8>,
        coordinates: Vec<[f64; 3]>,
        active_orbitals: Option<usize>,
        min_wavelength_nm: Option<f64>,
        max_wavelength_nm: Option<f64>,
        step_nm: Option<f64>,
        fwhm_nm: Option<f64>,
    ) -> PyResult<UvVisSpectrumPy> {
        uv_vis_spectrum(
            atomic_numbers,
            coordinates,
            Some(&self.method),
            active_orbitals,
            min_wavelength_nm,
            max_wavelength_nm,
            step_nm,
            fwhm_nm,
            Some(self.use_nddo),
        )
    }

    /// Calculate periodic boundary conditions Bloch SCF and band structure.
    #[pyo3(signature = (atomic_numbers, coordinates, translation_vectors, mers = None, k_grid = None, band_points = 40))]
    fn pbc(
        &self,
        atomic_numbers: Vec<u8>,
        coordinates: Vec<[f64; 3]>,
        translation_vectors: Vec<[f64; 3]>,
        mers: Option<[usize; 3]>,
        k_grid: Option<[usize; 3]>,
        band_points: Option<usize>,
    ) -> PyResult<PbcResultPy> {
        pbc(
            atomic_numbers,
            coordinates,
            translation_vectors,
            Some(&self.method),
            mers,
            k_grid,
            Some(self.use_nddo),
            Some(self.max_iter),
            band_points,
        )
    }
}

/// Periodic Boundary Condition calculation and band structure results.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct PbcResultPy {
    pub converged: bool,
    pub iterations: usize,
    pub total_energy_per_cell_ev: f64,
    pub electronic_energy_per_cell_ev: f64,
    pub nuclear_repulsion_per_cell_ev: f64,
    pub heat_of_formation_kcal_mol: f64,
    pub vbm_energy_ev: f64,
    pub cbm_energy_ev: f64,
    pub direct_bandgap_ev: f64,
    pub indirect_bandgap_ev: f64,
    pub band_energies_ev: Vec<Vec<f64>>,
    pub dos_energies_ev: Vec<f64>,
    pub dos_values: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coordinates,
    translation_vectors,
    method = "PM6",
    mers = None,
    k_grid = None,
    use_nddo = true,
    max_iter = 100,
    band_points = 40
))]
pub fn pbc(
    atomic_numbers: Vec<u8>,
    coordinates: Vec<[f64; 3]>,
    translation_vectors: Vec<[f64; 3]>,
    method: Option<&str>,
    mers: Option<[usize; 3]>,
    k_grid: Option<[usize; 3]>,
    use_nddo: Option<bool>,
    max_iter: Option<usize>,
    band_points: Option<usize>,
) -> PyResult<PbcResultPy> {
    if atomic_numbers.is_empty() {
        return Err(PyValueError::new_err("atomic_numbers must not be empty"));
    }
    if atomic_numbers.len() != coordinates.len() {
        return Err(PyValueError::new_err(
            "atomic_numbers and coordinates must have same length",
        ));
    }
    if translation_vectors.is_empty() || translation_vectors.len() > 3 {
        return Err(PyValueError::new_err(
            "translation_vectors must have 1, 2, or 3 vectors",
        ));
    }

    let model_str = method.unwrap_or("PM6");
    let model = get_model(model_str)?;
    let unit_cell =
        UnitCell::from_translation_vectors(&translation_vectors).map_err(PyValueError::new_err)?;

    let mut pbc_opts = PbcOptions::new(unit_cell);
    if let Some(m) = mers {
        pbc_opts.mers = m;
    }
    if let Some(k) = k_grid {
        pbc_opts.k_grid = k;
    }
    if let Some(nddo) = use_nddo {
        pbc_opts.use_nddo = nddo;
    }
    if let Some(mi) = max_iter {
        pbc_opts.max_iter = mi;
    }
    if let Some(bp) = band_points {
        pbc_opts.band_path_points = bp;
    }

    let batch = MolecularBatch::new_for_model(atomic_numbers, &coordinates, model.as_ref());
    let n_trans = pbc_opts.mers[0] * pbc_opts.mers[1] * pbc_opts.mers[2];
    let n_k = pbc_opts.k_grid[0] * pbc_opts.k_grid[1] * pbc_opts.k_grid[2];
    let mut ws = PbcWorkspace::allocate(batch.norbs, n_trans, n_k);

    let res =
        run_pbc_scf(&batch, model.as_ref(), &pbc_opts, &mut ws).map_err(PyValueError::new_err)?;

    Ok(PbcResultPy {
        converged: res.converged,
        iterations: res.iterations,
        total_energy_per_cell_ev: res.total_energy_per_cell_ev,
        electronic_energy_per_cell_ev: res.electronic_energy_per_cell_ev,
        nuclear_repulsion_per_cell_ev: res.nuclear_repulsion_per_cell_ev,
        heat_of_formation_kcal_mol: res.heat_of_formation_kcal_mol,
        vbm_energy_ev: res.vbm_energy_ev,
        cbm_energy_ev: res.cbm_energy_ev,
        direct_bandgap_ev: res.direct_bandgap_ev,
        indirect_bandgap_ev: res.indirect_bandgap_ev,
        band_energies_ev: res.band_energies_ev,
        dos_energies_ev: res.dos_energies_ev,
        dos_values: res.dos_values,
    })
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
    m.add_class::<IrcPointPy>()?;
    m.add_class::<IrcPyResult>()?;
    m.add_class::<DrcFramePy>()?;
    m.add_class::<DrcPyResult>()?;
    m.add_class::<CiStatePy>()?;
    m.add_class::<MeciPyResult>()?;
    m.add_class::<UvVisSpectrumPy>()?;
    m.add_class::<PbcResultPy>()?;
    m.add_class::<EspResultPy>()?;
    m.add_class::<MopacCalculator>()?;
    m.add_function(wrap_pyfunction!(calculate, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(transition_state, m)?)?;
    m.add_function(wrap_pyfunction!(frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(esp_charges, m)?)?;
    m.add_function(wrap_pyfunction!(irc, m)?)?;
    m.add_function(wrap_pyfunction!(drc, m)?)?;
    m.add_function(wrap_pyfunction!(meci, m)?)?;
    m.add_function(wrap_pyfunction!(uv_vis_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(pbc, m)?)?;

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
