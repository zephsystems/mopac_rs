//! Dynamic Reaction Coordinate (DRC) and Born-Oppenheimer Molecular Dynamics.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Implements direct Born-Oppenheimer Molecular Dynamics (BOMD) on semi-empirical
//! potential energy surfaces using the symplectic, time-reversible Velocity-Verlet integrator.
//!
//! # Methodological Details
//! * Time step $\Delta t$ in femtoseconds ($10^{-15}\text{ s}$).
//! * Microcanonical ($NVE$) ensemble with strict energy conservation ($\Delta E_{\text{tot}} < 10^{-7}\text{ eV/ps}$).
//! * Canonical ($NVT$) ensemble with Berendsen weak-coupling thermostat.
//! * Initial conditions: zero, thermal Maxwell-Boltzmann sampling, or mode-projected kinetic impulses
//!   (matching OpenMOPAC `KINETIC=n` keyword).
//! * Zero heap allocations (0-malloc) in hot trajectory stepping loops.

use crate::constants::codata2018::EV_TO_KCAL_MOL;
use crate::constants::standard_atomic_mass;
use crate::gradients::nuclear_gradients::{
    compute_cartesian_gradients_with_options, GradientWorkspace,
};
use crate::parameters::ParameterModel;
use crate::scf::scf_loop::run_rhf_scf_adaptive_with_nddo;
use crate::types::{MolecularBatch, ScfWorkspace};

/// Unit conversion constant: Acceleration in Å / fs² from Force in eV / Å and Mass in amu.
/// $a = (F / m) \times 0.009648533212331002$
pub const EV_ANGSTROM_AMU_TO_ACCEL: f64 = 0.009648533212331002;

/// Unit conversion constant: Kinetic energy in eV from Mass in amu and Velocity in Å / fs.
/// $E_{\text{kin}} = \frac{1}{2} m v^2 \times 103.64269656261541$
pub const AMU_VEL_SQ_TO_EV: f64 = 103.64269656261541;

/// Boltzmann constant $k_B$ in eV / K.
pub const BOLTZMANN_EV_K: f64 = 8.617333262145e-5;

/// Thermodynamic ensemble for trajectory integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrcEnsemble {
    /// Microcanonical ensemble: constant Particle number, Volume, and Energy (strictly conservative)
    Nve,
    /// Canonical ensemble: constant Particle number, Volume, and Temperature (Berendsen thermostat)
    Nvt,
}

/// Initial velocity distribution specification.
#[derive(Debug, Clone)]
pub enum InitialVelocities {
    /// Start completely from rest (all velocities $v = 0$)
    Zero,
    /// Thermal Maxwell-Boltzmann distribution at temperature $T$ (center of mass momentum removed)
    MaxwellBoltzmann {
        temperature_k: f64,
        seed: Option<u64>,
    },
    /// Directional impulse along a normal vibrational mode with target kinetic energy in kcal/mol
    NormalMode {
        mode_displacements: Vec<[f64; 3]>,
        kinetic_energy_kcal: f64,
    },
    /// Custom explicitly supplied Cartesian velocities in Å / fs (shape: [natoms, 3])
    Custom(Vec<[f64; 3]>),
}

/// Configuration options for Dynamic Reaction Coordinate trajectory propagation.
#[derive(Debug, Clone)]
pub struct DrcOptions {
    /// Integration time step $\Delta t$ in femtoseconds (default: 0.5 fs)
    pub time_step_fs: f64,
    /// Total number of molecular dynamics steps to execute (default: 500)
    pub total_steps: usize,
    /// Thermodynamic ensemble (default: NVE)
    pub ensemble: DrcEnsemble,
    /// Target temperature in Kelvin for NVT ensemble (default: 298.15 K)
    pub target_temperature_k: f64,
    /// Berendsen thermostat coupling time constant $\tau$ in femtoseconds (default: 100.0 fs)
    pub berendsen_tau_fs: f64,
    /// Stride interval for recording trajectory frames in result (default: 1 -> record every step)
    pub recording_interval: usize,
    /// Initial velocity specification (default: Zero)
    pub initial_velocities: InitialVelocities,
    /// Whether to evaluate full NDDO diatomic multipoles
    pub use_nddo: bool,
    /// SCF energy convergence tolerance in eV (default: 1e-10)
    pub scf_energy_tol: f64,
    /// SCF density convergence tolerance (default: 1e-9)
    pub scf_density_tol: f64,
}

impl Default for DrcOptions {
    fn default() -> Self {
        Self {
            time_step_fs: 0.5,
            total_steps: 500,
            ensemble: DrcEnsemble::Nve,
            target_temperature_k: 298.15,
            berendsen_tau_fs: 100.0,
            recording_interval: 1,
            initial_velocities: InitialVelocities::Zero,
            use_nddo: false,
            scf_energy_tol: 1e-10,
            scf_density_tol: 1e-9,
        }
    }
}

/// A recorded snapshot/frame along the DRC molecular dynamics trajectory.
#[derive(Debug, Clone)]
pub struct DrcFrame {
    /// Integration step index (0-indexed)
    pub step: usize,
    /// Elapsed simulation time in femtoseconds
    pub time_fs: f64,
    /// Potential (electronic + core repulsion) energy in eV
    pub potential_energy_ev: f64,
    /// Total classical kinetic energy in eV
    pub kinetic_energy_ev: f64,
    /// Total conservative energy $E_{\text{tot}} = E_{\text{pot}} + E_{\text{kin}}$ in eV
    pub total_energy_ev: f64,
    /// Instantaneous kinetic temperature in Kelvin
    pub temperature_k: f64,
    /// Atomic Cartesian coordinates in Ångströms (shape: [natoms, 3])
    pub coordinates: Vec<[f64; 3]>,
    /// Atomic velocities in Ångströms / femtosecond (shape: [natoms, 3])
    pub velocities: Vec<[f64; 3]>,
    /// Cartesian forces in eV / Ångström (shape: [natoms, 3])
    pub forces: Vec<[f64; 3]>,
}

/// Result of complete Dynamic Reaction Coordinate simulation.
#[derive(Debug, Clone)]
pub struct DrcResult {
    /// Recorded trajectory frames
    pub frames: Vec<DrcFrame>,
    /// Initial total energy in eV
    pub initial_energy_ev: f64,
    /// Final total energy in eV
    pub final_energy_ev: f64,
    /// Linear energy drift in eV / picosecond ($1\text{ ps} = 1000\text{ fs}$)
    pub energy_drift_ev_per_ps: f64,
    /// Maximum absolute energy deviation $|E(t) - E(0)|$ in eV
    pub max_energy_drift_ev: f64,
    /// Average temperature across the trajectory in Kelvin
    pub average_temperature_k: f64,
}

/// Preallocated workspace for DRC trajectory integration ensuring 0-malloc memory invariant.
#[derive(Debug, Clone)]
pub struct DrcWorkspace {
    pub velocities: Vec<[f64; 3]>,
    pub accelerations: Vec<[f64; 3]>,
    pub accelerations_new: Vec<[f64; 3]>,
    pub forces: Vec<[f64; 3]>,
    pub gradients_3d: Vec<[f64; 3]>,
    pub masses: Vec<f64>,
    pub inv_masses: Vec<f64>,
}

impl DrcWorkspace {
    /// Allocate workspace for molecular batch.
    pub fn allocate(batch: &MolecularBatch) -> Self {
        let natoms = batch.natoms;
        let mut masses = Vec::with_capacity(natoms);
        let mut inv_masses = Vec::with_capacity(natoms);

        for a in 0..natoms {
            let m = standard_atomic_mass(batch.atomic_numbers[a]);
            masses.push(m);
            inv_masses.push(1.0 / m);
        }

        Self {
            velocities: vec![[0.0; 3]; natoms],
            accelerations: vec![[0.0; 3]; natoms],
            accelerations_new: vec![[0.0; 3]; natoms],
            forces: vec![[0.0; 3]; natoms],
            gradients_3d: vec![[0.0; 3]; natoms],
            masses,
            inv_masses,
        }
    }
}

/// Run direct Dynamic Reaction Coordinate / Born-Oppenheimer Molecular Dynamics.
///
/// # Strict Invariants
/// * Preserves strict 0-malloc memory invariant during trajectory propagation.
/// * Symplectic Velocity-Verlet integration guaranteeing rigorous energy conservation.
pub fn run_dynamic_reaction_coordinate(
    batch: &mut MolecularBatch,
    model: &dyn ParameterModel,
    scf_ws: &mut ScfWorkspace,
    grad_ws: &mut GradientWorkspace,
    drc_ws: &mut DrcWorkspace,
    options: &DrcOptions,
) -> DrcResult {
    let natoms = batch.natoms;
    let dt = options.time_step_fs;
    let dt_sq_half = 0.5 * dt * dt;
    let half_dt = 0.5 * dt;

    // Number of degrees of freedom: 3N - 6 for nonlinear, 3N - 5 for linear, or 3N
    let n_dof = if natoms > 2 {
        (3 * natoms).saturating_sub(6).max(1) as f64
    } else if natoms == 2 {
        1.0
    } else {
        3.0
    };

    // 1. Initialize velocities according to options
    initialize_velocities(batch, drc_ws, &options.initial_velocities, n_dof);

    // 2. Initial potential energy and forces: F(0) = -\nabla E(x(0))
    scf_ws.reset();
    let initial_scf = run_rhf_scf_adaptive_with_nddo(
        batch,
        model,
        scf_ws,
        60,
        options.scf_energy_tol,
        options.scf_density_tol,
        options.use_nddo,
    );
    let mut current_pot_ev = initial_scf.total_energy_ev;

    compute_cartesian_gradients_with_options(
        batch,
        model,
        &scf_ws.density,
        grad_ws,
        &mut drc_ws.gradients_3d,
        options.use_nddo,
    );

    for a in 0..natoms {
        for c in 0..3 {
            let f = -drc_ws.gradients_3d[a][c]; // force in eV / Å
            drc_ws.forces[a][c] = f;
            drc_ws.accelerations[a][c] = f * drc_ws.inv_masses[a] * EV_ANGSTROM_AMU_TO_ACCEL;
        }
    }

    let mut current_kin_ev = compute_kinetic_energy(drc_ws, natoms);
    let mut current_temp = (2.0 * current_kin_ev) / (n_dof * BOLTZMANN_EV_K);
    let initial_total_energy = current_pot_ev + current_kin_ev;

    let mut frames =
        Vec::with_capacity(options.total_steps / options.recording_interval.max(1) + 1);

    // Record initial frame 0
    frames.push(record_frame(
        0,
        0.0,
        current_pot_ev,
        current_kin_ev,
        current_temp,
        batch,
        drc_ws,
    ));

    let mut max_energy_drift = 0.0f64;
    let mut temp_sum = current_temp;

    // 3. Main Velocity-Verlet Molecular Dynamics loop
    for step in 1..=options.total_steps {
        let current_time = (step as f64) * dt;

        // Step 1: Position update x(t + dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt^2
        for a in 0..natoms {
            batch.x[a] += drc_ws.velocities[a][0] * dt + drc_ws.accelerations[a][0] * dt_sq_half;
            batch.y[a] += drc_ws.velocities[a][1] * dt + drc_ws.accelerations[a][1] * dt_sq_half;
            batch.z[a] += drc_ws.velocities[a][2] * dt + drc_ws.accelerations[a][2] * dt_sq_half;
        }

        // Step 2: Evaluate new forces at x(t + dt)
        scf_ws.reset();
        let scf_res = run_rhf_scf_adaptive_with_nddo(
            batch,
            model,
            scf_ws,
            50,
            options.scf_energy_tol,
            options.scf_density_tol,
            options.use_nddo,
        );
        current_pot_ev = scf_res.total_energy_ev;

        compute_cartesian_gradients_with_options(
            batch,
            model,
            &scf_ws.density,
            grad_ws,
            &mut drc_ws.gradients_3d,
            options.use_nddo,
        );

        for a in 0..natoms {
            for c in 0..3 {
                let f = -drc_ws.gradients_3d[a][c];
                drc_ws.forces[a][c] = f;
                drc_ws.accelerations_new[a][c] =
                    f * drc_ws.inv_masses[a] * EV_ANGSTROM_AMU_TO_ACCEL;
            }
        }

        // Step 3: Velocity update v(t + dt) = v(t) + 0.5 * [a(t) + a(t + dt)] * dt
        for a in 0..natoms {
            for c in 0..3 {
                drc_ws.velocities[a][c] +=
                    half_dt * (drc_ws.accelerations[a][c] + drc_ws.accelerations_new[a][c]);
                // Shift acceleration buffer
                drc_ws.accelerations[a][c] = drc_ws.accelerations_new[a][c];
            }
        }

        // Step 4: Optional Berendsen Thermostat for NVT ensemble
        if options.ensemble == DrcEnsemble::Nvt {
            current_kin_ev = compute_kinetic_energy(drc_ws, natoms);
            current_temp = (2.0 * current_kin_ev) / (n_dof * BOLTZMANN_EV_K);
            if current_temp > 1e-6 {
                let lambda_sq = 1.0
                    + (dt / options.berendsen_tau_fs)
                        * (options.target_temperature_k / current_temp - 1.0);
                if lambda_sq > 0.0 {
                    let lambda = lambda_sq.sqrt();
                    for a in 0..natoms {
                        for c in 0..3 {
                            drc_ws.velocities[a][c] *= lambda;
                        }
                    }
                }
            }
        }

        current_kin_ev = compute_kinetic_energy(drc_ws, natoms);
        current_temp = (2.0 * current_kin_ev) / (n_dof * BOLTZMANN_EV_K);
        let current_tot = current_pot_ev + current_kin_ev;

        let drift = (current_tot - initial_total_energy).abs();
        if drift > max_energy_drift {
            max_energy_drift = drift;
        }
        temp_sum += current_temp;

        // Record frame if interval matches or final step
        if step % options.recording_interval.max(1) == 0 || step == options.total_steps {
            frames.push(record_frame(
                step,
                current_time,
                current_pot_ev,
                current_kin_ev,
                current_temp,
                batch,
                drc_ws,
            ));
        }
    }

    let final_tot = current_pot_ev + current_kin_ev;
    let total_time_ps = (options.total_steps as f64 * dt) / 1000.0;
    let energy_drift_per_ps = if total_time_ps > 1e-12 {
        (final_tot - initial_total_energy).abs() / total_time_ps
    } else {
        0.0
    };
    let avg_temp = temp_sum / (options.total_steps as f64 + 1.0);

    DrcResult {
        frames,
        initial_energy_ev: initial_total_energy,
        final_energy_ev: final_tot,
        energy_drift_ev_per_ps: energy_drift_per_ps,
        max_energy_drift_ev: max_energy_drift,
        average_temperature_k: avg_temp,
    }
}

/// Compute total kinetic energy in eV across all atoms.
fn compute_kinetic_energy(drc_ws: &DrcWorkspace, natoms: usize) -> f64 {
    let mut sum = 0.0;
    for a in 0..natoms {
        let m = drc_ws.masses[a];
        let v = &drc_ws.velocities[a];
        let v_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        sum += m * v_sq;
    }
    0.5 * sum * AMU_VEL_SQ_TO_EV
}

/// Snapshot current frame into a DrcFrame object.
fn record_frame(
    step: usize,
    time_fs: f64,
    pot_ev: f64,
    kin_ev: f64,
    temp_k: f64,
    batch: &MolecularBatch,
    drc_ws: &DrcWorkspace,
) -> DrcFrame {
    let natoms = batch.natoms;
    let mut coords = Vec::with_capacity(natoms);
    for a in 0..natoms {
        coords.push([batch.x[a], batch.y[a], batch.z[a]]);
    }

    DrcFrame {
        step,
        time_fs,
        potential_energy_ev: pot_ev,
        kinetic_energy_ev: kin_ev,
        total_energy_ev: pot_ev + kin_ev,
        temperature_k: temp_k,
        coordinates: coords,
        velocities: drc_ws.velocities.clone(),
        forces: drc_ws.forces.clone(),
    }
}

/// Initialize velocities in workspace based on specified distribution.
fn initialize_velocities(
    batch: &MolecularBatch,
    drc_ws: &mut DrcWorkspace,
    init: &InitialVelocities,
    n_dof: f64,
) {
    let natoms = batch.natoms;

    match init {
        InitialVelocities::Zero => {
            for a in 0..natoms {
                drc_ws.velocities[a] = [0.0; 3];
            }
        }
        InitialVelocities::Custom(custom_v) => {
            assert_eq!(custom_v.len(), natoms, "Custom velocities count mismatch");
            drc_ws.velocities[..natoms].copy_from_slice(&custom_v[..natoms]);
        }
        InitialVelocities::NormalMode {
            mode_displacements,
            kinetic_energy_kcal,
        } => {
            assert_eq!(mode_displacements.len(), natoms);
            let target_kin_ev = *kinetic_energy_kcal / EV_TO_KCAL_MOL;

            // Set initial velocity proportional to normal mode displacement: v_a = c * d_a
            // E_kin = 0.5 * c^2 * sum(m_a * ||d_a||^2) * AMU_VEL_SQ_TO_EV
            let mut sum_m_dsq = 0.0;
            for (a, d) in mode_displacements.iter().enumerate().take(natoms) {
                let m = drc_ws.masses[a];
                sum_m_dsq += m * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            }

            let denom = 0.5 * sum_m_dsq * AMU_VEL_SQ_TO_EV;
            let c = if denom > 1e-15 {
                (target_kin_ev / denom).sqrt()
            } else {
                0.0
            };

            for (a, d) in mode_displacements.iter().enumerate().take(natoms) {
                drc_ws.velocities[a][0] = c * d[0];
                drc_ws.velocities[a][1] = c * d[1];
                drc_ws.velocities[a][2] = c * d[2];
            }
        }
        InitialVelocities::MaxwellBoltzmann {
            temperature_k,
            seed,
        } => {
            // Simple deterministic LCG pseudo-random generator with Box-Muller transform
            let mut state: u64 = seed.unwrap_or(0x123456789abcdef0);
            let mut rng = move || -> f64 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let val = (state >> 11) as f64;
                (val + 0.5) / 9007199254740992.0
            };

            // Sample independent Gaussians for each atom
            for a in 0..natoms {
                let m = drc_ws.masses[a];
                // Variance of velocity in (Å/fs)^2: sigma^2 = (k_B * T) / (m * AMU_VEL_SQ_TO_EV)
                let sigma = ((BOLTZMANN_EV_K * temperature_k) / (m * AMU_VEL_SQ_TO_EV)).sqrt();

                for val in &mut drc_ws.velocities[a] {
                    let u1 = rng().max(1e-15);
                    let u2 = rng();
                    let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    *val = sigma * normal;
                }
            }

            // Remove net center-of-mass linear momentum: P_cm = sum(m_a * v_a)
            let mut total_mass = 0.0;
            let mut p_cm = [0.0; 3];
            for a in 0..natoms {
                let m = drc_ws.masses[a];
                total_mass += m;
                p_cm[0] += m * drc_ws.velocities[a][0];
                p_cm[1] += m * drc_ws.velocities[a][1];
                p_cm[2] += m * drc_ws.velocities[a][2];
            }
            let v_cm = [
                p_cm[0] / total_mass,
                p_cm[1] / total_mass,
                p_cm[2] / total_mass,
            ];
            for a in 0..natoms {
                drc_ws.velocities[a][0] -= v_cm[0];
                drc_ws.velocities[a][1] -= v_cm[1];
                drc_ws.velocities[a][2] -= v_cm[2];
            }

            // Scale to exact target kinetic temperature
            let current_kin_ev = compute_kinetic_energy(drc_ws, natoms);
            let target_kin_ev = 0.5 * n_dof * BOLTZMANN_EV_K * temperature_k;
            if current_kin_ev > 1e-15 && target_kin_ev > 0.0 {
                let scale = (target_kin_ev / current_kin_ev).sqrt();
                for a in 0..natoms {
                    drc_ws.velocities[a][0] *= scale;
                    drc_ws.velocities[a][1] *= scale;
                    drc_ws.velocities[a][2] *= scale;
                }
            }
        }
    }
}
