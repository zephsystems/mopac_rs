//! Canonical MOPAC Command-Line Interface (CLI).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Provides drop-in MOPAC CLI compatibility with CPU AVX2 and Vulkan GPU FP32/FP64 backends.

use clap::Parser;
use mopac_core::constants::codata2018::EV_TO_KCAL_MOL;
use mopac_core::corrections::{
    compute_dispersion_energy, compute_h4_energy, compute_hh_repulsion_energy_and_gradients,
    DispersionModel, H4Parameters,
};
use mopac_core::gradients::GradientWorkspace;
use mopac_core::opt::{
    optimize_geometry_lbfgs, optimize_transition_state, EigenvectorFollowingWorkspace,
    HessianUpdateScheme, OptimizationOptions, TransitionStateOptions,
};
use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::mndo::MndoModel;
use mopac_core::parameters::pm3::Pm3Model;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::parameters::pm7::Pm7Model;
use mopac_core::parameters::rm1::Rm1Model;
use mopac_core::parameters::ParameterModel;
use mopac_core::properties::{
    compute_bond_orders, compute_dipole_moment, compute_mulliken_population, DipoleResult,
};
use mopac_core::reactions::{
    run_dynamic_reaction_coordinate, trace_intrinsic_reaction_coordinate, DrcEnsemble, DrcOptions,
    DrcWorkspace, InitialVelocities, IrcDirection, IrcOptions, IrcWorkspace,
};
use mopac_core::scf::scf_loop::{
    run_rhf_scf_adaptive_with_nddo, run_rhf_scf_adaptive_with_nddo_and_cosmo, ScfOptions, ScfResult,
};
use mopac_core::solvation::{CosmoCavity, CosmoParams};
use mopac_core::types::{MolecularBatch, ScfWorkspace};
use mopac_core::vibrations::{compute_hessian_and_frequencies, HessianOptions};
use mopac_gpu::coulomb_fp32::GpuCoulombCalculatorFP32;
use mopac_gpu::{GpuCoulombCalculator, VulkanContext};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "mopac",
    version = "0.1.0-alpha",
    about = "MOPAC_RS: Modern Data-Oriented Semi-Empirical Quantum Chemistry Engine"
)]
struct Cli {
    /// Input file path (.mop)
    input: PathBuf,

    /// Force calculation mode (1SCF or OPT)
    #[arg(short, long)]
    mode: Option<String>,

    /// Semi-empirical method override (PM6, PM3, RM1, AM1, MNDO)
    #[arg(long)]
    method: Option<String>,

    /// Enable full NDDO diatomic 22-multipole integrals & 3D rotation frame
    #[arg(long)]
    nddo: bool,

    /// Enable geometry optimization (L-BFGS)
    #[arg(long)]
    opt: bool,

    /// Enable transition state optimization via Eigenvector Following (P-RFO Baker)
    #[arg(long)]
    ts: bool,

    /// Enable Intrinsic Reaction Coordinate (IRC) path tracing (González-Schlegel)
    #[arg(long)]
    irc: bool,

    /// Enable Dynamic Reaction Coordinate (DRC) molecular dynamics (Velocity-Verlet)
    #[arg(long)]
    drc: bool,

    /// Enable Cartesian Hessian, mass-weighting and normal mode vibrational analysis
    #[arg(long)]
    force: bool,

    /// Enable Mayer bond orders and atomic valencies calculation
    #[arg(long)]
    bonds: bool,

    /// Force single-point calculation (1SCF)
    #[arg(long = "1scf", default_value_t = false)]
    one_scf: bool,

    /// Enable Mulliken population analysis and Löwdin de-orthogonalization
    #[arg(long, alias = "mulliken")]
    mullik: bool,

    /// Enable Vulkan GPU compute acceleration
    #[arg(long)]
    gpu: bool,

    /// Use FP32 single-precision GPU pipeline (faster on consumer gaming GPUs)
    #[arg(long)]
    fp32: bool,

    /// Number of Rayon worker threads
    #[arg(long)]
    threads: Option<usize>,

    /// Solvent dielectric constant for COSMO implicit solvation (e.g. --eps 78.4)
    #[arg(long)]
    eps: Option<f64>,

    /// Non-covalent empirical dispersion model (e.g. --disp pm6-dh+ or --disp pm7)
    #[arg(long)]
    disp: Option<String>,

    /// Enable D3H4 composite correction (dispersion, H4 hydrogen bonding, and H-H repulsion)
    #[arg(long)]
    d3h4: bool,

    /// Custom output file path (.out)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Custom archive file path (.arc)
    #[arg(long)]
    arc: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, Default)]
struct NonCovalentBreakdown {
    dispersion_kcal: f64,
    h4_kcal: f64,
    hh_repulsion_kcal: f64,
}

#[derive(Debug, Clone)]
struct ParsedAtom {
    #[allow(dead_code)]
    symbol: String,
    z: u8,
    x: f64,
    y: f64,
    z_coord: f64,
    opt_x: bool,
    opt_y: bool,
    opt_z: bool,
}

#[derive(Debug)]
struct ParsedInput {
    keywords: Vec<String>,
    title: String,
    comment: String,
    atoms: Vec<ParsedAtom>,
    is_opt_requested: bool,
    is_ts_requested: bool,
    is_irc_requested: bool,
    is_drc_requested: bool,
    is_force_requested: bool,
    is_gpu_requested: bool,
    is_fp32_requested: bool,
    method: Option<String>,
    is_nddo_requested: bool,
    #[allow(dead_code)]
    is_dipole_requested: bool,
    is_bonds_requested: bool,
    is_mullik_requested: bool,
    eps: Option<f64>,
    dispersion: Option<DispersionModel>,
    use_h4: bool,
    use_hh: bool,
}

fn symbol_to_atomic_number(sym: &str) -> Option<u8> {
    match sym.to_uppercase().as_str() {
        "H" => Some(1),
        "HE" => Some(2),
        "LI" => Some(3),
        "BE" => Some(4),
        "B" => Some(5),
        "C" => Some(6),
        "N" => Some(7),
        "O" => Some(8),
        "F" => Some(9),
        "NE" => Some(10),
        "NA" => Some(11),
        "MG" => Some(12),
        "AL" => Some(13),
        "SI" => Some(14),
        "P" => Some(15),
        "S" => Some(16),
        "CL" => Some(17),
        "AR" => Some(18),
        "K" => Some(19),
        "CA" => Some(20),
        "BR" => Some(35),
        "I" => Some(53),
        _ => None,
    }
}

fn atomic_number_to_symbol(z: u8) -> &'static str {
    match z {
        1 => "H",
        2 => "He",
        3 => "Li",
        4 => "Be",
        5 => "B",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        10 => "Ne",
        11 => "Na",
        12 => "Mg",
        13 => "Al",
        14 => "Si",
        15 => "P",
        16 => "S",
        17 => "Cl",
        18 => "Ar",
        19 => "K",
        20 => "Ca",
        35 => "Br",
        53 => "I",
        _ => "X",
    }
}

fn get_isolated_atom_energy_and_heat(z: u8, model: &dyn ParameterModel) -> (f64, f64) {
    mopac_core::properties::get_isolated_atom_energy_and_heat(z, model)
}

fn parse_mopac_input(content: &str) -> io::Result<ParsedInput> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Empty input file",
        ));
    }

    let kw_line = lines[0].trim();
    let keywords: Vec<String> = kw_line.split_whitespace().map(|s| s.to_string()).collect();

    let title = if lines.len() > 1 {
        lines[1].trim().to_string()
    } else {
        String::new()
    };
    let comment = if lines.len() > 2 {
        lines[2].trim().to_string()
    } else {
        String::new()
    };

    let mut is_opt_requested = false;
    let mut is_ts_requested = false;
    let mut is_irc_requested = false;
    let mut is_drc_requested = false;
    let mut is_force_requested = false;
    let mut is_gpu = false;
    let mut is_fp32 = false;
    let mut is_1scf = false;
    let mut method = None;
    let mut is_nddo_requested = false;
    let mut is_dipole_requested = false;
    let mut is_bonds_requested = false;
    let mut is_mullik_requested = false;
    let mut eps = None;
    let mut dispersion = None;
    let mut use_h4 = false;
    let mut use_hh = false;

    for kw in &keywords {
        let u = kw.to_uppercase();
        if u == "TS" {
            is_ts_requested = true;
        } else if u == "IRC" || u.starts_with("IRC=") {
            is_irc_requested = true;
        } else if u == "DRC" {
            is_drc_requested = true;
        } else if u == "OPT" || u == "EF" || u == "BFGS" {
            is_opt_requested = true;
        } else if u == "FORCE" || u == "VIB" || u == "FREQ" || u == "THERMO" {
            is_force_requested = true;
        } else if u == "DIPOLE" {
            is_dipole_requested = true;
        } else if u == "BONDS" {
            is_bonds_requested = true;
        } else if u == "MULLIK" || u == "MULLIKEN" {
            is_mullik_requested = true;
        } else if u == "1SCF" {
            is_1scf = true;
        } else if u == "GPU" {
            is_gpu = true;
        } else if u == "FP32" {
            is_fp32 = true;
        } else if u == "PM6-D3H4" || u == "D3H4" {
            method = Some("PM6".to_string());
            dispersion = Some(DispersionModel::Pm6DhPlus);
            use_h4 = true;
            use_hh = true;
        } else if u == "PM6-DH+" || u == "PM6-DH2" || u == "DH+" || u == "DISP" {
            method = Some("PM6".to_string());
            dispersion = Some(DispersionModel::Pm6DhPlus);
        } else if u == "PM7" {
            method = Some("PM7".to_string());
            dispersion = Some(DispersionModel::Pm7);
        } else if u == "H4" {
            use_h4 = true;
        } else if u == "PM6" {
            method = Some("PM6".to_string());
        } else if u == "PM3" {
            method = Some("PM3".to_string());
        } else if u == "RM1" {
            method = Some("RM1".to_string());
        } else if u == "AM1" {
            method = Some("AM1".to_string());
        } else if u == "MNDO" {
            method = Some("MNDO".to_string());
        } else if u == "NDDO" {
            is_nddo_requested = true;
        } else if let Some(stripped) = u.strip_prefix("EPS=") {
            if let Ok(v) = stripped.parse::<f64>() {
                eps = Some(v);
            }
        }
    }

    let mut atoms = Vec::new();
    for line in lines.iter().skip(3) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        let sym = tokens[0];
        let z = match symbol_to_atomic_number(sym) {
            Some(num) => num,
            None => continue,
        };

        let mut x = 0.0;
        let mut y = 0.0;
        let mut z_c = 0.0;
        let mut opt_x = true;
        let mut opt_y = true;
        let mut opt_z = true;

        if tokens.len() >= 7 {
            x = tokens[1].parse().unwrap_or(0.0);
            opt_x = tokens[2] == "1";
            y = tokens[3].parse().unwrap_or(0.0);
            opt_y = tokens[4] == "1";
            z_c = tokens[5].parse().unwrap_or(0.0);
            opt_z = tokens[6] == "1";
        } else if tokens.len() >= 4 {
            x = tokens[1].parse().unwrap_or(0.0);
            y = tokens[2].parse().unwrap_or(0.0);
            z_c = tokens[3].parse().unwrap_or(0.0);
        }

        atoms.push(ParsedAtom {
            symbol: sym.to_string(),
            z,
            x,
            y,
            z_coord: z_c,
            opt_x,
            opt_y,
            opt_z,
        });
    }

    if !is_1scf && !is_opt_requested {
        let any_opt = atoms.iter().any(|a| a.opt_x || a.opt_y || a.opt_z);
        if any_opt {
            is_opt_requested = true;
        }
    }

    if is_1scf {
        is_opt_requested = false;
    }

    Ok(ParsedInput {
        keywords,
        title,
        comment,
        atoms,
        is_opt_requested,
        is_ts_requested,
        is_irc_requested,
        is_drc_requested,
        is_force_requested,
        is_gpu_requested: is_gpu,
        is_fp32_requested: is_fp32,
        method,
        is_nddo_requested,
        is_dipole_requested,
        is_bonds_requested,
        is_mullik_requested,
        eps,
        dispersion,
        use_h4,
        use_hh,
    })
}

fn build_batch(atoms: &[ParsedAtom]) -> MolecularBatch {
    let natoms = atoms.len();
    let mut atomic_numbers = Vec::with_capacity(natoms);
    let mut coords = Vec::with_capacity(natoms);

    for a in atoms {
        atomic_numbers.push(a.z);
        coords.push([a.x, a.y, a.z_coord]);
    }

    MolecularBatch::new(atomic_numbers, &coords)
}

fn compute_wavefunction_properties(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    ws: &ScfWorkspace,
    scf: &ScfResult,
    disp_model: Option<DispersionModel>,
    use_h4: bool,
    use_hh: bool,
) -> (
    f64,
    Vec<f64>,
    Vec<[f64; 3]>,
    DipoleResult,
    NonCovalentBreakdown,
) {
    let mut sum_eisol = 0.0;
    let mut sum_eheat = 0.0;
    for &z in &batch.atomic_numbers {
        let (eisol, eheat) = get_isolated_atom_energy_and_heat(z, model);
        sum_eisol += eisol;
        sum_eheat += eheat;
    }

    let mut non_cov = NonCovalentBreakdown::default();
    if let Some(dm) = disp_model {
        non_cov.dispersion_kcal = compute_dispersion_energy(batch, dm);
    }
    if use_h4 {
        let params = H4Parameters::default();
        non_cov.h4_kcal = compute_h4_energy(batch, &params);
    }
    if use_hh {
        let (e_hh, _) = compute_hh_repulsion_energy_and_gradients(batch);
        non_cov.hh_repulsion_kcal = e_hh;
    }
    let non_cov_total = non_cov.dispersion_kcal + non_cov.h4_kcal + non_cov.hh_repulsion_kcal;

    let binding_energy_ev = scf.total_energy_ev - sum_eisol;
    let heat_of_formation_kcal = binding_energy_ev * EV_TO_KCAL_MOL + sum_eheat + non_cov_total;

    let dipole = compute_dipole_moment(batch, model, &ws.density);
    let charges = dipole.atomic_charges.clone();

    let mut populations = Vec::with_capacity(batch.natoms);
    for i in 0..batch.natoms {
        let orb_start = batch.orbital_offsets[i];
        let norbs = batch.basis_types[i].num_orbitals();

        let s_pop = ws.density.get(orb_start, orb_start);
        let mut p_pop = 0.0;
        for o in 1..norbs {
            p_pop += ws.density.get(orb_start + o, orb_start + o);
        }
        populations.push([s_pop, p_pop, s_pop + p_pop]);
    }

    (
        heat_of_formation_kcal,
        charges,
        populations,
        dipole,
        non_cov,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let start_time = Instant::now();

    if let Some(nthr) = cli.threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(nthr)
            .build_global();
    }

    let input_path = &cli.input;
    let content = fs::read_to_string(input_path)?;
    let parsed = parse_mopac_input(&content)?;

    let input_stem = input_path.file_stem().unwrap().to_str().unwrap();
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new("."));

    let out_file = cli
        .output
        .unwrap_or_else(|| parent_dir.join(format!("{}.out", input_stem)));
    let arc_file = cli
        .arc
        .unwrap_or_else(|| parent_dir.join(format!("{}.arc", input_stem)));

    let use_gpu = cli.gpu || parsed.is_gpu_requested;
    let use_fp32 = cli.fp32 || parsed.is_fp32_requested;
    let is_irc = (cli.irc || parsed.is_irc_requested) && !cli.one_scf;
    let is_drc = (cli.drc || parsed.is_drc_requested) && !cli.one_scf && !is_irc;
    let is_ts = (cli.ts || parsed.is_ts_requested) && !cli.one_scf && !is_irc && !is_drc;
    let is_opt =
        (cli.opt || parsed.is_opt_requested) && !cli.one_scf && !is_ts && !is_irc && !is_drc;

    let method_name = cli
        .method
        .or(parsed.method)
        .unwrap_or_else(|| "AM1".to_string())
        .to_uppercase();

    let model: Box<dyn ParameterModel> = match method_name.as_str() {
        "PM7" => Box::new(Pm7Model),
        "PM6" => Box::new(Pm6Model),
        "PM3" => Box::new(Pm3Model),
        "RM1" => Box::new(Rm1Model),
        "MNDO" => Box::new(MndoModel),
        _ => Box::new(Am1Model),
    };

    let use_nddo = cli.nddo || parsed.is_nddo_requested;

    println!("===============================================================================");
    println!("                          MOPAC_RS CANONICAL QUANTUM ENGINE                     ");
    println!("                                Version 0.1.0-alpha                             ");
    println!("===============================================================================");
    println!(" Job Input File        : {}", input_path.display());
    println!(" Method / Hamiltonian   : {}", model.name());
    println!(
        " NDDO Multipoles       : {}",
        if use_nddo {
            "Enabled (Full 22 Multipoles)"
        } else {
            "Monopole Approximation"
        }
    );
    println!(
        " Calculation Mode      : {}",
        if is_irc {
            "Intrinsic Reaction Coordinate (IRC) Path Tracing"
        } else if is_drc {
            "Dynamic Reaction Coordinate (DRC) Molecular Dynamics"
        } else if is_ts {
            "Eigenvector Following (P-RFO) Transition State Search"
        } else if is_opt {
            "L-BFGS Geometry Optimization"
        } else {
            "1SCF (Single Point)"
        }
    );
    println!(
        " Compute Backend       : {}",
        if use_gpu {
            format!(
                "Vulkan GPU ({})",
                if use_fp32 { "FP32 (18 TFLOPS)" } else { "FP64" }
            )
        } else {
            "CPU SIMD AVX2".to_string()
        }
    );
    println!(" Number of Atoms       : {}", parsed.atoms.len());
    println!(" Title Line            : \"{}\"", parsed.title);
    println!("-------------------------------------------------------------------------------");

    let mut batch = build_batch(&parsed.atoms);
    let mut ws = ScfWorkspace::allocate(batch.norbs);

    // If GPU is requested, warm up and verify device acceleration
    if use_gpu {
        println!(" [Vulkan GPU] Initializing PCIe DMA buffers & GDDR6 device memory...");
        let ctx = match VulkanContext::new() {
            Ok(c) => Arc::new(c),
            Err(e) => {
                eprintln!(" [Vulkan GPU Warning] Failed to initialize Vulkan device: {}. Falling back to CPU.", e);
                Arc::new(VulkanContext::new()?)
            }
        };

        println!(
            " [Vulkan GPU] Device: {} (Discrete: {})",
            ctx.device_info.device_name, ctx.device_info.is_discrete
        );
        if use_fp32 {
            let gpu_calc = GpuCoulombCalculatorFP32::new(Arc::clone(&ctx))?;
            let _gpu_matrix = gpu_calc.compute_batch(&batch, model.as_ref())?;
            println!(
                " [Vulkan GPU] Evaluated pairwise Coulomb matrix using FP32 hardware pipeline."
            );
        } else {
            let gpu_calc = GpuCoulombCalculator::new(Arc::clone(&ctx))?;
            let _gpu_matrix = gpu_calc.compute_batch(&batch, model.as_ref())?;
            println!(
                " [Vulkan GPU] Evaluated pairwise Coulomb matrix using native Float64 pipeline."
            );
        }
    }

    let (scf_final, total_scf_cycles) = if is_irc {
        println!(
            " [IRC] Starting Intrinsic Reaction Coordinate Path Tracing (González-Schlegel)..."
        );
        let irc_opts = IrcOptions {
            step_size: 0.1,
            max_points: 50,
            corrector_max_iter: 25,
            corrector_tol: 1e-4,
            grad_rms_tol: 0.05,
            energy_increase_tol: 0.02,
            direction: IrcDirection::Both,
            use_nddo,
            transition_vector: None,
        };
        let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
        let mut irc_ws = IrcWorkspace::allocate(&batch);
        let irc_res = trace_intrinsic_reaction_coordinate(
            &mut batch,
            model.as_ref(),
            &mut ws,
            &mut grad_ws,
            &mut irc_ws,
            &irc_opts,
        );
        println!(
            " [IRC] Traced {} points along reaction coordinate.",
            irc_res.points.len()
        );
        println!("   TS Point Index : {} (s = 0.000)", irc_res.ts_point_index);
        if let Some(ts_pt) = irc_res.points.get(irc_res.ts_point_index) {
            println!(
                "   TS Energy      : {:12.6} eV | HoF: {:12.5} kcal/mol",
                ts_pt.energy_ev, ts_pt.heat_of_formation_kcal
            );
        }
        if let Some(first_pt) = irc_res.points.first() {
            println!(
                "   Reverse Extr   : s = {:+.4} | {:12.6} eV | HoF: {:12.5} kcal/mol",
                first_pt.path_coordinate, first_pt.energy_ev, first_pt.heat_of_formation_kcal
            );
        }
        if let Some(last_pt) = irc_res.points.last() {
            println!(
                "   Forward Extr   : s = {:+.4} | {:12.6} eV | HoF: {:12.5} kcal/mol",
                last_pt.path_coordinate, last_pt.energy_ev, last_pt.heat_of_formation_kcal
            );
        }
        let final_scf = run_rhf_scf_adaptive_with_nddo(
            &batch,
            model.as_ref(),
            &mut ws,
            60,
            1e-7,
            1e-6,
            use_nddo,
        );
        let niter = final_scf.iterations;
        (final_scf, niter)
    } else if is_drc {
        println!(
            " [DRC] Starting Dynamic Reaction Coordinate Simulation (Velocity-Verlet BOMD)..."
        );
        let drc_opts = DrcOptions {
            time_step_fs: 0.5,
            total_steps: 500,
            ensemble: DrcEnsemble::Nve,
            target_temperature_k: 298.15,
            berendsen_tau_fs: 100.0,
            recording_interval: 10,
            initial_velocities: InitialVelocities::Zero,
            use_nddo,
            scf_energy_tol: 1e-8,
            scf_density_tol: 1e-7,
        };
        let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
        let mut drc_ws = DrcWorkspace::allocate(&batch);
        let drc_res = run_dynamic_reaction_coordinate(
            &mut batch,
            model.as_ref(),
            &mut ws,
            &mut grad_ws,
            &mut drc_ws,
            &drc_opts,
        );
        println!(
            " [DRC] Trajectory complete: {} steps ({} frames recorded)",
            drc_opts.total_steps,
            drc_res.frames.len()
        );
        println!(
            "   Initial Energy : {:12.6} eV | Final Energy : {:12.6} eV",
            drc_res.initial_energy_ev, drc_res.final_energy_ev
        );
        println!(
            "   Energy Drift   : {:12.6e} eV/ps (Max Dev: {:12.6e} eV)",
            drc_res.energy_drift_ev_per_ps, drc_res.max_energy_drift_ev
        );
        println!(
            "   Avg Temperature: {:8.2} K",
            drc_res.average_temperature_k
        );
        let final_scf = run_rhf_scf_adaptive_with_nddo(
            &batch,
            model.as_ref(),
            &mut ws,
            60,
            1e-7,
            1e-6,
            use_nddo,
        );
        let niter = final_scf.iterations;
        (final_scf, niter)
    } else if is_ts {
        println!(" [Optimizer] Starting Transition State Search (Eigenvector Following P-RFO)...");
        let mut opt_mask = Vec::with_capacity(3 * parsed.atoms.len());
        for a in &parsed.atoms {
            opt_mask.push(a.opt_x);
            opt_mask.push(a.opt_y);
            opt_mask.push(a.opt_z);
        }

        let opts = TransitionStateOptions {
            max_cycles: 100,
            grad_rms_tol: 0.1,
            grad_max_tol: 0.2,
            trust_radius: 0.1,
            min_trust_radius: 0.005,
            max_trust_radius: 0.3,
            update_scheme: HessianUpdateScheme::Bofill,
            mode_following: true,
            target_mode: None,
            opt_mask: Some(opt_mask),
            use_nddo,
            hessian_delta: 0.005,
            initial_hessian: None,
        };

        let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
        let mut ef_ws = EigenvectorFollowingWorkspace::allocate(batch.natoms);
        let ts_res = optimize_transition_state(
            &mut batch,
            model.as_ref(),
            &mut ws,
            &mut grad_ws,
            &mut ef_ws,
            &opts,
        );

        println!(
            " [Optimizer] TS Search finished in {} cycles (Converged: {})",
            ts_res.cycles, ts_res.converged
        );
        println!(
            "   Transition Mode Eigenvalue : {:12.4} kcal/(mol*A^2) (Mode #{})",
            ts_res.ts_mode_eigenvalue,
            ts_res.ts_mode_index + 1
        );
        println!(
            "   Final Electronic Energy    : {:12.6} eV",
            ts_res.final_energy_ev
        );
        println!(
            "   Standard Heat of Formation : {:12.5} kcal/mol",
            ts_res.heat_of_formation_kcal
        );
        println!(
            "   Initial RMS G: {:10.4} | Final RMS G: {:10.4} kcal/(mol*A)",
            ts_res.initial_grad_rms, ts_res.final_grad_rms
        );
        let niter = ts_res.final_scf.iterations;
        (ts_res.final_scf, niter)
    } else if is_opt {
        println!(" [Optimizer] Starting Cartesian L-BFGS Relaxation...");
        let mut opt_mask = Vec::with_capacity(3 * parsed.atoms.len());
        for a in &parsed.atoms {
            opt_mask.push(a.opt_x);
            opt_mask.push(a.opt_y);
            opt_mask.push(a.opt_z);
        }

        let opts = OptimizationOptions {
            max_cycles: 100,
            grad_rms_tol: 0.5,
            grad_max_tol: 1.0,
            energy_tol_ev: 1e-6,
            max_step_size: 0.1,
            history_capacity: 6,
            use_nddo,
            opt_mask: Some(opt_mask),
        };

        let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
        let opt_res =
            optimize_geometry_lbfgs(&mut batch, model.as_ref(), &mut ws, &mut grad_ws, &opts);

        println!(
            " [Optimizer] Optimization finished in {} cycles (Converged: {})",
            opt_res.cycles, opt_res.converged
        );
        println!(
            "   Initial Energy: {:12.6} eV | Final Energy: {:12.6} eV",
            opt_res.initial_energy_ev, opt_res.final_energy_ev
        );
        println!(
            "   Initial RMS G : {:12.4} kcal/(mol*A) | Final RMS G: {:12.4} kcal/(mol*A)",
            opt_res.initial_grad_rms, opt_res.final_grad_rms
        );
        let niter = opt_res.final_scf.iterations;
        (opt_res.final_scf, niter)
    } else {
        println!(
            " [SCF] Running Roothaan-Hall Self-Consistent Field (NDDO: {})...",
            use_nddo
        );
        let cosmo_params = cli.eps.or(parsed.eps).map(|epsilon| CosmoParams {
            epsilon,
            rsolv: 1.30005,
        });
        if let Some(cp) = cosmo_params {
            println!(
                " [COSMO] Implicit solvation active: EPS = {:.2}, RSOLV = {:.5} A",
                cp.epsilon, cp.rsolv
            );
        }
        let res = run_rhf_scf_adaptive_with_nddo_and_cosmo(
            &batch,
            model.as_ref(),
            &mut ws,
            60,
            1e-7,
            1e-6,
            use_nddo,
            cosmo_params,
        );
        let niter = res.iterations;
        (res, niter)
    };

    let cosmo_params = cli.eps.or(parsed.eps).map(|epsilon| CosmoParams {
        epsilon,
        rsolv: 1.30005,
    });

    let is_force = cli.force || parsed.is_force_requested;
    let force_result = if is_force {
        println!(" [Vibrations] Computing Cartesian Hessian, mass-weighting & normal modes...");
        let scf_opts = ScfOptions {
            max_iter: 50,
            energy_tol_ev: 1e-8,
            density_tol: 1e-7,
            level_shift_ev: 0.0,
            damping: 0.5,
            use_nddo,
            cosmo: cosmo_params,
        };

        let hess_opts = HessianOptions {
            delta: 1.0e-3,
            recompute_scf: true,
            use_nddo,
            project_external: true,
            temperature_k: 298.15,
            pressure_atm: 1.0,
            rotational_symmetry_number: 1.0,
        };
        let h_res = compute_hessian_and_frequencies(
            &mut batch,
            model.as_ref(),
            &mut ws,
            &scf_opts,
            &hess_opts,
        );
        println!(
            " [Vibrations] Done: {} vibrational modes, ZPVE = {:.3} kcal/mol",
            h_res.vibrational_frequencies_cm1.len(),
            h_res.zpve_kcal_mol
        );
        Some(h_res)
    } else {
        None
    };

    let elapsed = start_time.elapsed();
    let disp_model = match cli.disp.as_deref() {
        Some("pm6-dh+") | Some("pm6-dh2") | Some("dh+") => Some(DispersionModel::Pm6DhPlus),
        Some("pm7") => Some(DispersionModel::Pm7),
        _ => parsed.dispersion,
    };
    let use_h4 = cli.d3h4 || parsed.use_h4;
    let use_hh = cli.d3h4 || parsed.use_hh;

    let (hof_kcal, charges, pops, dipole, non_cov) = compute_wavefunction_properties(
        &batch,
        model.as_ref(),
        &ws,
        &scf_final,
        disp_model,
        use_h4,
        use_hh,
    );

    println!("-------------------------------------------------------------------------------");
    println!("                             FINAL SCF RESULTS                                 ");
    println!("-------------------------------------------------------------------------------");
    println!(
        " Final Heat of Formation : {:15.5} kcal/mol ({:12.5} kJ/mol)",
        hof_kcal,
        hof_kcal * 4.184
    );
    if non_cov.dispersion_kcal.abs() > 1e-6 {
        println!(
            " Dispersion Energy (D3)  : {:15.5} kcal/mol",
            non_cov.dispersion_kcal
        );
    }
    if non_cov.h4_kcal.abs() > 1e-6 {
        println!(
            " H4 Hydrogen Bond Energy : {:15.5} kcal/mol",
            non_cov.h4_kcal
        );
    }
    if non_cov.hh_repulsion_kcal.abs() > 1e-6 {
        println!(
            " H-H Repulsion Energy    : {:15.5} kcal/mol",
            non_cov.hh_repulsion_kcal
        );
    }
    println!(
        " Total SCF Energy        : {:15.6} eV",
        scf_final.total_energy_ev
    );
    println!(
        " Electronic Energy       : {:15.6} eV",
        scf_final.electronic_energy_ev
    );
    println!(
        " Nuclear Repulsion       : {:15.6} eV",
        scf_final.nuclear_repulsion_ev
    );
    if let Some(diel_ev) = scf_final.dielectric_energy_ev {
        println!(
            " Dielectric Solv Energy  : {:15.6} eV ({:12.5} kcal/mol)",
            diel_ev,
            diel_ev * 23.06054801
        );
    }
    if let Some(cp) = cosmo_params {
        let cav = CosmoCavity::construct(&batch, cp.rsolv);
        println!(
            " COSMO Cavity Area       : {:15.2} Square Angstroms",
            cav.total_area_angstrom2
        );
        println!(
            " COSMO Cavity Volume     : {:15.2} Cubic Angstroms",
            cav.total_volume_angstrom3
        );
    }
    println!(
        " HOMO Energy (IP)        : {:15.4} eV",
        scf_final.homo_energy_ev
    );
    println!(
        " LUMO Energy             : {:15.4} eV",
        scf_final.lumo_energy_ev
    );
    println!(
        " HOMO-LUMO Gap           : {:15.4} eV",
        scf_final.lumo_energy_ev - scf_final.homo_energy_ev
    );
    println!(" Total Dipole Moment     : {:15.4} Debye", dipole.total[3]);
    println!(
        "   Point-Charge Dipole   : {:15.4} Debye",
        dipole.point_charge[3]
    );
    println!(
        "   Hybridization Dipole  : {:15.4} Debye",
        dipole.hybridization[3]
    );
    println!(" SCF Iterations Total    : {}", total_scf_cycles);
    println!(
        " Total Wall-Clock Time   : {:.4} seconds",
        elapsed.as_secs_f64()
    );

    if let Some(ref h_res) = force_result {
        println!("-------------------------------------------------------------------------------");
        println!("                   NORMAL MODES & VIBRATIONAL FREQUENCIES                      ");
        println!("-------------------------------------------------------------------------------");
        println!("  MODE        FREQUENCY (CM^-1)      FORCE CONST (MDYNE/A)     REDUCED MASS");
        for (i, &nu) in h_res.vibrational_frequencies_cm1.iter().enumerate() {
            let mode_idx =
                h_res.all_frequencies_cm1.len() - h_res.vibrational_frequencies_cm1.len() + i;
            let mode = &h_res.normal_modes[mode_idx];
            println!(
                "   {:3}             {:10.2}                 {:8.4}              {:8.4} amu",
                i + 1,
                nu,
                mode.force_constant_mdyne_a,
                mode.reduced_mass_amu
            );
        }
        println!(
            " Zero-Point Vibrational Energy : {:12.3} kcal/mol",
            h_res.zpve_kcal_mol
        );
        println!();
        println!(
            " CALCULATED THERMODYNAMIC PROPERTIES (T = {:.2} K, P = {:.2} atm):",
            h_res.thermo.temperature_k, h_res.thermo.pressure_atm
        );
        println!(
            "   Enthalpy (Thermal)          : {:12.4} cal/mol",
            h_res.thermo.enthalpy_thermal_cal_mol
        );
        println!(
            "   Heat Capacity (Cp)          : {:12.4} cal/(mol K)",
            h_res.thermo.cp_total_cal_k_mol
        );
        println!(
            "   Standard Entropy (S°)       : {:12.4} cal/(mol K)",
            h_res.thermo.entropy_total_cal_k_mol
        );
        println!(
            "   Gibbs Free Energy Corr.     : {:12.4} kcal/mol",
            h_res.thermo.gibbs_correction_kcal_mol
        );
    }

    let is_bonds = cli.bonds || parsed.is_bonds_requested;
    let bond_result = if is_bonds {
        Some(compute_bond_orders(&batch, &ws.density))
    } else {
        None
    };

    if let Some(ref b_res) = bond_result {
        println!("-------------------------------------------------------------------------------");
        println!("                         (VALENCIES)   BOND ORDERS                             ");
        println!("-------------------------------------------------------------------------------");
        for i in 0..batch.natoms {
            let sym_i = atomic_number_to_symbol(batch.atomic_numbers[i]);
            let mut line = format!(
                "   {:3}  {:2}     ({:6.3})",
                i + 1,
                sym_i,
                b_res.valencies[i]
            );
            for j in 0..batch.natoms {
                if i != j {
                    let b_val = b_res.bond_orders.get(i, j);
                    if b_val > 0.01 {
                        let sym_j = atomic_number_to_symbol(batch.atomic_numbers[j]);
                        line.push_str(&format!("     {:3}  {:2} {:5.3}", j + 1, sym_j, b_val));
                    }
                }
            }
            println!("{}", line);
        }
    }

    let is_mullik = cli.mullik || parsed.is_mullik_requested;
    let mullik_result = if is_mullik {
        let n_electrons: usize = batch
            .atomic_numbers
            .iter()
            .map(|&z| model.get_element(z).unwrap().core_charge as usize)
            .sum();
        let num_occupied = n_electrons / 2;
        Some(compute_mulliken_population(
            &batch,
            model.as_ref(),
            &ws.eigenvectors,
            num_occupied,
        ))
    } else {
        None
    };

    if let Some(ref m_res) = mullik_result {
        println!("-------------------------------------------------------------------------------");
        println!("                        MULLIKEN POPULATION ANALYSIS                           ");
        println!("-------------------------------------------------------------------------------");
        println!("      NO.  ATOM   POPULATION      CHARGE");
        for i in 0..batch.natoms {
            let sym = atomic_number_to_symbol(batch.atomic_numbers[i]);
            println!(
                "    {:4}    {:2}     {:10.6}     {:10.6}",
                i + 1,
                sym,
                m_res.atomic_populations[i],
                m_res.net_charges[i]
            );
        }
    }
    println!("===============================================================================");

    // Write .out file matching standard MOPAC format
    let mut out = File::create(&out_file)?;
    writeln!(
        out,
        " *******************************************************************************"
    )?;
    writeln!(
        out,
        " **                                                                           **"
    )?;
    writeln!(
        out,
        " **                              MOPAC_RS v0.1.0                              **"
    )?;
    writeln!(
        out,
        " **                Canonical Semi-Empirical Quantum Chemistry Engine          **"
    )?;
    writeln!(
        out,
        " **                                                                           **"
    )?;
    writeln!(
        out,
        " *******************************************************************************"
    )?;
    writeln!(out)?;
    writeln!(out, " KEYWORDS: {}", parsed.keywords.join(" "))?;
    writeln!(out, " TITLE:    {}", parsed.title)?;
    writeln!(out, " COMMENT:  {}", parsed.comment)?;
    writeln!(out)?;
    writeln!(out, " CALCULATION PARAMETERS:")?;
    writeln!(out, "   Method:    {}", model.name())?;
    writeln!(
        out,
        "   NDDO:      {}",
        if use_nddo {
            "Enabled (Full 22 Multipoles)"
        } else {
            "Monopole Approximation"
        }
    )?;
    writeln!(
        out,
        "   Mode:      {}",
        if is_opt {
            "L-BFGS Geometry Optimization"
        } else {
            "1SCF"
        }
    )?;
    writeln!(
        out,
        "   Backend:   {}",
        if use_gpu {
            "Vulkan GPU"
        } else {
            "CPU SIMD AVX2"
        }
    )?;
    writeln!(out)?;
    writeln!(
        out,
        " FINAL HEAT OF FORMATION = {:17.5} KCAL/MOL = {:14.5} KJ/MOL",
        hof_kcal,
        hof_kcal * 4.184
    )?;
    if non_cov.dispersion_kcal.abs() > 1e-6 {
        writeln!(
            out,
            " DISPERSION ENERGY       = {:17.5} KCAL/MOL",
            non_cov.dispersion_kcal
        )?;
    }
    if non_cov.h4_kcal.abs() > 1e-6 {
        writeln!(
            out,
            " H4 HYDROGEN BOND ENERGY = {:17.5} KCAL/MOL",
            non_cov.h4_kcal
        )?;
    }
    if non_cov.hh_repulsion_kcal.abs() > 1e-6 {
        writeln!(
            out,
            " H-H REPULSION ENERGY    = {:17.5} KCAL/MOL",
            non_cov.hh_repulsion_kcal
        )?;
    }
    writeln!(
        out,
        " TOTAL ENERGY            = {:17.6} EV",
        scf_final.total_energy_ev
    )?;
    writeln!(
        out,
        " ELECTRONIC ENERGY       = {:17.6} EV",
        scf_final.electronic_energy_ev
    )?;
    writeln!(
        out,
        " NUCLEAR REPULSION       = {:17.6} EV",
        scf_final.nuclear_repulsion_ev
    )?;
    if let Some(diel_ev) = scf_final.dielectric_energy_ev {
        writeln!(out, " DIELECTRIC ENERGY       = {:17.5} EV", diel_ev)?;
    }
    if let Some(cp) = cosmo_params {
        let cav = CosmoCavity::construct(&batch, cp.rsolv);
        writeln!(
            out,
            " COSMO AREA              = {:17.2} SQUARE ANGSTROMS",
            cav.total_area_angstrom2
        )?;
        writeln!(
            out,
            " COSMO VOLUME            = {:17.2} CUBIC ANGSTROMS",
            cav.total_volume_angstrom3
        )?;
    }
    writeln!(
        out,
        " IONIZATION POTENTIAL    = {:17.5} EV",
        -scf_final.homo_energy_ev
    )?;
    writeln!(
        out,
        " HOMO LUMO ENERGIES (EV) = {:12.4} {:12.4}",
        scf_final.homo_energy_ev, scf_final.lumo_energy_ev
    )?;
    writeln!(
        out,
        " DIPOLE MOMENT           = {:17.4} DEBYE",
        dipole.total[3]
    )?;
    writeln!(
        out,
        " WALL-CLOCK TIME         = {:17.4} SECONDS",
        elapsed.as_secs_f64()
    )?;
    writeln!(out)?;
    writeln!(out, " DIPOLE           X         Y         Z       TOTAL")?;
    writeln!(
        out,
        " POINT-CHG.   {:9.3} {:9.3} {:9.3} {:10.3}",
        dipole.point_charge[0],
        dipole.point_charge[1],
        dipole.point_charge[2],
        dipole.point_charge[3]
    )?;
    writeln!(
        out,
        " HYBRID       {:9.3} {:9.3} {:9.3} {:10.3}",
        dipole.hybridization[0],
        dipole.hybridization[1],
        dipole.hybridization[2],
        dipole.hybridization[3]
    )?;
    writeln!(
        out,
        " SUM          {:9.3} {:9.3} {:9.3} {:10.3}",
        dipole.total[0], dipole.total[1], dipole.total[2], dipole.total[3]
    )?;
    writeln!(out)?;
    writeln!(
        out,
        "              NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS"
    )?;
    writeln!(
        out,
        "  ATOM NO.   TYPE          CHARGE      No. of ELECS.   s-Pop       p-Pop"
    )?;
    for i in 0..batch.natoms {
        let sym = atomic_number_to_symbol(batch.atomic_numbers[i]);
        writeln!(
            out,
            "   {:4}       {:2}         {:10.6}        {:8.4}     {:8.4}    {:8.4}",
            i + 1,
            sym,
            charges[i],
            pops[i][2],
            pops[i][0],
            pops[i][1]
        )?;
    }
    writeln!(out)?;
    writeln!(out, "                             CARTESIAN COORDINATES")?;
    for i in 0..batch.natoms {
        let sym = atomic_number_to_symbol(batch.atomic_numbers[i]);
        let (x, y, z) = (batch.x[i], batch.y[i], batch.z[i]);
        writeln!(
            out,
            "  {:4}    {:2}       {:16.9}  {:16.9}  {:16.9}",
            i + 1,
            sym,
            x,
            y,
            z
        )?;
    }
    writeln!(out)?;

    if let Some(ref b_res) = bond_result {
        writeln!(out, "            (VALENCIES)   BOND ORDERS")?;
        writeln!(out)?;
        for i in 0..batch.natoms {
            let sym_i = atomic_number_to_symbol(batch.atomic_numbers[i]);
            let mut line = format!(
                "   {:3}  {:2}     ({:6.3})",
                i + 1,
                sym_i,
                b_res.valencies[i]
            );
            for j in 0..batch.natoms {
                if i != j {
                    let b_val = b_res.bond_orders.get(i, j);
                    if b_val > 0.01 {
                        let sym_j = atomic_number_to_symbol(batch.atomic_numbers[j]);
                        line.push_str(&format!("     {:3}  {:2} {:5.3}", j + 1, sym_j, b_val));
                    }
                }
            }
            writeln!(out, "{}", line)?;
        }
        writeln!(out)?;
    }

    if let Some(ref m_res) = mullik_result {
        writeln!(out, "           MULLIKEN POPULATION ANALYSIS")?;
        writeln!(out)?;
        writeln!(out, "        MULLIKEN POPULATIONS AND CHARGES")?;
        writeln!(out)?;
        writeln!(out, "      NO.  ATOM   POPULATION      CHARGE")?;
        for i in 0..batch.natoms {
            let sym = atomic_number_to_symbol(batch.atomic_numbers[i]);
            writeln!(
                out,
                "    {:4}    {:2}     {:10.6}     {:10.6}",
                i + 1,
                sym,
                m_res.atomic_populations[i],
                m_res.net_charges[i]
            )?;
        }
        writeln!(out)?;
    }

    if let Some(ref h_res) = force_result {
        writeln!(
            out,
            "           NORMAL COORDINATE ANALYSIS & VIBRATIONAL FREQUENCIES"
        )?;
        writeln!(
            out,
            "  ROOT NO.     FREQUENCY (CM^-1)      FORCE CONST (MDYNE/A)     REDUCED MASS"
        )?;
        for (i, &nu) in h_res.vibrational_frequencies_cm1.iter().enumerate() {
            let mode_idx =
                h_res.all_frequencies_cm1.len() - h_res.vibrational_frequencies_cm1.len() + i;
            let mode = &h_res.normal_modes[mode_idx];
            writeln!(
                out,
                "   {:3}             {:10.2}                 {:8.4}              {:8.4} amu",
                i + 1,
                nu,
                mode.force_constant_mdyne_a,
                mode.reduced_mass_amu
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            " ZERO POINT VIBRATIONAL ENERGY = {:12.3} KCAL/MOL",
            h_res.zpve_kcal_mol
        )?;
        writeln!(out)?;
        writeln!(
            out,
            " CALCULATED THERMODYNAMIC PROPERTIES (T = {:.2} K, P = {:.2} ATM):",
            h_res.thermo.temperature_k, h_res.thermo.pressure_atm
        )?;
        writeln!(
            out,
            "   ENTHALPY (THERMAL)          = {:12.4} CAL/MOL",
            h_res.thermo.enthalpy_thermal_cal_mol
        )?;
        writeln!(
            out,
            "   HEAT CAPACITY (CP)          = {:12.4} CAL/(MOL K)",
            h_res.thermo.cp_total_cal_k_mol
        )?;
        writeln!(
            out,
            "   STANDARD ENTROPY (S°)       = {:12.4} CAL/(MOL K)",
            h_res.thermo.entropy_total_cal_k_mol
        )?;
        writeln!(
            out,
            "   GIBBS FREE ENERGY CORR.     = {:12.4} KCAL/MOL",
            h_res.thermo.gibbs_correction_kcal_mol
        )?;
        writeln!(out)?;
    }

    writeln!(out, " == MOPAC_RS DONE ==")?;

    // Write .arc file with optimized geometry
    let mut arc = File::create(&arc_file)?;
    writeln!(
        arc,
        "{} {}",
        model.name(),
        if is_opt { "OPT" } else { "1SCF" }
    )?;
    writeln!(arc, "{}", parsed.title)?;
    writeln!(arc, "Final Heat of Formation: {:12.5} kcal/mol", hof_kcal)?;
    for i in 0..batch.natoms {
        let sym = atomic_number_to_symbol(batch.atomic_numbers[i]);
        let (x, y, z) = (batch.x[i], batch.y[i], batch.z[i]);
        writeln!(arc, " {:2}   {:14.8} 1  {:14.8} 1  {:14.8} 1", sym, x, y, z)?;
    }

    println!(" Output files written to:");
    println!("   .out Report : {}", out_file.display());
    println!("   .arc Archive: {}", arc_file.display());
    println!(" Done.");

    Ok(())
}
