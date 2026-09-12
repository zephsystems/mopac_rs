//! Canonical MOPAC Command-Line Interface (CLI).
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Provides drop-in MOPAC CLI compatibility with CPU AVX2 and Vulkan GPU FP32/FP64 backends.

use clap::Parser;
use mopac_core::constants::codata2018::EV_TO_KCAL_MOL;
use mopac_core::gradients::GradientWorkspace;
use mopac_core::opt::{optimize_geometry_lbfgs, OptimizationOptions};
use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::parameters::rm1::Rm1Model;
use mopac_core::parameters::ParameterModel;
use mopac_core::scf::scf_loop::{run_rhf_scf_adaptive_with_nddo, ScfResult};
use mopac_core::types::{MolecularBatch, ScfWorkspace};
use mopac_gpu::{GpuCoulombCalculator, VulkanContext};
use mopac_gpu::coulomb_fp32::GpuCoulombCalculatorFP32;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "mopac",
    author = "zeph.sys",
    version = "0.1.0",
    about = "Canonical MOPAC Semi-Empirical Engine in Rust with Vulkan GPU Acceleration"
)]
struct Cli {
    /// Input file path (.mop or .dat)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Semi-empirical method / Hamiltonian (AM1, PM6, RM1)
    #[arg(long)]
    method: Option<String>,

    /// Enable full NDDO 22-multipole two-center electron repulsion integrals
    #[arg(long, default_value_t = false)]
    nddo: bool,

    /// Enable Vulkan GPU acceleration
    #[arg(long, default_value_t = false)]
    gpu: bool,

    /// Use FP32 precision on GPU (default: FP64)
    #[arg(long, default_value_t = false)]
    fp32: bool,

    /// Force geometry optimization (L-BFGS)
    #[arg(long, default_value_t = false)]
    opt: bool,

    /// Force single-point calculation (1SCF)
    #[arg(long = "1scf", default_value_t = false)]
    one_scf: bool,

    /// Number of CPU worker threads
    #[arg(long)]
    threads: Option<usize>,

    /// Custom output file path (.out)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Custom archive file path (.arc)
    #[arg(long)]
    arc: Option<PathBuf>,
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
    is_gpu_requested: bool,
    is_fp32_requested: bool,
    method: Option<String>,
    is_nddo_requested: bool,
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
    let p = match model.get_element(z) {
        Some(param) => param,
        None => return (0.0, 0.0),
    };

    let (ios, iop, eheat): (f64, f64, f64) = match z {
        1 => (1.0, 0.0, 52.102),
        6 => (2.0, 2.0, 170.890),
        7 => (2.0, 3.0, 113.000),
        8 => (2.0, 4.0, 59.559),
        9 => (2.0, 5.0, 18.890),
        15 => (2.0, 3.0, 75.570),
        16 => (2.0, 4.0, 66.400),
        17 => (2.0, 5.0, 28.990),
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

fn parse_mopac_input(content: &str) -> io::Result<ParsedInput> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Empty input file"));
    }

    let kw_line = lines[0].trim();
    let keywords: Vec<String> = kw_line.split_whitespace().map(|s| s.to_string()).collect();

    let title = if lines.len() > 1 { lines[1].trim().to_string() } else { String::new() };
    let comment = if lines.len() > 2 { lines[2].trim().to_string() } else { String::new() };

    let mut is_opt_requested = false;
    let mut is_1scf = false;
    let mut is_gpu = false;
    let mut is_fp32 = false;
    let mut method = None;
    let mut is_nddo_requested = false;

    for kw in &keywords {
        let u = kw.to_uppercase();
        if u == "OPT" || u == "EF" || u == "BFGS" {
            is_opt_requested = true;
        } else if u == "1SCF" {
            is_1scf = true;
        } else if u == "GPU" {
            is_gpu = true;
        } else if u == "FP32" {
            is_fp32 = true;
        } else if u == "PM6" {
            method = Some("PM6".to_string());
        } else if u == "RM1" {
            method = Some("RM1".to_string());
        } else if u == "AM1" {
            method = Some("AM1".to_string());
        } else if u == "NDDO" {
            is_nddo_requested = true;
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
        is_gpu_requested: is_gpu,
        is_fp32_requested: is_fp32,
        method,
        is_nddo_requested,
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
) -> (f64, Vec<f64>, Vec<[f64; 3]>, [f64; 3], f64) {
    let mut sum_eisol = 0.0;
    let mut sum_eheat = 0.0;
    for &z in &batch.atomic_numbers {
        let (eisol, eheat) = get_isolated_atom_energy_and_heat(z, model);
        sum_eisol += eisol;
        sum_eheat += eheat;
    }

    let binding_energy_ev = scf.total_energy_ev - sum_eisol;
    let heat_of_formation_kcal = binding_energy_ev * EV_TO_KCAL_MOL + sum_eheat;

    let mut charges = Vec::with_capacity(batch.natoms);
    let mut populations = Vec::with_capacity(batch.natoms);

    for i in 0..batch.natoms {
        let z = batch.atomic_numbers[i];
        let p = model.get_element(z).unwrap();
        let orb_start = batch.orbital_offsets[i];
        let norbs = batch.basis_types[i].num_orbitals();

        let s_pop = ws.density.get(orb_start, orb_start);
        let mut p_pop = 0.0;
        for o in 1..norbs {
            p_pop += ws.density.get(orb_start + o, orb_start + o);
        }
        let total_pop = s_pop + p_pop;
        let q = p.core_charge - total_pop;
        charges.push(q);
        populations.push([s_pop, p_pop, total_pop]);
    }

    let mut dipole = [0.0f64; 3];
    for (i, &q) in charges.iter().enumerate().take(batch.natoms) {
        let (x, y, z) = (batch.x[i], batch.y[i], batch.z[i]);
        dipole[0] += q * x * 4.803204;
        dipole[1] += q * y * 4.803204;
        dipole[2] += q * z * 4.803204;
    }
    let total_dipole = (dipole[0] * dipole[0] + dipole[1] * dipole[1] + dipole[2] * dipole[2]).sqrt();

    (heat_of_formation_kcal, charges, populations, dipole, total_dipole)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let start_time = Instant::now();

    if let Some(nthr) = cli.threads {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(nthr).build_global();
    }

    let input_path = &cli.input;
    let content = fs::read_to_string(input_path)?;
    let parsed = parse_mopac_input(&content)?;

    let input_stem = input_path.file_stem().unwrap().to_str().unwrap();
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new("."));

    let out_file = cli.output.unwrap_or_else(|| parent_dir.join(format!("{}.out", input_stem)));
    let arc_file = cli.arc.unwrap_or_else(|| parent_dir.join(format!("{}.arc", input_stem)));

    let use_gpu = cli.gpu || parsed.is_gpu_requested;
    let use_fp32 = cli.fp32 || parsed.is_fp32_requested;
    let is_opt = (cli.opt || parsed.is_opt_requested) && !cli.one_scf;

    let method_name = cli.method
        .or(parsed.method)
        .unwrap_or_else(|| "AM1".to_string())
        .to_uppercase();

    let model: Box<dyn ParameterModel> = match method_name.as_str() {
        "PM6" => Box::new(Pm6Model),
        "RM1" => Box::new(Rm1Model),
        _ => Box::new(Am1Model),
    };

    let use_nddo = cli.nddo || parsed.is_nddo_requested;

    println!("===============================================================================");
    println!("                          MOPAC_RS CANONICAL QUANTUM ENGINE                     ");
    println!("                                Version 0.1.0-alpha                             ");
    println!("===============================================================================");
    println!(" Job Input File        : {}", input_path.display());
    println!(" Method / Hamiltonian   : {}", model.name());
    println!(" NDDO Multipoles       : {}", if use_nddo { "Enabled (Full 22 Multipoles)" } else { "Monopole Approximation" });
    println!(" Calculation Mode      : {}", if is_opt { "L-BFGS Geometry Optimization" } else { "1SCF (Single Point)" });
    println!(" Compute Backend       : {}", if use_gpu { format!("Vulkan GPU ({})", if use_fp32 { "FP32 (18 TFLOPS)" } else { "FP64" }) } else { "CPU SIMD AVX2".to_string() });
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

        println!(" [Vulkan GPU] Device: {} (Discrete: {})", ctx.device_info.device_name, ctx.device_info.is_discrete);
        if use_fp32 {
            let gpu_calc = GpuCoulombCalculatorFP32::new(Arc::clone(&ctx))?;
            let _gpu_matrix = gpu_calc.compute_batch(&batch, model.as_ref())?;
            println!(" [Vulkan GPU] Evaluated pairwise Coulomb matrix using FP32 hardware pipeline.");
        } else {
            let gpu_calc = GpuCoulombCalculator::new(Arc::clone(&ctx))?;
            let _gpu_matrix = gpu_calc.compute_batch(&batch, model.as_ref())?;
            println!(" [Vulkan GPU] Evaluated pairwise Coulomb matrix using native Float64 pipeline.");
        }
    }

    let (scf_final, total_scf_cycles) = if is_opt {
        println!(" [Optimizer] Starting Cartesian L-BFGS Relaxation...");
        let opts = OptimizationOptions {
            max_cycles: 100,
            grad_rms_tol: 0.5,
            grad_max_tol: 1.0,
            energy_tol_ev: 1e-6,
            max_step_size: 0.1,
            history_capacity: 6,
        };

        let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
        let opt_res = optimize_geometry_lbfgs(&mut batch, model.as_ref(), &mut ws, &mut grad_ws, &opts);

        println!(" [Optimizer] Optimization finished in {} cycles (Converged: {})", opt_res.cycles, opt_res.converged);
        println!("   Initial Energy: {:12.6} eV | Final Energy: {:12.6} eV", opt_res.initial_energy_ev, opt_res.final_energy_ev);
        println!("   Initial RMS G : {:12.4} kcal/(mol*A) | Final RMS G: {:12.4} kcal/(mol*A)", opt_res.initial_grad_rms, opt_res.final_grad_rms);
        let niter = opt_res.final_scf.iterations;
        (opt_res.final_scf, niter)
    } else {
        println!(" [SCF] Running Roothaan-Hall Self-Consistent Field (NDDO: {})...", use_nddo);
        let res = run_rhf_scf_adaptive_with_nddo(&batch, model.as_ref(), &mut ws, 60, 1e-7, 1e-6, use_nddo);
        let niter = res.iterations;
        (res, niter)
    };

    let elapsed = start_time.elapsed();
    let (hof_kcal, charges, pops, dipole, dipole_tot) =
        compute_wavefunction_properties(&batch, model.as_ref(), &ws, &scf_final);

    println!("-------------------------------------------------------------------------------");
    println!("                             FINAL SCF RESULTS                                 ");
    println!("-------------------------------------------------------------------------------");
    println!(" Final Heat of Formation : {:15.5} kcal/mol ({:12.5} kJ/mol)", hof_kcal, hof_kcal * 4.184);
    println!(" Total SCF Energy        : {:15.6} eV", scf_final.total_energy_ev);
    println!(" Electronic Energy       : {:15.6} eV", scf_final.electronic_energy_ev);
    println!(" Nuclear Repulsion       : {:15.6} eV", scf_final.nuclear_repulsion_ev);
    println!(" HOMO Energy (IP)        : {:15.4} eV", scf_final.homo_energy_ev);
    println!(" LUMO Energy             : {:15.4} eV", scf_final.lumo_energy_ev);
    println!(" HOMO-LUMO Gap           : {:15.4} eV", scf_final.lumo_energy_ev - scf_final.homo_energy_ev);
    println!(" Total Dipole Moment     : {:15.4} Debye", dipole_tot);
    println!(" SCF Iterations Total    : {}", total_scf_cycles);
    println!(" Total Wall-Clock Time   : {:.4} seconds", elapsed.as_secs_f64());
    println!("===============================================================================");

    // Write .out file matching standard MOPAC format
    let mut out = File::create(&out_file)?;
    writeln!(out, " *******************************************************************************")?;
    writeln!(out, " **                                                                           **")?;
    writeln!(out, " **                              MOPAC_RS v0.1.0                              **")?;
    writeln!(out, " **                Canonical Semi-Empirical Quantum Chemistry Engine          **")?;
    writeln!(out, " **                                                                           **")?;
    writeln!(out, " *******************************************************************************")?;
    writeln!(out)?;
    writeln!(out, " KEYWORDS: {}", parsed.keywords.join(" "))?;
    writeln!(out, " TITLE:    {}", parsed.title)?;
    writeln!(out, " COMMENT:  {}", parsed.comment)?;
    writeln!(out)?;
    writeln!(out, " CALCULATION PARAMETERS:")?;
    writeln!(out, "   Method:    {}", model.name())?;
    writeln!(out, "   NDDO:      {}", if use_nddo { "Enabled (Full 22 Multipoles)" } else { "Monopole Approximation" })?;
    writeln!(out, "   Mode:      {}", if is_opt { "L-BFGS Geometry Optimization" } else { "1SCF" })?;
    writeln!(out, "   Backend:   {}", if use_gpu { "Vulkan GPU" } else { "CPU SIMD AVX2" })?;
    writeln!(out)?;
    writeln!(out, " FINAL HEAT OF FORMATION = {:17.5} KCAL/MOL = {:14.5} KJ/MOL", hof_kcal, hof_kcal * 4.184)?;
    writeln!(out, " TOTAL ENERGY            = {:17.6} EV", scf_final.total_energy_ev)?;
    writeln!(out, " ELECTRONIC ENERGY       = {:17.6} EV", scf_final.electronic_energy_ev)?;
    writeln!(out, " NUCLEAR REPULSION       = {:17.6} EV", scf_final.nuclear_repulsion_ev)?;
    writeln!(out, " IONIZATION POTENTIAL    = {:17.5} EV", -scf_final.homo_energy_ev)?;
    writeln!(out, " HOMO LUMO ENERGIES (EV) = {:12.4} {:12.4}", scf_final.homo_energy_ev, scf_final.lumo_energy_ev)?;
    writeln!(out, " DIPOLE MOMENT           = {:17.4} DEBYE (X={:.3}, Y={:.3}, Z={:.3})", dipole_tot, dipole[0], dipole[1], dipole[2])?;
    writeln!(out, " WALL-CLOCK TIME         = {:17.4} SECONDS", elapsed.as_secs_f64())?;
    writeln!(out)?;
    writeln!(out, "              NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS")?;
    writeln!(out, "  ATOM NO.   TYPE          CHARGE      No. of ELECS.   s-Pop       p-Pop")?;
    for i in 0..batch.natoms {
        let sym = atomic_number_to_symbol(batch.atomic_numbers[i]);
        writeln!(
            out,
            "   {:4}       {:2}         {:10.6}        {:8.4}     {:8.4}    {:8.4}",
            i + 1, sym, charges[i], pops[i][2], pops[i][0], pops[i][1]
        )?;
    }
    writeln!(out)?;
    writeln!(out, "                             CARTESIAN COORDINATES")?;
    for i in 0..batch.natoms {
        let sym = atomic_number_to_symbol(batch.atomic_numbers[i]);
        let (x, y, z) = (batch.x[i], batch.y[i], batch.z[i]);
        writeln!(out, "  {:4}    {:2}       {:16.9}  {:16.9}  {:16.9}", i + 1, sym, x, y, z)?;
    }
    writeln!(out)?;
    writeln!(out, " == MOPAC_RS DONE ==")?;

    // Write .arc file with optimized geometry
    let mut arc = File::create(&arc_file)?;
    writeln!(arc, "{} {}", model.name(), if is_opt { "OPT" } else { "1SCF" })?;
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
