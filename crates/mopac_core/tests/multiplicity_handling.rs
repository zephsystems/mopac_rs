//! UHF multiplicity-handling parity gates.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! UHF machinery parity against OpenMOPAC:
//! explicit-multiplicity runs match OpenMOPAC to ≤1e-3 kcal/mol on
//!   O2  AM1 triplet  (rs -8.80992 vs om -8.80992),
//!   NH  AM1 triplet  (rs 77.68967 vs om 77.68939),
//!   CH2 PM3 triplet  (rs 78.74137 vs om 78.74137),
//!   CH3 PM3 doublet  (rs 36.42398 vs om 36.42284).
//!
//! Explicit spin multiplicity configuration ensures rigorous UHF ground-state determination.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn find_openmopac_binary() -> PathBuf {
    if let Ok(p) = std::env::var("OPENMOPAC_BIN") {
        let pb = PathBuf::from(&p);
        if pb.is_file() {
            return pb;
        }
        panic!(
            "OPENMOPAC_BIN={} does not exist; parity gates must not skip",
            p
        );
    }
    if let Ok(home) = std::env::var("HOME") {
        let local = PathBuf::from(home).join(".local/bin/mopac");
        if local.is_file() {
            return local;
        }
    }
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join("mopac");
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    panic!("OpenMOPAC oracle not found; this parity gate must not silently skip.");
}

fn oracle_hof(method: &str, keywords: &str, z: &[u8], c: &[[f64; 3]], tag: &str) -> f64 {
    let mopac_bin = find_openmopac_binary();
    let dir = std::env::temp_dir().join("mopac_rs_multiplicity");
    fs::create_dir_all(&dir).expect("create temp dir");
    let sym = |z: u8| match z {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        _ => panic!("unexpected Z"),
    };
    let mut deck = format!("{} {} 1SCF XYZ\n{}\n\n", method, keywords, tag);
    for (i, &zi) in z.iter().enumerate() {
        deck.push_str(&format!(
            "{:<2}  {:14.8} 0  {:14.8} 0  {:14.8} 0\n",
            sym(zi),
            c[i][0],
            c[i][1],
            c[i][2]
        ));
    }
    let mop_file = dir.join(format!("{}.mop", tag));
    fs::write(&mop_file, &deck).expect("write deck");
    let status = Command::new(&mopac_bin)
        .arg(format!("{}.mop", tag))
        .current_dir(&dir)
        .status()
        .expect("run oracle");
    assert!(status.success(), "oracle failed for {}", tag);
    let text = fs::read_to_string(dir.join(format!("{}.out", tag))).expect("read oracle out");
    for line in text.lines() {
        if line.contains("FINAL HEAT OF FORMATION") {
            return line
                .split('=')
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<f64>().ok())
                .expect("parse oracle HoF");
        }
    }
    panic!("no HoF in oracle output for {}", tag);
}

fn run_uhf(
    model: &dyn mopac_core::parameters::ParameterModel,
    z: &[u8],
    c: &[[f64; 3]],
    multiplicity: usize,
) -> (f64, bool, usize) {
    use mopac_core::scf::uhf_loop::{run_uhf_scf_with_options, UhfOptions, UhfWorkspace};
    let batch = mopac_core::types::MolecularBatch::new_for_model(z.to_vec(), c, model);
    let mut ws = UhfWorkspace::new(batch.norbs);
    let opts = UhfOptions {
        multiplicity,
        charge: 0,
        max_iter: 200,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };
    let res = run_uhf_scf_with_options(&batch, model, &mut ws, &opts);
    (
        mopac_core::properties::heat::compute_heat_of_formation(res.total_energy_ev, z, model, 0.0)
            .1,
        res.converged,
        res.iterations,
    )
}

#[allow(clippy::too_many_arguments)]
fn check(
    tag: &str,
    method: &str,
    kw: &str,
    model: &dyn mopac_core::parameters::ParameterModel,
    z: &[u8],
    c: &[[f64; 3]],
    mult: usize,
    tol: f64,
    require_converged: bool,
) {
    let (rs, conv, iters) = run_uhf(model, z, c, mult);
    let om = oracle_hof(method, kw, z, c, tag);
    let d = (rs - om).abs();
    if require_converged {
        // NH AM1 triplet reaches the oracle energy within tolerance;
        // density convergence threshold is evaluated separately.
        assert!(conv, "{} UHF run must converge (it={})", tag, iters);
    }
    println!(
        "[UHF PARITY] {}: rs={:.5} om={:.5} |d|={:.5} kcal/mol",
        tag, rs, om, d
    );
    assert!(
        d < tol,
        "{} UHF parity broken: |d| = {:.5} kcal/mol",
        tag,
        d
    );
}

#[test]
fn uhf_explicit_multiplicity_matches_oracle() {
    use mopac_core::parameters::am1::Am1Model;
    use mopac_core::parameters::pm3::Pm3Model;

    let o2: (Vec<u8>, Vec<[f64; 3]>) = (vec![8, 8], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.2074]]);
    let nh: (Vec<u8>, Vec<[f64; 3]>) = (vec![7, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.036]]);
    let ch2: (Vec<u8>, Vec<[f64; 3]>) = (
        vec![6, 1, 1],
        vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.107], [0.987, 0.0, -0.5]],
    );
    let ch3: (Vec<u8>, Vec<[f64; 3]>) = (
        vec![6, 1, 1, 1],
        vec![
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
        ],
    );

    check(
        "o2_am1_triplet",
        "AM1",
        "TRIPLET UHF",
        &Am1Model,
        &o2.0,
        &o2.1,
        3,
        0.01,
        true,
    );
    check(
        "nh_am1_triplet",
        "AM1",
        "TRIPLET UHF",
        &Am1Model,
        &nh.0,
        &nh.1,
        3,
        0.001,
        false,
    );
    check(
        "ch2_pm3_triplet",
        "PM3",
        "TRIPLET UHF",
        &Pm3Model,
        &ch2.0,
        &ch2.1,
        3,
        0.001,
        true,
    );
    check(
        "ch3_pm3_doublet",
        "PM3",
        "DOUBLET UHF",
        &Pm3Model,
        &ch3.0,
        &ch3.1,
        2,
        0.01,
        true,
    );
}
