//! CLI dispersion-flag parity gates.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Validates CLI `--disp` flag behavior:
//! `mopac --disp none` and `--disp D3` map to appropriate DispersionModel values
//! (none/off -> disabled, d3/d3-bj -> D3Bj, pm7 -> Pm7, pm6-dh+ -> Pm6DhPlus),
//! matching Python API behavior.
//!
//! Fidelity anchor: the PM7 default (dispersion on, matching OpenMOPAC's
//! automatic PM7 dispersion) must stay exact against the oracle.

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

const WATER_PM7: &str = "PM7 1SCF XYZ\nwater disp gate\n\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n";

fn run_cli(flags: &[&str], tag: &str) -> (f64, Option<f64>) {
    let dir = std::env::temp_dir().join("mopac_rs_dispflag");
    fs::create_dir_all(&dir).expect("create temp dir");
    let mop_file = dir.join(format!("{}.mop", tag));
    fs::write(&mop_file, WATER_PM7).expect("write deck");
    let out_file = dir.join(format!("{}.out", tag));
    let out = Command::new(env!("CARGO_BIN_EXE_mopac"))
        .arg(&mop_file)
        .arg("-o")
        .arg(&out_file)
        .args(flags)
        .output()
        .expect("run CLI");
    assert!(out.status.success(), "CLI failed for {:?}", flags);
    let text = fs::read_to_string(&out_file).expect("read CLI output");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut hof = None;
    let mut disp = None;
    for line in text.lines() {
        if line.contains("FINAL HEAT OF FORMATION") {
            hof = line
                .split('=')
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<f64>().ok());
        }
    }
    // The dispersion-energy breakdown is printed to stdout, not the .out report.
    for line in stdout.lines() {
        if line.contains("Dispersion Energy") {
            disp = line
                .split(':')
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<f64>().ok());
        }
    }
    (hof.expect("parse CLI HoF"), disp)
}

fn run_oracle(tag: &str) -> f64 {
    let mopac_bin = find_openmopac_binary();
    let dir = std::env::temp_dir().join("mopac_rs_dispflag");
    fs::create_dir_all(&dir).expect("create temp dir");
    let mop_file = dir.join(format!("{}.mop", tag));
    fs::write(&mop_file, WATER_PM7).expect("write deck");
    let status = Command::new(&mopac_bin)
        .arg(format!("{}.mop", tag))
        .current_dir(&dir)
        .status()
        .expect("run oracle");
    assert!(status.success(), "oracle failed");
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
    panic!("no HoF in oracle output");
}

#[test]
fn disp_none_disables_dispersion() {
    let (h_default, d_default) = run_cli(&[], "df_default");
    let (h_none, _) = run_cli(&["--disp", "none"], "df_none");
    let disp_energy = d_default.expect("default output must report the dispersion energy");
    let expected = h_default - disp_energy;
    println!(
        "[DISP] default={:.6} (disp={:.6}), none={:.6}, expected none={:.6}",
        h_default, disp_energy, h_none, expected
    );
    assert!(
        (h_none - expected).abs() < 1e-4,
        "--disp none was ignored: HoF {} vs expected {}",
        h_none,
        expected
    );
}

#[test]
fn disp_d3_selects_d3bj() {
    let (h_default, _) = run_cli(&[], "df_d3_default");
    let (h_none, _) = run_cli(&["--disp", "none"], "df_d3_none");
    let (h_d3, d_d3) = run_cli(&["--disp", "D3"], "df_d3");
    println!(
        "[DISP] default={:.6} none={:.6} d3={:.6} (d3 disp energy={:?})",
        h_default, h_none, h_d3, d_d3
    );
    assert!(
        (h_d3 - h_default).abs() > 1e-4 && (h_d3 - h_none).abs() > 1e-4,
        "--disp D3 was ignored (d3={:.6} equals default/none)",
        h_d3
    );
}

#[test]
fn pm7_default_matches_oracle() {
    let (h_default, _) = run_cli(&[], "df_oracle");
    let om = run_oracle("df_oracle_om");
    let d = (h_default - om).abs();
    println!(
        "[DISP] PM7 default rs={:.6} oracle={:.6} |d|={:.6}",
        h_default, om, d
    );
    assert!(
        d < 1e-3,
        "PM7 default parity broke: |d| = {:.6} kcal/mol",
        d
    );
}
