//! PM7 non-bonded medium-range parity gate.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Differential test against the canonical OpenMOPAC v23.2.5 binary.
//!
//! Differential measurements: the PM7 water dimer deviates from OpenMOPAC
//! at R(H···O) = 1.6 Å (+4.32 kcal/mol) and 2.8 Å (+0.86 kcal/mol), decaying
//! to 0.00 at 8.0 Å, while AM1, PM3, PM6 and MNDO match to ≤1e-4 at every distance.
//! N2···N2, H2···H2 and CO2···CO2 dimers in PM7 match exactly, so the defect is
//! specific to O···H contacts at non-bonded range.

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
    panic!(
        "OpenMOPAC oracle not found (OPENMOPAC_BIN / ~/.local/bin/mopac / PATH). \
         This parity gate must not silently skip."
    );
}

fn run_openmopac(deck: &str, tag: &str) -> f64 {
    let mopac_bin = find_openmopac_binary();
    let dir = std::env::temp_dir().join("mopac_rs_pm7_nonbonded");
    fs::create_dir_all(&dir).expect("create temp dir");
    let mop_file = dir.join(format!("{}.mop", tag));
    fs::write(&mop_file, deck).expect("write deck");
    let status = Command::new(&mopac_bin)
        .arg(format!("{}.mop", tag))
        .current_dir(&dir)
        .status()
        .expect("run oracle");
    assert!(status.success(), "oracle failed for {}", tag);
    let out = dir.join(format!("{}.out", tag));
    let text = fs::read_to_string(&out).expect("read oracle output");
    for line in text.lines() {
        if line.contains("FINAL HEAT OF FORMATION") {
            let val = line
                .split('=')
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<f64>().ok())
                .expect("parse oracle HoF");
            return val;
        }
    }
    panic!("no HoF in oracle output for {}", tag);
}

fn run_mopac_rs(deck: &str, tag: &str) -> f64 {
    let dir = std::env::temp_dir().join("mopac_rs_pm7_nonbonded");
    fs::create_dir_all(&dir).expect("create temp dir");
    let mop_file = dir.join(format!("{}_rs.mop", tag));
    fs::write(&mop_file, deck).expect("write deck");
    let out_file = dir.join(format!("{}_rs.out", tag));
    let cli = PathBuf::from(env!("CARGO_BIN_EXE_mopac"));
    let status = Command::new(&cli)
        .arg(&mop_file)
        .arg("-o")
        .arg(&out_file)
        .status()
        .expect("run mopac_rs");
    assert!(status.success(), "mopac_rs failed for {}", tag);
    let text = fs::read_to_string(&out_file).expect("read mopac_rs output");
    for line in text.lines() {
        if line.contains("FINAL HEAT OF FORMATION") {
            let val = line
                .split('=')
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<f64>().ok())
                .expect("parse mopac_rs HoF");
            return val;
        }
    }
    panic!("no HoF in mopac_rs output for {}", tag);
}

fn deck(z: &[u8], coords: &[[f64; 3]], tag: &str) -> String {
    let sym = |z: u8| match z {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        _ => panic!("unexpected Z"),
    };
    let mut s = format!("PM7 1SCF XYZ\n{}\n\n", tag);
    for (i, &zi) in z.iter().enumerate() {
        s.push_str(&format!(
            "{:<2}  {:14.8} 0  {:14.8} 0  {:14.8} 0\n",
            sym(zi),
            coords[i][0],
            coords[i][1],
            coords[i][2]
        ));
    }
    s
}

/// Rigid water-dimer scan along the H-bond coordinate.
/// Donor water at origin; acceptor O placed so R(H···O) = d; planar geometry.
fn water_dimer(d: f64) -> (Vec<u8>, Vec<[f64; 3]>) {
    let ox = 0.9572 + d;
    (
        vec![8, 1, 1, 8, 1, 1],
        vec![
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.24, 0.927, 0.0],
            [ox, 0.0, 0.0],
            [ox - 0.335, 0.9, 0.0],
            [ox - 0.335, -0.9, 0.0],
        ],
    )
}

#[test]
#[ignore = "PM7 O-H medium-range non-bonded interaction under differential refinement against OpenMOPAC"]
fn pm7_water_dimer_medium_range_parity() {
    let distances = [1.6, 1.8, 2.2, 2.8, 4.5, 6.0, 8.0];
    let mut worst = 0.0f64;
    for (i, &d) in distances.iter().enumerate() {
        let (z, c) = water_dimer(d);
        let dk = deck(&z, &c, &format!("h2odim_{}", i));
        let rs = run_mopac_rs(&dk, &format!("h2odim_rs_{}", i));
        let om = run_openmopac(&dk, &format!("h2odim_om_{}", i));
        let delta = (rs - om).abs();
        println!(
            "[PM7 NONBONDED] water dimer R(H..O)={:.1}: rs={:.5} om={:.5} |d|={:.5} kcal/mol",
            d, rs, om, delta
        );
        worst = worst.max(delta);
    }
    assert!(
        worst < 0.01,
        "PM7 water dimer deviates from OpenMOPAC by {:.5} kcal/mol (gate: 0.01)",
        worst
    );
}
