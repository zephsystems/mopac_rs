//! PM6 additive heat-of-formation correction parity gates.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//!
//! Measured defect at HEAD (2026-10-30): PM6 runs on nitro/nitrate compounds are
//! exactly +0.500 kcal/mol above OpenMOPAC (CH3NO2, CH3NO3, C2H5NO3, FNO3) and
//! +0.980 for N2O5. The missing term is the nsp2 (trigonal-planar nitrogen)
//! molecular-mechanics correction: OpenMOPAC `nsp2_correction` /
//! `nsp2_atom_correction` (set_up_dentate.F90 lines 139-210), applied to PM6
//! only (compfg.F90 line 371):
//!   for every N with exactly 3 ligands of which < 2 are H:
//!     tot = 2π − (sum of the three angles at N)   [radians]
//!     correction += −0.5 · exp(−10 · tot) kcal/mol.
//!
//! The CCCBDB geometries are embedded; the oracle is run per molecule.

use std::fs;
use std::path::{Path, PathBuf};
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

fn deck(z: &[u8], c: &[[f64; 3]], tag: &str) -> String {
    let sym = |z: u8| match z {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        _ => panic!("unexpected Z"),
    };
    let mut s = format!("PM6 1SCF XYZ\n{}\n\n", tag);
    for (i, &zi) in z.iter().enumerate() {
        s.push_str(&format!(
            "{:<2}  {:14.8} 0  {:14.8} 0  {:14.8} 0\n",
            sym(zi),
            c[i][0],
            c[i][1],
            c[i][2]
        ));
    }
    s
}

fn run(engine: &str, dk: &str, tag: &str, binary: &Path) -> f64 {
    let dir = std::env::temp_dir().join("mopac_rs_additive");
    fs::create_dir_all(&dir).expect("create temp dir");
    let mop_file = dir.join(format!("{}.mop", tag));
    fs::write(&mop_file, dk).expect("write deck");
    let status = Command::new(binary)
        .arg(format!("{}.mop", tag))
        .current_dir(&dir)
        .status()
        .expect("run engine");
    assert!(status.success(), "{} failed for {}", engine, tag);
    let text = fs::read_to_string(dir.join(format!("{}.out", tag))).expect("read output");
    for line in text.lines() {
        if line.contains("FINAL HEAT OF FORMATION") {
            return line
                .split('=')
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<f64>().ok())
                .expect("parse HoF");
        }
    }
    panic!("no HoF in {} output for {}", engine, tag);
}

struct TestCase {
    tag: &'static str,
    z: Vec<u8>,
    c: Vec<[f64; 3]>,
}

#[test]
fn pm6_nitro_nitrate_nsp2_correction_parity() {
    let cases = [
        TestCase {
            tag: "ch3no2",
            z: vec![6, 7, 1, 1, 1, 8, 8],
            c: vec![
                [0.041164, -1.324937, 0.0],
                [0.0, 0.173934, 0.0],
                [1.081821, -1.641042, 0.0],
                [-0.489294, -1.662661, 0.891454],
                [-0.489294, -1.662661, -0.891454],
                [1.069946, 0.773463, 0.0],
                [-1.113724, 0.688842, 0.0],
            ],
        },
        TestCase {
            tag: "ch3no3",
            z: vec![7, 8, 8, 8, 6, 1, 1, 1],
            c: vec![
                [0.0, 0.621497, 0.0],
                [-1.213737, 0.560731, 0.0],
                [0.726023, 1.585251, 0.0],
                [0.714363, -0.601515, 0.0],
                [-0.140995, -1.758068, 0.0],
                [0.565996, -2.589184, 0.0],
                [-0.766607, -1.78431, 0.895191],
                [-0.766607, -1.78431, -0.895191],
            ],
        },
        TestCase {
            tag: "c2h5no3",
            z: vec![7, 8, 8, 8, 6, 6, 1, 1, 1, 1, 1],
            c: vec![
                [1.10887, -0.31156, 0.0],
                [0.0, 0.563065, 0.0],
                [2.165871, 0.272211, 0.0],
                [0.883208, -1.506213, 0.0],
                [-1.283224, -0.113164, 0.0],
                [-2.326774, 0.990182, 0.0],
                [-1.353277, -0.748519, 0.88771],
                [-1.353277, -0.748519, -0.88771],
                [-3.325486, 0.540888, 0.0],
                [-2.231351, 1.621228, 0.888511],
                [-2.231351, 1.621228, -0.888511],
            ],
        },
        TestCase {
            tag: "fno3",
            z: vec![7, 8, 8, 8, 9],
            c: vec![
                [0.0, 0.62073, 0.0],
                [-0.700297, -0.714474, 0.0],
                [1.19029, 0.603868, 0.0],
                [-0.834184, 1.480473, 0.0],
                [0.305947, -1.700449, 0.0],
            ],
        },
        TestCase {
            tag: "n2o5",
            z: vec![8, 7, 7, 8, 8, 8, 8],
            c: vec![
                [0.0, 0.0, 0.825319],
                [0.0, 1.265453, 0.011207],
                [0.0, -1.265453, 0.011207],
                [0.52408, 2.14692, 0.626348],
                [-0.52408, -2.14692, 0.626348],
                [-0.552818, 1.222325, -1.048814],
                [0.552818, -1.222325, -1.048814],
            ],
        },
    ];
    let om_bin = find_openmopac_binary();
    let cli = PathBuf::from(env!("CARGO_BIN_EXE_mopac"));
    let mut worst = 0.0f64;
    for (i, tc) in cases.iter().enumerate() {
        let dk = deck(&tc.z, &tc.c, &format!("pm6_{}", tc.tag));
        let rs = run("mopac_rs", &dk, &format!("rs_{}", i), &cli);
        let om = run("openmopac", &dk, &format!("om_{}", i), &om_bin);
        let d = (rs - om).abs();
        println!(
            "[NSP2] {}: rs={:.5} om={:.5} |d|={:.5} kcal/mol",
            tc.tag, rs, om, d
        );
        worst = worst.max(d);
    }
    assert!(
        worst < 0.001,
        "PM6 nsp2 correction missing or wrong: worst |d| = {:.5} kcal/mol",
        worst
    );
}

#[test]
fn pm6_control_without_sp2_nitrogen_unchanged() {
    // NH3 has no trigonal-planar N: the correction must not touch it.
    let z = vec![7u8, 1, 1, 1];
    let c = vec![
        [0.0, 0.0, 0.119205],
        [0.0, 0.938797, -0.278145],
        [0.813022, -0.469398, -0.278145],
        [-0.813022, -0.469398, -0.278145],
    ];
    let dk = deck(&z, &c, "pm6_nh3");
    let om_bin = find_openmopac_binary();
    let cli = PathBuf::from(env!("CARGO_BIN_EXE_mopac"));
    let rs = run("mopac_rs", &dk, "rs_nh3", &cli);
    let om = run("openmopac", &dk, "om_nh3", &om_bin);
    let d = (rs - om).abs();
    println!(
        "[NSP2] control NH3: rs={:.5} om={:.5} |d|={:.5} kcal/mol",
        rs, om, d
    );
    assert!(d < 0.001, "NH3 PM6 parity broke: |d| = {:.5} kcal/mol", d);
}
