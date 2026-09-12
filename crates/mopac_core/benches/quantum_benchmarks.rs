//! High-Performance Criterion Benchmark Suite for MOPAC_RS.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Benchmarks the 4 strategic pillars and core SCF solvers:
//! 1. Closed-shell RHF SCF on Water (AM1).
//! 2. Boron chemistry BH3 SCF convergence (AM1 & PM6).
//! 3. Open-shell UHF SCF on Methyl Radical (AM1 doublet).
//! 4. COSMO dielectric analytical gradients on Solvated Water.

use criterion::{criterion_group, criterion_main, Criterion};
use mopac_core::gradients::nuclear_gradients::{
    compute_cartesian_gradients_full, GradientWorkspace,
};
use mopac_core::parameters::am1::Am1Model;
use mopac_core::parameters::pm6::Pm6Model;
use mopac_core::scf::scf_loop::{run_rhf_scf_with_options, ScfOptions};
use mopac_core::scf::uhf_loop::{run_uhf_scf_with_options, UhfOptions, UhfWorkspace};
use mopac_core::solvation::cosmo::{CosmoParams, CosmoState};
use mopac_core::types::{MolecularBatch, ScfWorkspace};
use std::hint::black_box;

fn bench_rhf_scf(c: &mut Criterion) {
    let z = vec![8, 1, 1];
    let coords = vec![
        [0.000, 0.000, 0.000],
        [0.000, 0.757, 0.586],
        [0.000, -0.757, 0.586],
    ];
    let batch = MolecularBatch::new(z, &coords);
    let model = Am1Model;
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_opts = ScfOptions {
        max_iter: 50,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    c.bench_function("scf/rhf_am1_water", |b| {
        b.iter(|| {
            let res = run_rhf_scf_with_options(
                black_box(&batch),
                black_box(&model),
                black_box(&mut ws),
                black_box(&scf_opts),
            );
            black_box(res.total_energy_ev);
        });
    });
}

fn bench_boron_bh3(c: &mut Criterion) {
    let z = vec![5, 1, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [1.19, 0.0, 0.0],
        [-0.595, 1.030569, 0.0],
        [-0.595, -1.030569, 0.0],
    ];
    let batch = MolecularBatch::new(z, &coords);
    let model = Pm6Model;
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_opts = ScfOptions {
        max_iter: 50,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    c.bench_function("boron/bh3_pm6_scf", |b| {
        b.iter(|| {
            let res = run_rhf_scf_with_options(
                black_box(&batch),
                black_box(&model),
                black_box(&mut ws),
                black_box(&scf_opts),
            );
            black_box(res.total_energy_ev);
        });
    });
}

fn bench_uhf_radicals(c: &mut Criterion) {
    let z = vec![6, 1, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [1.079, 0.0, 0.0],
        [-0.5395, 0.934441, 0.0],
        [-0.5395, -0.934441, 0.0],
    ];
    let batch = MolecularBatch::new(z, &coords);
    let model = Am1Model;
    let mut ws = UhfWorkspace::new(batch.norbs);
    let options = UhfOptions {
        multiplicity: 2,
        charge: 0,
        max_iter: 60,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        damping: 0.5,
        use_nddo: true,
        cosmo: None,
    };

    c.bench_function("uhf/ch3_radical_am1_doublet", |b| {
        b.iter(|| {
            let res = run_uhf_scf_with_options(
                black_box(&batch),
                black_box(&model),
                black_box(&mut ws),
                black_box(&options),
            );
            black_box(res.s_squared);
        });
    });
}

fn bench_cosmo_gradients(c: &mut Criterion) {
    let z = vec![8, 1, 1];
    let coords = vec![
        [0.000, 0.000, 0.000],
        [0.000, 0.757, 0.586],
        [0.000, -0.757, 0.586],
    ];
    let mut batch = MolecularBatch::new(z, &coords);
    let model = Am1Model;
    let params = CosmoParams {
        epsilon: 78.4,
        rsolv: 1.30005,
    };
    let state =
        CosmoState::initialize(&batch, &model, params).expect("COSMO initialization failed");
    let mut ws = ScfWorkspace::allocate(batch.norbs);
    let scf_opts = ScfOptions {
        max_iter: 50,
        energy_tol_ev: 1e-7,
        density_tol: 1e-6,
        level_shift_ev: 0.0,
        damping: 0.5,
        use_nddo: true,
        cosmo: Some(params),
    };
    let _ = run_rhf_scf_with_options(&batch, &model, &mut ws, &scf_opts);
    let mut grad_ws = GradientWorkspace::allocate(batch.norbs);
    let mut grads = vec![[0.0f64; 3]; batch.natoms];

    c.bench_function("cosmo/dielectric_gradients_water", |b| {
        b.iter(|| {
            compute_cartesian_gradients_full(
                black_box(&mut batch),
                black_box(&model),
                black_box(&ws.density),
                black_box(&mut grad_ws),
                black_box(&mut grads),
                black_box(true),
                black_box(Some(&state)),
            );
            black_box(grads[0][0]);
        });
    });
}

criterion_group!(
    benches,
    bench_rhf_scf,
    bench_boron_bh3,
    bench_uhf_radicals,
    bench_cosmo_gradients
);
criterion_main!(benches);
