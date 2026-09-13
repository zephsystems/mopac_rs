//! Electrostatic Potential (ESP) Fitting and Merz-Singh-Kollman Partial Charges.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Computes the molecular quantum electrostatic potential on Connolly/vdW concentric
//! spherical grid shells, then determines atom-centered partial charges via
//! Lagrange-multiplier constrained linear least-squares regression.

use crate::constants::codata2018::BOHR_RADIUS_ANGSTROMS as A0_BOHR;
use crate::integrals::multipoles::DerivedMultipoleParams;
use crate::parameters::ParameterModel;
use crate::properties::dipole::E_ANGSTROM_TO_DEBYE;
use crate::types::{AlignedMatrix, MolecularBatch};
use std::f64::consts::PI;

/// Conversion factor from Angstroms to Bohr (atomic units of length).
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / A0_BOHR;

/// Conversion factor from atomic units of potential (Hartree/e) to electronvolts (eV).
pub const AU_TO_EV: f64 = 27.211386245988;

/// Retrieve Bondi standard van der Waals radius in Angstroms for an element Z.
pub fn bondi_vdw_radius_angstrom(z: u8) -> f64 {
    match z {
        1 => 1.20,  // H
        2 => 1.40,  // He
        3 => 1.82,  // Li
        4 => 1.53,  // Be
        5 => 1.92,  // B
        6 => 1.70,  // C
        7 => 1.55,  // N
        8 => 1.52,  // O
        9 => 1.47,  // F
        10 => 1.54, // Ne
        11 => 2.27, // Na
        12 => 1.73, // Mg
        13 => 1.84, // Al
        14 => 2.10, // Si
        15 => 1.80, // P
        16 => 1.80, // S
        17 => 1.75, // Cl
        18 => 1.88, // Ar
        19 => 2.75, // K
        20 => 2.31, // Ca
        26 => 2.05, // Fe
        28 => 1.63, // Ni
        29 => 1.40, // Cu
        30 => 1.39, // Zn
        35 => 1.85, // Br
        53 => 1.98, // I
        _ => 1.80,
    }
}

/// Options controlling Electrostatic Potential (ESP) grid generation and fitting.
#[derive(Debug, Clone)]
pub struct EspOptions {
    /// Radial multipliers for concentric Connolly/vdW grid shells (default: [1.4, 1.6, 1.8, 2.0])
    pub shell_multipliers: Vec<f64>,
    /// Number of Fibonacci sample points per atom per radial shell (default: 64)
    pub points_per_shell: usize,
    /// Total molecular net charge constraint (default: 0.0 for neutral molecules)
    pub net_charge: f64,
}

impl Default for EspOptions {
    fn default() -> Self {
        Self {
            shell_multipliers: vec![1.4, 1.6, 1.8, 2.0],
            points_per_shell: 64,
            net_charge: 0.0,
        }
    }
}

/// Results of Electrostatic Potential (ESP) fitting.
#[derive(Debug, Clone)]
pub struct EspResult {
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

/// Generate uniform Fibonacci spiral points on unit sphere S^2.
fn generate_fibonacci_sphere_points(n: usize) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(n);
    let phi_golden = PI * (3.0 - 5.0f64.sqrt()); // Golden angle ~2.39996 rad

    for i in 0..n {
        let y = 1.0 - (i as f64 / ((n - 1).max(1) as f64)) * 2.0;
        let radius = (1.0 - y * y).max(0.0).sqrt();
        let theta = phi_golden * (i as f64);
        let x = theta.cos() * radius;
        let z = theta.sin() * radius;
        points.push([x, y, z]);
    }
    points
}

/// Solve linear system A * x = b via Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn solve_linear_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>, String> {
    let n = b.len();
    assert_eq!(a.len(), n);

    for k in 0..n {
        // Find pivot
        let mut max_val = a[k][k].abs();
        let mut pivot_row = k;
        for p in (k + 1)..n {
            let val = a[p][k].abs();
            if val > max_val {
                max_val = val;
                pivot_row = p;
            }
        }

        if max_val < 1e-14 {
            return Err("Singular or ill-conditioned matrix in ESP charge regression".to_string());
        }

        // Swap rows
        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        // Eliminate column k
        let pivot = a[k][k];
        for i in (k + 1)..n {
            let factor = a[i][k] / pivot;
            for j in k..n {
                let val = a[k][j];
                a[i][j] -= factor * val;
            }
            b[i] -= factor * b[k];
        }
    }

    // Back-substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    Ok(x)
}

/// Compute Merz-Singh-Kollman atom-centered Electrostatic Potential (ESP) partial charges.
///
/// Parameters:
/// - `batch`: MolecularBatch containing Cartesian coordinates in Angstroms and atomic numbers.
/// - `model`: Semi-empirical parameter model providing core charges $Z_A^{\text{core}}$ and hybridization parameters.
/// - `density`: Converged Self-Consistent Field density matrix $P$.
/// - `opts`: EspOptions controlling grid shell distances and net charge constraints.
#[allow(clippy::needless_range_loop)]
pub fn compute_esp_charges(
    batch: &MolecularBatch,
    model: &dyn ParameterModel,
    density: &AlignedMatrix<f64>,
    opts: &EspOptions,
) -> Result<EspResult, String> {
    let natoms = batch.natoms;
    if natoms == 0 {
        return Err("Cannot compute ESP charges for empty molecular batch".to_string());
    }

    let vdw_radii: Vec<f64> = batch
        .atomic_numbers
        .iter()
        .map(|&z| bondi_vdw_radius_angstrom(z))
        .collect();

    // 1. Generate Merz-Singh-Kollman grid points on concentric vdW shells
    let sphere_points = generate_fibonacci_sphere_points(opts.points_per_shell);
    let mut grid_points: Vec<[f64; 3]> = Vec::new();

    for (a, &r_vdw) in vdw_radii.iter().enumerate().take(natoms) {
        let ra = [batch.x[a], batch.y[a], batch.z[a]];
        for &multiplier in &opts.shell_multipliers {
            let shell_radius = multiplier * r_vdw;
            for &p in &sphere_points {
                let candidate = [
                    ra[0] + shell_radius * p[0],
                    ra[1] + shell_radius * p[1],
                    ra[2] + shell_radius * p[2],
                ];

                // Point exclusion: discard if inside the vdW envelope of any other atom B
                let mut inside_any = false;
                for (b, &rb_vdw) in vdw_radii.iter().enumerate().take(natoms) {
                    if a == b {
                        continue;
                    }
                    let rb = [batch.x[b], batch.y[b], batch.z[b]];
                    let dist_sq = (candidate[0] - rb[0]).powi(2)
                        + (candidate[1] - rb[1]).powi(2)
                        + (candidate[2] - rb[2]).powi(2);
                    if dist_sq < rb_vdw * rb_vdw {
                        inside_any = true;
                        break;
                    }
                }

                if !inside_any {
                    grid_points.push(candidate);
                }
            }
        }
    }

    let m_grid = grid_points.len();
    if m_grid < natoms {
        return Err(format!(
            "Insufficient grid points ({}) for {} atoms in ESP fitting",
            m_grid, natoms
        ));
    }

    // 2. Precompute atomic core charges, electronic populations, and hybridization dipoles
    let mut core_charges = Vec::with_capacity(natoms);
    let mut elec_pops = Vec::with_capacity(natoms);
    let mut hyb_dipoles = Vec::with_capacity(natoms);

    for a in 0..natoms {
        let z = batch.atomic_numbers[a];
        let p = model
            .get_element(z)
            .ok_or_else(|| format!("Unsupported element Z={} in ESP parameter model", z))?;
        core_charges.push(p.core_charge);

        let start = batch.orbital_offsets[a];
        let norbs_a = batch.basis_types[a].num_orbitals();

        let mut pop_a = 0.0;
        for o in 0..norbs_a {
            pop_a += density.get(start + o, start + o);
        }
        elec_pops.push(pop_a);

        // Atomic hybridization dipole: d_x = 2 P(s, px) D1, d_y = 2 P(s, py) D1, d_z = 2 P(s, pz) D1
        let mut d_vec = [0.0; 3];
        if norbs_a >= 4 {
            let mp = DerivedMultipoleParams::from_element(&p);
            let d1 = mp.dd; // in Bohr
            let s_orb = start;
            let px_orb = start + 1;
            let py_orb = start + 2;
            let pz_orb = start + 3;

            // Dipole in Bohr * e
            d_vec[0] = 2.0 * density.get(s_orb, px_orb) * d1;
            d_vec[1] = 2.0 * density.get(s_orb, py_orb) * d1;
            d_vec[2] = 2.0 * density.get(s_orb, pz_orb) * d1;
        }
        hyb_dipoles.push(d_vec);
    }

    // 3. Compute quantum electrostatic potential V(r_k) at each grid point
    let mut v_quantum = Vec::with_capacity(m_grid);
    let mut inv_dist = vec![vec![0.0; natoms]; m_grid];

    for (k, r_k) in grid_points.iter().enumerate() {
        let mut v_k = 0.0;
        for a in 0..natoms {
            let dx = r_k[0] - batch.x[a];
            let dy = r_k[1] - batch.y[a];
            let dz = r_k[2] - batch.z[a];
            let dist_ang = (dx * dx + dy * dy + dz * dz).sqrt();
            let dist_bohr = dist_ang * ANGSTROM_TO_BOHR;
            let inv_r_bohr = 1.0 / dist_bohr;
            inv_dist[k][a] = inv_r_bohr;

            // Monopole potential (Core - Valence)
            let net_atom_charge = core_charges[a] - elec_pops[a];
            v_k += net_atom_charge * inv_r_bohr;

            // Hybridization dipole potential: (d . r) / r^3
            let d_vec = hyb_dipoles[a];
            let dx_bohr = dx * ANGSTROM_TO_BOHR;
            let dy_bohr = dy * ANGSTROM_TO_BOHR;
            let dz_bohr = dz * ANGSTROM_TO_BOHR;
            let dot_product_bohr = d_vec[0] * dx_bohr + d_vec[1] * dy_bohr + d_vec[2] * dz_bohr;
            let inv_r3_bohr = inv_r_bohr.powi(3);
            v_k -= dot_product_bohr * inv_r3_bohr;
        }
        v_quantum.push(v_k);
    }

    // 4. Build Lagrange-constrained least-squares system (natoms + 1) x (natoms + 1)
    // Minimizing sum_k (sum_A q_A / r_kA - V_k)^2 subject to sum_A q_A = Q_net
    let dim = natoms + 1;
    let mut a_mat = vec![vec![0.0; dim]; dim];
    let mut b_vec = vec![0.0; dim];

    for i in 0..natoms {
        for j in 0..natoms {
            let mut sum_inv_prod = 0.0;
            for k in 0..m_grid {
                sum_inv_prod += inv_dist[k][i] * inv_dist[k][j];
            }
            a_mat[i][j] = sum_inv_prod;
        }
        // Lagrange constraint column and row
        a_mat[i][natoms] = 1.0;
        a_mat[natoms][i] = 1.0;

        // b vector: sum_k V_k / r_ki
        let mut sum_v_inv = 0.0;
        for k in 0..m_grid {
            sum_v_inv += v_quantum[k] * inv_dist[k][i];
        }
        b_vec[i] = sum_v_inv;
    }
    // Net charge constraint
    a_mat[natoms][natoms] = 0.0;
    b_vec[natoms] = opts.net_charge;

    // 5. Solve linear system for fitted partial charges
    let solution = solve_linear_system(&mut a_mat, &mut b_vec)?;
    let charges: Vec<f64> = solution[0..natoms].to_vec();

    // 6. Compute RMS error and electric dipole moment from fitted charges
    let mut sum_err_sq = 0.0;
    for k in 0..m_grid {
        let mut v_fitted = 0.0;
        for a in 0..natoms {
            v_fitted += charges[a] * inv_dist[k][a];
        }
        let diff = (v_quantum[k] - v_fitted) * AU_TO_EV;
        sum_err_sq += diff * diff;
    }
    let rms_error_ev = (sum_err_sq / m_grid as f64).sqrt();

    // Electric dipole from point charges: mu = sum_A q_A * R_A * e_angstrom_to_debye
    let mut dipole_debye = [0.0; 3];
    for a in 0..natoms {
        let q = charges[a];
        dipole_debye[0] += q * batch.x[a] * E_ANGSTROM_TO_DEBYE;
        dipole_debye[1] += q * batch.y[a] * E_ANGSTROM_TO_DEBYE;
        dipole_debye[2] += q * batch.z[a] * E_ANGSTROM_TO_DEBYE;
    }
    let dipole_magnitude_debye =
        (dipole_debye[0].powi(2) + dipole_debye[1].powi(2) + dipole_debye[2].powi(2)).sqrt();

    Ok(EspResult {
        charges,
        dipole_debye,
        dipole_magnitude_debye,
        rms_error_ev,
        num_grid_points: m_grid,
    })
}
