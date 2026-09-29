//! PDB Exporter Optimized for Macromolecular Visualizers (Mol*, PyMOL, WebGL).
//!
//! Maps continuous partial charges q in [-q_max, +q_max] to the crystallographic
//! B-factor domain [10.0, 90.0] via affine transformation:
//!
//! B_i = 50.0 - 40.0 * (clamp(q_i, -q_max, +q_max) / q_max)
//!
//! Ensuring:
//! 1. Nucleophilic centers (delta-, e.g. oxygens) map to B ~ 90.0 (Bright Red in Mol* Uncertainty theme).
//! 2. Neutral centers (q ~ 0.0, aliphatic carbons) map to B ~ 50.0 (White/Cyan).
//! 3. Electrophilic centers (delta+, carbonyl carbons) map to B ~ 10.0 (Deep Blue).
//! 4. B_i >= 10.0 > 0.0 strictly, preventing Mol* from collapsing atomic radii to zero
//!    and causing surface punctures or polygonal tearing.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").

use super::sdf::z_to_symbol;

/// Export molecular structure as a PDB string formatted specifically for Mol* visualization.
///
/// Ensures high-contrast Diverging Blue-White-Red coloring in Mol* 'Uncertainty' color theme
/// without surface clipping artifacts or radius collapse.
pub fn export_molstar_pdb(
    atomic_numbers: &[u8],
    coordinates: &[[f64; 3]],
    charges: &[f64],
    max_charge: Option<f64>,
) -> Result<String, String> {
    let natoms = atomic_numbers.len();
    if coordinates.len() != natoms {
        return Err(format!(
            "Mismatch between atomic_numbers ({}) and coordinates ({}) length",
            natoms,
            coordinates.len()
        ));
    }
    if charges.len() != natoms {
        return Err(format!(
            "Mismatch between atomic_numbers ({}) and charges ({}) length",
            natoms,
            charges.len()
        ));
    }
    if natoms == 0 {
        return Err("Cannot export empty molecular batch to PDB".to_string());
    }

    let q_max = max_charge.unwrap_or(0.6).abs().max(1e-3);
    let mut out = String::with_capacity(natoms * 82 + 128);
    out.push_str("REMARK   MOPAC_RS MOLSTAR-OPTIMIZED PDB WITH AFFINE B-FACTOR CHARGES\n");
    out.push_str("REMARK   B-FACTOR RANGE: 10.0 (ELECTROPHILIC/BLUE) TO 90.0 (NUCLEOPHILIC/RED)\n");

    for (i, (&z, c)) in atomic_numbers.iter().zip(coordinates.iter()).enumerate() {
        let sym = z_to_symbol(z);
        let q = charges[i];
        let q_clamped = if q.is_finite() {
            q.clamp(-q_max, q_max)
        } else if q.is_nan() {
            0.0
        } else if q > 0.0 {
            q_max
        } else {
            -q_max
        };

        // Negative charge -> Red (high B-factor ~90.0)
        // Neutral charge  -> White/Cyan (mid B-factor ~50.0)
        // Positive charge -> Blue (low B-factor ~10.0)
        let b_scaled = 50.0 - (q_clamped / q_max) * 40.0;

        let atom_num = (i + 1) % 100000;
        let line = format!(
            "ATOM  {:5} {:^4} STO A   1    {:8.3}{:8.3}{:8.3}  1.00{:6.2}          {:>2}\n",
            atom_num, sym, c[0], c[1], c[2], b_scaled, sym
        );
        out.push_str(&line);
    }
    out.push_str("TER\nEND\n");
    Ok(out)
}
