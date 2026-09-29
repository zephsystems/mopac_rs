//! Canonical PDBx/mmCIF Exporter with Native Partial Charge Loop.
//!
//! Emits standardized mmCIF files containing `_atom_site.partial_charge` and
//! `_atom_site.B_iso_or_equiv`, enabling native Mol* `Color Theme: Partial Charge`
//! without manipulating crystallographic B-factors.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").

use super::sdf::z_to_symbol;

/// Export molecular structure as a canonical PDBx/mmCIF string with partial charges.
pub fn export_mmcif_with_charges(
    atomic_numbers: &[u8],
    coordinates: &[[f64; 3]],
    charges: &[f64],
    molecule_name: Option<&str>,
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
        return Err("Cannot export empty molecular batch to mmCIF".to_string());
    }

    let mol_name = molecule_name.unwrap_or("MOL");
    let sanitized_name: String = mol_name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    let entry_name = if sanitized_name.is_empty() {
        "MOL"
    } else {
        &sanitized_name
    };

    let mut out = String::with_capacity(natoms * 100 + 512);

    out.push_str(&format!("data_{}\n#\n", entry_name));
    out.push_str(&format!("_entry.id   {}\n#\n", entry_name));
    out.push_str(
        "loop_\n\
_atom_site.group_PDB\n\
_atom_site.id\n\
_atom_site.type_symbol\n\
_atom_site.label_atom_id\n\
_atom_site.label_comp_id\n\
_atom_site.label_asym_id\n\
_atom_site.label_seq_id\n\
_atom_site.Cartn_x\n\
_atom_site.Cartn_y\n\
_atom_site.Cartn_z\n\
_atom_site.occupancy\n\
_atom_site.B_iso_or_equiv\n\
_atom_site.partial_charge\n",
    );

    for (i, (&z, c)) in atomic_numbers.iter().zip(coordinates.iter()).enumerate() {
        let sym = z_to_symbol(z);
        let atom_label = format!("{}{}", sym, i + 1);
        let q = charges[i];
        let q_clean = if q.is_finite() { q } else { 0.0 };

        let line = format!(
            "ATOM {:<5} {:<2} {:<5} STO A 1 {:10.3} {:10.3} {:10.3}  1.00  20.00 {:9.4}\n",
            i + 1,
            sym,
            atom_label,
            c[0],
            c[1],
            c[2],
            q_clean
        );
        out.push_str(&line);
    }
    out.push_str("#\n");

    Ok(out)
}
