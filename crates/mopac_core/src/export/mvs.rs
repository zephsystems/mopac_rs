//! MolViewSpec (MVS v1.0) Declarative Session Generator for Mol*.
//!
//! Generates JSON session descriptions enabling single-click loading into Mol*
//! with an optimized covalent skeleton (Ball & Stick) and a translucent Gaussian Surface.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").

/// Export a MolViewSpec (MVS v1.0) JSON session string.
///
/// Sets up:
/// 1. Download node referencing `structure_filename`.
/// 2. Parse node with appropriate format ("pdb", "cif", or "mmcif").
/// 3. Structure and Component 'all' nodes.
/// 4. Ball & Stick representation with 'element_symbol' coloring.
/// 5. Gaussian Surface representation with translucent opacity and 'uncertainty' coloring.
pub fn export_mvs_session(
    structure_filename: &str,
    format: &str,
    surface_opacity: f64,
) -> Result<String, String> {
    if structure_filename.trim().is_empty() {
        return Err("Structure filename cannot be empty in MVS session".to_string());
    }

    let fmt_lower = format.to_ascii_lowercase();
    let valid_fmt = match fmt_lower.as_str() {
        "pdb" | "cif" | "mmcif" => fmt_lower.as_str(),
        _ => {
            return Err(format!(
                "Unsupported format '{}' for MVS session (expected pdb, cif, or mmcif)",
                format
            ))
        }
    };

    let opacity_clamped = surface_opacity.clamp(0.05, 1.0);

    // Escape string values for valid JSON output
    let escaped_url = structure_filename
        .replace('\\', "\\\\")
        .replace('"', "\\\"");

    let mvs_json = format!(
        r#"{{
  "metadata": {{
    "version": "1.0",
    "generator": "mopac_rs"
  }},
  "root": {{
    "kind": "root",
    "children": [
      {{
        "kind": "download",
        "params": {{
          "url": "{}"
        }},
        "children": [
          {{
            "kind": "parse",
            "params": {{
              "format": "{}"
            }},
            "children": [
              {{
                "kind": "structure",
                "children": [
                  {{
                    "kind": "component",
                    "params": {{
                      "selector": "all"
                    }},
                    "children": [
                      {{
                        "kind": "representation",
                        "params": {{
                          "type": "ball_and_stick",
                          "color_theme": "element_symbol"
                        }}
                      }},
                      {{
                        "kind": "representation",
                        "params": {{
                          "type": "gaussian_surface",
                          "color_theme": "uncertainty",
                          "opacity": {:.2},
                          "size_theme": "physical"
                        }}
                      }}
                    ]
                  }}
                ]
              }}
            ]
          }}
        ]
      }}
    ]
  }}
}}"#,
        escaped_url, valid_fmt, opacity_clamped
    );

    Ok(mvs_json)
}
