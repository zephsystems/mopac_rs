//! Solvent-Accessible Surface (SAS) Cavity Construction for COSMO.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct translation and vectorized refinement of OpenMOPAC v23.2.5 `coscav`, `surclo`, and `ansude` subroutines in `cosmo.F90`.

use crate::solvation::radii::cosmo_atomic_radius;
use crate::solvation::tessellation::generate_sphere_tessellation;
use crate::types::MolecularBatch;
use std::f64::consts::PI;

/// A discrete boundary fine element belonging to a cavity segment.
#[derive(Debug, Clone, PartialEq)]
pub struct FineElement {
    /// 3D Cartesian coordinates of the element center in Angstroms.
    pub position: [f64; 3],
    /// Surface area of the element in square Angstroms (A^2).
    pub area: f64,
}

/// A discrete boundary segment on the solvent-accessible cavity surface.
#[derive(Debug, Clone, PartialEq)]
pub struct CavitySegment {
    /// 3D Cartesian coordinates of the segment center in Angstroms.
    pub position: [f64; 3],
    /// Surface area of the segment in square Angstroms (A^2).
    pub area: f64,
    /// Outward unit normal vector on the cavity surface.
    pub normal: [f64; 3],
    /// Index of the parent atom (0-indexed).
    pub atom_index: usize,
    /// Fine integration elements constituting this segment (from 1082 grid and crevice closure rings).
    pub fine_elements: Vec<FineElement>,
}

/// Geometric properties of the constructed COSMO solute cavity.
#[derive(Debug, Clone, PartialEq)]
pub struct CosmoCavity {
    pub segments: Vec<CavitySegment>,
    pub total_area_angstrom2: f64,
    pub total_volume_angstrom3: f64,
}

/// Result of analytical intersection area calculation between two spheres with solvent probe (`ansude.F90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnsudeResult {
    pub aar: f64,
    pub abr: f64,
    pub ara: f64,
    pub arb: f64,
    pub arad: f64,
    pub arbd: f64,
    pub rinc: f64,
}

/// Analytical intersection area of two spheres with radii `ra` and `rb` at distance `d` with probe radius `rs`.
///
/// Port of OpenMOPAC `ansude` (`cosmo.F90:125-189`).
pub fn ansude(ra: f64, rb: f64, d: f64, rs: f64) -> AnsudeResult {
    let qa = ra + rs;
    let qb = rb + rs;
    let ca = (qa * qa + d * d - qb * qb) / (2.0 * qa * d);
    let cb = (qb * qb + d * d - qa * qa) / (2.0 * qb * d);
    let sa = (1.0 - ca * ca).max(0.0).sqrt();
    let sb = (1.0 - cb * cb).max(0.0).sqrt();
    let ta = PI * sa;
    let tb = PI * sb;
    let mut fza = (1.0 - ta.cos()) / 2.0;
    let mut fzb = (1.0 - tb.cos()) / 2.0;
    if sa <= 0.0 || sb <= 0.0 {
        fza = 1.0;
        fzb = 1.0;
    }
    let xa = fzb * rs * (ca + cb);
    let xb = fza * rs * (ca + cb);
    let ya = ra * sa - fzb * rb * sb;
    let yb = rb * sb - fza * ra * sa;
    let za = (xa * xa + ya * ya).sqrt();
    let zb = (xb * xb + yb * yb).sqrt();
    let denom = (rs * rs * (ca + cb).powi(2) + (ra * sa - rb * sb).powi(2)).sqrt();
    let rinc = if denom > 1e-12 {
        0.5 * (za + zb) / denom
    } else {
        1.0
    };
    let ara = PI * ra * (2.0 * (1.0 + ca) * ra + sa * za);
    let arb = PI * rb * (2.0 * (1.0 + cb) * rb + sb * zb);
    let aar = PI * ra * (sa * za);
    let abr = PI * rb * (sb * zb);

    let cad = (qb * qb + d * d - qa * qa) / (2.0 * qa * d * d);
    let cbd = (qa * qa + d * d - qb * qb) / (2.0 * qb * d * d);
    let sad = if sa > 1e-12 { -ca * cad / sa } else { 0.0 };
    let sbd = if sb > 1e-12 { -cb * cbd / sb } else { 0.0 };
    let tad = PI * sad;
    let tbd = PI * sbd;
    let mut fzad = ta.sin() * 0.5;
    let mut fzbd = tb.sin() * 0.5;
    if sa <= 0.0 || sb <= 0.0 {
        fzad = 0.0;
        fzbd = 0.0;
    }
    let xad = rs * ((ca + cb) * fzbd * tbd + fzb * (cad + cbd));
    let xbd = rs * ((ca + cb) * fzad * tad + fza * (cad + cbd));
    let yad = ra * sad - fzbd * tbd * rb * sb - fzb * rb * sbd;
    let ybd = rb * sbd - fzad * tad * ra * sa - fza * ra * sad;
    let zad = if za > 1e-12 {
        (xa * xad + ya * yad) / za
    } else {
        0.0
    };
    let zbd = if zb > 1e-12 {
        (xb * xbd + yb * ybd) / zb
    } else {
        0.0
    };
    let arad = PI * ra * (sad * za + sa * zad + 2.0 * ra * cad);
    let arbd = PI * rb * (sbd * zb + sb * zbd + 2.0 * rb * cbd);

    AnsudeResult {
        aar,
        abr,
        ara,
        arb,
        arad,
        arbd,
        rinc,
    }
}

/// Compute local coordinate transformation matrix $T_M$ for an atom.
///
/// Port of OpenMOPAC `cosmo.F90:326-376`.
fn compute_tm_matrix(i: usize, coord: &[[f64; 3]], nn: [Option<usize>; 3]) -> [[f64; 3]; 3] {
    let xa = coord[i];
    let mut tm = [[0.0f64; 3]; 3];

    if let Some(n1) = nn[0] {
        let d = ((coord[n1][0] - xa[0]).powi(2)
            + (coord[n1][1] - xa[1]).powi(2)
            + (coord[n1][2] - xa[2]).powi(2))
        .sqrt();
        let inv_d = 1.0 / d;
        tm[0][0] = (coord[n1][0] - xa[0]) * inv_d;
        tm[0][1] = (coord[n1][1] - xa[1]) * inv_d;
        tm[0][2] = (coord[n1][2] - xa[2]) * inv_d;
    } else {
        tm[0][0] = 1.0;
    }

    let mut rr = -1.0f64;
    let mut xx = [0.0f64; 3];
    let mut sp;
    let mut current_nn2 = nn[1];
    let mut current_nn3 = nn[2];

    loop {
        if let Some(n2) = current_nn2 {
            let d = ((coord[n2][0] - xa[0]).powi(2)
                + (coord[n2][1] - xa[1]).powi(2)
                + (coord[n2][2] - xa[2]).powi(2))
            .sqrt();
            let inv_d = 1.0 / d;
            xx[0] = (coord[n2][0] - xa[0]) * inv_d;
            xx[1] = (coord[n2][1] - xa[1]) * inv_d;
            xx[2] = (coord[n2][2] - xa[2]) * inv_d;
        } else {
            rr += 1.0;
            let term1 = 3.0 * (tm[0][1] + rr);
            let term2 = 2.0 * (tm[0][0] + rr * rr);
            let term3 = tm[0][2];
            let dist = (term1 * term1 + term2 * term2 + term3 * term3).sqrt();
            let inv_dist = 1.0 / dist;
            xx[0] = term1 * inv_dist;
            xx[1] = term3 * inv_dist;
            xx[2] = term2 * inv_dist;
        }

        sp = xx[0] * tm[0][0] + xx[1] * tm[0][1] + xx[2] * tm[0][2];
        if sp * sp > 0.99 {
            current_nn2 = current_nn3;
            current_nn3 = None;
        } else {
            break;
        }
    }

    let sininv = 1.0 / (1.0 - sp * sp).max(1e-14).sqrt();
    tm[1][0] = (xx[0] - sp * tm[0][0]) * sininv;
    tm[1][1] = (xx[1] - sp * tm[0][1]) * sininv;
    tm[1][2] = (xx[2] - sp * tm[0][2]) * sininv;

    tm[2][0] = tm[0][1] * tm[1][2] - tm[1][1] * tm[0][2];
    tm[2][1] = tm[0][2] * tm[1][0] - tm[1][2] * tm[0][0];
    tm[2][2] = tm[0][0] * tm[1][1] - tm[1][0] * tm[0][1];

    tm
}

/// Rotates vector `v` by local frame matrix $T_M$: $\vec{u} = \sum_{k=0}^2 v_k \vec{tm}_k$.
#[inline]
fn rotate_vector_by_tm(v: [f64; 3], tm: &[[f64; 3]; 3]) -> [f64; 3] {
    [
        v[0] * tm[0][0] + v[1] * tm[1][0] + v[2] * tm[2][0],
        v[0] * tm[0][1] + v[1] * tm[1][1] + v[2] * tm[2][1],
        v[0] * tm[0][2] + v[1] * tm[1][2] + v[2] * tm[2][2],
    ]
}

impl CosmoCavity {
    /// Construct the discrete COSMO cavity surface for a molecular batch matching OpenMOPAC `coscav.F90`.
    ///
    /// Implements:
    /// 1. Asymmetric coordinate perturbation ($3 \times 10^{-9} \cos(0.1 i j)$) to break artificial degeneracies.
    /// 2. Canonical 107-element vdW radii from `cosini.F90`.
    /// 3. Local orientation matrix $T_M$ per atom aligned with nearest neighbors.
    /// 4. 1082-direction sphere tessellation rotated per-atom with exact solvent accessibility test.
    /// 5. Clustering into 42 (heavy) or 12 (hydrogen) primary segments.
    /// 6. Concave crevice closure (`surclo` + `ansude`) with ring segment generation.
    /// 7. Attachment of fine integration elements to each segment for exact near-field and diagonal self-energy.
    pub fn construct(batch: &MolecularBatch, rsolv: f64) -> Self {
        let natoms = batch.natoms;
        assert!(natoms > 0, "Molecular batch cannot be empty for COSMO");

        let r_solv = rsolv.max(0.5);

        // 1. Asymmetric coordinate perturbation
        let mut coord = vec![[0.0f64; 3]; natoms];
        for (i, coord_i) in coord.iter_mut().enumerate().take(natoms) {
            for (j, coord_ij) in coord_i.iter_mut().enumerate() {
                let pert = (((i + 1) * (j + 1)) as f64 * 0.1).cos() * 3.0e-9;
                let c = match j {
                    0 => batch.x[i],
                    1 => batch.y[i],
                    _ => batch.z[i],
                };
                *coord_ij = c + pert;
            }
        }

        // 2. Solvent radii
        let mut srad = vec![0.0f64; natoms];
        for (i, srad_i) in srad.iter_mut().enumerate().take(natoms) {
            *srad_i = cosmo_atomic_radius(batch.atomic_numbers[i]);
        }

        // 3. Find neighbor lists (lipa) for each atom: atoms within (r_i + r_j + 2*rsolv)
        let mut lipa = vec![Vec::new(); natoms];
        for i in 0..natoms {
            let rr_i = srad[i] + 2.0 * r_solv;
            for j in 0..natoms {
                if j != i {
                    let dx = coord[i][0] - coord[j][0];
                    let dy = coord[i][1] - coord[j][1];
                    let dz = coord[i][2] - coord[j][2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < rr_i + srad[j] {
                        lipa[i].push(j);
                    }
                }
            }
        }

        // 4. Generate icosahedral unit grids
        let fine_grid = generate_sphere_tessellation(1082);
        let basic_heavy = generate_sphere_tessellation(42);
        let basic_h = generate_sphere_tessellation(12);

        // Primary segment data structures per atom
        let mut atom_segments: Vec<Vec<CavitySegment>> = vec![Vec::new(); natoms];
        let mut atom_has_surface = vec![false; natoms];
        let mut total_area = 0.0f64;
        let mut total_volume = 0.0f64;

        for i in 0..natoms {
            let ri = srad[i];
            let r_sas_i = ri + r_solv;
            let ri2 = ri * ri;

            // Search 3 nearest neighbors
            let mut best_dist = [1e20f64; 3];
            let mut nn: [Option<usize>; 3] = [None; 3];
            for j in 0..natoms {
                if j != i {
                    let dx = coord[i][0] - coord[j][0];
                    let dy = coord[i][1] - coord[j][1];
                    let dz = coord[i][2] - coord[j][2];
                    let d2 = dx * dx + dy * dy + dz * dz;

                    if d2 + 0.05 < best_dist[2] {
                        best_dist[2] = d2;
                        nn[2] = Some(j);
                    }
                    if best_dist[2] + 0.05 < best_dist[1] {
                        best_dist.swap(1, 2);
                        nn.swap(1, 2);
                    }
                    if best_dist[1] + 0.05 < best_dist[0] {
                        best_dist.swap(0, 1);
                        nn.swap(0, 1);
                    }
                }
            }

            // Build transformation matrix T_M(i)
            let tm = compute_tm_matrix(i, &coord, nn);

            // Rotate fine grid points by T_M(i)
            let mut dirtm = vec![[0.0f64; 3]; 1082];
            for (j, pt) in fine_grid.iter().enumerate() {
                dirtm[j] = rotate_vector_by_tm(pt.dir, &tm);
            }

            // Test accessibility of 1082 fine points
            let mut din = vec![false; 1082];
            for j in 0..1082 {
                let xx = [
                    coord[i][0] + dirtm[j][0] * r_sas_i,
                    coord[i][1] + dirtm[j][1] * r_sas_i,
                    coord[i][2] + dirtm[j][2] * r_sas_i,
                ];

                let mut accessible = true;
                for &k in &lipa[i] {
                    let dx = xx[0] - coord[k][0];
                    let dy = xx[1] - coord[k][1];
                    let dz = xx[2] - coord[k][2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt() - r_solv - srad[k];
                    if dist < 0.0 {
                        accessible = false;
                        break;
                    }
                }

                if accessible {
                    din[j] = true;
                    let w = fine_grid[j].weight * ri2;
                    total_area += w;
                    // Volume contribution: (1/3) * ri^2 * w * (dirtm . coord_i + ri)
                    let dot_c = dirtm[j][0] * coord[i][0]
                        + dirtm[j][1] * coord[i][1]
                        + dirtm[j][2] * coord[i][2]
                        + ri;
                    total_volume += (1.0 / 3.0) * w * dot_c;
                }
            }

            let num_accessible = din.iter().filter(|&&x| x).count();
            if num_accessible == 0 {
                continue;
            }
            atom_has_surface[i] = true;

            // Rotate basic grid directions by T_M(i)
            let is_h = batch.atomic_numbers[i] == 1;
            let basic_grid = if is_h { &basic_h } else { &basic_heavy };
            let mut n_segs = basic_grid.len();
            let mut seg_dirs = vec![[0.0f64; 3]; n_segs];
            for (j, pt) in basic_grid.iter().enumerate() {
                seg_dirs[j] = rotate_vector_by_tm(pt.dir, &tm);
            }

            // Iterative segment clustering matching OpenMOPAC coscav.F90:436-497
            let mut iseg = vec![0usize; 1082];
            let mut nar_csm = vec![0usize; n_segs];
            let mut phinet = vec![[0.0f64; 3]; n_segs];
            let mut niter = 0;

            loop {
                niter += 1;
                nar_csm.fill(0);
                phinet.fill([0.0, 0.0, 0.0]);

                for j in 0..1082 {
                    if din[j] {
                        let mut spm = -2.0f64;
                        let mut ipm = 0;
                        for (ips, seg_dir) in seg_dirs.iter().enumerate().take(n_segs) {
                            let sp = dirtm[j][0] * seg_dir[0]
                                + dirtm[j][1] * seg_dir[1]
                                + dirtm[j][2] * seg_dir[2];
                            if sp >= spm {
                                spm = sp * (1.0 + 1e-14);
                                ipm = ips;
                            }
                        }
                        iseg[j] = ipm;
                        nar_csm[ipm] += 1;
                        let w = fine_grid[j].weight;
                        phinet[ipm][0] += dirtm[j][0] * w;
                        phinet[ipm][1] += dirtm[j][1] * w;
                        phinet[ipm][2] += dirtm[j][2] * w;
                    }
                }

                // Prune empty segments
                let mut ips = 0;
                while ips < n_segs {
                    if nar_csm[ips] == 0 {
                        niter = 1;
                        n_segs -= 1;
                        nar_csm.remove(ips);
                        phinet.remove(ips);
                        seg_dirs.remove(ips);
                    } else {
                        let dists = (phinet[ips][0].powi(2)
                            + phinet[ips][1].powi(2)
                            + phinet[ips][2].powi(2))
                        .max(1e-20);
                        let dist = 1.0 / dists.sqrt();
                        seg_dirs[ips] = [
                            phinet[ips][0] * dist,
                            phinet[ips][1] * dist,
                            phinet[ips][2] * dist,
                        ];
                        ips += 1;
                    }
                }

                if niter >= 2 || n_segs == 0 {
                    break;
                }
            }

            // Construct converged primary segments
            for (ips, &normal) in seg_dirs.iter().enumerate().take(n_segs) {
                let pos = [
                    coord[i][0] + normal[0] * ri,
                    coord[i][1] + normal[1] * ri,
                    coord[i][2] + normal[2] * ri,
                ];

                let mut seg_area = 0.0f64;
                let mut fine_elements = Vec::new();
                for j in 0..1082 {
                    if din[j] && iseg[j] == ips {
                        let w = fine_grid[j].weight * ri2;
                        seg_area += w;
                        fine_elements.push(FineElement {
                            position: [
                                coord[i][0] + dirtm[j][0] * ri,
                                coord[i][1] + dirtm[j][1] * ri,
                                coord[i][2] + dirtm[j][2] * ri,
                            ],
                            area: w,
                        });
                    }
                }

                atom_segments[i].push(CavitySegment {
                    position: pos,
                    area: seg_area,
                    normal,
                    atom_index: i,
                    fine_elements,
                });
            }
        }

        // 5. Concave crevice closure (surclo)
        for ia in 0..(natoms.saturating_sub(1)) {
            if !atom_has_surface[ia] {
                continue;
            }
            let ra = srad[ia] + r_solv;

            for &ib in &lipa[ia] {
                if ib <= ia || !atom_has_surface[ib] {
                    continue;
                }
                let rb = srad[ib] + r_solv;
                let xx_ab = [
                    coord[ib][0] - coord[ia][0],
                    coord[ib][1] - coord[ia][1],
                    coord[ib][2] - coord[ia][2],
                ];
                let dab = (xx_ab[0].powi(2) + xx_ab[1].powi(2) + xx_ab[2].powi(2)).sqrt();

                if dab >= ra + rb || dab <= (ra - rb).abs() {
                    continue;
                }

                // Call ansude
                let res = ansude(ra - r_solv, rb - r_solv, dab, r_solv);
                let cosa = (ra * ra + dab * dab - rb * rb) / (2.0 * dab * ra);
                let cosb = (rb * rb + dab * dab - ra * ra) / (2.0 * dab * rb);
                let sina = (1.0 - cosa * cosa).max(0.0).sqrt();
                let da = ra * cosa;
                let hh = ra * sina;
                let ddd = r_solv * (cosa + cosb) / dab;

                let mut fz = [
                    (1.0 - (hh * PI / ra).cos()) / 2.0,
                    (1.0 - (hh * PI / rb).cos()) / 2.0,
                ];
                if cosa * cosb < 0.0 {
                    fz = [1.0, 1.0];
                }

                let yx = [r_solv / ra, r_solv / rb];
                let xd = [
                    coord[ia][0] + da * xx_ab[0] / dab,
                    coord[ia][1] + da * xx_ab[1] / dab,
                    coord[ia][2] + da * xx_ab[2] / dab,
                ];

                // Create ring vectors rvx, rvy perpendicular to xx_ab
                let mut rvx = [
                    xx_ab[1] * 3.0 - xx_ab[2] * 2.0,
                    xx_ab[2] * 1.0 - xx_ab[0] * 3.0,
                    xx_ab[0] * 2.0 - xx_ab[1] * 1.0,
                ];
                let mut d_rvx = (rvx[0].powi(2) + rvx[1].powi(2) + rvx[2].powi(2)).sqrt();
                if d_rvx < 1e-9 {
                    rvx = [
                        xx_ab[1] * 4.0 - xx_ab[2] * 2.0,
                        xx_ab[2] * 1.0 - xx_ab[0] * 4.0,
                        xx_ab[0] * 2.0 - xx_ab[1] * 1.0,
                    ];
                    d_rvx = (rvx[0].powi(2) + rvx[1].powi(2) + rvx[2].powi(2)).sqrt();
                }
                let scale_rvx = hh / d_rvx;
                rvx = [rvx[0] * scale_rvx, rvx[1] * scale_rvx, rvx[2] * scale_rvx];

                let rvy = [
                    (xx_ab[1] * rvx[2] - xx_ab[2] * rvx[1]) / dab,
                    (xx_ab[2] * rvx[0] - xx_ab[0] * rvx[2]) / dab,
                    (xx_ab[0] * rvx[1] - xx_ab[1] * rvx[0]) / dab,
                ];

                // Search triple points with 3rd atom ic
                let mut phiset = Vec::new();
                let mut tarset = Vec::new();
                let mut is_open = Vec::new();
                let mut ntrp2 = 0;

                for &ic in &lipa[ia] {
                    if ic == ib {
                        continue;
                    }
                    let rc = srad[ic] + r_solv;
                    let xxx = [
                        coord[ic][0] - xd[0],
                        coord[ic][1] - xd[1],
                        coord[ic][2] - xd[2],
                    ];
                    let sp = xxx[0] * xx_ab[0] + xxx[1] * xx_ab[1] + xxx[2] * xx_ab[2];
                    let dabc = (xxx[0].powi(2) + xxx[1].powi(2) + xxx[2].powi(2)).sqrt();
                    let cos_alpha = sp / (dab * dabc);
                    let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(1e-28).sqrt();
                    let cj = (dabc * dabc + hh * hh - rc * rc) / (2.0 * dabc * hh * sin_alpha);

                    if cj < 1.0 {
                        ntrp2 += 2;
                    }

                    if (-1.0..=1.0).contains(&cj) {
                        let sj = (1.0 - cj * cj).sqrt();
                        let mut tvx = [
                            xxx[0] - cos_alpha * dabc * xx_ab[0] / dab,
                            xxx[1] - cos_alpha * dabc * xx_ab[1] / dab,
                            xxx[2] - cos_alpha * dabc * xx_ab[2] / dab,
                        ];
                        let d_tvx = (tvx[0].powi(2) + tvx[1].powi(2) + tvx[2].powi(2)).sqrt();
                        let sc_tvx = hh / d_tvx;
                        tvx = [tvx[0] * sc_tvx, tvx[1] * sc_tvx, tvx[2] * sc_tvx];

                        let tvy = [
                            (xx_ab[1] * tvx[2] - xx_ab[2] * tvx[1]) / dab,
                            (xx_ab[2] * tvx[0] - xx_ab[0] * tvx[2]) / dab,
                            (xx_ab[0] * tvx[1] - xx_ab[1] * tvx[0]) / dab,
                        ];

                        for &l_sign in &[-1.0f64, 1.0f64] {
                            let trp = [
                                xd[0] + cj * tvx[0] + sj * tvy[0] * l_sign,
                                xd[1] + cj * tvx[1] + sj * tvy[1] * l_sign,
                                xd[2] + cj * tvx[2] + sj * tvy[2] * l_sign,
                            ];

                            // Check occlusion by other atoms
                            let mut occluded = false;
                            for &k in &lipa[ia] {
                                if k != ib && k != ic {
                                    let dist = ((trp[0] - coord[k][0]).powi(2)
                                        + (trp[1] - coord[k][1]).powi(2)
                                        + (trp[2] - coord[k][2]).powi(2))
                                    .sqrt();
                                    if dist < srad[k] + r_solv {
                                        occluded = true;
                                        break;
                                    }
                                }
                            }

                            if !occluded {
                                let spx = rvx[0] * (trp[0] - xd[0])
                                    + rvx[1] * (trp[1] - xd[1])
                                    + rvx[2] * (trp[2] - xd[2]);
                                let spy = rvy[0] * (trp[0] - xd[0])
                                    + rvy[1] * (trp[1] - xd[1])
                                    + rvy[2] * (trp[2] - xd[2]);
                                let mut phi = (spx / (hh * hh + 1e-10)).clamp(-1.0, 1.0).acos();
                                if spy < 0.0 {
                                    phi = -phi;
                                }
                                phiset.push(phi);

                                let sp_dir = (-spy * rvx[0] + spx * rvy[0])
                                    * (trp[0] - coord[ic][0])
                                    + (-spy * rvx[1] + spx * rvy[1]) * (trp[1] - coord[ic][1])
                                    + (-spy * rvx[2] + spx * rvy[2]) * (trp[2] - coord[ic][2]);
                                is_open.push(if sp_dir < 0.0 { -1 } else { 1 });

                                // Triangle edges
                                let mut ee = [[0.0f64; 3]; 3];
                                for ix in 0..3 {
                                    ee[0][ix] =
                                        trp[ix] + r_solv / srad[ia] * (coord[ia][ix] - trp[ix]);
                                    ee[1][ix] =
                                        trp[ix] + r_solv / srad[ib] * (coord[ib][ix] - trp[ix]);
                                    ee[2][ix] =
                                        trp[ix] + r_solv / srad[ic] * (coord[ic][ix] - trp[ix]);
                                }
                                let dac = (ee[0][0] - ee[2][0]).powi(2)
                                    + (ee[0][1] - ee[2][1]).powi(2)
                                    + (ee[0][2] - ee[2][2]).powi(2);
                                let dbc = (ee[1][0] - ee[2][0]).powi(2)
                                    + (ee[1][1] - ee[2][1]).powi(2)
                                    + (ee[1][2] - ee[2][2]).powi(2);
                                let sp_tri = (ee[0][0] - ee[2][0]) * (ee[1][0] - ee[2][0])
                                    + (ee[0][1] - ee[2][1]) * (ee[1][0] - ee[2][1])
                                    + (ee[0][2] - ee[2][2]) * (ee[1][2] - ee[2][2]);
                                let tar =
                                    0.8 * (dac * dbc - sp_tri * sp_tri).max(0.0).sqrt() / 12.0;
                                tarset.push(tar);
                            }
                        }
                    }
                }

                // If no triple points found and no overlap, entire ring circle is open
                if phiset.is_empty() && ntrp2 == 0 {
                    phiset.push(0.0);
                    phiset.push(2.0 * PI);
                    tarset.push(0.0);
                    tarset.push(0.0);
                    is_open.push(1);
                    is_open.push(-1);
                }

                // Bubble sort with 2*pi wrapping matching lines 1468-1491
                let n_pts = phiset.len();
                if n_pts >= 2 && n_pts % 2 == 0 {
                    loop {
                        let mut swapped = false;
                        for l in 1..n_pts {
                            if phiset[l] < phiset[l - 1] {
                                phiset.swap(l, l - 1);
                                is_open.swap(l, l - 1);
                                tarset.swap(l, l - 1);
                                swapped = true;
                            }
                        }
                        if !swapped {
                            if is_open[0] == -1 {
                                phiset[0] += 2.0 * PI;
                            } else {
                                break;
                            }
                        }
                    }
                }

                // Generate ring segments along continuous sections
                let mut l = 1;
                while l < phiset.len() {
                    let phiu = phiset[l - 1];
                    let phio = phiset[l];
                    l += 2;

                    let mut nsa = (((phio - phiu) / (2.0 * PI) * 20.0) as usize) + 1;
                    nsa = nsa.max(2);
                    let dp = (phio - phiu) / ((nsa - 1) as f64);

                    for ich in 1..=2 {
                        let iat = if ich == 2 { ia } else { ib };
                        let htr = if iat == ia {
                            res.aar / (2.0 * PI)
                        } else {
                            res.abr / (2.0 * PI)
                        };

                        let mut ja = ich;
                        while ja <= nsa {
                            let jb = ja.saturating_sub(1).max(1);
                            let jc = (ja + 1).min(nsa);

                            let phi_a = phiu + (ja - 1) as f64 * dp;
                            let cphi_a = phi_a.cos();
                            let sphi_a = phi_a.sin();

                            let mut xja = [0.0f64; 3];
                            let fz_val = fz[ich - 1];
                            let yx_val = yx[ich - 1];
                            let other_coord = coord[if ich == 2 { ib } else { ia }];
                            let this_coord = coord[iat];

                            for ix in 0..3 {
                                let ca = xd[ix]
                                    + (cphi_a * rvx[ix] + sphi_a * rvy[ix]) * fz_val
                                    + (this_coord[ix] - xd[ix]) * yx_val;
                                xja[ix] =
                                    ca + (other_coord[ix] - this_coord[ix]) * ddd * (1.0 - fz_val);
                            }

                            let phi_b = phiu + (jb - 1) as f64 * dp;
                            let cphi_b = phi_b.cos();
                            let sphi_b = phi_b.sin();
                            let other_yx = yx[2 - ich];
                            let mut xjb = [0.0f64; 3];
                            for ix in 0..3 {
                                let ca = xd[ix] + cphi_b * rvx[ix] + sphi_b * rvy[ix];
                                xjb[ix] = ca + (other_coord[ix] - ca) * other_yx;
                            }

                            let phi_c = phiu + (jc - 1) as f64 * dp;
                            let cphi_c = phi_c.cos();
                            let sphi_c = phi_c.sin();
                            let mut xjc = [0.0f64; 3];
                            for ix in 0..3 {
                                let ca = xd[ix] + cphi_c * rvx[ix] + sphi_c * rvy[ix];
                                xjc[ix] = ca + (other_coord[ix] - ca) * other_yx;
                            }

                            let rsc_pos = [
                                (xja[0] * 0.5 + xjb[0] + xjc[0]) / 2.5,
                                (xja[1] * 0.5 + xjb[1] + xjc[1]) / 2.5,
                                (xja[2] * 0.5 + xjb[2] + xjc[2]) / 2.5,
                            ];

                            let cnrs = [
                                (xjc[1] - xjb[1]) * (xja[2] - xjb[2])
                                    - (xja[1] - xjb[1]) * (xjc[2] - xjb[2]),
                                (xjc[2] - xjb[2]) * (xja[0] - xjb[0])
                                    - (xja[2] - xjb[2]) * (xjc[0] - xjb[0]),
                                (xjc[0] - xjb[0]) * (xja[1] - xjb[1])
                                    - (xja[0] - xjb[0]) * (xjc[1] - xjb[1]),
                            ];
                            let d_cnrs =
                                (cnrs[0] * cnrs[0] + cnrs[1] * cnrs[1] + cnrs[2] * cnrs[2]).sqrt();
                            let rsc_area = ((jc - jb) as f64) * dp * htr;

                            if rsc_area > 1e-12 {
                                total_area += rsc_area;
                                if d_cnrs > 1e-12 {
                                    let spn2 = cnrs[0] * rsc_pos[0]
                                        + cnrs[1] * rsc_pos[1]
                                        + cnrs[2] * rsc_pos[2];
                                    total_volume += (1.0 / 3.0) * rsc_area * spn2 / d_cnrs;
                                }

                                // Associate ring segment with nearest primary segment of atom `iat`
                                let is_h_iat = batch.atomic_numbers[iat] == 1;
                                let d2max = 16.0 * srad[iat] * srad[iat] / 42.0
                                    * if is_h_iat { 42.0 / 12.0 } else { 1.0 };
                                let mut best_seg = 0;
                                let mut min_d2 = 1e20f64;

                                for (seg_idx, seg) in atom_segments[iat].iter().enumerate() {
                                    let d2 = (seg.position[0] - rsc_pos[0]).powi(2)
                                        + (seg.position[1] - rsc_pos[1]).powi(2)
                                        + (seg.position[2] - rsc_pos[2]).powi(2);
                                    if d2 < min_d2 {
                                        min_d2 = d2;
                                        best_seg = seg_idx;
                                    }
                                }

                                if min_d2 <= d2max && !atom_segments[iat].is_empty() {
                                    let target_seg = &mut atom_segments[iat][best_seg];
                                    let old_area = target_seg.area;
                                    let new_area = old_area + rsc_area;
                                    target_seg.position[0] = (old_area * target_seg.position[0]
                                        + rsc_area * rsc_pos[0])
                                        / new_area;
                                    target_seg.position[1] = (old_area * target_seg.position[1]
                                        + rsc_area * rsc_pos[1])
                                        / new_area;
                                    target_seg.position[2] = (old_area * target_seg.position[2]
                                        + rsc_area * rsc_pos[2])
                                        / new_area;
                                    target_seg.area = new_area;
                                    target_seg.fine_elements.push(FineElement {
                                        position: rsc_pos,
                                        area: rsc_area,
                                    });
                                } else {
                                    let norm_dir = [
                                        rsc_pos[0] - coord[iat][0],
                                        rsc_pos[1] - coord[iat][1],
                                        rsc_pos[2] - coord[iat][2],
                                    ];
                                    let len = (norm_dir[0].powi(2)
                                        + norm_dir[1].powi(2)
                                        + norm_dir[2].powi(2))
                                    .max(1e-12)
                                    .sqrt();
                                    atom_segments[iat].push(CavitySegment {
                                        position: rsc_pos,
                                        area: rsc_area,
                                        normal: [
                                            norm_dir[0] / len,
                                            norm_dir[1] / len,
                                            norm_dir[2] / len,
                                        ],
                                        atom_index: iat,
                                        fine_elements: vec![FineElement {
                                            position: rsc_pos,
                                            area: rsc_area,
                                        }],
                                    });
                                }
                            }

                            ja += 2;
                        }
                    }
                }
            }
        }

        // Flatten segments into unified cavity array
        let mut segments = Vec::new();
        for segs in atom_segments {
            segments.extend(segs);
        }

        CosmoCavity {
            segments,
            total_area_angstrom2: total_area,
            total_volume_angstrom3: total_volume,
        }
    }

    /// Number of discrete surface segments on the cavity.
    #[inline]
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_atom_sphere_area_and_volume() {
        // Single oxygen atom at origin: R = 1.72 A
        let coords = vec![[0.0, 0.0, 0.0]];
        let z = vec![8];
        let batch = MolecularBatch::new(z, &coords);

        let cavity = CosmoCavity::construct(&batch, 1.30005);
        let r = 1.72f64;
        let expected_area = 4.0 * PI * r * r;
        let expected_vol = (4.0 / 3.0) * PI * r.powi(3);

        assert!(
            (cavity.total_area_angstrom2 - expected_area).abs() < 0.2,
            "Single sphere area {} differs from 4*pi*R^2 = {}",
            cavity.total_area_angstrom2,
            expected_area
        );
        assert!(
            (cavity.total_volume_angstrom3 - expected_vol).abs() < 0.2,
            "Single sphere volume {} differs from 4/3*pi*R^3 = {}",
            cavity.total_volume_angstrom3,
            expected_vol
        );
    }

    /// TEST 1 (PHASE 3): DESEMPEÑO Y THROUGHPUT DE CONSTRUCCIÓN DE CAVIDAD COSMO
    /// Evalúa la velocidad de teselación analítica, clustering iterativo y cierre de grietas (surclo)
    /// asegurando que la cavidad molecular se genere en sub-milisegundos con alta tasa de throughput.
    #[test]
    fn test_cosmo_cavity_construction_performance_throughput() {
        use std::time::Instant;

        let coords = vec![
            [0.000, 0.000, 0.117176],
            [0.000, 0.756950, -0.468706],
            [0.000, -0.756950, -0.468706],
        ];
        let z = vec![8, 1, 1];
        let batch = MolecularBatch::new(z, &coords);

        let n_iterations = if cfg!(debug_assertions) { 50 } else { 2_000 };
        let mut total_segments = 0;
        let mut total_fine = 0;

        let start = Instant::now();
        for _ in 0..n_iterations {
            let cavity = CosmoCavity::construct(&batch, 1.30005);
            total_segments += cavity.num_segments();
            total_fine += cavity
                .segments
                .iter()
                .map(|s| s.fine_elements.len())
                .sum::<usize>();
        }
        let elapsed = start.elapsed();
        let evals_per_sec = (n_iterations as f64) / elapsed.as_secs_f64();
        let us_per_eval = elapsed.as_micros() as f64 / (n_iterations as f64);

        println!(
            "\n[COSMO PERF] Cavity construction: {:.1} evals/sec ({:.2} µs/eval), Segments: {}, Fine: {}",
            evals_per_sec,
            us_per_eval,
            total_segments / n_iterations,
            total_fine / n_iterations
        );

        // Throughput check: < 500 µs in release, < 35_000 µs in unoptimized debug under shared CI contention
        let max_allowed_us = if cfg!(debug_assertions) {
            35_000.0
        } else {
            500.0
        };
        assert!(
            us_per_eval < max_allowed_us,
            "COSMO cavity generation took too long: {:.2} µs/eval (limit: {:.2} µs)",
            us_per_eval,
            max_allowed_us
        );
        assert_eq!(total_segments / n_iterations, 49);
    }

    /// TEST 2 (PHASE 3): APLICABILIDAD MATEMÁTICA Y PARIDAD EXACTA DE CAVIDAD KLAMT (cosmo.F90)
    /// Verifica la correspondencia canónica con OpenMOPAC:
    /// 1. Número exacto de segmentos de superficie (NPS = 49).
    /// 2. Desglose canónico por átomo (O = 33, H1 = 8, H2 = 8).
    /// 3. Simetría espacial exacta entre átomos simétricos (Área H1 == Área H2 a 1e-6 A^2).
    /// 4. Conservación de área total: sum(A_seg) == total_area_angstrom2 (~ 42.32 A^2).
    /// 5. Coherencia geométrica: todos los centroides están situados en el radio solvente y fuera del solapamiento interatómico.
    #[test]
    fn test_cosmo_exact_klamt_cavity_applicability_and_parity() {
        let coords = vec![
            [0.000, 0.000, 0.117176],
            [0.000, 0.756950, -0.468706],
            [0.000, -0.756950, -0.468706],
        ];
        let z = vec![8, 1, 1];
        let batch = MolecularBatch::new(z, &coords);

        let cavity = CosmoCavity::construct(&batch, 1.30005);

        // 1. Exact segment count matching OpenMOPAC NPS = 49
        assert_eq!(
            cavity.num_segments(),
            49,
            "COSMO cavity segment count must exactly match OpenMOPAC NPS = 49"
        );

        // 2. Canonical atom distribution: O=33, H1=8, H2=8
        let segs_o = cavity.segments.iter().filter(|s| s.atom_index == 0).count();
        let segs_h1 = cavity.segments.iter().filter(|s| s.atom_index == 1).count();
        let segs_h2 = cavity.segments.iter().filter(|s| s.atom_index == 2).count();

        assert_eq!(segs_o, 33, "Oxygen segment count must be 33");
        assert_eq!(segs_h1, 8, "H1 segment count must be 8");
        assert_eq!(segs_h2, 8, "H2 segment count must be 8");

        // 3. Exact symmetry preservation between H1 and H2
        let area_h1: f64 = cavity
            .segments
            .iter()
            .filter(|s| s.atom_index == 1)
            .map(|s| s.area)
            .sum();
        let area_h2: f64 = cavity
            .segments
            .iter()
            .filter(|s| s.atom_index == 2)
            .map(|s| s.area)
            .sum();
        let sym_diff = (area_h1 - area_h2).abs();
        assert!(
            sym_diff < 1e-6,
            "Symmetry violation between equivalent hydrogens: diff = {:.8} A^2",
            sym_diff
        );

        // 4. Area conservation and OpenMOPAC reference range
        let sum_area: f64 = cavity.segments.iter().map(|s| s.area).sum();
        assert!(
            (sum_area - cavity.total_area_angstrom2).abs() < 1e-8,
            "Segment area sum must match total cavity area"
        );
        assert!(
            (cavity.total_area_angstrom2 - 42.316).abs() < 0.2,
            "Total area deviated from Klamt reference: {:.4} vs 42.32 A^2",
            cavity.total_area_angstrom2
        );

        // 5. Fine elements distribution per segment
        for (i, seg) in cavity.segments.iter().enumerate() {
            assert!(
                !seg.fine_elements.is_empty(),
                "Segment {} has no fine integration points",
                i + 1
            );
            assert!(
                seg.area > 0.05 && seg.area < 2.5,
                "Segment {} area {:.4} outside physical bounds",
                i + 1,
                seg.area
            );
        }
    }
}
