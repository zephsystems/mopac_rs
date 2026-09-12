//! Molecular Bond Order and Valency Analysis Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Direct mathematical translation of OpenMOPAC `bonds.F90`.
//! Reference: Armstrong, D.R., Perkins, P.G., Stewart, J.J.P., J.C.S. Dalton, 838 (1973).

use crate::types::{AlignedMatrix, MolecularBatch};

/// Comprehensive bond order, valency, and charge partition analysis result.
#[derive(Debug, Clone, PartialEq)]
pub struct BondOrderResult {
    /// Mayer / Armstrong-Perkins-Stewart bond order matrix $B_{AB}$ of dimension $N_{\text{atoms}} \times N_{\text{atoms}}$.
    pub bond_orders: AlignedMatrix<f64>,
    /// Total atomic valencies $V_A = 2 \sum_{\mu \in A} P_{\mu\mu} - \sum_{\mu \in A} \sum_{\nu \in A} P_{\mu\nu}^2$.
    pub valencies: Vec<f64>,
    /// Active charge used in bonding: $AQ_A = \sum_{B \neq A} B_{AB}$.
    pub active_charges: Vec<f64>,
    /// Self / inactive charge: $SQ_A = (V_A - AQ_A) / 2$.
    pub self_charges: Vec<f64>,
    /// Free valence remaining on atom: $FV_A = V_A - AQ_A$.
    pub free_valencies: Vec<f64>,
}

/// Compute Armstrong-Perkins-Stewart / Mayer bond orders and atomic valencies matching `bonds.F90`.
///
/// For closed-shell RHF systems with density matrix $P_{\mu\nu}$:
/// 1. The diatomic bond order index between atoms $A$ and $B$ is:
///    $$B_{AB} = \sum_{\mu \in A} \sum_{\nu \in B} P_{\mu\nu}^2$$
/// 2. The total atomic valency of atom $A$ is:
///    $$V_A = 2 \sum_{\mu \in A} P_{\mu\mu} - \sum_{\mu \in A} \sum_{\nu \in A} P_{\mu\nu}^2$$
/// 3. The free valence represents unshared or radical density: $FV_A = V_A - \sum_{B \neq A} B_{AB}$.
#[allow(clippy::needless_range_loop)]
pub fn compute_bond_orders(
    batch: &MolecularBatch,
    density: &AlignedMatrix<f64>,
) -> BondOrderResult {
    let natoms = batch.natoms;
    let mut bond_orders = AlignedMatrix::zeroed(natoms, natoms);
    let mut valencies = vec![0.0; natoms];
    let mut active_charges = vec![0.0; natoms];
    let mut self_charges = vec![0.0; natoms];
    let mut free_valencies = vec![0.0; natoms];

    // 1. Calculate intra-atomic valency V_A and inter-atomic bond order B_{AB}
    for a in 0..natoms {
        let la = batch.orbital_offsets[a];
        let norbs_a = batch.basis_types[a].num_orbitals();
        let lla = la + norbs_a;

        // Diagonal population sum: \sum_{\mu \in A} P_{\mu\mu}
        let mut sum_diag = 0.0;
        for mu in la..lla {
            sum_diag += density.get(mu, mu);
        }

        // Intra-atomic self-contraction: \sum_{\mu \in A}\sum_{\nu \in A} P_{\mu\nu}^2
        let mut self_term = 0.0;
        for mu in la..lla {
            for nu in la..lla {
                let p = density.get(mu, nu);
                self_term += p * p;
            }
        }

        // Valency V_A = 2.0 * \sum P_{\mu\mu} - self_term
        let va = 2.0 * sum_diag - self_term;
        valencies[a] = va;

        // Inter-atomic bond orders B_{AB} for B > A
        for b in (a + 1)..natoms {
            let lb = batch.orbital_offsets[b];
            let norbs_b = batch.basis_types[b].num_orbitals();
            let llb = lb + norbs_b;

            let mut b_ab = 0.0;
            for mu in la..lla {
                for nu in lb..llb {
                    let p = density.get(mu, nu);
                    b_ab += p * p;
                }
            }

            bond_orders.set(a, b, b_ab);
            bond_orders.set(b, a, b_ab);
        }
    }

    // 2. Compute active charges, free valency, and self charge
    for a in 0..natoms {
        let mut aq = 0.0;
        for b in 0..natoms {
            if a != b {
                aq += bond_orders.get(a, b);
            }
        }
        active_charges[a] = aq;
        free_valencies[a] = valencies[a] - aq;
        self_charges[a] = (valencies[a] - aq) * 0.5;
    }

    BondOrderResult {
        bond_orders,
        valencies,
        active_charges,
        self_charges,
        free_valencies,
    }
}
