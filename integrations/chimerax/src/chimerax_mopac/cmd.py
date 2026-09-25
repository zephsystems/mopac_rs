# ChimeraX Command Implementations for MOPAC_RS.
#
# Licensed under the Apache License, Version 2.0 (the "License").

import mopac_py

def mopac_calculate(session, atoms_spec, method="PM6", dispersion="D3-BJ", cosmo_eps=None):
    """Run single-point quantum calculation on selected atoms."""
    atoms = atoms_spec.atoms
    if len(atoms) == 0:
        session.logger.warning("No atoms selected for MOPAC calculation.")
        return

    atomic_numbers = [atom.element.number for atom in atoms]
    coords = [list(atom.coord) for atom in atoms]

    res = mopac_py.calculate(
        atomic_numbers,
        coords,
        method=method,
        dispersion=dispersion,
        cosmo_eps=cosmo_eps,
        use_nddo=True
    )

    # Assign charges to atoms for surface coloring
    for i, atom in enumerate(atoms):
        atom.charge = res.mulliken_charges[i]

    session.logger.info(
        f"[MOPAC_RS] Total Energy: {res.total_energy_ev:.5f} eV | "
        f"Heat of Formation: {res.heat_of_formation_kcal:.2f} kcal/mol | "
        f"Dipole: {res.dipole_debye[3]:.3f} D"
    )

def mopac_bcc(session, atoms_spec, method="AM1"):
    """Assign AM1-BCC partial atomic charges and prepare MEP surface."""
    atoms = atoms_spec.atoms
    if len(atoms) == 0:
        session.logger.warning("No atoms selected for AM1-BCC charges.")
        return

    atomic_numbers = [atom.element.number for atom in atoms]
    coords = [list(atom.coord) for atom in atoms]

    bcc_res = mopac_py.am1_bcc(atomic_numbers, coords, method=method)

    for i, atom in enumerate(atoms):
        atom.charge = bcc_res.bcc_charges[i]

    session.logger.info(
        f"[MOPAC_RS AM1-BCC] Assigned {len(atoms)} AM1-BCC partial charges. "
        f"Total charge = {bcc_res.total_charge:.6f} e."
    )

def mopac_optimize(session, atoms_spec, method="PM6", max_cycles=100):
    """Perform in-place L-BFGS geometry optimization on selected atoms."""
    atoms = atoms_spec.atoms
    if len(atoms) == 0:
        session.logger.warning("No atoms selected for geometry optimization.")
        return

    atomic_numbers = [atom.element.number for atom in atoms]
    coords = [list(atom.coord) for atom in atoms]

    opt_res = mopac_py.optimize(
        atomic_numbers,
        coords,
        method=method,
        max_cycles=max_cycles
    )

    for i, atom in enumerate(atoms):
        atom.coord = opt_res.final_coordinates[i]

    session.logger.info(
        f"[MOPAC_RS OPT] Converged in {opt_res.iterations} cycles | "
        f"Final Energy: {opt_res.final_energy_ev:.5f} eV"
    )

def mopac_mozyme(session, atoms_spec, method="PM6"):
    """Execute linear-scaling MOZYME SCF on macromolecule."""
    atoms = atoms_spec.atoms
    if len(atoms) == 0:
        session.logger.warning("No atoms selected for MOZYME.")
        return

    atomic_numbers = [atom.element.number for atom in atoms]
    coords = [list(atom.coord) for atom in atoms]

    moz_res = mopac_py.mozyme(atomic_numbers, coords, method=method)

    for i, atom in enumerate(atoms):
        atom.charge = moz_res.atomic_charges[i]

    session.logger.info(
        f"[MOPAC_RS MOZYME] Macromolecular SCF Converged in {moz_res.iterations} macro-sweeps | "
        f"Heat of Formation: {moz_res.heat_of_formation_kcal:.2f} kcal/mol"
    )

def register_mopac_commands(session):
    """Register 'mopac' CLI commands within ChimeraX command system."""
    from chimerax.core.commands import register, CmdDesc, AtomsArg, StringArg, FloatArg, IntArg

    desc_calc = CmdDesc(
        required=[("atoms_spec", AtomsArg)],
        keyword=[
            ("method", StringArg),
            ("dispersion", StringArg),
            ("cosmo_eps", FloatArg),
        ],
        synopsis="Run MOPAC_RS quantum calculation and assign Mulliken charges"
    )
    register("mopac calculate", desc_calc, mopac_calculate)

    desc_bcc = CmdDesc(
        required=[("atoms_spec", AtomsArg)],
        keyword=[("method", StringArg)],
        synopsis="Calculate and assign AM1-BCC partial atomic charges"
    )
    register("mopac bcc", desc_bcc, mopac_bcc)

    desc_opt = CmdDesc(
        required=[("atoms_spec", AtomsArg)],
        keyword=[
            ("method", StringArg),
            ("max_cycles", IntArg),
        ],
        synopsis="Optimize atomic coordinates using MOPAC_RS L-BFGS minimizer"
    )
    register("mopac optimize", desc_opt, mopac_optimize)

    desc_moz = CmdDesc(
        required=[("atoms_spec", AtomsArg)],
        keyword=[("method", StringArg)],
        synopsis="Run linear-scaling MOZYME calculation on macromolecular structure"
    )
    register("mopac mozyme", desc_moz, mopac_mozyme)
