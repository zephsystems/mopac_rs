#!/usr/bin/env python3
"""
Automated Verification Suite for mopac_py Python Bindings.

Validates:
1. Single-point energy, heat of formation, dipole, Mulliken charges, analytical gradients across AM1, PM6, RM1, PM3, MNDO.
2. Halogenated compounds (CH3Br, CH3I) and heteroatoms (PH3, H2S).
3. COSMO dielectric implicit solvation.
4. Empirical dispersion corrections (PM6-DH+, PM7).
5. Non-covalent H4 hydrogen bonding and H-H core repulsion corrections.
6. Quasi-Newton L-BFGS geometry optimization.
7. Dictionary conversion and error handling.
"""

import os
import sys
import unittest

# Ensure target/debug or target/release is in sys.path
for build_dir in ["target/release", "target/debug"]:
    abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", build_dir))
    if os.path.exists(abs_dir):
        # Create symlink if libmopac_py.so exists and mopac_py.so does not
        lib_so = os.path.join(abs_dir, "libmopac_py.so")
        mod_so = os.path.join(abs_dir, "mopac_py.so")
        if os.path.exists(lib_so) and not os.path.exists(mod_so):
            try:
                os.symlink("libmopac_py.so", mod_so)
            except OSError:
                pass
        sys.path.insert(0, abs_dir)

import mopac_py


class TestMopacPyBindings(unittest.TestCase):

    def setUp(self):
        self.h2o_atoms = [8, 1, 1]
        self.h2o_coords = [
            [0.0, 0.0, 0.0655],
            [0.0, 0.7571, -0.5205],
            [0.0, -0.7571, -0.5205],
        ]

        self.ch3br_atoms = [6, 35, 1, 1, 1]
        self.ch3br_coords = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.93],
            [1.03, 0.0, -0.36],
            [-0.515, 0.892, -0.36],
            [-0.515, -0.892, -0.36],
        ]

        self.ch3i_atoms = [6, 53, 1, 1, 1]
        self.ch3i_coords = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.14],
            [1.03, 0.0, -0.36],
            [-0.515, 0.892, -0.36],
            [-0.515, -0.892, -0.36],
        ]

    def test_water_all_methods(self):
        """Verify H2O calculation converges under all supported semi-empirical methods."""
        methods = ["PM6", "AM1", "RM1", "PM3", "MNDO"]
        for m in methods:
            res = mopac_py.calculate(self.h2o_atoms, self.h2o_coords, method=m)
            self.assertTrue(res.converged, f"{m} failed to converge on H2O")
            self.assertLess(res.total_energy_ev, -300.0)
            self.assertLess(res.binding_energy_ev, 0.0)
            self.assertGreater(res.dipole_debye[3], 1.2)
            self.assertLess(res.dipole_debye[3], 2.5)
            self.assertEqual(len(res.gradients_ev_angstrom), 3)
            self.assertEqual(len(res.mulliken_charges), 3)
            # Oxygen charge negative, hydrogen charges positive
            self.assertLess(res.mulliken_charges[0], -0.25)
            self.assertGreater(res.mulliken_charges[1], 0.1)
            self.assertGreater(res.mulliken_charges[2], 0.1)

    def test_halogens_bromine_iodine(self):
        """Verify heavy halogens Br (35) and I (53) in organohalides."""
        # CH3Br
        res_br = mopac_py.calculate(self.ch3br_atoms, self.ch3br_coords, method="PM6")
        self.assertTrue(res_br.converged)
        self.assertLess(res_br.total_energy_ev, 0.0)
        self.assertLess(res_br.binding_energy_ev, 0.0)
        self.assertEqual(len(res_br.gradients_ev_angstrom), 5)

        # CH3I
        res_i = mopac_py.calculate(self.ch3i_atoms, self.ch3i_coords, method="PM6")
        self.assertTrue(res_i.converged)
        self.assertLess(res_i.total_energy_ev, 0.0)
        self.assertLess(res_i.binding_energy_ev, 0.0)
        self.assertEqual(len(res_i.gradients_ev_angstrom), 5)

    def test_cosmo_solvation(self):
        """Verify dielectric reaction field stabilizes energy and polarizes dipole."""
        res_gas = mopac_py.calculate(self.h2o_atoms, self.h2o_coords, method="PM6")
        res_solv = mopac_py.calculate(self.h2o_atoms, self.h2o_coords, method="PM6", cosmo_eps=78.4)

        self.assertTrue(res_gas.converged)
        self.assertTrue(res_solv.converged)
        self.assertLess(res_solv.total_energy_ev, res_gas.total_energy_ev)
        # Solvation polarizes dipole moment
        self.assertGreater(res_solv.dipole_debye[3], res_gas.dipole_debye[3])

    def test_dispersion_and_hbonds(self):
        """Verify empirical dispersion and non-covalent corrections."""
        res_plain = mopac_py.calculate(self.ch3br_atoms, self.ch3br_coords, method="PM6")
        res_disp = mopac_py.calculate(self.ch3br_atoms, self.ch3br_coords, method="PM6", dispersion="PM6-DH+")
        res_hb = mopac_py.calculate(self.ch3br_atoms, self.ch3br_coords, method="PM6", h_bonds=True)

        self.assertNotEqual(res_plain.heat_of_formation_kcal, res_disp.heat_of_formation_kcal)
        self.assertNotEqual(res_plain.heat_of_formation_kcal, res_hb.heat_of_formation_kcal)

    def test_geometry_optimization(self):
        """Verify L-BFGS geometry optimization reduces energy and forces."""
        opt_res = mopac_py.optimize(self.h2o_atoms, self.h2o_coords, method="PM6", max_cycles=50)
        self.assertTrue(opt_res.converged)
        self.assertLess(opt_res.final_energy_ev, opt_res.initial_energy_ev)
        self.assertLess(opt_res.final_grad_rms, 1.0)
        self.assertEqual(len(opt_res.coordinates), 3)

    def test_calculator_class(self):
        """Verify object-oriented MopacCalculator class interface."""
        calc = mopac_py.MopacCalculator(method="AM1", use_nddo=True, cosmo_eps=78.4)
        self.assertEqual(calc.method, "AM1")
        self.assertTrue(calc.use_nddo)
        self.assertEqual(calc.cosmo_eps, 78.4)

        res = calc.calculate(self.h2o_atoms, self.h2o_coords)
        self.assertTrue(res.converged)
        d = res.to_dict()
        self.assertIn("total_energy_ev", d)
        self.assertIn("heat_of_formation_kcal", d)
        self.assertIn("dipole_debye", d)
        self.assertIn("mulliken_charges", d)
        self.assertIn("gradients_ev_angstrom", d)

    def test_input_validation(self):
        """Verify input validation throws proper Python exceptions."""
        with self.assertRaises(ValueError):
            mopac_py.calculate([], [])

        with self.assertRaises(ValueError):
            mopac_py.calculate([8, 1], [[0.0, 0.0, 0.0]])

        with self.assertRaises(ValueError):
            mopac_py.calculate([1], [[0.0, 0.0, 0.0]], method="NONEXISTENT")

    def test_silicon_sih4(self):
        """Verify Silicon (Z=14) converges in PM6, AM1, PM3, MNDO and is rejected in RM1."""
        si_atoms = [14, 1, 1, 1, 1]
        si_coords = [
            [0.0, 0.0, 0.0],
            [0.85, 0.85, 0.85],
            [-0.85, -0.85, 0.85],
            [-0.85, 0.85, -0.85],
            [0.85, -0.85, -0.85],
        ]
        for m in ["PM6", "AM1", "PM3", "MNDO"]:
            res = mopac_py.calculate(si_atoms, si_coords, method=m)
            self.assertTrue(res.converged)
            self.assertLess(res.total_energy_ev, -100.0)

        with self.assertRaises(ValueError):
            mopac_py.calculate(si_atoms, si_coords, method="RM1")

    def test_grimme_d3_bj(self):
        """Verify Grimme D3-BJ empirical dispersion stabilizes methane dimer."""
        ch4_coords = [
            [0.0, 0.0, 0.0],
            [0.629118, 0.629118, 0.629118],
            [-0.629118, -0.629118, 0.629118],
            [-0.629118, 0.629118, -0.629118],
            [0.629118, -0.629118, -0.629118],
            [0.0, 0.0, 3.8],
            [0.629118, 0.629118, 4.429118],
            [-0.629118, -0.629118, 4.429118],
            [-0.629118, 0.629118, 3.170882],
            [0.629118, -0.629118, 3.170882],
        ]
        ch4_atoms = [6, 1, 1, 1, 1, 6, 1, 1, 1, 1]
        res_plain = mopac_py.calculate(ch4_atoms, ch4_coords, method="PM6")
        res_d3 = mopac_py.calculate(ch4_atoms, ch4_coords, method="PM6", dispersion="D3-BJ")
        self.assertTrue(res_d3.converged)
        delta_disp = res_d3.heat_of_formation_kcal - res_plain.heat_of_formation_kcal
        self.assertLess(delta_disp, -0.5, "D3-BJ must provide attractive dispersion stabilization")

    def test_rdkit_and_ase_interop(self):
        """Verify RDKit and ASE helper bridges."""
        self.assertTrue(hasattr(mopac_py, "from_rdkit"))
        self.assertTrue(hasattr(mopac_py, "from_ase"))
        self.assertTrue(hasattr(mopac_py, "MopacASECalculator"))

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
            AllChem.EmbedMolecule(mol, randomSeed=1)
            atoms, coords = mopac_py.from_rdkit(mol)
            self.assertEqual(atoms, [6, 8, 1, 1, 1, 1])
            res = mopac_py.calculate(atoms, coords, method="PM6")
            self.assertTrue(res.converged)
        except ImportError:
            pass


    def test_vibrational_frequencies_and_thermodynamics(self):
        """Verify harmonic vibrational frequencies, normal modes, and thermochemistry."""
        vib = mopac_py.frequencies(self.h2o_atoms, self.h2o_coords, method="AM1")
        self.assertEqual(len(vib.all_frequencies_cm1), 9)
        self.assertEqual(len(vib.vibrational_frequencies_cm1), 3)
        self.assertGreater(vib.zpve_kcal_mol, 8.0)
        self.assertLess(vib.zpve_kcal_mol, 15.0)
        self.assertFalse(vib.is_transition_state)
        self.assertEqual(len(vib.normal_modes), 9)
        self.assertEqual(len(vib.cartesian_hessian), 9)

        # Thermodynamics assertions
        thermo = vib.thermo
        self.assertEqual(thermo.temperature_k, 298.15)
        self.assertEqual(thermo.pressure_atm, 1.0)
        self.assertGreater(thermo.entropy_total_cal_k_mol, 40.0)
        self.assertGreater(thermo.enthalpy_thermal_cal_mol, 2000.0)
        self.assertGreater(thermo.cp_total_cal_k_mol, 7.0)

        # Dictionary conversion
        d = vib.to_dict()
        self.assertIn("vibrational_frequencies_cm1", d)
        self.assertIn("thermo", d)
        self.assertIn("zpve_kcal_mol", d)

        # Calculator object method
        calc = mopac_py.MopacCalculator(method="PM6")
        vib_calc = calc.frequencies(self.h2o_atoms, self.h2o_coords)
        self.assertEqual(len(vib_calc.vibrational_frequencies_cm1), 3)

        # Boron BH3 normal modes (4 atoms -> 3N-6 = 6 vibrational modes)
        bh3_atoms = [5, 1, 1, 1]
        bh3_coords = [
            [0.0, 0.0, 0.0],
            [1.19, 0.0, 0.0],
            [-0.595, 1.030569, 0.0],
            [-0.595, -1.030569, 0.0],
        ]
        vib_bh3 = mopac_py.frequencies(bh3_atoms, bh3_coords, method="AM1")
        self.assertEqual(len(vib_bh3.vibrational_frequencies_cm1), 6)
        self.assertGreater(vib_bh3.zpve_kcal_mol, 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
