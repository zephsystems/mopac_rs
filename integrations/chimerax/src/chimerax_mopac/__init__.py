# UCSF ChimeraX Bundle for MOPAC_RS.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# Enables in-viewport semi-empirical quantum calculations, AM1-BCC charge assignment,
# L-BFGS geometry optimization, and Molecular Electrostatic Potential (MEP) surface coloring.

from chimerax.core.toolshed import BundleInfo

class _MopacBundleInfo(BundleInfo):
    def initialize(self, session):
        super().initialize(session)
        from .cmd import register_mopac_commands
        register_mopac_commands(session)

bundle_info = _MopacBundleInfo()
