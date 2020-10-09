"""See the notebook with the empirical pseudopotential plots.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from bzi_3D.plots import plot_band_structure, plot_paths
from bzi_3D.pseudopots import *
from numpy.linalg import norm, inv, det
from conftest import run

tests = run("all pseudopotential")

@pytest.mark.skipif("test_pseudopotentials" not in tests, reason="different tests")
def test_pseudopotentials():
    assert True

    free_energy_shift = free_EPM.eval([0.]*3,1)[0]
    free_args = {"materials_list": ["free"],
               "PPlist": [free_EPM],
               "PPargs_list": [{"neigvals": 1}],
               "lattice": free_EPM.lattice,
               "npts": 1000,
               "neigvals": 1,
               "energy_shift": free_energy_shift,
               "energy_limits": [0,35],
               "show": True}

    # plot_band_structure(**free_args)

    Si_energy_shift = Si_EPM.eval([0.]*3,10)[1]
    Si_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    Si_args = {"materials_list": ["Si"],
               "PPlist": [Si_EPM],
               "PPargs_list": [{}],
               "lattice": Si_lattice,
               "npts": 100,
               "neigvals": 9,
               "energy_shift": Si_energy_shift,
               "energy_limits": [-5.5,6.5],
               "show": True,
               "save": False}

    # plot_band_structure(**Si_args)

    Ge_energy_shift = Ge_EPM.eval([0.]*3,10)[1]
    Ge_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    Ge_args = {"materials_list": ["Ge"],
               "PPlist": [Ge_EPM],
               "PPargs_list": [{}],
               "lattice": Ge_lattice,
               "npts": 100,
               "neigvals": 10,
               "energy_shift": Ge_energy_shift,
               "energy_limits": [-5,7],
               "save": True}

    # plot_band_structure(**Ge_args)

    cSn_energy_shift = cSn_EPM.eval([0.]*3,10)[1]
    cSn_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    cSn_args = {"materials_list": ["Sn"],
               "PPlist": [cSn_EPM],
               "PPargs_list": [{}],
               "lattice": cSn_lattice,
               "npts": 100,
               "neigvals": 10,
               "energy_shift": cSn_energy_shift,
               "energy_limits": [-4,6],
               "save": True}

    # plot_band_structure(**cSn_args)

    GaP_energy_shift = GaP_EPM.eval([0.]*3,10)[1]
    GaP_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    GaP_args = {"materials_list": ["GaP"],
               "PPlist": [GaP_EPM],
               "PPargs_list": [{}],
               "lattice": GaP_lattice,
               "npts": 100,
               "neigvals": 10,
               "energy_shift": GaP_energy_shift,
               "energy_limits": [-4,7]}

    # plot_band_structure(**GaP_args)

    GaAs_energy_shift = GaAs_EPM.eval([0.]*3,10)[1]
    GaAs_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    GaAs_args = {"materials_list": ["GaAs"],
               "PPlist": [GaAs_EPM],
               "PPargs_list": [{}],
               "lattice": GaAs_lattice,
               "npts": 100,
               "neigvals": 10,
               "energy_shift": GaAs_energy_shift,
               "energy_limits": [-4,7]}

    # plot_band_structure(**GaAs_args)

    AlSb_energy_shift = AlSb_EPM.eval([0.]*3,10)[1]
    AlSb_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    AlSb_args = {"materials_list": ["AlSb"],
               "PPlist": [AlSb_EPM],
               "PPargs_list": [{}],
               "lattice": AlSb_lattice,
               "npts": 50,
               "neigvals": 10,
               "energy_shift": AlSb_energy_shift,
               "energy_limits": [-4,7]}

    # plot_band_structure(**AlSb_args)

    InP_energy_shift = InP_EPM.eval([0.]*3,10)[1]
    InP_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    InP_args = {"materials_list": ["InP"],
               "PPlist": [InP_EPM],
               "PPargs_list": [{}],
               "lattice": InP_lattice,
               "npts": 50,
               "neigvals": 10,
               "energy_shift": InP_energy_shift,
               "energy_limits": [-4,7]}

    # plot_band_structure(**InP_args)
    
    GaSb_energy_shift = GaSb_EPM.eval([0.]*3,10)[1]
    GaSb_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    GaSb_args = {"materials_list": ["GaSb"],
               "PPlist": [GaSb_EPM],
               "PPargs_list": [{}],
               "lattice": GaSb_lattice,
               "npts": 50,
               "neigvals": 10,
               "energy_shift": GaSb_energy_shift,
               "energy_limits": [-3,6.5]}

    # plot_band_structure(**GaSb_args)

    InAs_energy_shift = InAs_EPM.eval([0.]*3,10)[1]
    InAs_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    InAs_args = {"materials_list": ["InAs"],
               "PPlist": [InAs_EPM],
               "PPargs_list": [{}],
               "lattice": InAs_lattice,
               "npts": 50,
               "neigvals": 10,
               "energy_shift": InAs_energy_shift,
               "energy_limits": [-4,7]}

    # plot_band_structure(**InAs_args)

    InSb_energy_shift = InSb_EPM.eval([0.]*3,10)[1]
    InSb_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    InSb_args = {"materials_list": ["InSb"],
               "PPlist": [InSb_EPM],
               "PPargs_list": [{}],
               "lattice": InSb_lattice,
               "npts": 50,
               "neigvals": 10,
               "energy_shift": InSb_energy_shift,
               "energy_limits": [-3,6]}

    # plot_band_structure(**InSb_args)
    
    ZnS_energy_shift = ZnS_EPM.eval([0.]*3,10)[1]
    ZnS_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    ZnS_args = {"materials_list": ["ZnS"],
               "PPlist": [ZnS_EPM],
               "PPargs_list": [{}],
               "lattice": ZnS_lattice,
               "npts": 50,
               "neigvals": 10,
               "energy_shift": ZnS_energy_shift,
               "energy_limits": [-3,10]}

    # plot_band_structure(**ZnS_args)
    
    ZnSe_energy_shift = ZnSe_EPM.eval([0.]*3,10)[1]
    ZnSe_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    ZnSe_args = {"materials_list": ["ZnSe"],
               "PPlist": [ZnSe_EPM],
               "PPargs_list": [{}],
               "lattice": ZnSe_lattice,
               "npts": 50,
               "neigvals": 10,
               "energy_shift": ZnSe_energy_shift,
               "energy_limits": [-3,9]}

    # plot_band_structure(**ZnSe_args)

    ZnTe_energy_shift = ZnTe_EPM.eval([0.]*3,10)[1]
    ZnTe_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    ZnTe_args = {"materials_list": ["ZnTe"],
               "PPlist": [ZnTe_EPM],
               "PPargs_list": [{}],
               "lattice": ZnTe_lattice,
               "npts": 50,
               "neigvals": 10,
               "energy_shift": ZnTe_energy_shift,
               "energy_limits": [-3,8.5]}

    # plot_band_structure(**ZnTe_args)

    CdTe_energy_shift = CdTe_EPM.eval([0.]*3,10)[1]
    CdTe_EPM.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    CdTe_args = {"materials_list": ["CdTe"],
               "PPlist": [CdTe_EPM],
               "PPargs_list": [{}],
               "lattice": CdTe_lattice,
               "npts": 50,
               "neigvals": 10,
               "energy_shift": CdTe_energy_shift,
               "energy_limits": [-3.5,8]}

    # plot_band_structure(**CdTe_args)

    Li_energy_shift = 3
    Li_params = {"materials_list": ["Li"],
                 "PPlist": [Li_EPM],
                 "PPargs_list": [{}],
                 "lattice": Li_lattice,
                 "npts": 500,
                 "neigvals": 10,
                 "energy_shift": Li_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True}

    # plot_band_structure(**Li_params)

    Al_energy_shift = 10
    Al_params = {"materials_list": ["Al"],
                 "PPlist": [Al_EPM],
                 "PPargs_list": [{}],
                 "lattice": Al_lattice,
                 "npts": 500,
                 "neigvals": 10,
                 "energy_shift": Al_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True,
                 "save": True}

    # plot_band_structure(**Al_params)

    Al_energy_shift = 10
    Al_params = {"materials_list": ["Al"],
                 "PPlist": [Al_EPM],
                 "PPargs_list": [{}],
                 "lattice": Al_lattice,
                 "npts": 50,
                 "neigvals": 10,
                 "energy_shift": Al_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True,
                 "save": True}

    # plot_band_structure(**Al_params)

    Na_energy_shift = 4
    Na_params = {"materials_list": ["Na"],
                 "PPlist": [Na_EPM],
                 "PPargs_list": [{}],
                 "lattice": Na_lattice,
                 "npts": 1000,
                 "neigvals": 10,
                 "energy_shift": Na_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True,
                 "save": False}

    # plot_band_structure(**Na_params)

    K_energy_shift = 2.4
    K_params = {"materials_list": ["K"],
                 "PPlist": [K_EPM],
                 "PPargs_list": [{}],
                 "lattice": K_lattice,
                 "npts": 1000,
                 "neigvals": 15,
                 "energy_shift": K_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True,           
                 "save": True}

    # plot_band_structure(**K_params)

    # Rb was a very rough estimate
    Rb_energy_shift = 2.2
    Rb_params = {"materials_list": ["Rb"],
                 "PPlist": [Rb_EPM],
                 "PPargs_list": [{}],
                 "lattice": Rb_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": Rb_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True,
                 "save": True}

    # plot_band_structure(**Rb_params)

    # Cs doesn't look right.

    Cs_energy_shift = 3.
    Cs_params = {"materials_list": ["Cs"],
                 "PPlist": [Cs_EPM],
                 "PPargs_list": [{}],
                 "lattice": Cs_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": Cs_energy_shift,
                 "energy_limits": [-5,10],
                 "show": True,
                 "save": True}

    # plot_band_structure(**Cs_params)

    # Cu doesn't look right.
    Cu_energy_shift = 4
    Cu_params = {"materials_list": ["Cu"],
                 "PPlist": [Cu_EPM],
                 "PPargs_list": [{}],
                 "lattice": Cu_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": Cu_energy_shift,
                 "energy_limits": [-6,25],
                 "show": True,
                 "save": True}             

    # plot_band_structure(**Cu_params)

    # Ag doesn't look correct.
    Ag_energy_shift = 4.
    Ag_params = {"materials_list": ["Ag"],
                 "PPlist": [Ag_EPM],
                 "PPargs_list": [{}],
                 "lattice": Ag_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": Ag_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True}

    # plot_band_structure(**Ag_params)

    # Au doesn't look right.
    Au_energy_shift = 4.
    Au_params = {"materials_list": ["Au"],
                 "PPlist": [Au_EPM],
                 "PPargs_list": [{}],
                 "lattice": Au_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": Au_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True}

    # plot_band_structure(**Au_params)

    Pb_energy_shift = 10.
    Pb_params = {"materials_list": ["Pb"],
                 "PPlist": [Pb_EPM],
                 "PPargs_list": [{}],
                 "lattice": Pb_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": Pb_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True}

    # plot_band_structure(**Pb_params)

    Mg_energy_shift =6.
    Mg_params = {"materials_list": ["Mg"],
                 "PPlist": [Mg_EPM],
                 "PPargs_list": [{}],
                 "lattice": Mg_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": Mg_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True,
                 "save": True}
    
    # plot_band_structure(**Mg_params)

    Zn_energy_shift = 10.
    Zn_params = {"materials_list": ["Zn"],
                 "PPlist": [Zn_EPM],
                 "PPargs_list": [{}],
                 "lattice": Zn_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": Zn_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True,
                 "save": True}

    # plot_band_structure(**Zn_params)

    Cd_energy_shift = 10.
    Cd_params = {"materials_list": ["Cd"],
                 "PPlist": [Cd_EPM],
                 "PPargs_list": [{}],
                 "lattice": Cd_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": Cd_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True}
    
    # plot_band_structure(**Cd_params)

    Hg_energy_shift = 10.
    Hg_params = {"materials_list": ["Hg"],
                 "PPlist": [Hg_EPM],
                 "PPargs_list": [{}],
                 "lattice": Hg_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": Hg_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True}

    # plot_band_structure(**Hg_params)

    In_energy_shift = 10.
    In_params = {"materials_list": ["In"],
                 "PPlist": [In_EPM],
                 "PPargs_list": [{}],
                 "lattice": In_lattice,
                 "npts": 1000,
                 "neigvals": 12,
                 "energy_shift": In_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True}

    # plot_band_structure(**In_params)

    Sn_energy_shift = 10.
    Sn_params = {"materials_list": ["Sn"],
                 "PPlist": [Sn_EPM],
                 "PPargs_list": [{}],
                 "lattice": Sn_lattice,
                 "npts": 100,
                 "neigvals": 12,
                 "energy_shift": Sn_energy_shift,
                 "energy_limits": [-10,10],
                 "show": True}

    # plot_band_structure(**Sn_params)
