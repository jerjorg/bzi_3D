"""Unit tests for pseudopots.py module.
"""

import pytest
import numpy as np
from BZI.pseudopots import *
from BZI.plots import plot_band_structure

def test_pseudopots():


    Si_energy_shift = Si_PP.eval([0.]*3,10)[1]
    Si_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    Si_args = {"materials_list": ["Si"],
               "PPlist": [Si_PP],
               "PPargs_list": [{}],
               "lattice": Si_lattice,
               "npts": 3,
               "neigvals": 10,
               "energy_shift": Si_energy_shift,
               "energy_limits": [-5.5,6.5],
               "show": False}

    plot_band_structure(**Si_args)
    
    Ge_energy_shift = Ge_PP.eval([0.]*3,10)[1]
    Ge_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    Ge_args = {"materials_list": ["Ge"],
               "PPlist": [Ge_PP],
               "PPargs_list": [{}],
               "lattice": Ge_lattice,
               "npts": 3,
               "neigvals": 10,
               "energy_shift": Ge_energy_shift,
               "energy_limits": [-5,7],
               "show": False}
    
    plot_band_structure(**Ge_args)

    cSn_energy_shift = Sn_PP.eval([0.]*3,10)[1]
    cSn_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    cSn_args = {"materials_list": ["Sn"],
               "PPlist": [cSn_PP],
               "PPargs_list": [{}],
               "lattice": cSn_lattice,
               "npts": 3,
               "neigvals": 10,
               "energy_shift": cSn_energy_shift,
               "energy_limits": [-4,6],
               "show": False}

    plot_band_structure(**cSn_args)

    GaP_energy_shift = GaP_PP.eval([0.]*3,10)[1]
    GaP_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    GaP_args = {"materials_list": ["GaP"],
                "PPlist": [GaP_PP],
                "PPargs_list": [{}],
                "lattice": GaP_lattice,
                "npts": 3,
                "neigvals": 10,
                "energy_shift": GaP_energy_shift,
                "energy_limits": [-4,7],
                "show": False}
    
    plot_band_structure(**GaP_args)

    GaAs_energy_shift = GaAs_PP.eval([0.]*3,10)[1]
    GaAs_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    GaAs_args = {"materials_list": ["GaAs"],
                 "PPlist": [GaAs_PP],
                 "PPargs_list": [{}],
                 "lattice": GaAs_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": GaAs_energy_shift,
                 "energy_limits": [-4,7],
                 "show": False}

    plot_band_structure(**GaAs_args)    

    AlSb_energy_shift = AlSb_PP.eval([0.]*3,10)[1]
    AlSb_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    AlSb_args = {"materials_list": ["AlSb"],
                 "PPlist": [AlSb_PP],
                 "PPargs_list": [{}],
                 "lattice": AlSb_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": AlSb_energy_shift,
                 "energy_limits": [-4,7],
                 "show": False}
    
    plot_band_structure(**AlSb_args)
    
    InP_energy_shift = InP_PP.eval([0.]*3,10)[1]
    InP_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    InP_args = {"materials_list": ["InP"],
                "PPlist": [InP_PP],
                "PPargs_list": [{}],
                "lattice": InP_lattice,
                "npts": 3,
                "neigvals": 10,
                "energy_shift": InP_energy_shift,
                "energy_limits": [-4,7],
                "show": False}

    plot_band_structure(**InP_args)
    
    GaSb_energy_shift = GaSb_PP.eval([0.]*3,10)[1]
    GaSb_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    GaSb_args = {"materials_list": ["GaSb"],
                 "PPlist": [GaSb_PP],
                 "PPargs_list": [{}],
                 "lattice": GaSb_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": GaSb_energy_shift,
                 "energy_limits": [-3,6.5],
                 "show": False}

    plot_band_structure(**GaSb_args)

    InAs_energy_shift = InAs_PP.eval([0.]*3,10)[1]
    InAs_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    InAs_args = {"materials_list": ["InAs"],
                 "PPlist": [InAs_PP],
                 "PPargs_list": [{}],
                 "lattice": InAs_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": InAs_energy_shift,
                 "energy_limits": [-4,7],
                 "show": False}

    plot_band_structure(**InAs_args)    

    InSb_energy_shift = InSb_PP.eval([0.]*3,10)[1]
    InSb_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    InSb_args = {"materials_list": ["InSb"],
                 "PPlist": [InSb_PP],
                 "PPargs_list": [{}],
                 "lattice": InSb_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": InSb_energy_shift,
                 "energy_limits": [-3,6],
                 "show": False}

    plot_band_structure(**InSb_args)    

    ZnS_energy_shift = ZnS_PP.eval([0.]*3,10)[1]
    ZnS_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    ZnS_args = {"materials_list": ["ZnS"],
                "PPlist": [ZnS_PP],
                "PPargs_list": [{}],
                "lattice": ZnS_lattice,
                "npts": 3,
                "neigvals": 10,
                "energy_shift": ZnS_energy_shift,
                "energy_limits": [-3,10],
                "show": False}

    plot_band_structure(**ZnS_args)

    ZnSe_energy_shift = ZnSe_PP.eval([0.]*3,10)[1]
    ZnSe_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    ZnSe_args = {"materials_list": ["ZnSe"],
                 "PPlist": [ZnSe_PP],
                 "PPargs_list": [{}],
                 "lattice": ZnSe_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": ZnSe_energy_shift,
                 "energy_limits": [-3,9],
                 "show": False}

    plot_band_structure(**ZnSe_args)

    ZnTe_energy_shift = ZnTe_PP.eval([0.]*3,10)[1]
    ZnTe_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    ZnTe_args = {"materials_list": ["ZnTe"],
                 "PPlist": [ZnTe_PP],
                 "PPargs_list": [{}],
                 "lattice": ZnTe_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": ZnTe_energy_shift,
                 "energy_limits": [-3,8.5],
                 "show": False}

    plot_band_structure(**ZnTe_args)

    CdTe_energy_shift = CdTe_PP.eval([0.]*3,10)[1]
    CdTe_PP.lattice.symmetry_paths = [["L", "G"],["G", "X"],["X", "U"],["U", "G2"]]
    CdTe_args = {"materials_list": ["CdTe"],
                 "PPlist": [CdTe_PP],
                 "PPargs_list": [{}],
                 "lattice": CdTe_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": CdTe_energy_shift,
                 "energy_limits": [-3.5,8],
                 "show": False}

    plot_band_structure(**CdTe_args)

    # The shifts that follow are random.
    Li_energy_shift = 3.7
    Li_params = {"materials_list": ["Li"],
                 "PPlist": [Li_PP],
                 "PPargs_list": [{}],
                 "lattice": Li_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": Li_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}
    
    plot_band_structure(**Li_params)

    Al_energy_shift = 12.
    Al_params = {"materials_list": ["Al"],
                 "PPlist": [Al_PP],
                 "PPargs_list": [{}],
                 "lattice": Al_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": Al_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Al_params)

    Na_energy_shift = 4
    Na_params = {"materials_list": ["Na"],
                 "PPlist": [Na_PP],
                 "PPargs_list": [{}],
                 "lattice": Na_lattice,
                 "npts": 3,
                 "neigvals": 10,
                 "energy_shift": Na_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Na_params)

    K_energy_shift = 2
    K_params = {"materials_list": ["K"],
                 "PPlist": [K_PP],
                 "PPargs_list": [{}],
                 "lattice": K_lattice,
                 "npts": 3,
                 "neigvals": 15,
                 "energy_shift": K_energy_shift,
                 "energy_limits": [-2.5,5],
                 "show": False}
    
    plot_band_structure(**K_params)

    # Rb doesn't look right. Maybe the lattice is wrong (constant or type).
    # It could also be a different empirical pseudopotential formalism.
    Rb_energy_shift = 2
    Rb_params = {"materials_list": ["Rb"],
                 "PPlist": [Rb_PP],
                 "PPargs_list": [{}],
                 "lattice": Rb_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Rb_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Rb_params)

    # Cs doesn't look right.
    Cs_energy_shift = 3.
    Cs_params = {"materials_list": ["Cs"],
                 "PPlist": [Cs_PP],
                 "PPargs_list": [{}],
                 "lattice": Cs_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Cs_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Cs_params)

    # Cu doesn't look right.
    Cu_energy_shift = 4
    Cu_params = {"materials_list": ["Cu"],
                 "PPlist": [Cu_PP],
                 "PPargs_list": [{}],
                 "lattice": Cu_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Cu_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Cu_params)

    # Ag doesn't look correct.
    Ag_energy_shift = 4.
    Ag_params = {"materials_list": ["Ag"],
                 "PPlist": [Ag_PP],
                 "PPargs_list": [{}],
                 "lattice": Ag_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Ag_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Ag_params)

    # Au doesn't look right.
    Au_energy_shift = 4.
    Au_params = {"materials_list": ["Au"],
                 "PPlist": [Au_PP],
                 "PPargs_list": [{}],
                 "lattice": Au_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Au_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Au_params)

    # Pb_energy_shift = Pb_PP.eval([0.]*3,10)[1]
    Pb_energy_shift = 4.
    Pb_params = {"materials_list": ["Pb"],
                 "PPlist": [Pb_PP],
                 "PPargs_list": [{}],
                 "lattice": Pb_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Pb_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Pb_params)

    Mg_energy_shift = 6.
    Mg_params = {"materials_list": ["Mg"],
                 "PPlist": [Mg_PP],
                 "PPargs_list": [{}],
                 "lattice": Mg_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Mg_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Mg_params)

    Zn_energy_shift = 10.
    Zn_params = {"materials_list": ["Zn"],
                 "PPlist": [Zn_PP],
                 "PPargs_list": [{}],
                 "lattice": Zn_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Zn_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Zn_params)

    Cd_energy_shift = 10.
    Cd_params = {"materials_list": ["Cd"],
                 "PPlist": [Cd_PP],
                 "PPargs_list": [{}],
                 "lattice": Cd_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Cd_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Cd_params)

    Hg_energy_shift = 10.
    Hg_params = {"materials_list": ["Hg"],
                 "PPlist": [Hg_PP],
                 "PPargs_list": [{}],
                 "lattice": Hg_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Hg_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Hg_params)

    In_energy_shift = 10.
    In_params = {"materials_list": ["In"],
                 "PPlist": [In_PP],
                 "PPargs_list": [{}],
                 "lattice": In_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": In_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**In_params)

    Sn_energy_shift = 10.
    Sn_params = {"materials_list": ["Sn"],
                 "PPlist": [Sn_PP],
                 "PPargs_list": [{}],
                 "lattice": Sn_lattice,
                 "npts": 3,
                 "neigvals": 12,
                 "energy_shift": Sn_energy_shift,
                 "energy_limits": [-10,10],
                 "show": False}

    plot_band_structure(**Sn_params)    
