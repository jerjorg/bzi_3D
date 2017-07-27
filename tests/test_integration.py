"""Unit tests for integration module."""

import numpy as np
import pytest
from BZI.pseudopots import FreeElectronModel
from BZI.integration import *
from BZI.symmetry import Lattice, make_ptvecs, make_rptvecs
from BZI.sampling import make_grid

def test_rectangular():
    """This will test the rectangular methods of finding the total energy and the
    Fermi level.
    """

    degree_list = range(1,5)
    for degree in degree_list:
        # Verify the Fermi level of the free electron model.
        lat_angles =[np.pi/2]*3
        lat_consts = [1]*3
        lat_centering = "prim"
        lattice = Lattice(lat_centering, lat_consts, lat_angles)
        
        free = FreeElectronModel(lattice, degree)
        
        grid_consts = [40]*3
        grid_angles = [np.pi/2]*3
        grid_centering = "prim"
        grid_vecs = make_ptvecs(grid_centering, grid_consts, grid_angles)
        rgrid_vecs = make_rptvecs(grid_vecs)
        offset = -np.dot(np.linalg.inv(rgrid_vecs), 
                         np.dot(lattice.reciprocal_vectors, [.5]*3))
        grid = make_grid(free.lattice.reciprocal_vectors, rgrid_vecs, offset)

        free.fermi_level = rectangular_fermi_level(free, grid)
        sphere_volume = 4./3*np.pi*free.fermi_level**(3./degree)
        occupied_volume = free.lattice.reciprocal_volume*free.nvalence_electrons/2
        fl_answer = (3*occupied_volume/(4*np.pi))**(degree/3.)

        print("degree ", degree)
        print("shere volume ", sphere_volume)
        print("occupied_volume ", occupied_volume)
        
        assert np.isclose(sphere_volume, occupied_volume, 1e-1, 1e-1)
        assert np.isclose(free.fermi_level, fl_answer, 1e-2,1e-2)
        
        total_energy = rectangular_method(free, grid)
        rf = free.fermi_level**(1./degree)
        te_answer = 4*np.pi*(rf**(3 + degree)/(3. + degree))
        assert np.isclose(total_energy, te_answer, 1e-1, 1e-1)
