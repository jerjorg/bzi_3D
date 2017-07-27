"""Unit tests for integration module."""

import numpy as np
import pytest
import itertools

from BZI.tetrahedron import *
from BZI.pseudopots import FreeElectronModel, Al_PP
from BZI.integration import rectangular_fermi_level
from BZI.symmetry import make_ptvecs, make_rptvecs, Lattice

# def test_find_tetrahedra():
#     vertices = np.array([[.5,0,0],[1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1],
#                          [0,1,1], [1,1,1]])

#     tetrahedra = np.sort([[2,4,1,8],
#                           [4,1,3,8],
#                           [1,3,8,7],
#                           [1,8,5,7],
#                           [1,6,8,5],
#                           [2,1,6,8]])
#     assert np.allclose(tetrahedra, find_tetrahedra(vertices))
    
#     vertices = np.array([[0,0,0],[.5,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1],
#                          [0,1,1], [1,1,1]])
#     tetrahedra = np.sort([[4,3,2,7],
#                           [3,2,1,7],
#                           [2,1,7,5],
#                           [2,7,6,5],
#                           [2,8,7,6],
#                           [4,2,8,7]])
#     assert np.allclose(tetrahedra, find_tetrahedra(vertices))

#     vertices = np.array([[0,0,0],[1,0,0], [.5,1,0], [1,1,0], [0,0,1], [1,0,1],
#                          [0,1,1], [1,1,1]])
#     tetrahedra = np.sort([[1,2,3,6],
#                           [2,3,4,6],
#                           [3,4,6,8],
#                           [3,6,7,8],
#                           [3,5,6,7],
#                           [1,3,5,6]])
#     assert np.allclose(tetrahedra, find_tetrahedra(vertices))

#     vertices = np.array([[0,0,0],[1,0,0], [0,1,0], [.5,1,0], [0,0,1], [1,0,1],
#                          [0,1,1], [1,1,1]])
#     tetrahedra = np.sort([[3,1,4,5],
#                           [1,4,2,5],
#                           [4,2,5,6],
#                           [4,5,8,6],
#                           [4,7,5,8],
#                           [3,4,7,5]])
#     assert np.allclose(tetrahedra, find_tetrahedra(vertices))

    
# def test_number_of_states():
#     VG = 11
#     VT = 2.5
#     energies = [1.3, 2.5, np.pi, 6.6]

#     e = 0    
#     assert np.isclose(number_of_states(VG, VT, energies, e), 0.)
    
#     e = 1.4
#     assert np.isclose(number_of_states(VG, VT, energies, e), 0.0000194042)
    
#     e = 2.7
#     assert np.isclose(number_of_states(VG, VT, energies, e), 0.0526692)

#     e = 3.8
#     assert np.isclose(number_of_states(VG, VT, energies, e), 0.160885)

#     e = 23./3
#     assert np.isclose(number_of_states(VG, VT, energies, e), 0.227273)    


# def test_integration_weights():
#     VG = 12.3
#     VT = 4./3
#     energies = [1.3, 2.8, np.pi, 5.65]

#     e = 1.1
#     weights =  [0.]*4
#     assert np.allclose(integration_weights(VT, energies, e), weights)
    
#     e = 1.9
#     weights = np.array([0.0015278, 0.000194856, 0.000158712, 0.0000671916])*(
#         VG)
#     assert np.allclose(integration_weights(VT, energies, e), weights)
    
#     e = 3.11
#     weights = np.array([0.0185508, 0.0139896, 0.0119783, 0.00535825])*VG
#     assert np.allclose(integration_weights(VT, energies, e), weights)

#     e = 5.07
#     weights = np.array([0.0270776, 0.0270657, 0.027061, 0.0265167])*VG
#     assert np.allclose(integration_weights(VT, energies, e), weights)

#     e = 19./3
#     weights = np.array([VT/(4*VG)]*4)*VG
#     assert np.allclose(integration_weights(VT, energies, e), weights)

# def test_density_of_states():

#     VG = 12.3
#     VT = 4./3
#     energies = [1.3, 2.8, np.pi, 5.65]

#     e = 1.1
#     dos =  0.
#     assert np.isclose(density_of_states(VG, VT, energies, e), dos)
    
#     e = 1.9
#     dos = 0.00974279
#     assert np.isclose(density_of_states(VG, VT, energies, e), dos)
    
#     e = 3.11
#     dos = 0.0672611
#     assert np.isclose(density_of_states(VG, VT, energies, e), dos)

#     e = 5.07
#     dos = 0.00351786
#     assert np.isclose(density_of_states(VG, VT, energies, e), dos)

#     e = 19./3
#     dos = 0.
#     assert np.isclose(density_of_states(VG, VT, energies, e), dos)

    
# def test_grid_and_tetrahedra():
#     grid, tetrahedra1 = grid_and_tetrahedra(Al_PP, [2,2,2], [0,0,0])

#     # These are the tetrahedra that would be created if the common diagonal
#     # were [1,8].
#     tet_map = np.sort([[2,4,1,8],
#                        [4,1,3,8],
#                        [1,3,8,7],
#                        [1,8,5,7],
#                        [1,6,8,5],
#                        [2,1,6,8]])
    
#     submesh_cells = np.array([[0,9,3,12,1,10,4,13],
#                               [1,10,4,13,2,11,5,14],
#                               [3,12,6,15,4,13,7,16],
#                               [4,13,7,16,5,14,8,17],
#                               [9,18,12,21,10,19,13,22],
#                               [10,19,13,22,11,20,14,23],
#                               [12,21,15,24,13,22,16,25],
#                               [13,22,16,25,14,23,17,26]])
    
#     tetrahedra2 = []
#     for i,submesh_cell in enumerate(submesh_cells):
#         for j, tmap in enumerate(tet_map):
#             tetrahedra2.append(submesh_cell[tmap-1])

#     for t1 in tetrahedra1:
#         contained = False
#         for t2 in tetrahedra2:
#             if np.allclose(t1,t2):
#                 contained = True
#         assert contained


#     def submesh_tetrahedra(vertices, indices):
#         """Create a function determines the indices of the tetrahedra
#         vertices from a list of vertices and indices.
#         """
            
#         tetrahedra = find_tetrahedra(vertices)
#         return indices[tetrahedra-1]        


#     ndiv_list = range(1, 5)
#     for ndiv in ndiv_list:
#         ndiv = 2
#         lat_angles =[np.pi/2]*3
#         lat_consts = [1]*3
#         lat_centering = "prim"
#         lattice = Lattice(lat_centering, lat_consts, lat_angles)
#         lat_shift = [1./2]*3

#         nvalence = 1
#         degree = 2
#         free = FreeElectronModel(lattice, nvalence, degree)
        
#         grid_shift = [0,0,0]
#         grid, tetrahedra = grid_and_tetrahedra(free, ndiv, lat_shift, grid_shift)

#         test_tetrahedra = np.array([])
#         indices = np.reshape(list(range(27)), [ndiv+1]*3)
#         for i,j,k in itertools.product(range(2), repeat=3):
#             vertices = []
#             vindices = []
#             for l,m,n in itertools.product(range(2), repeat=3):
#                 vertices.append(np.reshape(grid, [ndiv+1]*3)[i+l, j+m, k+n])
#                 vindices.append(indices[i+l, j+m, k+n])
#             vindices = np.array(vindices)
#             vertices = np.array(vertices)
#             if len(test_tetrahedra) == 0:
#                 test_tetrahedra = submesh_tetrahedra(vertices, vindices)
#             else:
#                 test_tetrahedra = np.append(test_tetrahedra,
#                                         submesh_tetrahedra(vertices, vindices), 0)
#         assert all([np.allclose(t1,t2) for t1,t2 in
#                     zip(tetrahedra, test_tetrahedra)])
    
# def test_find_irreducible_tetrahedra():
#     """Check that symmetry reduction is performed correcly by comparing the
#     value of the Fermi level with and without reduction.
#     """

#     ndiv0 = np.array([2,2,2])
#     offset = [0]*3
#     grid, tetrahedra = grid_and_tetrahedra(Al_PP, ndiv0, offset)
#     irreducible_tetrahedra, weights1 = find_irreducible_tetrahedra(Al_PP, tetrahedra, grid)
#     fermi_level1 = calc_fermi_level(Al_PP, irreducible_tetrahedra, weights1, grid)
#     weights2 = np.ones(len(tetrahedra))
#     fermi_level2 = calc_fermi_level(Al_PP, tetrahedra, weights2, grid)

#     print(fermi_level1)
#     print(fermi_level2)

#     assert np.isclose(fermi_level1, fermi_level2)

def test_integrals():
    degree_list = range(2,3)
    nvalence_list = [1]
    
    for degree in degree_list:
        print("degree ", degree)
        for nvalence in nvalence_list:
            print("nvalence ", nvalence)
            
            # Verify the Fermi level of the free electron model.
            lat_angles =[np.pi/2]*3
            lat_consts = [1]*3
            lat_centering = "prim"
            lattice = Lattice(lat_centering, lat_consts, lat_angles)
            nvalence = 1
            free = FreeElectronModel(lattice, nvalence, degree)

            lat_offset = [-1./2]*3
            grid, tetrahedra = grid_and_tetrahedra(free, 20, lat_offset)
            weights = np.ones(len(tetrahedra))
            free.fermi_level = calc_fermi_level(free, tetrahedra, weights, grid)

            sphere_volume = 4./3*np.pi*free.fermi_level**(3./degree)
            occupied_volume = free.lattice.reciprocal_volume*free.nvalence_electrons/2
            fl_answer = (3*occupied_volume/(4*np.pi))**(degree/3.)
    
            assert np.isclose(sphere_volume, occupied_volume, 1e-1, 1e-1)
            assert np.isclose(free.fermi_level, fl_answer, 1e-2,1e-2)

            total_energy = calc_total_energy(free, tetrahedra, weights, grid)
            rf_answer = fl_answer**(1./degree)
            te_answer = 4*np.pi*(rf_answer**(3 + degree)/(3. + degree))
            assert np.isclose(total_energy, te_answer, 1e-1, 1e-1)
