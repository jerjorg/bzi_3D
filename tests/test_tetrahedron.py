"""Unit tests for tetrahedron module."""

import numpy as np
import pytest
import itertools

from BZI.tetrahedron import *
from BZI.pseudopots import FreeElectronModel, Al_PP
from BZI.integration import rectangular_fermi_level
from BZI.symmetry import make_ptvecs, make_rptvecs, Lattice
from conftest import run

tests = run("all tetrahedra")
@pytest.mark.skipif("test_find_tetrahedra" not in tests, reason="different tests")
def test_find_tetrahedra():
    vertices = np.array([[.5,0,0],[1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1],
                         [0,1,1], [1,1,1]])

    tetrahedra = np.sort([[2,4,1,8],
                          [4,1,3,8],
                          [1,3,8,7],
                          [1,8,5,7],
                          [1,6,8,5],
                          [2,1,6,8]])
    assert np.allclose(tetrahedra, find_tetrahedra(vertices))
    
    vertices = np.array([[0,0,0],[.5,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1],
                         [0,1,1], [1,1,1]])
    tetrahedra = np.sort([[4,3,2,7],
                          [3,2,1,7],
                          [2,1,7,5],
                          [2,7,6,5],
                          [2,8,7,6],
                          [4,2,8,7]])
    assert np.allclose(tetrahedra, find_tetrahedra(vertices))

    vertices = np.array([[0,0,0],[1,0,0], [.5,1,0], [1,1,0], [0,0,1], [1,0,1],
                         [0,1,1], [1,1,1]])
    tetrahedra = np.sort([[1,2,3,6],
                          [2,3,4,6],
                          [3,4,6,8],
                          [3,6,7,8],
                          [3,5,6,7],
                          [1,3,5,6]])
    assert np.allclose(tetrahedra, find_tetrahedra(vertices))

    vertices = np.array([[0,0,0],[1,0,0], [0,1,0], [.5,1,0], [0,0,1], [1,0,1],
                         [0,1,1], [1,1,1]])
    tetrahedra = np.sort([[3,1,4,5],
                          [1,4,2,5],
                          [4,2,5,6],
                          [4,5,8,6],
                          [4,7,5,8],
                          [3,4,7,5]])
    assert np.allclose(tetrahedra, find_tetrahedra(vertices))
    
@pytest.mark.skipif("test_number_of_states" not in tests, reason="different tests")
def test_number_of_states():
    VG = np.pi*30
    VT = 2.5
    energies = [1.3, 2.5, np.pi, 6.6]

    e = 0    
    assert np.isclose(number_of_states(VG, VT, energies, e), 0.)
    
    e = 1.4
    nos = 0.00021344663828890635/VG*2
    assert np.isclose(number_of_states(VG, VT, energies, e), nos)
    
    e = 2.7
    nos = 0.5793617163683986/VG*2
    assert np.isclose(number_of_states(VG, VT, energies, e), nos)

    e = 3.8
    nos = 1.7697387918381358/VG*2
    assert np.isclose(number_of_states(VG, VT, energies, e), nos)

    e = 23./3
    nos = VT/VG*2
    assert np.isclose(number_of_states(VG, VT, energies, e), nos)

    VT = 2
    energies = [2,4,6,8]
    assert number_of_states(VG, VT, energies, 1) == 0
    assert number_of_states(VG, VT, energies, 3) == (VT/VG)/48*2
    assert number_of_states(VG, VT, energies, 5) == (VT/VG)/2.*2
    assert np.isclose(number_of_states(VG, VT, energies, 7), (47.*VT)/(VG*48.)*2.)
    assert number_of_states(VG, VT, energies, 9) == (VT/VG)*2

    VT = np.log(3)
    energies = [1.3, 4.8, 6.7, 8.9]
    assert number_of_states(VG, VT, energies, np.sqrt(2)/2) == 0
    assert np.isclose(number_of_states(VG, VT, energies, 2*np.sqrt(2)), 0.0248576*(VT/VG)*2)
    assert np.isclose(number_of_states(VG, VT, energies, 4*np.sqrt(2)), 0.552689*(VT/VG)*2)
    assert np.isclose(number_of_states(VG, VT, energies, 5*np.sqrt(2)), 0.910757*(VT/VG)*2)
    assert number_of_states(VG, VT, energies, 7*np.sqrt(2)) == (VT/VG)*2
    

@pytest.mark.skipif("test_integration_weights" not in tests, reason="different tests")
def test_integration_weights():
    VT = 1.
    energies = [0.221562, 0.771927, 0.907994, 0.989497]

    ef = 0.1384938485930
    weights0 = [0.]*4
    weights1 = integration_weights(VT, energies, ef)
    assert np.allclose(weights0, weights1)

    ef = 0.33216584858
    weights0 = np.array([0.0040737, 0.000234316, 0.000187869, 0.00016793])
    weights1 = integration_weights(VT, energies, ef)
    assert np.allclose(weights0, weights1)

    ef = 0.8847473833938
    weights0 = np.array([0.247136, 0.239911, 0.224564, 0.205636])
    weights1 = integration_weights(VT, energies, ef)
    assert np.allclose(weights0, weights1)

    ef = 0.959475937294
    weights0 = np.array([0.249981, 0.249931, 0.249817, 0.248284])
    weights1 = integration_weights(VT, energies, ef)

    ef = 0.99837499374
    weights0 = [VT/4.]*4
    weights1 = integration_weights(VT, energies, ef)
    assert np.allclose(weights0, weights1)
    

@pytest.mark.skipif("test_density_of_states" not in tests, reason="different tests")
def test_density_of_states():
    # Didn't included primitive cell volume in calculation and I believe the weights
    # in Blochl's paper didn't include spin degeneracy, so all weights scaled by 2.
    VG = np.pi
    VT = 4./3
    energies = [1.3, 2.8, np.pi, 5.65]

    e = 1.1
    dos =  0.
    assert np.isclose(density_of_states(VG, VT, energies, e), dos)
    
    e = 1.9
    dos = 0.11983630296429897/VG*2
    assert np.isclose(density_of_states(VG, VT, energies, e), dos)
    
    e = 3.11
    dos = 0.827311611858529/VG*2
    assert np.isclose(density_of_states(VG, VT, energies, e), dos)

    e = 5.07
    dos = 0.04326969136461653/VG*2
    assert np.isclose(density_of_states(VG, VT, energies, e), dos)

    e = 19./3
    dos = 0.
    assert np.isclose(density_of_states(VG, VT, energies, e), dos)

    
@pytest.mark.skipif("test_grid_and_tetrahedra" not in tests, reason="different tests")
def test_grid_and_tetrahedra():
    grid, tetrahedra1 = grid_and_tetrahedra(Al_PP, [2,2,2], [0,0,0])

    # These are the tetrahedra that would be created if the common diagonal
    # were [1,8].
    tet_map = np.sort([[2,4,1,8],
                       [4,1,3,8],
                       [1,3,8,7],
                       [1,8,5,7],
                       [1,6,8,5],
                       [2,1,6,8]])
    
    submesh_cells = np.array([[0,9,3,12,1,10,4,13],
                              [1,10,4,13,2,11,5,14],
                              [3,12,6,15,4,13,7,16],
                              [4,13,7,16,5,14,8,17],
                              [9,18,12,21,10,19,13,22],
                              [10,19,13,22,11,20,14,23],
                              [12,21,15,24,13,22,16,25],
                              [13,22,16,25,14,23,17,26]])

    tetrahedra2 = []
    for i,submesh_cell in enumerate(submesh_cells):
        for j, tmap in enumerate(tet_map):
            tetrahedra2.append(submesh_cell[tmap-1])

    for t1 in tetrahedra1:
        contained = False
        for t2 in tetrahedra2:
            if np.allclose(t1,t2):
                contained = True
        assert contained


    def submesh_tetrahedra(vertices, indices):
        """Create a function that determines the indices of the tetrahedra
        vertices from a list of vertices and indices.
        """
            
        tetrahedra = find_tetrahedra(vertices)
        return indices[tetrahedra-1]

    # This unit test verifies that the function submesh_tetrahedra is working correctly.
    # This is done by creating a grid, selecting each submesh cell, creating tetrahedra
    # from the submesh cell, collecting all the tetrahedra from each submesh cell, and
    # comparing them to the tetrahedra produced by the function grid_and_tetrahedra.
    
    lat_angles =[np.pi/2]*3
    lat_consts = [1]*3
    lat_centering = "prim"
    lattice = Lattice(lat_centering, lat_consts, lat_angles)

    nvalence = 1
    degree = 2
    free = FreeElectronModel(lattice, nvalence, degree)

    ndiv_list = range(2, 8)
    for ndiv in ndiv_list:
        ndiv1 = [ndiv + 1]*3
        lat_shift = [1./2]*3
        grid_shift = [0,0,0]
        grid, tetrahedra = grid_and_tetrahedra(free, ndiv, lat_shift, grid_shift)

        test_tetrahedra = np.array([])
        npts = (ndiv + 1)**3
        indices = np.reshape(list(range(npts)), ndiv1)
        for i,j,k in itertools.product(range(ndiv), repeat=3):
            vertices = []
            vindices = []
            for l,m,n in itertools.product(range(2), repeat=3):
                vertices.append(np.reshape(grid, ndiv1 + [3])[i+l, j+m, k+n])
                vindices.append(indices[i+l, j+m, k+n])
            vindices = np.array(vindices)
            vertices = np.array(vertices)
            if len(test_tetrahedra) == 0:
                test_tetrahedra = submesh_tetrahedra(vertices, vindices)
            else:
                test_tetrahedra = np.append(test_tetrahedra,
                                        submesh_tetrahedra(vertices, vindices), 0)
        assert all([np.allclose(t1,t2) for t1,t2 in
                    zip(tetrahedra, test_tetrahedra)])

    lat_angles =[np.pi/2]*3
    lat_consts = [2*np.pi]*3
    lat_centering = "prim"
    lattice = Lattice(lat_centering, lat_consts, lat_angles)
    lat_offset = [-1./2]*3
    degree = 2
    nvalence = 1
    free = FreeElectronModel(lattice, degree)
    grid, tetrahedra = grid_and_tetrahedra(free, 2, lat_offset)

    true_tetra = [[0, 1, 4, 13],
                  [0, 3, 4, 13],
                  [0, 3, 12, 13],
                  [0, 9, 12, 13],
                  [0, 9, 10, 13],
                  [0, 1, 10, 13],
                  [1, 2, 5, 14],
                  [1, 4, 5, 14],
                  [1, 4, 13, 14],
                  [1, 10, 13, 14],
                  [1, 10, 11, 14],
                  [1, 2, 11, 14],
                  [3, 4, 7, 16],
                  [3, 6, 7, 16],
                  [3, 6, 15, 16],
                  [3, 12, 15, 16],
                  [3, 12, 13, 16],
                  [3, 4, 13, 16],
                  [4, 5, 8, 17],
                  [4, 7, 8, 17],
                  [4, 7, 16, 17],
                  [4, 13, 16, 17],
                  [4, 13, 14, 17],
                  [4, 5, 14, 17],
                  [9, 10, 13, 22],
                  [9, 12, 13, 22],
                  [9, 12, 21, 22],
                  [9, 18, 21, 22],
                  [9, 18, 19, 22],
                  [9, 10, 19, 22],
                  [10, 11, 14, 23],
                  [10, 13, 14, 23],
                  [10, 13, 22, 23],
                  [10, 19, 22, 23],
                  [10, 19, 20, 23],
                  [10, 11, 20, 23],
                  [12, 13, 16, 25],
                  [12, 15, 16, 25],
                  [12, 15, 24, 25],
                  [12, 21, 24, 25],
                  [12, 21, 22, 25],
                  [12, 13, 22, 25],
                  [13, 14, 17, 26],
                  [13, 16, 17, 26],
                  [13, 16, 25, 26],
                  [13, 22, 25, 26],
                  [13, 22, 23, 26],
                  [13, 14, 23, 26]]
    assert np.allclose(tetrahedra, true_tetra)

@pytest.mark.skipif("test_find_irreducible_tetrahedra" not in tests,
                    reason="different tests")
def test_find_irreducible_tetrahedra():
    """Check that symmetry reduction is performed correcly by comparing the
    value of the Fermi level with and without reduction.
    """

    ndiv0 = np.array([2,2,2])
    offset = [0]*3
    grid, tetrahedra = grid_and_tetrahedra(Al_PP, ndiv0, offset)
    weights_unreduced = np.ones(len(tetrahedra))
    fermi_level_unreduced = calc_fermi_level(Al_PP, tetrahedra,
                                             weights_unreduced, grid)    
    irreducible_tetrahedra, weights_reduced = find_irreducible_tetrahedra(Al_PP,
                                                            tetrahedra, grid)
    fermi_level_reduced = calc_fermi_level(Al_PP, irreducible_tetrahedra,
                                    weights_reduced, grid)

    assert np.isclose(fermi_level_unreduced, fermi_level_reduced)


@pytest.mark.skipif("test_find_irreducible_tetrahedra" not in tests, reason="different tests")
def test_find_irreducible_tetrahedra():

    lat_angles =[np.pi/2]*3
    lat_consts = [2*np.pi]*3
    lat_centering = "prim"
    lattice = Lattice(lat_centering, lat_consts, lat_angles)
    lat_offset = [-1./2]*3
    degree = 2
    free = FreeElectronModel(lattice, degree)
    grid, tetrahedra = grid_and_tetrahedra(free, 2, lat_offset)
    
    cells = [[0,1,3,4,9,10,12,13],
             [1,2,4,5,10,11,13,14],
             [3,4,6,7,12,13,15,16],
             [4,5,7,8,13,14,16,17],
             [9,10,12,13,18,19,21,22],
             [10,11,13,14,19,20,22,23],
             [12,13,15,16,21,22,24,25],
             [13,14,16,17,22,23,25,26]]

    tetrah = [find_tetrahedra(c)for c in cells]
    tetrahedra = []
    for i,t in enumerate(tetrah):
        for tet in t:
            real_tet = []
            for ind in tet:
                real_tet.append(cells[i][ind-1])
            tetrahedra.append(real_tet)

    index_map ={0:0,
                1:1,
                2:0,
                3:1,
                4:4,
                5:1,
                6:0,
                7:1,
                8:0,
                9:1,
                10:4,
                11:1,
                12:4,
                13:13,
                14:4,
                15:1,
                16:4,
                17:1,
                18:0,
                19:1,
                20:0,
                21:1,
                22:4,
                23:1,
                24:0,
                25:1,
                26:0}    

    for i in range(len(tetrahedra)):
        for j in range(len(tetrahedra[i])):
            tetrahedra[i][j] = index_map[tetrahedra[i][j]]
        tetrahedra[i] = np.sort(tetrahedra[i]).tolist()

    unitet = []
    weights = []
    for tet in tetrahedra:
        if any([np.allclose(tet, ut) for ut in unitet]):
            loc = np.where([np.allclose(tet,ut) for ut in
                                unitet])[0][0]                                    
            weights[loc] += 1.
            continue
        else:
            unitet.append(tet)
            weights.append(1.)
            
    grid, tetrahedra = grid_and_tetrahedra(free, 2, lat_offset)
    irr_tet, weights_irr = find_irreducible_tetrahedra(free, tetrahedra, grid)
    
    assert np.allclose(weights, weights_irr)

    for it in irr_tet:
        it = np.sort(it)
        contained = False
        for ut in unitet:
            ut = np.sort(ut)
            if np.allclose(it,ut):
                contained = True
        assert contained == True

    # Instead of labeling a tetrahedra by the index of the points at its
    # vertices, label it by the grid points themselves. Then evaluate the
    # potential at these grid points and find the unique energies.
    
    # Do this for tetrahedra with all k-point indices and for ones which
    # have had their k-point indices replaced by the index of the k-point
    # that represents its orbit.

    # Do it without replacing indices in tetrahedra.
    grid, tetrahedra = grid_and_tetrahedra(free, 2, lat_offset)        
    energies = np.empty(np.shape(tetrahedra), dtype=list).tolist()

    tet_grid_pts = np.empty(np.shape(tetrahedra), dtype=list).tolist()
    for i,tet in enumerate(tetrahedra):
        for j,ind in enumerate(tet):
            if j == 0:
                tet_grid_pts[i] = [grid[ind]]
            else:
                tet_grid_pts[i].append(grid[ind])
                
    for i,tet in enumerate(tet_grid_pts):
        for j,pt in enumerate(tet):
            if j == 0:            
                energies[i] = [free.eval(pt, 1)[0]]
            else:
                energies[i].append(free.eval(pt, 1)[0])
    weights = []
    unique_energies = []
    for i,en in enumerate(energies):
        en = np.sort(en)
        if any([np.allclose(en, ue) for ue in unique_energies]):
            loc = np.where([np.isclose(en,ue) for ue in unique_energies])[0][0]
            weights[loc] += 1.
        else:
            unique_energies.append(np.sort(en))
            weights.append(1.)
            
    # Do it replacing indices in tetrahedra
    index_map ={0:26,
                1:25,
                2:26,
                3:25,
                4:22,
                5:25,
                6:26,
                7:25,
                8:26,
                9:25,
                10:22,
                11:25,
                12:22,
                13:13,
                14:22,
                15:25,
                16:22,
                17:25,
                18:26,
                19:25,
                20:26,
                21:25,
                22:22,
                23:25,
                24:26,
                25:25,
                26:26}
    grid, tetrahedra = grid_and_tetrahedra(free, 2, lat_offset)
    for i in range(len(tetrahedra)):
        for j in range(len(tetrahedra[i])):
            tetrahedra[i][j] = index_map[tetrahedra[i][j]]
        tetrahedra[i] = np.sort(tetrahedra[i]).tolist()

    tet_grid_pts = np.empty(np.shape(tetrahedra), dtype=list).tolist()
    for i,tet in enumerate(tetrahedra):
        for j,ind in enumerate(tet):
            if j == 0:
                tet_grid_pts[i] = [grid[ind]]
            else:
                tet_grid_pts[i].append(grid[ind])
        
    sym_energies = np.empty(np.shape(tetrahedra), dtype=list).tolist()
    for i,tet in enumerate(tet_grid_pts):
        for j,pt in enumerate(tet):
            if j == 0:            
                sym_energies[i] = [free.eval(pt, 1)[0]]
            else:
                sym_energies[i].append(free.eval(pt, 1)[0])
    sym_weights = []
    sym_unique_energies = []
    for i,en in enumerate(sym_energies):
        en = np.sort(en)
        if any([np.allclose(en, ue) for ue in sym_unique_energies]):
            loc = np.where([np.isclose(en,ue) for ue in sym_unique_energies])[0][0]
            sym_weights[loc] += 1.
        else:
            sym_unique_energies.append(np.sort(en))
            sym_weights.append(1.)

    assert np.allclose(weights, sym_weights)
    assert np.allclose(unique_energies, sym_unique_energies)

    # Verify these energies are the same as those given when we find the irreducible
    # tetrahedra.    
    grid, tetrahedra = grid_and_tetrahedra(free, 2, lat_offset)
    irreducible_tetrahedra, weights_reduced = find_irreducible_tetrahedra(free,
                                                                          tetrahedra,
                                                                          grid)
    tet_grid_pts = np.empty(np.shape(irreducible_tetrahedra), dtype=list).tolist()
    for i,tet in enumerate(irreducible_tetrahedra):
        for j,ind in enumerate(tet):
            if j == 0:
                tet_grid_pts[i] = [grid[ind]]
            else:
                tet_grid_pts[i].append(grid[ind])

    irr_energies = np.empty(np.shape(irreducible_tetrahedra), dtype=list).tolist()
    for i,tet in enumerate(tet_grid_pts):
        for j,pt in enumerate(tet):
            if j == 0:            
                irr_energies[i] = [free.eval(pt, 1)[0]]
            else:
                irr_energies[i].append(free.eval(pt, 1)[0])
            irr_energies[i] = np.sort(irr_energies[i]).tolist()

    assert np.allclose(irr_energies, unique_energies)
    
@pytest.mark.skipif("test_integrals" not in tests, reason="different tests")
def test_integrals():
    degree_list = range(2,4)
    nvalence = 1
    
    for degree in degree_list:
        
        # Verify the Fermi level of the free electron model.
        lat_angles =[np.pi/2]*3
        lat_consts = [1]*3
        lat_centering = "prim"
        lattice = Lattice(lat_centering, lat_consts, lat_angles)
        free = FreeElectronModel(lattice, degree)

        lat_offset = [-1./2]*3
        grid, tetrahedra = grid_and_tetrahedra(free, 25, lat_offset)
        weights = np.ones(len(tetrahedra))
        free.fermi_level = calc_fermi_level(free, tetrahedra, weights, grid)

        sphere_volume = 4./3*np.pi*free.fermi_level**(3./degree)
        # The fact that the Fermi surface encloses a volume equal to the the reciprocal
        # primitive cell volume divided by the number of electrons comes from p. 230
        # of Kittel.
        occupied_volume = free.lattice.reciprocal_volume*free.nvalence_electrons/2

        assert np.isclose(sphere_volume, occupied_volume, 1e-1, 1e-1)
        assert np.isclose(free.fermi_level, free.fermi_level_ans, 1e-2,1e-2)

        total_energy = calc_total_energy(free, tetrahedra, weights, grid)
        rf_answer = free.fermi_level_ans**(1./degree)
        te_answer = 4*np.pi*(rf_answer**(3 + degree)/(3. + degree))
        assert np.isclose(total_energy, te_answer, 1e-1, 1e-1)

@pytest.mark.skipif("test_adjacent_tetrahedra" not in tests, reason="different tests")
def test_adjacent_tetrahedra():

    lat_angles =[np.pi/2]*3
    lat_consts = [1]*3
    lat_centering = "prim"
    lattice = Lattice(lat_centering, lat_consts, lat_angles)
    lat_offset = [-1./2]*3
    degree = 2
    nvalence = 1
    free = FreeElectronModel(lattice, nvalence, degree)
    grid, tetrahedra = grid_and_tetrahedra(free, 1, lat_offset)

    zeros = [[0, 1, 3, 7],
             [0, 2, 3, 7],
             [0, 2, 6, 7],
             [0, 4, 6, 7],
             [0, 4, 5, 7],
             [0, 1, 5, 7]]
    ones = [[0, 1, 3, 7], [0, 1, 5, 7]]
    twos = [[0, 2, 3, 7], [0, 2, 6, 7]]
    threes = [[0, 1, 3, 7], [0, 2, 3, 7]]
    fours = [[0, 4, 6, 7], [0, 4, 5, 7]]
    fives = [[0, 4, 5, 7], [0, 1, 5, 7]]
    sixes = [[0, 2, 6, 7], [0, 4, 6, 7]]
    sevens = [[0, 1, 3, 7],
              [0, 2, 3, 7],
              [0, 2, 6, 7],
              [0, 4, 6, 7],
              [0, 4, 5, 7],
              [0, 1, 5, 7]]
    
    assert find_adjacent_tetrahedra(tetrahedra, 0) == zeros
    assert find_adjacent_tetrahedra(tetrahedra, 1) == ones
    assert find_adjacent_tetrahedra(tetrahedra, 2) == twos
    assert find_adjacent_tetrahedra(tetrahedra, 3) == threes
    assert find_adjacent_tetrahedra(tetrahedra, 4) == fours
    assert find_adjacent_tetrahedra(tetrahedra, 5) == fives
    assert find_adjacent_tetrahedra(tetrahedra, 6) == sixes
    assert find_adjacent_tetrahedra(tetrahedra, 7) == sevens    


    grid, tetrahedra = grid_and_tetrahedra(free, 2, lat_offset)

    zeros = [[0,1,4,13],
             [0,3,4,13],
             [0,3,12,13],
             [0,9,12,13],
             [0,9,10,13],
             [0,1,10,13]]
    ones = [[0,1,4,13],
            [0,1,10,13],
            [1,2,5,14],
            [1,4,5,14],
            [1,4,13,14],
            [1,10,13,14],
            [1,10,11,14],
            [1,2,11,14]]
    twos = [[1,2,5,14],
            [1,2,11,14]]
    threes = [[0,3,4,13],
              [0,3,12,13],
              [3,4,7,16],
              [3,6,7,16],
              [3,6,15,16],
              [3,12,15,16],
              [3,12,13,16],
              [3,4,13,16]]
    fours = [[0,1,4,13],
             [0,3,4,13],
             [1,4,5,14],
             [1,4,13,14],
             [3,4,7,16],
             [3,4,13,16],
             [4,5,8,17],
             [4,7,8,17],
             [4,7,16,17],
             [4,13,16,17],
             [4,13,14,17],
             [4,5,14,17]]
    fives = [[1,2,5,14],
             [1,4,5,14],
             [4,5,8,17],
             [4,5,14,17]]
    sixes = [[3,6,7,16],
             [3,6,15,16]]
    sevens = [[3,4,7,16],
              [3,6,7,16],
              [4,7,8,17],
              [4,7,16,17]]
    eights = [[4,5,8,17],
              [4,7,8,17]]
    nines = [[0,9,12,13],
             [0,9,10,13],
             [9,10,13,22],
             [9,12,13,22],
             [9,12,21,22],
             [9,18,21,22],
             [9,18,19,22],
             [9,10,19,22]]
    tens = [[0,9,10,13],
            [0,1,10,13],
            [1,10,13,14],
            [1,10,11,14],
            [9,10,13,22],
            [9,10,19,22],
            [10,11,14,23],
            [10,13,14,23],
            [10,13,22,23],
            [10,19,22,23],
            [10,19,20,23],
            [10,11,20,23]]    
    elevens = [[1,10,11,14],
               [1,2,11,14],
               [10,11,14,23],
               [10,11,20,23]]
    twelves = [[0,3,12,13],
               [0,9,12,13],
               [3,12,15,16],
               [3,12,13,16],
               [9,12,13,22],
               [9,12,21,22],
               [12,13,16,25],
               [12,15,16,25],
               [12,15,24,25],
               [12,21,24,25],
               [12,21,22,25],
               [12,13,22,25]]
    thirteens = [[0,1,4,13],
                 [0,3,4,13],
                 [0,3,12,13],
                 [0,9,12,13],
                 [0,9,10,13],
                 [0,1,10,13],
                 [1,4,13,14],
                 [1,10,13,14],
                 [3,12,13,16],
                 [3,4,13,16],
                 [4,13,16,17],
                 [4,13,14,17],
                 [9,10,13,22],
                 [9,12,13,22],
                 [10,13,14,23],
                 [10,13,22,23],
                 [12,13,16,25],
                 [12,13,22,25],
                 [13,14,17,26],
                 [13,16,17,26],
                 [13,16,25,26],
                 [13,22,25,26],
                 [13,22,23,26],
                 [13,14,23,26]]
    fourteens = [[1,2,5,14],
                 [1,4,5,14],
                 [1,4,13,14],
                 [1,10,13,14],
                 [1,10,11,14],
                 [1,2,11,14],
                 [4,13,14,17],
                 [4,5,14,17],
                 [10,11,14,23],
                 [10,13,14,23],
                 [13,14,17,26],
                 [13,14,23,26]]
    fifteens = [[3,6,15,16],
                [3,12,15,16],
                [12,15,16,25],
                [12,15,24,25]]
    sixteens = [[3,4,7,16],
                [3,6,7,16],
                [3,6,15,16],
                [3,12,15,16],
                [3,12,13,16],
                [3,4,13,16],
                [4,7,16,17],
                [4,13,16,17],
                [12,13,16,25],
                [12,15,16,25],
                [13,16,17,26],
                [13,16,25,26]]
    seventeens = [[4,5,8,17],
                  [4,7,8,17],
                  [4,7,16,17],
                  [4,13,16,17],
                  [4,13,14,17],
                  [4,5,14,17],
                  [13,14,17,26],
                  [13,16,17,26]]
    eighteens = [[9,18,21,22],
                 [9,18,19,22]]
    nineteens = [[9,18,19,22],
                 [9,10,19,22],
                 [10,19,22,23],
                 [10,19,20,23]]
    twenties = [[10,19,20,23],
                [10,11,20,23]]
    twenty_ones = [[9,12,21,22],
                   [9,18,21,22],
                   [12,21,24,25],
                   [12,21,22,25]]
    twenty_twos = [[9,10,13,22],
                   [9,12,13,22],
                   [9,12,21,22],
                   [9,18,21,22],
                   [9,18,19,22],
                   [9,10,19,22],
                   [10,13,22,23],
                   [10,19,22,23],
                   [12,21,22,25],
                   [12,13,22,25],
                   [13,22,25,26],
                   [13,22,23,26]]
    twenty_threes = [[10,11,14,23],
                     [10,13,14,23],
                     [10,13,22,23],
                     [10,19,22,23],
                     [10,19,20,23],
                     [10,11,20,23],
                     [13,22,23,26],
                     [13,14,23,26]]
    twenty_fours = [[12,15,24,25],
                    [12,21,24,25]]
    twenty_fives = [[12,13,16,25],
                    [12,15,16,25],
                    [12,15,24,25],
                    [12,21,24,25],
                    [12,21,22,25],
                    [12,13,22,25],
                    [13,16,25,26],
                    [13,22,25,26]]
    twenty_sixes = [[13,14,17,26],
                    [13,16,17,26],
                    [13,16,25,26],
                    [13,22,25,26],
                    [13,22,23,26],
                    [13,14,23,26]]
    assert find_adjacent_tetrahedra(tetrahedra, 0) == zeros
    assert find_adjacent_tetrahedra(tetrahedra, 1) == ones
    assert find_adjacent_tetrahedra(tetrahedra, 2) == twos
    assert find_adjacent_tetrahedra(tetrahedra, 3) == threes
    assert find_adjacent_tetrahedra(tetrahedra, 4) == fours
    assert find_adjacent_tetrahedra(tetrahedra, 5) == fives
    assert find_adjacent_tetrahedra(tetrahedra, 6) == sixes
    assert find_adjacent_tetrahedra(tetrahedra, 7) == sevens    
    assert find_adjacent_tetrahedra(tetrahedra, 8) == eights
    assert find_adjacent_tetrahedra(tetrahedra, 9) == nines
    assert find_adjacent_tetrahedra(tetrahedra, 10) == tens
    assert find_adjacent_tetrahedra(tetrahedra, 11) == elevens
    assert find_adjacent_tetrahedra(tetrahedra, 12) == twelves
    assert find_adjacent_tetrahedra(tetrahedra, 13) == thirteens
    assert find_adjacent_tetrahedra(tetrahedra, 14) == fourteens
    assert find_adjacent_tetrahedra(tetrahedra, 15) == fifteens  
    assert find_adjacent_tetrahedra(tetrahedra, 16) == sixteens
    assert find_adjacent_tetrahedra(tetrahedra, 17) == seventeens
    assert find_adjacent_tetrahedra(tetrahedra, 18) == eighteens 
    assert find_adjacent_tetrahedra(tetrahedra, 19) == nineteens
    assert find_adjacent_tetrahedra(tetrahedra, 20) == twenties
    assert find_adjacent_tetrahedra(tetrahedra, 21) == twenty_ones
    assert find_adjacent_tetrahedra(tetrahedra, 22) == twenty_twos
    assert find_adjacent_tetrahedra(tetrahedra, 23) == twenty_threes
    assert find_adjacent_tetrahedra(tetrahedra, 24) == twenty_fours
    assert find_adjacent_tetrahedra(tetrahedra, 25) == twenty_fives
    assert find_adjacent_tetrahedra(tetrahedra, 26) == twenty_sixes


@pytest.mark.skipif("test_convert_tet_index" not in tests, reason="different tests")
def test_convert_tet_index():
    lat_angles =[np.pi/2]*3
    lat_consts = [2*np.pi]*3
    lat_centering = "prim"
    lattice = Lattice(lat_centering, lat_consts, lat_angles)
    lat_shift = [-1./2]*3
    grid_shift = [0.]*3
    degree = 2
    nvalence = 1
    free = FreeElectronModel(lattice, nvalence, degree)
    ndivs = 3
    ndiv0 = np.array([ndivs]*3)
    ndiv1 = ndiv0 + 1
    ndiv2 = ndiv0 + 2
    ndiv3 = ndiv0 + 3
    npts = np.prod(ndiv1)

    extended_indices = np.zeros(np.array(ndiv3), dtype=int)

    extended_indices = np.empty(np.array(ndiv3), dtype=int)
    for k,j,i in product(range(ndiv3[0]), range(ndiv3[1]), range(ndiv3[2])):
        if ((i > 0 and i < ndiv2[0]) and (j > 0 and j < ndiv2[1]) and
            (k > 0 and k < ndiv2[2])):
            extended_indices[k,j,i] = (i-1) + (j-1)*ndiv1[1] + (k-1)*ndiv1[0]*ndiv1[1]
        else:
            extended_indices[k,j,i] = i + j*ndiv3[1] + k*ndiv3[0]*ndiv3[1] + npts
            
    for i, ind in enumerate(extended_indices.flatten()):
        assert i == convert_tet_index(ind, ndiv0)

@pytest.mark.skipif("test_get_grid_tetrahedra" not in tests, reason="different tests")
def test_get_grid_tetrahedra():
    # Compare the new method of getting tetrahedra, with periodic
    # boundary conditions, to the old.
    lat_angles =[np.pi/2]*3
    lat_consts = [1]*3
    lat_centering = "prim"
    lattice = Lattice(lat_centering, lat_consts, lat_angles)
    lat_shift = [-1./2]*3
    grid_shift = [0.]*3
    degree = 2
    nvalence = 1
    free = FreeElectronModel(lattice, nvalence, degree)
    ndivs = 2

    grid, tetrahedra = grid_and_tetrahedra(free, ndivs, lat_shift)
    new_grid, new_tetrahedra = get_grid_tetrahedra(free, ndivs, lat_shift)
    
    weights = np.ones(len(tetrahedra))
    fermi_level = calc_fermi_level(free, tetrahedra, weights, grid)
    new_fermi_level = calc_fermi_level(free, new_tetrahedra, weights, new_grid)

    assert fermi_level == new_fermi_level

    irr_tet, weights = find_irreducible_tetrahedra(free, tetrahedra, grid)
    new_irr_tet, new_weights = find_irreducible_tetrahedra(free,
                                                    new_tetrahedra, new_grid)
    fermi_level = calc_fermi_level(free, irr_tet, weights, grid)
    new_fermi_level = calc_fermi_level(free, new_irr_tet, new_weights, new_grid)

    assert fermi_level == new_fermi_level

    # Make sure the tetrahedra for the twe methods are the same.
    lat_shift = [-1./2]*3
    grid, tetrahedra = grid_and_tetrahedra(Al_PP, ndivs, lat_shift)
    new_grid, new_tetrahedra = get_grid_tetrahedra(Al_PP, ndivs, lat_shift)
    map = {0:0,
       1:1,
       2:0,
       3:2,
       4:3,
       5:2,
       6:0,
       7:1,
       8:0,
       9:4,
       10:5,
       11:4,
       12:6,
       13:7,
       14:6,
       15:4,
       16:5,
       17:4,
       18:0,
       19:1,
       20:0,
       21:2,
       22:3,
       23:2,
       24:0,
       25:1,
       26:0}
    check_tet = np.zeros(np.shape(tetrahedra), dtype=int)
    for i,t in enumerate(tetrahedra):
        for j,ind in enumerate(t):
            check_tet[i,j] = map[tetrahedra[i,j]]
            
    assert all(np.equal(check_tet, new_tetrahedra).flatten())

    # Verify the Fermi level calculated with the new method gives the value
    # as the old.

    grid, tetrahedra = grid_and_tetrahedra(Al_PP, ndivs, lat_shift)
    new_grid, new_tetrahedra = get_grid_tetrahedra(Al_PP, ndivs, lat_shift)

    weights = np.ones(len(tetrahedra))
    fermi_level = calc_fermi_level(Al_PP, tetrahedra, weights, grid)

    weights = np.ones(len(tetrahedra))
    new_fermi_level = calc_fermi_level(Al_PP, new_tetrahedra, weights, new_grid)

    assert fermi_level == new_fermi_level
    irr_tet, weights = find_irreducible_tetrahedra(Al_PP, tetrahedra, grid)
    new_irr_tet, new_weights = find_irreducible_tetrahedra(Al_PP,
                                                        new_tetrahedra, new_grid)

    red_fermi_level = calc_fermi_level(Al_PP, irr_tet, weights, grid)
    new_red_fermi_level = calc_fermi_level(Al_PP, new_irr_tet,
                                           new_weights, new_grid)
    
    assert fermi_level == red_fermi_level
    assert new_red_fermi_level == red_fermi_level

@pytest.mark.skipif("test_find_adjacent_tetrahedra" not in tests, reason="different tests")    
def test_find_adjacent_tetrahedra():
    lat_angles =[np.pi/2]*3
    lat_consts = [2*np.pi]*3
    lat_centering = "prim"
    lattice = Lattice(lat_centering, lat_consts, lat_angles)
    lat_shift = [-1./2]*3
    grid_shift = [0.]*3
    degree = 2
    nvalence = 1
    free = FreeElectronModel(lattice, nvalence, degree)
    ndivs = 1    
    
    grid, tetrahedra = grid_and_tetrahedra(free, ndivs, lat_shift)
    extended_grid, extended_tetrahedra = get_extended_tetrahedra(free, ndivs, lat_shift)
    
    calc_adj_tet = find_adjacent_tetrahedra(extended_tetrahedra, 0)
    
    cells = [[8,9,12,13,24,25,28,0],
             [9,10,13,14,25,26,0,1],
             [12,13,16,17,28,0,32,2],
             [13,14,17,18,0,1,2,3],
             [24,25,28,0,40,41,44,4],
             [25,26,0,1,41,42,4,5],
             [28,0,32,2,44,4,48,6],
             [0,1,2,3,4,5,6,7]]

    adjacent_tet_list = []
    for cell in cells:
        for tet in tetrahedra:
            adjacent_tet_list.append([])
            for index in tet:
                adjacent_tet_list[-1].append(cell[index])
            adjacent_tet_list[-1] = np.sort(adjacent_tet_list[-1]).tolist()
            if adjacent_tet_list[-1][0] != 0:
                del adjacent_tet_list[-1]
                
    for tet1 in adjacent_tet_list:
        contained = False
        for tet2 in calc_adj_tet:
            tet2 = np.sort(tet2)
            if np.allclose(tet1, tet2):
                contained = True
        assert contained == True

@pytest.mark.skipif("test_corrections" not in tests, reason="different tests")
def test_corrections():
    """Unit test for the corrected integration weights.
    """
        
    lat_angles =[np.pi/2]*3
    lat_consts = [2*np.pi]*3
    lat_centering = "prim"
    lattice = Lattice(lat_centering, lat_consts, lat_angles)

    degree = 1
    free = FreeElectronModel(lattice, degree)

    lat_offset0 = [0,0,0]
    ndivisions = 1
    grid, tetrahedra = grid_and_tetrahedra(free, ndivisions, lat_offset0)

    # I probably only need to calculate the Fermi level once.
    weights = np.ones(len(grid))
    fermi_level = calc_fermi_level(free, tetrahedra, weights, grid)
    free.fermi_level = fermi_level

    energies = [0, 1, np.sqrt(2), np.sqrt(3)]
    VG = free.lattice.reciprocal_volume
    VT = free.lattice.reciprocal_volume/len(tetrahedra)
    uncorrected_weights = integration_weights(VT, energies, fermi_level)

    # Cell 0
    energies0 = [0,1,np.sqrt(2),np.sqrt(3)]
    EST0 = np.sum(energies0)
    DOS0 = density_of_states(VG, VT, energies0, fermi_level)
    correction0 = 6*DOS0*EST0

    # Cell 1
    energies1 = [0,1,1,np.sqrt(2)]
    EST1 = np.sum(energies1)
    DOS1 = density_of_states(VG, VT, energies1, fermi_level)
    correction1 = 2*DOS1*EST1

    # Cell 2
    energies2 = [0,1,1,np.sqrt(2)]
    EST2 = np.sum(energies2)
    DOS2 = density_of_states(VG, VT, energies2, fermi_level)
    correction2 = 2*DOS2*EST2

    # Cell 3
    energies3 = [0,1,1,np.sqrt(2)]
    EST3 = np.sum(energies3)
    DOS3 = density_of_states(VG, VT, energies3, fermi_level)
    correction3 = 2*DOS3*EST3

    # Cell 4
    energies4 = [0,1,1,np.sqrt(2)]
    EST4 = np.sum(energies4)
    DOS4 = density_of_states(VG, VT, energies4, fermi_level)
    correction4 = 2*DOS4*EST4

    # Cell 5
    energies5 = [0,1,1,np.sqrt(2)]
    EST5 = np.sum(energies5)
    DOS5 = density_of_states(VG, VT, energies5, fermi_level)
    correction5 = 2*DOS5*EST5

    # Cell 6
    energies6 = [0,1,1,np.sqrt(2)]
    EST6 = np.sum(energies6)
    DOS6 = density_of_states(VG, VT, energies6, fermi_level)
    correction6 = 2*DOS6*EST6

    # Cell 7
    energies7 = [0,1,np.sqrt(2), np.sqrt(3)]
    EST7 = np.sum(energies7)
    DOS7 = density_of_states(VG, VT, energies7, fermi_level)
    correction7 = 6*DOS7*EST7

    correction = (correction0 + 
                  correction1 + 
                  correction2 + 
                  correction3 + 
                  correction4 + 
                  correction5 + 
                  correction6 + 
                  correction7)/40

    ex_grid, ex_tetrahedra = get_extended_tetrahedra(free, ndivisions)

    c_energies, corrected_weights = corrected_integration_weights(free, ex_grid,
                                                                  ex_tetrahedra,
                                                                  [0,1,3,7],
                                                                  0, [1,1,1])

    assert np.isclose(corrected_weights[0], uncorrected_weights[0] + correction)

    # Calculate correction for point with index 1. Change to degree 2.
    lat_angles =[np.pi/2]*3
    lat_consts = [2*np.pi]*3
    lat_centering = "prim"
    lattice = Lattice(lat_centering, lat_consts, lat_angles)

    degree = 2
    free = FreeElectronModel(lattice, degree)

    lat_offset0 = [0,0,0]
    ndivisions = 1
    grid, tetrahedra = grid_and_tetrahedra(free, ndivisions, lat_offset0)

    # I probably only need to calculate the Fermi level once.
    weights = np.ones(len(grid))
    fermi_level = calc_fermi_level(free, tetrahedra, weights, grid)
    free.fermi_level = fermi_level

    energies = [0, 1, 2, 3]
    VG = free.lattice.reciprocal_volume
    VT = free.lattice.reciprocal_volume/len(tetrahedra)
    uncorrected_weights = integration_weights(VT, energies, fermi_level)

    # Cell 0    
    enT1 = [1, 2, 5, 6]
    enT2 = [1, 2, 3, 6]
    enT3 = [1, 4, 5, 6]

    DOST1 = density_of_states(VG, VT, enT1, fermi_level)
    DOST2 = density_of_states(VG, VT, enT2, fermi_level)
    DOST3 = density_of_states(VG, VT, enT3, fermi_level)

    correction0 = 2*DOST1*(np.sum(enT1)-4) + (
                  2*DOST2*(np.sum(enT2)-4) +
                  2*DOST3*(np.sum(enT3) - 4))

    # Cell 1
    enT4 = [0, 1, 2, 3]
    DOST4 = density_of_states(VG, VT, enT4, fermi_level)
    correction1 = 2*DOST4*(np.sum(enT4) - 4)

    # Cell 2
    enT5 = [1, 2, 4, 5]
    enT6 = [1, 2, 2, 5]
    DOST5 = density_of_states(VG, VT, enT5, fermi_level)
    DOST6 = density_of_states(VG, VT, enT6, fermi_level)

    correction2 = DOST5*(np.sum(enT5) - 4) + DOST6*(np.sum(enT6) - 4)
    
    # Cell 3
    enT7 = [1, 1, 2, 2]
    enT8 = [0, 1, 1, 2]
    DOST7 = density_of_states(VG, VT, enT7, fermi_level)
    DOST8 = density_of_states(VG, VT, enT8, fermi_level)

    correction3 = DOST7*(np.sum(enT7) - 4) + DOST8*(np.sum(enT8) - 4)

    # Cell 4
    enT10 = [1, 2, 2, 5]
    enT11 = [1, 2, 4, 5]
    DOST10 = density_of_states(VG, VT, enT10, fermi_level)
    DOST11 = density_of_states(VG, VT, enT11, fermi_level)

    correction4 = DOST10*(np.sum(enT10) - 4) + DOST11*(np.sum(enT11) - 4)
    
    # Cell 5
    enT12 = [1, 1, 2, 2]
    DOST12 = density_of_states(VG, VT, enT12, fermi_level)

    correction5 = DOST12*(np.sum(enT12) - 4)

    # Cell 6
    enT9 = [1, 2, 3, 4]
    DOST9 = density_of_states(VG, VT, enT9, fermi_level)

    correction6 = 2*DOST9*(np.sum(enT9) - 4)

    # Cell 7
    enT13 = [1, 1, 2, 2]
    enT14 = [1, 2, 2, 3]
    enT16 = [0, 1, 1, 2]
    DOST13 = density_of_states(VG, VT, enT13, fermi_level)
    DOST14 = density_of_states(VG, VT, enT14, fermi_level)


    correction7 = 2*DOST13*(np.sum(enT13) - 4) + (
                  2*DOST14*(np.sum(enT14) - 4))
                
    correction = (correction0 + 
                  correction1 + 
                  correction2 + 
                  correction3 + 
                  correction4 + 
                  correction5 + 
                  correction6 + 
                  correction7)/40

    ex_grid, ex_tetrahedra = get_extended_tetrahedra(free, ndivisions)
    c_energies, corrected_weights = corrected_integration_weights(free, ex_grid, ex_tetrahedra, [0,1,3,7],
                                                                  0, [1,1,1])
    assert np.isclose(corrected_weights[1], uncorrected_weights[1] + correction)
