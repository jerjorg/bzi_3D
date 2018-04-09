"""Test the 2D methods.
"""

import pytest
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from BZI.all_2D import *
from BZI.utilities import check_contained

from conftest import run

tests = run("all 2D")

@pytest.mark.skipif("test_make2D_lattice_basis" not in tests, reason="different tests")
def test_get_bragg_planes():
    
    lattice_constants = [2.1, 3.2]
    lattice_angle = np.pi/3
    
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)
    a1 = lattice_basis[:,0]
    a2 = lattice_basis[:,1]
    
    assert np.allclose(lattice_constants, norm(lattice_basis, axis=0))
    assert np.isclose(lattice_angle, np.arccos( np.dot(a1, a2)/( norm(a1)*norm(a2) )))

    
@pytest.mark.skipif("test_get_2Dlattice_type" not in tests, reason="different tests")
def test_get_2Dlattice_type():

    # square
    lattice_constants = [3.3, 3.3]
    lattice_angle = np.pi/2    
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)
    assert get_2Dlattice_type(lattice_basis) == "square"

    # hexagonal
    lattice_constants = [3.3, 3.3]
    lattice_angle = np.pi/3
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)
    assert get_2Dlattice_type(lattice_basis) == "hexagonal"
    
    # rhombic
    lattice_constants = [3.3, 3.3]
    lattice_angle = np.pi/4
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)
    assert get_2Dlattice_type(lattice_basis) == "rhombic"

    # centered rectangular
    lattice_constants = [1.4, 1.8]
    lattice_angle = np.arccos( lattice_constants[1]/(2*lattice_constants[0]) )
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)
    assert get_2Dlattice_type(lattice_basis) == "centered rectangular"

    # centered rectangular
    lattice_constants = [3.2, 1.7]
    lattice_angle = np.arccos( lattice_constants[0]/(2*lattice_constants[1]) )
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)
    assert get_2Dlattice_type(lattice_basis) == "centered rectangular"    

    # oblique
    lattice_constants = [1.4, 3.3]
    lattice_angle = np.pi/3
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)
    assert get_2Dlattice_type(lattice_basis) == "oblique"

    # oblique
    lattice_constants = [3.3, 2.9]
    lattice_angle = np.pi/3
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)
    assert get_2Dlattice_type(lattice_basis) == "oblique"


@pytest.mark.skipif("test_HermiteNormalForm2D" not in tests, reason="different tests")
def test_HermiteNormalForm():
    
    N = np.array([[1, 2], [3, 4]])
    
    H,B = HermiteNormalForm2D(N)

    # B acting on N should give H
    assert np.allclose(np.dot(N, B), H)

    # Verify B is a unimodular matrix
    assert np.isclose(abs(det(B)), 1)
    assert np.allclose(np.round(B), B)
    
    # Verify H is an integer matrix.
    assert np.allclose(np.round(H), H)

    assert np.isclose(H[0, 1], 0)
    assert H[1, 0] < H[1,1]
    assert (H >= 0).all()
    
    
@pytest.mark.skipif("test_make_cell_points2D" not in tests, reason="different tests")
def test_make_cell_points2D():

    lattice_constants = [1]*2
    lattice_angle = np.pi/2
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)

    grid_basis = lattice_basis/10
    offset = [0.5]*2

    norms = [[-1, 0], [1, 0], [0, 1], [-1, 0]]
    distances = [0, 1, 1, 0]
    lines = [[n, d] for n,d in zip(norms, distances)]

    grid = make_cell_points2D(lattice_basis, grid_basis, offset)

    plot_mesh2D(grid, lattice_basis, offset = np.array([0,0]))

    for point in grid:
        for line in lines:        
            assert point_line_location(point, line) == "inside"


@pytest.mark.skipif("test_plot_mesh2D" not in tests, reason="different tests")
def test_plot_mesh2D():
    
    lattice_constants = [3, 4]
    lattice_angle = np.pi/3
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)
    
    grid_basis = lattice_basis/10
    offset = [0.5]*2
    
    grid = make_cell_points2D(lattice_basis, grid_basis, offset)
    
    fig,ax = plt.subplots()    
    plot_mesh2D(grid, lattice_basis, offset = np.array([0,0]), ax=ax)
    
    assert True    


@pytest.mark.skipif("test_get_circle_pts" not in tests, reason="different tests")
def test_get_circle_pts():

    lattice_constants = [1]*2
    lattice_angle = np.pi/2
    lattice_basis = make2D_lattice_basis(lattice_constants, lattice_angle)

    radius = 2

    pts = [ [0, 0], [1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [-1, -1], [-1, 1], [1, -1]]
    circle_pts = get_circle_pts(lattice_basis, radius)
    
    assert check_contained(pts, circle_pts)
    assert len(pts) == len(circle_pts)
