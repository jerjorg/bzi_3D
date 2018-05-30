"""Test the 2D methods.
"""

import pytest
import numpy as np
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt

from BZI.all_2D import *

from BZI.utilities import check_contained, make_unique

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


@pytest.mark.skipif("test_get_perpendicular_vector2D" not in tests,
                    reason="different tests")
def test_get_perpendicular_vector2D():
    
    pt1 = np.array([0, 0])
    pt2 = np.array([1, 0])

    r12 = pt2 - pt1
    r_perp = get_perpendicular_vector2D(r12)

    assert np.allclose(r_perp, [0,1])
    
    r12 = [0,0]
    r_perp = get_perpendicular_vector2D(r12)
    
    assert np.allclose(r_perp, [1,0])

    
@pytest.mark.skipif("test_get_line_equation2D" not in tests,
                    reason="different tests")
def test_get_line_equation2D():
    
    pt1 = np.array([1, 0])
    pt2 = np.array([1, 2])
    
    line_eq = get_line_equation2D(pt1, pt2)

    # For some reason np.allclose doesn't work when comparing line equations.
    assert np.allclose(line_eq[0], [-1, 0])
    assert np.isclose(line_eq[1], -1)
    
    
@pytest.mark.skipif("test_square_tesselation" not in tests,
                    reason="different tests")
def test_square_tesselation():

    ndivs = 2
    grid_basis = make2D_lattice_basis([1/ndivs,1/ndivs], np.pi/2)
    
    lattice_basis = make2D_lattice_basis([1,1], np.pi/2)
    
    offset = [0,0]
    grid = make_cell_points2D(lattice_basis, grid_basis, offset,
                              grid_type="closed")

    square_tess = square_tesselation(grid)

    subsquares_list = [[[0, 0], [.5, 0], [.5, .5], [0, .5]],
                       [[0, .5], [.5, .5], [.5, 1], [0, 1]],
                       [[.5, 0], [1, 0], [1, .5], [.5, .5]],
                       [[.5, .5], [1, .5], [1, 1], [.5, 1]]]

    assert (check_contained(square_tess, subsquares_list) and
            check_contained(subsquares_list, square_tess))
    
    
@pytest.mark.skipif("test_refine_square" not in tests,
                    reason="different tests")
def test_refine_square():

    lattice_basis = make2D_lattice_basis([1,1], np.pi/2)
    ndivs = 6
    
    grid_basis = make2D_lattice_basis([1/ndivs,1/ndivs], np.pi/2)
    offset = np.dot(inv(grid_basis), np.dot(lattice_basis, [-.5]*2))

    grid = make_cell_points2D(lattice_basis, grid_basis, offset, grid_type="closed")

    args = {"lattice_basis": lattice_basis,
            "degree": 2,
            "prefactor":1,
            "nvalence_electrons": 3}

    free_2D = FreeElectron2D(**args)

    weights = np.ones(len(grid))
    fermi_level, band_energy = rectangular_integration2D(free_2D,
                                                         grid,
                                                         weights)
    free_2D.fermi_level = fermi_level
    free_2D.band_energy = band_energy

    tess = square_tesselation(grid)    

    all_squares, areas = refine_square(tess[2], free_2D, method="interpolate",
                                       ndivisions=2, derivative_tol=1e1)

    nsquares = int(np.prod(np.shape(all_squares))/2)
    all_pts = np.reshape(all_squares, (nsquares, 2))    

    unique_pts = make_unique(all_pts)

    all_squares2 = [[[1/3,0],[5/12, 0],[5/12, 1/12],[1/3, 1/12]],
                    [[5/12, 0],[1/2, 0],[1/2, 1/12],[5/12, 1/12]],
                    [[5/12, 1/12],[1/2, 1/12],[1/2, 1/6],[5/12, 1/6]],
                    [[1/3, 1/12],[5/12, 1/12],[5/12, 1/6],[1/3, 1/6]]]

    unique_pts2 = [[1/3, 0],[5/12, 0],[1/2, 0],[1/3, 1/12],[5/12, 1/12],[1/2, 1/12],
                   [1/3, 1/6],[5/12, 1/6],[1/2, 1/6]]


    assert (check_contained(unique_pts2, unique_pts) and 
            check_contained(unique_pts, unique_pts2))
    
    assert (check_contained(all_squares, all_squares2) and 
            check_contained(all_squares2, all_squares))
    
    
@pytest.mark.skipif("test_get_bilin_coeffs" not in tests,
                    reason="different tests")
def test_get_bilin_coeffs():

    
    pts = [[0,0],[1,0],[1,1],[0,1]]
    values = [5]*4

    coord_mat = [[1,0,0,0],[1,1,0,0],[1,1,1,1],[1,0,1,0]]

    coeffs1 = np.dot(inv(coord_mat), values)

    coeffs2 = get_bilin_coeffs(pts, values)

    assert np.allclose(coeffs1, coeffs2, [5,0,0,0])


@pytest.mark.skipif("test_eval_bilin" not in tests,
                    reason="different tests")
def test_eval_bilin():

    coeffs= [1, 2, 3, 4]
    pt = [2, 3]

    value1 = coeffs[0] + coeffs[1]*pt[0] + coeffs[2]*pt[1] + coeffs[3]*pt[0]*pt[1]
    value2 = eval_bilin(coeffs, pt)

    assert np.isclose(value1, value2)

    
    # This will create a plot of a bilinear interpolation. It can be visually inspected
    # to verify eval_bilin is working correctly.
    points = [[-.5,-.5], [.5,-.5], [.5,.5],[-.5,.5]]
    values = [0,100,0,100]

    coeffs = get_bilin_coeffs(points, values)

    ndivs = 100
    grid_basis = make2D_lattice_basis([1/ndivs,1/ndivs], np.pi/2)
    lattice_basis = make2D_lattice_basis([1,1], np.pi/2)

    offset = np.dot(inv(grid_basis), np.dot(lattice_basis, [-.5]*2)) # + [.5]*2
    grid = make_cell_points2D(lattice_basis, grid_basis, offset, grid_type="closed")

    values = values = [eval_bilin(coeffs,pt) for pt in grid]

    grid = np.array(grid)

    xs = grid[:,0]
    ys = grid[:,1]

    ax = plt.subplot(1,1,1,projection="3d")
    ax.scatter(xs, ys, values,s=.5)


@pytest.mark.skipif("test_integrate_bilinear" not in tests,
                    reason="different tests")
def test_integrate_bilinear():

    lattice_basis = make2D_lattice_basis([1,1], np.pi/2)
    args = {"lattice_basis": lattice_basis,
            "degree": 2,
            "prefactor":1,
            "nvalence_electrons": 3}

    free_2D = FreeElectron2D(**args)    

    a = 1/6
    test = np.array([[-a, -a],[0,-a],[0,0],[-a,0]])
    offset = test[0]
    values = [free_2D.eval(pt, sigma=True) for pt in test]
    integral1 = integrate_bilinear(test, values, offset)
    
    a = 1/6
    test = np.array([[-a, 0],[0,0],[0,a],[-a,a]])
    offset = test[0]
    values = [free_2D.eval(pt, sigma=True) for pt in test]
    integral2 = integrate_bilinear(test, values, offset)    

    a = 1/6
    test = np.array([[0, -a],[a,-a],[a,0],[0,0]])
    offset = test[0]
    values = [free_2D.eval(pt, sigma=True) for pt in test]
    integral3 = integrate_bilinear(test, values, offset)

    a = 1/6
    test = np.array([[0, 0],[a,0],[a,a],[0,a]])
    offset = test[0]
    values = [free_2D.eval(pt, sigma=True) for pt in test]
    integral4 = integrate_bilinear(test, values, offset)

    # These subcells are symmetrically equivalent and should give the same integral value.
    assert np.allclose(integral1, integral2, integral3, integral4)


@pytest.mark.skipif("test_find_param_intersect" not in tests,
                    reason="different tests")
def test_find_param_intersect():
    
    a = 0.25
    square_pts = [[0,0],[a,0],[a,a],[0,a]]
    
    f = lambda x,y: np.cos(2*np.pi*x) + np.cos(2*np.pi*y)
    
    values = [f(*pt) for pt in square_pts]
    
    coeffs = get_bilin_coeffs(square_pts, values)
    isovalue = 1.3
    
    (edge_params, isocurve_params, edge_indices,
     edge_intersects, intersect_coords) = find_param_intersect(square_pts, coeffs, isovalue)
    
    values_list = []
    for coord in intersect_coords:
        values_list.append(eval_bilin(coeffs, coord))
    
    assert np.allclose(values_list, isovalue)
    
    intersects_coords2 = [[0.175, 0],[0, 0.175]]
    assert np.allclose(intersect_coords, intersects_coords2)
    
    curve_vals = [0.175, 0]
    assert np.allclose(isocurve_params, curve_vals)
    
    
    # This plots the isocurve. You should see it intersecting the boundaries of the
    # square at [0.175, 0] and [0, 0.175].
    fig,ax = plt.subplots()
    
    test_basis = np.array([[0,a],[a,0]])
    grid_basis = test_basis/100
    offset = [0,0]
    
    grid = make_cell_points2D(test_basis, grid_basis, offset, grid_type="closed")
    
    kx = [grid[i][0] for i in range(len(grid))]
    ky = [grid[i][1] for i in range(len(grid))]
    kz = [eval_bilin(coeffs, pt) for pt in grid]
    
    kx2d = np.reshape(kx, (101,101))
    ky2d = np.reshape(ky, (101,101))
    kz2d = np.reshape(kz, (101,101))
    
    # ax.contour(kx2d, ky2d, kz2d, isovalue)
    
    square_pts = [[-.4, -.4], [.6, -.4], [.6, .6], [-.4, .6]]
    coeffs = [0,0,0,10]
    isovalue = 1
    
    (param_edge, param_isocurve, edge_indices,
     intersecting_edges, xy)  = find_param_intersect(square_pts, coeffs, isovalue)
    
    assert np.allclose(xy, [[-0.25, -0.4], [0.6, 0.16666667], [0.16666667, 0.6], [-0.4, -0.25]])
    
    # This is a corner intersection that required changing the relative tolerance to pass.
    atol = 1e-6
    rtol = 1e-4
    
    square_pts =  [[-0.3, -0.4], [ 0.7, -0.4], [ 0.7,  0.6], [-0.3,  0.6]]
    coeffs = [0.41667, 1.16667, 0.16667, 0.]
    isovalue = 0.0
    
    (edge_params, isocurve_params, edge_indices,
     edge_intersects, intersect_pts) = find_param_intersect(square_pts, coeffs, isovalue,
                                                            rtol=rtol, atol=atol)
    
    assert np.allclose(intersect_pts, [[-0.3000008571404082, -0.4000000000000001]])
        
    # Case where both intersections lie on the same edge.
    square_pts = [[0, 0], [8/5, 4/5], [12/5, 16/5], [4/5, 12/5]];
    basis = np.array([[8/5, 4/5], [4/5, 12/5]])
    coeffs = [1, 1, -3.5, 4]
    isovalue = 0

    (edge_params, isocurve_params, edge_indices,
     edges, intersect_pts) = find_param_intersect(square_pts, coeffs, isovalue)

    # Check that the correct edges get intersected.
    assert np.allclose(edge_indices, [3, 3])

    # Check that the location of intersection is correct.
    assert np.allclose(intersect_pts, [[0.666667, 2.], [0.125, 0.375]])

    # Check that the edge parameters are correct.
    assert np.allclose(edge_params, [0.16666666666666666, 0.84375])

    # Check that the curve parameters are correct.
    assert np.allclose(isocurve_params, [0.666667, 0.125])
    
    # Check that the edges are correct.
    assert np.allclose(edges, [[[4/5, 12/5], [0, 0]], [[4/5, 12/5], [0, 0]]])    
    

    # Case where the intersections lie on adjacent edges.
    square_pts = [[0, 0], [8/5, 4/5], [12/5, 16/5], [4/5, 12/5]];    
    coeffs = [1, 1, -3.5, 4]
    isovalue = 7.5

    (edge_params, isocurve_params, edge_indices,
     edge_intersects, intersect_pts) = find_param_intersect(square_pts, coeffs, isovalue)

    # Check that the correct edges get intersected.
    assert np.allclose(edge_indices, [1, 2])

    # Check that the location of intersection is correct.
    assert np.allclose(intersect_pts, [[1.77236, 1.31709], [1.35533, 2.67767]])

    # Check that the edge parameters are correct.
    assert np.allclose(edge_params, [0.21545442903129275, 0.6529178285196583])

    # Check that the curve parameters are correct.
    assert np.allclose(isocurve_params, [1.7723635432250342, 1.3553314743685467])

    # Check that the edges are correct.
    assert np.allclose(edge_intersects, [[[8/5, 4/5], [12/5, 16/5]], [[12/5, 16/5],
                                                                      [4/5, 12/5]]])
        
    # Case where the intersections lie on opposite edges.
    square_pts = [[0, 0], [8/5, 4/5], [12/5, 16/5], [4/5, 12/5]];    
    coeffs = [1, 1, -3.5, 4]
    isovalue = 2.5

    (edge_params, isocurve_params, edge_indices,
     edges, intersect_pts) = find_param_intersect(square_pts, coeffs, isovalue)

    # Check that the correct edges get intersected.
    assert np.allclose(edge_indices, [0, 2])

    # Check that the location of intersection is correct.
    assert np.allclose(intersect_pts, [[1.07359, 0.536795], [0.932524, 2.46626]])

    # Check that the edge parameters are correct.
    assert np.allclose(edge_params, [0.6709940187014776, 0.9171725581251758])

    # Check that the curve parameters are correct.
    assert np.allclose(isocurve_params, [1.073590429922364, 0.9325239069997187])

    # Check that the edges are correct.
    assert np.allclose(edges, [[[0, 0], [8/5, 4/5]], [[12/5, 16/5], [4/5, 12/5]]])
    
    
@pytest.mark.skipif("test_eval_bilin" not in tests,
                    reason="different tests")
def test_eval_bilin():

    a = 0.25
    square_pts = [[0,0],[a,0],[a,a],[0,a]]

    f = lambda x,y: np.cos(2*np.pi*x) + np.cos(2*np.pi*y)

    values = [f(*pt) for pt in square_pts]

    coeffs = get_bilin_coeffs(square_pts, values)


    bilin_values = [eval_bilin(coeffs, pt) for pt in square_pts]

    # Verify the value of the bilinear interpolation values at the corners of the square
    # are the same as the function values.
    assert np.allclose(values, bilin_values)

    # Verify that the values of the bilinear interpolation are within the range of the
    # values of the function at the corners of the square.
    test_basis = np.array([[0,a],[a,0]])
    grid_basis = test_basis/100
    offset = [0,0]

    grid = make_cell_points2D(test_basis, grid_basis, offset, grid_type="closed")

    bilin_values = [eval_bilin(coeffs, g) for g in grid]

    # Lower and upper bounds
    lb, ub = np.min(values), np.max(values)
    bilin_values2 = [check_inside(val, lb=lb, ub=ub) for val in bilin_values]
    
    assert np.allclose(bilin_values, bilin_values2)


@pytest.mark.skipif("test_group_bilinear_intersections" not in tests,
                    reason="different tests")
def test_group_bilinear_intersections():


    coeffs = [1, 0, 0, 1]    
    test_pts = np.array([[.1, .1], [-.1, .2], [-.3, -.4], [.3, -.5],
                         [.3, .5], [-.8, .2], [-.9, -.7], [.6, -.7]])    
    param_edge = [1, 2, 3, 4, 5, 6, 7, 8]
    param_isocurve = [8, 7, 6, 5, 4, 3, 2, 1]
    edge_indices = [9, 10, 11, 12, 13, 14, 15, 16]
    
    intersecting_edges = [[[1]*2, [1]*2],
                          [[2]*2,[2]*2],
                          [[3]*2,[3]*2],
                          [[4]*2,[4]*2],
                          [[5]*2,[5]*2],
                          [[6]*2,[6]*2],
                          [[7]*2,[7]*2],
                          [[8]*2,[8]*2]]
    
    (grouped_pts, grouped_param_edge, grouped_param_isocurve,
     grouped_edge_indices, grouped_intersecting_edges) = (
        group_bilinear_intersections(coeffs, test_pts, param_edge, param_isocurve,
                                     edge_indices, intersecting_edges))    
    
    assert np.allclose(grouped_edge_indices[0], [9, 13])
    assert np.allclose(grouped_edge_indices[1], [10, 14])    
    assert np.allclose(grouped_edge_indices[2], [11, 15])
    assert np.allclose(grouped_edge_indices[3], [12, 16])
    assert len(grouped_edge_indices) == 4
                       
    assert np.allclose(grouped_pts[0], [[.1, .1], [.3, .5]])
    assert np.allclose(grouped_pts[1], [[-.1, .2], [-.8, .2]])
    assert np.allclose(grouped_pts[2], [[-.3, -.4], [-.9, -.7]])
    assert np.allclose(grouped_pts[3], [[.3, -.5], [.6, -.7]])
    assert len(grouped_pts) == 4

    assert np.allclose(grouped_param_edge[0], [1, 5])
    assert np.allclose(grouped_param_edge[1], [2, 6])
    assert np.allclose(grouped_param_edge[2], [3, 7])
    assert np.allclose(grouped_param_edge[3], [4, 8])    
    assert len(grouped_param_edge) == 4

    assert np.allclose(grouped_param_isocurve[0], [8, 4])
    assert np.allclose(grouped_param_isocurve[1], [7, 3])
    assert np.allclose(grouped_param_isocurve[2], [6, 2])
    assert np.allclose(grouped_param_isocurve[3], [5, 1])
    assert len(grouped_param_isocurve) == 4

    assert np.allclose(grouped_intersecting_edges[0], [[[1]*2, [1]*2], [[5]*2,[5]*2]])
    assert np.allclose(grouped_intersecting_edges[1], [[[2]*2, [2]*2], [[6]*2,[6]*2]])
    assert np.allclose(grouped_intersecting_edges[2], [[[3]*2, [3]*2], [[7]*2,[7]*2]])
    assert np.allclose(grouped_intersecting_edges[3], [[[4]*2, [4]*2], [[8]*2,[8]*2]])
    assert len(grouped_intersecting_edges) == 4


    coeffs = [1, 0, 0, 1]
    test_pts = np.array([[-0.4       , -0.25      ],
                         [-0.25      , -0.4       ],
                         [ 0.16666667,  0.6       ],
                         [ 0.6       ,  0.16666667]])
    param_edge = [0.15000000000000002, 0.5666666666666668, 0.4333333333333333, 0.85]
    param_isocurve = [-0.25, 0.6, 0.16666666666666666, -0.4]

    edge_indices = [3, 0, 2, 1]
    
    intersecting_edges = [[[-0.4, -0.4], [0.6, -0.4]],
                          [[0.6, -0.4], [0.6, 0.6]],
                          [[0.6, 0.6], [-0.4, 0.6]],
                          [[-0.4, 0.6], [-0.4, -0.4]]]
    
    (grouped_pts, grouped_param_edge, grouped_param_isocurve,
     grouped_edge_indices, grouped_intersecting_edges) = (
        group_bilinear_intersections(coeffs, test_pts, param_edge, param_isocurve,
                                     edge_indices, intersecting_edges))

    assert np.allclose(grouped_pts[0], [[0.16666667, 0.6],[0.6, 0.16666667]])
    assert np.allclose(grouped_pts[1], [[-0.4, -0.25], [-0.25, -0.4]])
    assert len(grouped_pts) == 2

    assert np.allclose(grouped_param_edge[0], [0.433333333333, 0.85])
    assert np.allclose(grouped_param_edge[1], [0.15, 0.566666666666])
    assert len(grouped_param_edge) == 2

    assert np.allclose(grouped_param_isocurve[0], [0.166666666666, -0.4])
    assert np.allclose(grouped_param_isocurve[1], [-0.25, 0.6])
    assert len(grouped_param_isocurve) == 2

    assert np.allclose(grouped_edge_indices[0], [2, 1])
    assert np.allclose(grouped_edge_indices[1], [3, 0])
    assert len(grouped_edge_indices) == 2

    assert np.allclose(grouped_intersecting_edges[0], np.array([[[ 0.6,  0.6],
                                                                 [-0.4,  0.6]],
                                                                [[-0.4,  0.6],
                                                                 [-0.4, -0.4]]]))

    assert np.allclose(grouped_intersecting_edges[1], np.array([[[-0.4, -0.4],
                                                                 [ 0.6, -0.4]],
                                                                [[ 0.6, -0.4],
                                                                 [ 0.6,  0.6]]]))
    assert len(grouped_intersecting_edges) == 2

    square_pts = [[-.8, -.8], [.2, -.8], [.2, .2], [-.8, .2]]
    coeffs = [0, 0, 0, 10]
    isovalue = 1

    (param_edge, param_isocurve, edge_indices,
     intersecting_edges, intersecting_pts) = find_param_intersect(square_pts, coeffs,
                                                                  isovalue)

    p, pe, pi, ei, ie = group_bilinear_intersections(coeffs, intersecting_pts, param_edge,
                                                     param_isocurve, edge_indices,
                                                     intersecting_edges)

    assert np.allclose(p[0], intersecting_pts)
    assert np.allclose(pe[0], param_edge)
    assert np.allclose(pi[0], param_isocurve)
    assert np.allclose(ei[0], edge_indices)
    assert np.allclose(ie[0], intersecting_edges)


@pytest.mark.skipif("test_get_integration_case" not in tests,
                    reason="different tests")
def test_get_integration_case():

    # Case 0
    edges = [0, 2]
    assert get_integration_case(edges) == 0

    edges = [2, 0]
    assert get_integration_case(edges) == 0
    
    edges = [1, 3]
    assert get_integration_case(edges) == 0

    edges = [3, 1]
    assert get_integration_case(edges) == 0

    # Case 1
    edges = [0, 1]
    assert get_integration_case(edges) == 1

    edges = [1, 0]
    assert get_integration_case(edges) == 1
    
    edges = [1, 2]
    assert get_integration_case(edges) == 1

    edges = [2, 1]
    assert get_integration_case(edges) == 1

    edges = [2, 3]
    assert get_integration_case(edges) == 1

    edges = [3, 2]
    assert get_integration_case(edges) == 1

    edges = [0, 3]
    assert get_integration_case(edges) == 1

    edges = [3, 0]
    assert get_integration_case(edges) == 1
    
    # Case 2
    edges = [0, 0]
    assert get_integration_case(edges) == 2

    edges = [1, 1]
    assert get_integration_case(edges) == 2

    edges = [2, 2]
    assert get_integration_case(edges) == 2

    edges = [3, 3]
    assert get_integration_case(edges) == 2

    with pytest.raises(ValueError):
        edges = [4, 4]
        get_integration_case(edges)

    with pytest.raises(ValueError):
        edges = [-3, 4]
        get_integration_case(edges)

    with pytest.raises(ValueError):
        edges = [2, 5]
        get_integration_case(edges)

@pytest.mark.skipif("test_get_integration_cases" not in tests,
                    reason="different tests")
def test_get_integration_cases():

    # Case 2, two intersections same edge
    square_pts = [[0, 0], [8/5, 4/5], [12/5, 16/5], [4/5, 12/5]]
    coeffs = [1, 1, -3.5, 4]
    isovalue = 0
    cases = get_integration_cases(square_pts, coeffs, isovalue)
    assert all(cases == np.array([[2], ["inside"]]))

    # Case 0, intesections on opposite edges
    isovalue = 2.5
    cases = get_integration_cases(square_pts, coeffs, isovalue)
    assert all(cases == np.array([[0], ["inside"]]))
    
    # Case 1, intesections on adjacent edges
    isovalue = 7.5
    cases = get_integration_cases(square_pts, coeffs, isovalue)
    assert all(cases == np.array([[1], ["outside"]]))

    # Case 1, 1, intersections on opposite, adjacent edges
    square_pts = [[0, 0], [8/5, 4/5], [12/5, 16/5], [4/5, 12/5]]
    coeffs = [1, -4, -2.5, 2]
    isovalue = -3.5
    
    cases = get_integration_cases(square_pts, coeffs, isovalue)
    assert np.array(cases == np.array([[1, 1], ["outside", "outside"]]),
                    dtype="bool").all()
