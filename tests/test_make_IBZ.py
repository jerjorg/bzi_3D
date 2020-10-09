"""Test the irreducible Brillouin zone construction methods.
"""

import pytest
import numpy as np
from numpy.linalg import det
from bzi_3D.symmetry import *
from bzi_3D.make_IBZ import *

from conftest import run

tests = run("all make_IBZ")

@pytest.mark.skipif("test_get_bragg_planes" not in tests, reason="different tests")
def test_get_bragg_planes():
    
    lat_vecs = np.eye(3)
    bplanes = get_bragg_planes(lat_vecs)

    dir_lat_vecs = [[1,0,0],
                    [0,1,0],
                    [1,1,0],
                    [0,0,1],
                    [1,0,1],
                    [0,1,1],
                    [-1,0,0],
                    [0,-1,0],
                    [-1,-1,0],
                    [0,0,-1],
                    [-1,0,-1],
                    [0,-1,-1],
                    [-1,-1,-1],
                    [-1,1,0],
                    [1,-1,0],
                    [0,1,-1],
                    [0,-1,1],
                    [-1,0,1],
                    [1,0,-1]]    
    for dvec in dir_lat_vecs:
        assert any(map(lambda elem: np.allclose(dvec, elem), bplanes["Direct"]))

    car_lat_vecs = dir_lat_vecs        
    for cvec in car_lat_vecs:
        assert any(map(lambda elem: np.allclose(dvec, elem), bplanes["Cartesian"]))

    distances = np.sort(list(map(lambda x: norm(x)/2, car_lat_vecs)))
    for i,d in enumerate(distances):
        assert np.isclose(bplanes["Distance"][i], d)

    planes = [np.append(i/norm(i), norm(i)/2) for i in car_lat_vecs]
    for plane in planes:
        assert any(map(lambda elem: np.allclose(plane, elem), bplanes["Bragg plane"]))
    
    
    lat_vecs = np.array([[1,1,0],[1,0,1],[0,1,1]])
    bplanes = get_bragg_planes(lat_vecs)

    dir_lat_vecs = [[1,0,0],
                    [0,1,0],
                    [1,1,0],
                    [0,0,1],
                    [1,0,1],
                    [0,1,1],
                    [-1,0,0],
                    [0,-1,0],
                    [-1,-1,0],
                    [0,0,-1],
                    [-1,0,-1],
                    [0,-1,-1],
                    [-1,-1,-1],
                    [-1,1,0],
                    [1,-1,0],
                    [0,1,-1],
                    [0,-1,1],
                    [-1,0,1],
                    [1,0,-1]]
    
    for dvec in dir_lat_vecs:
        assert any(map(lambda elem: np.allclose(dvec, elem), bplanes["Direct"]))

    car_lat_vecs = list(map(lambda x: np.dot(lat_vecs, x), dir_lat_vecs))        
    for cvec in car_lat_vecs:
        assert any(map(lambda elem: np.allclose(dvec, elem), bplanes["Cartesian"]))

    distances = np.sort(list(map(lambda x: norm(x)/2, car_lat_vecs)))
    for i,d in enumerate(distances):
        assert any(map(lambda dist: np.isclose(dist, d), distances))

    planes = [np.append(i/norm(i), norm(i)/2) for i in car_lat_vecs]
    for plane in planes:
        assert any(map(lambda elem: np.allclose(plane, elem), bplanes["Bragg plane"]))

@pytest.mark.skipif("test_trim_small" not in tests, reason="different tests")
def test_trim_small():
    vec = np.array([[1.,1e-8, 1.], [1,1e-8,1]])
    assert (trim_small(vec) == np.array([[1,0,1],[1,0,1]])).all

    vec = np.array([[1.8e-7, 2.2, 7.6], [1e-5,1e-8, 8], [2.3, 1e-7, 3.4]])
    assert (trim_small(vec) == np.array([[0, 2.2, 7.6], [1e-5,0, 8], [2.3, 0, 3.4]])).all
    
@pytest.mark.skipif("test_three_planes_intersect" not in tests, reason="different tests")
def test_three_planes_intersect():    
    lat_vecs = make_lattice_vectors("simple cubic", [2]*3, [np.pi/2]*3)
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    # In order to concatenate the distances to the lattice vectors, the distances
    # need to by a 2D array.
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)
    assert np.allclose(three_planes_intersect(three_planes), [1]*3)

    lat_vecs = make_lattice_vectors("body-centered cubic", [2]*3, [np.pi/2]*3)
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)    
    assert np.allclose(three_planes_intersect(three_planes), [3/2]*3)
    
    lat_vecs = make_lattice_vectors("face-centered cubic", [2]*3, [np.pi/2]*3)
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)    
    assert np.allclose(three_planes_intersect(three_planes), [1/2]*3)

    lat_vecs = make_lattice_vectors("tetragonal", [1,1,2], [np.pi/2]*3)
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)    
    assert np.allclose(three_planes_intersect(three_planes), [.5, .5, 1])

    lat_vecs = make_lattice_vectors("body-centered tetragonal", [3,3,4], [np.pi/2]*3)
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)    
    assert np.allclose(three_planes_intersect(three_planes),
                       [2.83333333, 2.83333333, 2.125])

    lat_vecs = make_lattice_vectors("orthorhombic", [2,3,4], [np.pi/2]*3)
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)    
    assert np.allclose(three_planes_intersect(three_planes), [ 1. ,  1.5,  2. ])

    lat_vecs = make_lattice_vectors("body-centered orthorhombic", [1,3,4], [np.pi/2]*3)
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)    
    assert np.allclose(three_planes_intersect(three_planes), [6.5, 2.16667, 1.625])
    
    lat_vecs = make_lattice_vectors("face-centered orthorhombic", [1,2,4], [np.pi/2]*3)
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)
    assert np.allclose(three_planes_intersect(three_planes), [0.25, 0.5, 1.])

    lat_vecs = make_lattice_vectors("base-centered orthorhombic", [1,2,3], [np.pi/2]*3)
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)    
    assert np.allclose(three_planes_intersect(three_planes), [1.25, 0., 1.5])

    lat_vecs = make_lattice_vectors("hexagonal", [1,1,3], [np.pi/2, np.pi/2, 2*np.pi/3])
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)    
    assert np.allclose(three_planes_intersect(three_planes), [ 1., 0., 1.5 ])

    lat_vecs = make_lattice_vectors("rhombohedral", [2,2,2], [.3*np.pi]*3)
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)
    assert np.allclose(three_planes_intersect(three_planes),
                       [ 1.12232624,  0.        ,  0.34544531])

    lat_vecs = make_lattice_vectors("monoclinic", [1,2,3], [5*np.pi/11, np.pi/2, np.pi/2])
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)    
    assert np.allclose(three_planes_intersect(three_planes),
                       [0.5, 1., 1.37165])
    
    lat_vecs = make_lattice_vectors("base-centered monoclinic", [1,2,3],
                                    [5*np.pi/11, np.pi/2, np.pi/2])
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)
    assert np.allclose(three_planes_intersect(three_planes),
                       [0., 0.625, 1.42556])

    lat_vecs = make_lattice_vectors("triclinic", [1,2,3],
                                    [5*np.pi/11, 3*np.pi/5, 4*np.pi/7])
    distances = np.linalg.norm(lat_vecs, axis=0)/2
    distances = np.reshape(distances, (3, 1))
    unit_vecs = lat_vecs/(2*distances.T)
    three_planes = np.append(unit_vecs.T, distances, axis=1)    
    assert np.allclose(three_planes_intersect(three_planes),
                       [0.5, 1.13984, 1.65445])

@pytest.mark.skipif("test_find_bragg_shells" not in tests, reason="different tests")    
def test_find_bragg_shells():
    lat_vecs_list = [make_lattice_vectors("simple cubic", [2]*3, [np.pi/2]*3),
                     make_lattice_vectors("body-centered cubic", [2]*3, [np.pi/2]*3),
                     make_lattice_vectors("face-centered cubic", [2]*3, [np.pi/2]*3),
                     make_lattice_vectors("tetragonal", [1,1,2], [np.pi/2]*3),
                     make_lattice_vectors("body-centered tetragonal", [3,3,4],
                                          [np.pi/2]*3),
                     make_lattice_vectors("orthorhombic", [2,3,4], [np.pi/2]*3),
                     make_lattice_vectors("body-centered orthorhombic", [1,3,4],
                                          [np.pi/2]*3),
                     make_lattice_vectors("face-centered orthorhombic", [1,2,4],
                                          [np.pi/2]*3),
                     make_lattice_vectors("base-centered orthorhombic", [1,2,3],
                                          [np.pi/2]*3),
                     make_lattice_vectors("hexagonal", [1,1,3],
                                          [np.pi/2, np.pi/2, 2*np.pi/3]),
                     make_lattice_vectors("rhombohedral", [2,2,2], [.3*np.pi]*3),
                     make_lattice_vectors("monoclinic", [1,2,3],
                                          [5*np.pi/11, np.pi/2, np.pi/2]),
                     make_lattice_vectors("base-centered monoclinic", [1,2,3],
                                          [5*np.pi/11, np.pi/2, np.pi/2]),
                     make_lattice_vectors("triclinic", [1,2,3],
                                          [5*np.pi/11, 3*np.pi/5, 4*np.pi/7])]
                     
    for lat_vecs in lat_vecs_list:
        bragg_planes = get_bragg_planes(lat_vecs)
        bragg_shells = get_bragg_shells(bragg_planes)
        for i in range(len(bragg_shells)-1):
            shell_distance = bragg_planes["Distance"][bragg_shells[i]]
            all_same = True
            for j in range(bragg_shells[i]+1, bragg_shells[i+1]):
                if not np.isclose(bragg_planes["Distance"][j], shell_distance):
                    all_same = False
            assert all_same == True

@pytest.mark.skipif("test_find_bz" not in tests, reason="different tests")
def test_find_bz():
    # Try it out for all possible Bravais lattices. It won't be every possible Brillouin
    # zone but it's a good start. It would be hard to get very close to the exact volume
    # if there were a bug in the code, so the fact that they finish is a unit test.
    lat_type = "simple cubic"
    lat_vecs = make_lattice_vectors(lat_type, [2]*3, [np.pi/2]*3)
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    sympts = sc_sympts

    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert len(get_unique_planes(bz)) == 6
    assert len(bz.vertices) == 8
    assert np.isclose(bz.volume, det(rlat_vecs))
    assert np.isclose(bz.area, 1/4*6)
    assert np.isclose(bz.volume/nops, ibz.volume)


    lat_type = "body-centered cubic"
    lat_vecs = make_lattice_vectors(lat_type, [2]*3, [np.pi/2]*3)
    rlat_vecs = make_rptvecs(lat_vecs)
    sympts = bcc_sympts
    bz = find_bz(rlat_vecs)

    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert len(get_unique_planes(bz)) == 12
    assert len(bz.vertices) == 14
    assert np.isclose(bz.volume, det(rlat_vecs))
    assert np.isclose(bz.volume/nops, ibz.volume)
    
    
    lat_type = "face-centered cubic"
    lat_vecs = make_lattice_vectors(lat_type, [2]*3, [np.pi/2]*3)
    rlat_vecs = make_rptvecs(lat_vecs)
    sympts = fcc_sympts
    bz = find_bz(rlat_vecs)

    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])        
    assert len(get_unique_planes(bz)) == 14
    assert len(bz.vertices) == 24
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)
    

    lat_type = "tetragonal"
    lat_vecs = make_lattice_vectors(lat_type, [3,3,4],[np.pi/2]*3)
    rlat_vecs = make_rptvecs(lat_vecs)
    sympts = tet_sympts
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)

    
    lat_type = "body-centered tetragonal"
    lat_vecs = make_lattice_vectors(lat_type, [6,6,3],[np.pi/2]*3)
    rlat_vecs = make_rptvecs(lat_vecs)
    sympts = bct1_sympts(6,3)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)
    

    lat_type = "body-centered tetragonal"
    lat_vecs = make_lattice_vectors(lat_type, [3,3,6],[np.pi/2]*3)
    sympts = bct2_sympts(3,6)
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)

    lat_type = "orthorhombic"
    lat_vecs = make_lattice_vectors("orthorhombic", [10, 10.1, 10.2], [np.pi/2]*3)
    sympts = orc_sympts
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)

    
    lat_type = "face-centered orthorhombic"
    lat_vecs = make_lattice_vectors(lat_type, [4, 10, 11],
                                    [np.pi/2]*3)
    sympts = orcf13_sympts(4, 10, 11)
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)


    lat_type = "face-centered orthorhombic"
    lat_vecs = make_lattice_vectors(lat_type, [10, 11, 12],
                                [np.pi/2]*3)
    sympts = orcf2_sympts(10, 11, 12)
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)
    

    a = 8.108695542208157
    lat_type = "face-centered orthorhombic"
    lat_vecs = make_lattice_vectors(lat_type, [a, 11, 12],
                                [np.pi/2]*3)
    sympts = orcf13_sympts(a, 11, 12)
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)

    
    lat_type = "body-centered orthorhombic"
    lat_vecs = make_lattice_vectors(lat_type, [10, 12, 14],
                                [np.pi/2]*3)
    sympts = orci_sympts(10, 11, 12)
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)
    

    lat_type = "base-centered orthorhombic"
    lat_vecs = make_lattice_vectors(lat_type, [5, 9, 10],
                                [np.pi/2]*3)
    sympts = orcc_sympts(5, 6)
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)

    
    lat_type = "hexagonal"
    lat_vecs = make_lattice_vectors(lat_type, [5, 5, 6],
                                    [np.pi/2, np.pi/2, 2*np.pi/3])
    sympts = hex_sympts
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)


    lat_type = "rhombohedral"
    lat_vecs = make_lattice_vectors(lat_type, [5,5,5], [.3*np.pi]*3)
    sympts = rhl1_sympts(np.pi/3)
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)


    lat_type = "rhombohedral"
    lat_vecs = make_lattice_vectors(lat_type, [5]*3, [7*np.pi/12]*3)
    sympts = rhl2_sympts(7*np.pi/12)
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)


    lat_type = "monoclinic"
    lat_vecs = make_lattice_vectors(lat_type, [10, 11, 12],
                                    [5*np.pi/11, np.pi/2, np.pi/2])
    sympts = mcl_sympts(11, 12, 5*np.pi/11)
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    
    # Get the vertices of the IBZ.
    ibz_vertices = list(sympts.values())
    ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]

    # Get the symmetry points in Cartesian coordinates.
    sympts_cart = {}
    for spt in sympts.keys():
        sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])
    
    ibz = ConvexHull(ibz_vertices)
    nops = number_of_point_operators(lat_type.split()[-1])
    assert np.isclose(bz.volume, det(rlat_vecs))    
    assert np.isclose(bz.volume/nops, ibz.volume)
    
    lat_type = "base-centered monoclinic"
    lat_vecs = make_lattice_vectors(lat_type, [10, 11, 12],
                                    [5*np.pi/11, np.pi/2, np.pi/2])
    rlat_vecs = make_rptvecs(lat_vecs)
    bz = find_bz(rlat_vecs)
    assert np.isclose(bz.volume, det(rlat_vecs))
