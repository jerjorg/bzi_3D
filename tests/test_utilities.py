"""Unit tests for utilities.py module.
"""

import pytest
import numpy as np
from copy import deepcopy

from bzi_3D.utilities import *
from conftest import run

tests = run("all utilities")

@pytest.mark.skipif("test_swap_rows_columns" not in tests, reason="different tests")
def test_swap_rows_columns():

    test = np.random.uniform(-4, 4, size=(3,3))
    test_copy = swap_rows_columns(test, 0, 1, rows=True)

    all(test[0,:] == test_copy[1,:])
    all(test[1,:] == test_copy[0,:])

    test_copy = swap_rows_columns(test, 0, 1, rows=False)

    all(test[:,0] == test_copy[:,1])
    all(test[:,1] == test_copy[:,0])


@pytest.mark.skipif("test_remove_points" not in tests, reason="different tests")
def test_remove_points():

    # Check that it works for a single point.
    for _ in range(10):
        s = np.random.randint(51, 1001)
        point_list = np.random.uniform(-10, 10, size=(s, 3))
        
        pos = np.random.randint(0, s)
        point = point_list[pos]
        
        point_list = remove_points(point, point_list)        
        assert not check_contained(point, point_list)

    # Check that it works for many points.
    for _ in range(10):
        s = np.random.randint(51, 1001)
        point_list = np.random.uniform(-10, 10, size=(s, 3))
                                   
        pos = np.unique([np.random.randint(0, s) for _ in range(5)])           
        points = point_list[pos]

        point_list = remove_points(points, point_list)
        
        for point in points:
            assert not check_contained(point, point_list)
    
    
@pytest.mark.skipif("test_find_point_indices" not in tests, reason="different tests")
def test_find_point_indices():

    # Check that it works for a single point.
    for _ in range(10):
        s = np.random.randint(51, 1001)
        point_list = np.random.uniform(-10, 10, size=(s, 3))
        
        pos = np.random.randint(0, s)
        point = point_list[pos]
        
        assert find_point_indices(point, point_list) == [pos]
    
    # Check that it works for many points.
    for _ in range(10):
        s = np.random.randint(51, 1001)
        point_list = np.random.uniform(-10, 10, size=(s, 3))
                                   
        pos = np.unique([np.random.randint(0, s) for _ in range(5)])
        points = point_list[pos]        

        assert all(find_point_indices(points, point_list) == pos)
        
        
@pytest.mark.skipif("test_trim_small" not in tests, reason="different tests")
def test_trim_small():
    test = [[1.2, 3.4], [1e-4, 10]]
    assert ( trim_small(test, 1e-4) == [[1.2, 3.4], [0, 10]] ).all()

    
@pytest.mark.skipif("test_check_contained" not in tests, reason="different tests")
def test_check_contained():    
    # Check that it works for a single point.
    for _ in range(10):
        s = np.random.randint(51, 1001)
        point_list = np.random.uniform(-10, 10, size=(s, 3))
        
        pos = np.random.randint(0, s)
        point = point_list[pos]

        assert check_contained(point, point_list)        
    
    # Check that it works for many points.
    for _ in range(10):
        s = np.random.randint(51, 1001)
        point_list = np.random.uniform(-10, 10, size=(s, 3))
                                   
        pos = np.unique([np.random.randint(0, s) for _ in range(5)])
        points = point_list[pos]        

        assert check_contained(points, point_list)

@pytest.mark.skipif("test_inside" not in tests, reason="different tests")
def test_check_inside():


    a = 1e-8
    assert check_inside(a) == a

    a = 1 + 1e-8
    assert check_inside(a) == a

    a = 1.2
    assert check_inside(a) is None

    a = -1.001
    assert check_inside(a) is None    
