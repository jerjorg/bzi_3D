"""This module contains general, useful functions"""

import numpy as np
from copy import deepcopy


def swap_rows_columns(array, a, b, rows=True):
    """Swap the rows or columns of an array.

    Args:
        array (numpy.ndarray): an 2D numpy array.
        a (int): a row or column to be swapped.
        b (int): a row or column to be swapped.
        rows (bool): if true, swap the rows of the array. Otherwise, swap the columns.
        copy (bool): if `False`, the input argument `array` will be modified in place.

    Returns:
        _ (numpy.ndarray): a 2D numpy array with columns a and b swapped.
    """

    array_copy = deepcopy(array)                

    if rows:
        array_copy[a, :], array_copy[b, :] = array_copy[b, :], array_copy[a, :].copy()
    else:
        array_copy[:, a], array_copy[:, b] = array_copy[:, b], array_copy[:, a].copy()

    return array_copy        


def find_point_indices(points, point_list, rtol=1e-5, atol=1e-8):
    """Find the location of points in a list of points. Each point in the list of 
    points should be unique.
    
    Args:
        point (numpy.ndarray): an array of floats.
        point_list (numpy.ndarray): an array of arrays of floats.
        rtol (float): finite precision parameter for relative tolerance.

    Returns:
        indices (list): a list of indices that correspond to the location of `points`
            in `point_list`.
    """

    if np.shape(points) == (3,):
        points = [points]

    try:
        indices = list(map(lambda y:
                    np.where(list(map(lambda x:
                                      np.allclose(x, y, rtol=rtol, atol=atol),
                                      point_list)))[0], points))
    except:
        indices = []

    if len(indices) == 1:
        return indices[0]
    else:
        flat_indices = []
        for index in indices:
            for i in index:
                flat_indices.append(i)
            
        return np.array(flat_indices)


def remove_points(points, point_list, rtol=1e-5, atol=1e-8):
    """Remove points from a list of points in 3-space.
    
    Args:
        points (list or numpy.ndarray): a list of points in 3-space.
        point_list (list or numpy.ndarray): an array of points in 3-space.

    Returns:
        _ (numpy.ndarray): an array of points in 3-space.
    """
    
    # If only one point is provided and it isn't in a nested list, put it in one.
    if np.shape(points) == (3,):
        points = [points]

    indices = range(len(point_list))

    # Find the indices of elements to remove.
    remove_indices = find_point_indices(points, point_list,
                                        rtol=rtol, atol=atol)

    # Find the indices of elements to keep.
    keep_indices = list(set(indices).symmetric_difference(set(remove_indices)))

    point_list = np.array(point_list)
    return point_list[keep_indices]


def trim_small(array, atol=1e-6):
    """Replace elements in an array close to zero with exactly zero.

    Args:
        array (numpy.ndarray): an array of floats.

    Returns:
        array_copy (numpy.ndarray): an array of floats.
    """

    array_copy = np.array(deepcopy(array))
    
    array_copy[np.isclose(array_copy, 0, atol=atol)] = 0
    return array_copy


def check_contained(points, point_list, rtol=1e-5, atol=1e-8):
    """Check if a point or a list of points is contained in a list of points.
    
    Args:
        points (list or numpy.ndarray): a list of points in 3-space. If a single point
            is provided, it must still be in a nested list.
        point_list (numpy.ndarray): a list of points in 3-space.
    
    Returns:
        _ (bool): if the points are contained in the list of points, return `True`;
            return `False` otherwise.
    """

    # This will catch some cases where a single point is provided, but not all of them.
    if np.shape(points) == (3,):
        points = [points]

    return all(map(lambda x:
                   any(map(lambda y:
                           np.allclose(x, y, rtol=rtol, atol=atol),
                           point_list)), points))


def print_fortran_grid(lat_vecs, rlat_vecs, atom_labels, atom_positions, grid_vecs,
                       offset):
    """Print the lattice vectors, atom labels, atom positions, grid vectors, and offset
    in a format that is easy to copy and paste into Fortran code.

    Args:
        lat_vecs (numpy.ndarray): the lattice vectors in Cartesian coordinates as columns
            of a 3x3 array.

        rlat_vecs (numpy.ndarray): the reciprocal lattice vectors in Cartesian coordinates
            as columns of a 3x3 array.
        atom_labels (list): a list of atoms labels. Each label should be distince for each
            atomic species. The labels must start at zero and should be in the same order 
            as atomic basis.
        atom_positions (list or numpy.ndarray): a list of atomic positions in Cartesian 
            (default) or lattice coordinates.
        grid_vecs (numpy.ndarray): the grid generating vectors as columns of a 3x3 array.
        offset (list): a list of grid offsets in grid coordinates.   
    """

    # Print the lattice vectors in a format that is 
    # easy to copy and paste into Fortran code.
    for i, r in enumerate(lat_vecs):
        if i == 0:
            print("lat_vecs = transpose(reshape((/ " + 
                  "_dp, ".join(map(str, np.round(r, 15))) + "_dp, &")
        elif i == 1:
            print("                          " + 
                  "_dp, ".join(map(str, np.round(r, 15))) + "_dp, &")
        else: 
            print("                          " + 
                  "_dp, ".join(map(str, np.round(r, 15))) + "_dp /)," + 
                 "(/3,3/)))")
            
    # Print the reciprocal lattice vectors.
    for i, r in enumerate(rlat_vecs):
        if i == 0:
            print("rlat_vecs = transpose(reshape((/ " + 
                  "_dp, ".join(map(str, np.round(r, 15))) + "_dp, &")
        elif i == 1:
            print("                          " + 
                  "_dp, ".join(map(str, np.round(r, 15))) + "_dp, &")
        else: 
            print("                          " + 
                  "_dp, ".join(map(str, np.round(r, 15))) + "_dp /)," + 
                 "(/3,3/)))")

    # Fortran needs memory allocated for the atom labels and positions.
    print("allocate(atom_labels(1:{}))".format(len(atom_labels)))
    print("allocate(atom_positions(1:3, 1:{}))".format(len(atom_labels)))    
    
    # Print the species labels.
    print("atom_labels = (/ " + ", ".join(map(str, atom_labels)) + " /)")

    # Print the atom positions.
    natoms = len(atom_labels)

    if natoms == 1:
        r = atom_positions[0]
        print("atom_positions = transpose(reshape((/ " +
              ", ".join(map(str, np.round(r, 15))) + " /)" +
              ", (/{},3/)))".format(natoms))
    else:    
        for i, r in enumerate(atom_positions):
            if i == 0:
                print("atom_positions = transpose(reshape((/ " +
                      ", ".join(map(str, np.round(r, 15))) + ", &")
            elif i == (natoms-1):
                print("                          " + 
                      ", ".join(map(str, np.round(r, 15))) + " /)," +
                      "(/{},3/)))".format(natoms))
            else:
                print("                          " +
                      ", ".join(map(str, np.round(r, 15))) + ", &")

    # Print the grid vectors.
    for i, r in enumerate(grid_vecs):
        if i == 0:
            print("grid_vecs = transpose(reshape((/ " + 
                  ", ".join(map(str, np.round(r, 15))) + ", &")
        elif i == 1:
            print("                          " + 
                  ", ".join(map(str, np.round(r, 15))) + ", &")
        else:
            print("                          " + 
                  ", ".join(map(str, np.round(r, 15))) + " /)," + 
                 "(/3,3/)))")

    # Print the shift.
    print("shift = (/ " + "_dp, ".join(map(str, np.round(offset, 4))) + "_dp" + " /)")


def make_unique(array, rtol=1e-5, atol=1e-8):
    """Find the unique points from an array.
    
    Args:
        array (list or numpy.ndarray): a list of arrays.
        
    Returns:
        unique_array (np.ndarray): the unique elements from the array.
    """
    
    # Copy the array.
    array_copy = deepcopy(array)

    # Initialize the unique array.
    unique_array = []

    while len(array_copy) > 0:
        # Grab and remove a point from the grid array.
        pt = array_copy[-1]
        array_copy = array_copy[:-1]
        
        if not check_contained([pt], unique_array, atol=atol, rtol=rtol):
            unique_array.append(pt)        
    
    return unique_array    

def rprint(label, arg, dec=5):
    print(label + "\n", np.round(arg, dec))


def check_inside(t, lb=0, ub=1, rtol=1e-8, atol=1e-5):
    """Check that a real number is within or lies on the boundary of 
    a given range.
    
    Args:
        t (float): the number to test.
        lb (float): the lower bound.
        up (float): the upper bound.
        rtol (float): the relative tolerance used to check if the lower
            or upper bound is equivalent to the provided number.
        atol (float): the absolute tolerance used to check if the lower
            or upper bound is equivalent to the provided number.
    Returns:
        _ (float): if the number lies in the given range, return the number. 
            Otherwise, return "None".
    """
    if (np.isclose(t, lb, rtol=rtol, atol=atol) or
        np.isclose(t, ub, rtol=rtol, atol=atol) or
        (lb < t and t < ub)):
        return t
    else:
        return None
