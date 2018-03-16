"""This module contains general, useful functions"""

import numpy as np

def remove_points(point, point_list):
    """Remove a point from a list of points.
    
    Args:
        point (list or numpy.ndarray): an array of floats.
        point_list (list or numpy.ndarray): an array of arrays of floats.
    """

    point_list = np.array(point_list)
    return point_list[list(map(lambda x: not np.allclose(x, point), point_list))]


def find_point_index(point, point_list):
    """Find the location of a point in a list of points. Each point in the list of 
    points should be unique.
    
    Args:
        point (numpy.ndarray): an array of floats.
        point_list (numpy.ndarray): an array of arrays of floats.
    """

    loc = np.where(list(map(lambda x: np.allclose(x, point), point_list)))[0]
    
    if len(loc) > 1:
        msg = "There are duplicate points in the list of points."
        raise ValueError(msg.format(point_list))
    elif len(loc) == 0:
        msg = "This point isn't in the list of points."
        raise ValueError(msg.format(point_list))
    else:
        return loc[0]

def find_point_indices(point, point_list, atol=1e-6):
    """Find the location of a point in a list of points. Each point in the list of 
    points should be unique.
    
    Args:
        point (numpy.ndarray): an array of floats.
        point_list (numpy.ndarray): an array of arrays of floats.
    """
    
    return np.where(list(map(lambda x: np.allclose(x, point, atol=atol), point_list)))[0]

    
def trim_small(matrix, atol=1e-6):
    """Replace elements in an array close to zero with 0.

    Args:
        matrix (numpy.ndarray): any numpy array of floats.
    """
    
    matrix[np.isclose(matrix, 0, atol=atol)] = 0
    return matrix


def check_contained(vector, vector_list):
    """Check if a vector is contained in a list of vectors.
    
    Args:
        vector_list(numpy.ndarray): a 2D array.
        vector (numpy.ndarray): a 1D array.
        
    Returns:
        (bool): if the vector is contained in the list of vectors, return True;
            return False otherwise.
    """
    
    return any(map(lambda elem: np.allclose(vector, elem), vector_list))

def print_fortran_grid(rlatvecs, grid_vecs, offset):
    """Print the lattice vectors , grid vectors, and offset in a format that is easy to
    copy and paste into Fortran code.
    """

    print("shift = (/ " + "_dp, ".join(map(str, np.round(offset, 4))) + "_dp" + " /)")

    # Print the lattice vectors in a format that is 
    # easy to copy and paste into Fortran code.
    for i, r in enumerate(rlatvecs):
        if i == 0:
            print("R = transpose(reshape((/ " + 
                  "_dp, ".join(map(str, np.round(r, 15))) + "_dp, &")
        elif i == 1:
            print("                          " + 
                  "_dp, ".join(map(str, np.round(r, 15))) + "_dp, &")
        else: 
            print("                          " + 
                  "_dp, ".join(map(str, np.round(r, 15))) + "_dp /)," + 
                 "(/3,3/)))")
            
    # Print the reciprocal lattice vectors in a format that is 
    # easy to copy and paste into Fortran code.
    for i, r in enumerate(gridvecs):
        if i == 0:
            print("K = transpose(reshape((/ " + 
                  ", ".join(map(str, np.round(r, 15))) + ", &")
        elif i == 1:
            print("                          " + 
                  ", ".join(map(str, np.round(r, 15))) + ", &")
        else:
            print("                          " + 
                  ", ".join(map(str, np.round(r, 15))) + " /)," + 
                 "(/3,3/)))")    

