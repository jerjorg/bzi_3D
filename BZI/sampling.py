"""Various methods of sampling the Brillouin zone.
"""
 
import numpy as np
from numpy.linalg import inv, norm
from copy import deepcopy
from itertools import product
from math import ceil

from BZI.symmetry import (make_ptvecs, UpperHermiteNormalForm,
                          HermiteNormalForm)

def make_grid(lat_vecs, grid_vecs, offset, cart=True):
    """Sample within a parallelepiped using any regular grid.

    Args:
        lat_vecs (numpy.ndarray): the vectors defining the volume in which 
            to sample. The vectors are the columns of the matrix.
        grid_vecs (numpy.ndarray): the vectors that generate the grid as 
            columns of the matrix..
        offset: the offset of the coordinate system in grid coordinates.
        cart (bool): if true, return the grid in Cartesian coordinates; other-
            wise, return the grid in cell coordinates.

    Returns:
        grid (list): an array of sampling-point coordinates.

    Examples:
        >>> cell_centering = "face"
        >>> cell_consts = [1.]*3
        >>> cell_angles = [np.pi/2]*3
        >>> lat_vecs = make_ptvecs(cell_centering, cell_consts, cell_angles)
        >>> grid_centering = "base"
        >>> grid_consts = [cell_const/140]*3
        >>> grid_angles = [np.pi/2]*3
        >>> grid_vecs = make_ptvecs(grid_centering, grid_consts, grid_angles)
        >>> offset = .5*numpy.sum(lat_vecs,1)
        >>> grid = make_grid(lat_vecs, grid_vecs, offset)
    """    

    # Put the offset in Cartesian coordinates.
    offset = np.dot(grid_vecs, offset)

    # Offset in lattice coordinates in the first unit cell
    # offset = np.round(np.dot(inv(lat_vecs), offset), 15)%1

    # # Offset in cell coordinates
    # offset = np.dot(lat_vecs, offset)

    # Integer matrix
    N = np.dot(inv(grid_vecs), lat_vecs)
    # Check that N is an integer matrix.
    for i in range(len(N[:,0])):
        for j in range(len(N[0,:])):
            if np.isclose(N[i,j]%1, 0) or np.isclose(N[i,j]%1, 1):
                N[i,j] = int(np.round(N[i,j]))
            else:
                raise ValueError("The cell and grid vectors are incommensurate.")
            
    # H is an HNF and U is the transform.
    H, U = HermiteNormalForm(N)
    a = H[0,0]
    b = H[0,1]
    c = H[0,2]
    d = H[1,1]
    e = H[1,2]
    f = H[2,2]
    cell_const = norm(lat_vecs[:,0])

    if cart:
        grid = []
        z3pl = 0 # Limits for first loop of integers
        z3pu = int(f)
        for z3p in range(z3pl + 1, z3pu + 1):
            z2pl = int(e*z3p/f) # lower and upper limits
            z2pu = int(z2pl + d)
            for z2p in range(z2pl + 1, z2pu + 1):
                z1pl = int((c - b*e/d)*z3p/f + b/d*z2p)
                z1pu = int(z1pl + a)
                for z1p in range(z1pl + 1, z1pu + 1):
                    z = np.dot(inv(U), [z1p,z2p,z3p])
                    pt = np.dot(grid_vecs, z)
                    gpt = np.round(np.dot(inv(lat_vecs), pt), 12)%1
                    grid.append(np.dot(lat_vecs, gpt) + offset)
        return grid
    else:
        grid = []
        z3pl = 0
        z3pu = int(f)
        for z3p in range(z3pl + 1, z3pu + 1):
            z2pl = int(e*z3p/f) # lower and upper limits
            z2pu = int(z2pl + d)
            for z2p in range(z2pl + 1, z2pu + 1):
                z1pl = int((c - b*e/d)*z3p/f + b/d*z2p)
                z1pu = int(z1pl + a)
                for z1p in range(z1pl + 1, z1pu + 1):
                    z = np.dot(inv(U), [z1p,z2p,z3p])
                    pt = np.dot(grid_vecs, z)
                    gpt = np.round(np.dot(inv(lat_vecs), pt), 12)%1
                    grid.append(gpt + offset)
        return grid
                    
def make_large_grid(cell_vectors, grid_vectors, offset, cart=True, eps = 1e-9):
    """This function is similar to make_grid except it samples a volume
    that is larger and saves the points that are thrown away in make_grid.
    It returns two grids: the first is the unique points within the first unit
    cell. The second is all the points generated outside the first unit cell.

    ARGS:
        cell_vectors (numpy.ndarray): the vectors defining the volume in which
            to sample. The vectors are the columns of the matrix.
        grid_vecs (numpy.ndarray): the vectors that generate the grid as 
            columns of the matrix..
        offset: the origin of the coordinate system in grid coordinates.
        cart (bool): if true, return the grid in Cartesian coordinates. Other-
            wise return the grid in cell coordinates.        

    Returns:
        grid (numpy.ndarray): an array of sampling-point coordinates.
        null_grid (numpy.ndarray): an array of sampling-point coordinates 
            outside volume.
    """

    # Find a grid point close to the offset
    oi = np.round(offset).astype(int)

    # Put the offset in Cartesian coordinates.
    offset = np.dot(grid_vectors, offset)

    r = np.max(norm(cell_vectors, axis=0))
    V = np.linalg.det(grid_vectors)

    # Add one to account for the offset.
    # Multiply by two to cover larger volume.
    n = np.round(np.array([norm(np.cross(grid_vectors[:,(i+1)%3],
                                grid_vectors[:,(i+2)%3]))*r/V + 1
                           for i in range(3)])*1.5).astype(int)
    grid = []
    null_grid = []
    for nv in product(range(-n[0] + oi[0], n[0] + oi[0]),
                      range(-n[1] + oi[1], n[1] + oi[1]),
                      range(-n[2] + oi[2], n[2] + oi[2])):    
        grid_pt = np.dot(grid_vectors, nv) + offset
        
        grid_pt_cell = np.round(np.dot(inv(cell_vectors), grid_pt), 15)
        if any(abs(grid_pt_cell) > 1):
            null_grid.append(grid_pt)
        else:
            grid_pt_cell = grid_pt_cell%1
            grid_pt = np.dot(cell_vectors, grid_pt_cell)
            if any([np.allclose(grid_pt, g) for g in grid]):
                continue
            else:
                grid.append(grid_pt)
            
    return (np.asarray(grid), np.asarray(null_grid))

def sphere_pts(A, r2, offset=[0.,0.,0.], eps=1e-9):
    """ Calculate all the points within a sphere that are
    given by an integer linear combination of the columns of 
    A.
    
    Args:
        A (numpy.ndarray): the columns representing basis vectors.
        r2 (float): the squared radius of the sphere.
        offset(list or numpy.ndarray): a vector that points to the center
            of the sphere in Cartesian coordinates.
        
    Returns:
        grid (list): an array of grid coordinates in cartesian
            coordinates.
    """

    offset = np.asarray(offset)
    
    # Put the offset in cell coordinates and find a cell point close to the
    # offset.
    oi= np.round(np.dot(inv(A),offset)).astype(int)
    r = np.sqrt(r2)
    V = np.linalg.det(A)
    n = [int(np.ceil(norm(np.cross(A[:,(i+1)%3],A[:,(i+2)%3]))*r/V) + 1)
         for i in range(3)]

    ints = np.array(list(product(range(-n[0] + oi[0], n[0] + oi[0] + 1),
                   range(-n[1] + oi[1], n[1] + oi[1] + 1),
                   range(-n[2] + oi[2], n[2] + oi[2] + 1))))
    
    grid = np.dot(A, ints.T).T - offset
    # grid = np.array([np.dot(A, comb) - offset for comb in ints])
    norms = np.array([np.dot(p,p) for p in grid])
    return grid[np.where(norms < (r2 + eps))]


# def sphere_pts(A,r2,offset=[0.,0.,0.]):
#     """ Calculate all the points within a sphere that are
#     given by an integer linear combination of the columns of 
#     A.
    
#     Args:
#         A (numpy.ndarray): the columns representing basis vectors.
#         r2 (float): the squared radius of the sphere.
#         offset(list or numpy.ndarray): a vector that points to the center
#             of the sphere in Cartesian coordinates.
        
#     Returns:
#         grid (list): an array of grid coordinates in cartesian
#             coordinates.
#     """

#     # This is a parameter that should help deal with rounding error.
#     eps = 1e-9
#     offset = np.asarray(offset)
    
#     # Put the offset in cell coordinates
#     oi= np.round(np.dot(inv(A),offset))
#     # Find a cell point close to the offset
#     oi = oi.astype(int)
    
#     r = np.sqrt(r2)
#     V = np.linalg.det(A)
#     n = [0,0,0]
#     for i in range(3):
#         # Add 1 because the offset was rounded to the nearest cell point.
#         n[i] = int(np.ceil(norm(np.cross(A[:,(i+1)%3],A[:,(i+2)%3]))*r/V) + 1)

#     ints = product(range(-n[0] + oi[0], n[0] + oi[0]),
#                    range(-n[1] + oi[1], n[1] + oi[1]),
#                    range(-n[2] + oi[2], n[2] + oi[2]))

#     grid = np.array([np.dot(A, comb) for comb in ints])
#     norms = np.array([np.dot(p,p) for p in grid])
#     return grid[np.where(norms < (r2 + eps))]
        
#     # grid = []
#     # for i,j,k in product(range(-n[0] + oi[0], n[0] + oi[0]),
#     #                      range(-n[1] + oi[1], n[1] + oi[1]),
#     #                      range(-n[2] + oi[2], n[2] + oi[2])):
#     #     pt = np.dot(A,[i,j,k])
#     #     if np.dot(pt+offset,pt+offset) <= r2 + eps:
#     #         grid.append(np.array(pt))
#     #     else:
#     #         continue                
#     return grid

def large_sphere_pts(A,r2,offset=[0.,0.,0.]):
    """ Calculate all the points within a sphere that are 
    given by an integer linear combination of the columns of 
    A.
    
    Args:
        A (numpy.ndarray): the grid basis with the columns 
            representing basis vectors.
        r2 (float): the squared radius of the sphere.
        offset(list or numpy.ndarray): a vector that points to the center
            of the sphere in Cartesian coordinates.
        
    Returns:
        grid (list): an array of grid coordinates in cartesian
            coordinates.
    """
    
    # This is a parameter that should help deal with rounding error.
    eps = 1e-9
    offset = np.asarray(offset)
    # Put the offset in cell coordinates
    oi= np.round(np.dot(inv(A),offset))
    # Find a cell point close to the offset
    oi = oi.astype(int)
    # scale the integers by about 100% to ensure all points are enclosed.
    scale = 2.
    imax,jmax,kmax = map(int,np.ceil(scale*np.sqrt(r2/np.sum(np.dot(A,A),0))))
    
    grid = []
    for i,j,k in product(range(-imax + oi[0],imax + oi[0]),
                         range(-jmax + oi[1],jmax + oi[1]),
                         range(-kmax + oi[2],kmax + oi[2])):
        pt = np.dot(A,[i,j,k])
        if np.dot(pt+offset,pt+offset) <= r2 + eps:
            grid.append(pt)
        else:
            continue                
    return grid

def make_cell_points(lat_vecs, grid_vecs, offset=[0,0,0], cart=True):
    """Sample within a parallelepiped using any regular grid.

    Args:
        lat_vecs (numpy.ndarray): the vectors defining the volume in which 
            to sample. The vectors are the columns of the matrix.
        grid_vecs (numpy.ndarray): the vectors that generate the grid as 
            columns of a matrix..
        offset: the offset of the coordinate system in grid coordinates.
        cart (bool): if true, return the grid in Cartesian coordinates; other-
            wise, return the grid in cell coordinates. If cart == True, the
            offset must be in Cartesian coordinates. If cart == False, the offset 
            must be in cell coordinates.

    Returns:
        grid (list): an array of sampling-point coordinates.

    Examples:
        >>> cell_centering = "face"
        >>> cell_consts = [1.]*3
        >>> cell_angles = [np.pi/2]*3
        >>> lat_vecs = make_ptvecs(cell_centering, cell_consts, cell_angles)
        >>> grid_centering = "base"
        >>> grid_consts = [cell_const/140]*3
        >>> grid_angles = [np.pi/2]*3
        >>> grid_vecs = make_ptvecs(grid_centering, grid_consts, grid_angles)
        >>> offset = .5*numpy.sum(lat_vecs,1)
        >>> grid = make_grid(lat_vecs, grid_vecs, offset)
    """

    # Offset in Cartesian coordinates
    car_offset = np.dot(grid_vecs, offset)
    
    # Offset in lattice coordinates.
    lat_offset = np.dot(inv(lat_vecs), car_offset)

    # Integer matrix
    N = np.dot(inv(grid_vecs), lat_vecs)

    # Check that N is an integer matrix.
    for i in range(len(N[:,0])):
        for j in range(len(N[0,:])):
            if np.isclose(N[i,j]%1, 0) or np.isclose(N[i,j]%1, 1):
                N[i,j] = int(np.round(N[i,j]))
            else:
                raise ValueError("The cell and grid vectors are incommensurate.")

    H, U = HermiteNormalForm(N)
    D = np.diag(H).astype(int)    
    grid = []
    if cart == True:
        # Loop through the diagonal of the HNF matrix.
        for i,j,k in product(range(D[0]), range(D[1]), range(D[2])):
            # Find the point in Cartesian coordinates.
            pt = np.dot(grid_vecs, [i,j,k]) + car_offset
            
            # Put the point in cell coordinates and move it to the 
            # first unit cell.
            # pt = np.round(np.dot(pt, inv(lat_vecs)),12)%1
            pt = np.round(np.dot(inv(lat_vecs), pt),12)%1

            # Put the point back into Cartesian coordinates.
            pt = np.dot(lat_vecs, pt)
            grid.append(pt)
        return grid
    else:
        for i,j,k in product(range(D[0]), range(D[1]), range(D[2])):
            # Find the point in cartesian coordinates.
            pt = np.dot(grid_vecs, [i,j,k])
            # Put the point in cell coordinates and move it to the 
            # first unit cell.
            # pt = np.round(np.dot(inv(lat_vecs), pt),12)%1 + offset
            pt = np.round(np.dot(inv(lat_vecs), pt) + offset, 12)%1
            grid.append(pt)
        return grid
