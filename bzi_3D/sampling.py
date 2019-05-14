"""Various methods of sampling the Brillouin zone.
"""
 
import numpy as np
from numpy.linalg import inv, norm
from copy import deepcopy
import itertools as it
from math import ceil
import os, pickle

from bzi_3D.symmetry import (make_ptvecs, UpperHermiteNormalForm, HermiteNormalForm,
                          just_map_to_bz, bring_into_cell, check_commensurate)

# make_grid has a bug for some triclinic lattices. Fix then uncomment.
def make_grid(rlat_vecs, grid_vecs, offset, coords="Cart", rtol=1e-5, atol=1e-8):
    """Create a regular grid within a parallelepiped.

    Args:
        rlat_vecs (numpy.ndarray): the vectors defining the volume in which 
            to sample. The vectors are the columns of the array.
        grid_vecs (numpy.ndarray): the vectors that generate the grid as 
            columns of the matrix..
        offset: the offset of the coordinate system in grid coordinates. The offset can
            move points outside the first unit cell.
        coords (str): a string indicating the coordinate system of the returned k-points.
            Options include Cartesian ("Cart") and lattice ("lat") coordinates.
        rtol (float): a relative tolerance for checking if the grid and lattice vectors
            are commensurate and for bringing points into the first unit cell.
        atol (float): an absolute tolerance for checking if the grid and lattice vectors
            are commensurate and for bringing points into the first unit cell.

    Returns:
        grid (numpy.ndarray): an array of point coordinates in 3-space.

    Examples:
        >>> cell_centering = "face"
        >>> cell_consts = [1.]*3
        >>> cell_angles = [np.pi/2]*3
        >>> rlat_vecs = make_ptvecs(cell_centering, cell_consts, cell_angles)
        >>> grid_centering = "base"
        >>> grid_consts = [cell_const/140]*3
        >>> grid_angles = [np.pi/2]*3
        >>> grid_vecs = make_ptvecs(grid_centering, grid_consts, grid_angles)
        >>> offset = .5*numpy.sum(rlat_vecs,1)
        >>> grid = make_grid(rlat_vecs, grid_vecs, offset)
    """    

    # Put the offset in Cartesian coordinates.
    offset_car = np.dot(grid_vecs, offset)

    # Put the offset in lattice coordinates.
    offset_lat = np.dot(inv(rlat_vecs), offset_car)

    # Check that the lattice and grid vectors are commensurate.
    check, N = check_commensurate(grid_vecs, rlat_vecs, rtol=rtol, atol=atol)
    if not check:
        msg = "The lattice and grid vectors are incommensurate."
        raise ValueError(msg.format(grid_vectors))        
            
    # H is an HNF and U is the transform.
    H, U = HermiteNormalForm(N)
    a = H[0,0]
    b = H[0,1]
    c = H[0,2]
    d = H[1,1]
    e = H[1,2]
    f = H[2,2]

    grid = []
    
    # Lower and upper limits for first loop of integers
    z3pl = 0 
    z3pu = int(f)
    
    # for z3p in range(z3pl + 1, z3pu + 1):
    for z3p in range(z3pl, z3pu):
        # Lower and upper limits
        z2pl = int(e*z3p/f) 
        z2pu = int(z2pl + d)
        
        # for z2p in range(z2pl + 1, z2pu + 1):
        for z2p in range(z2pl, z2pu):
            # Lower and upper limits
            z1pl = int((c - b*e/d)*z3p/f + b/d*z2p)
            z1pu = int(z1pl + a)
            
            # for z1p in range(z1pl + 1, z1pu + 1):
            for z1p in range(z1pl, z1pu):
                
                z = np.dot(inv(U), [z1p,z2p,z3p])
                pt = bring_into_cell(np.dot(grid_vecs, z), rlat_vecs,
                                     rtol=rtol, atol=atol) + offset_lat
                grid.append(pt)

    grid = np.array(grid)

    if coords == "Cart":
        return grid
    elif coords == "lat":
        grid = np.dot(inv(rlat_vecs), grid.T).T
    else:
        raise ValueError("Coordinate options include 'Cart' and 'lat'.")

    return grid
                    
def make_large_grid(cell_vectors, grid_vectors, offset, cart=True):
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
    for nv in it.product(range(-n[0] + oi[0], n[0] + oi[0]),
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

def large_sphere_pts(A, r2, offset=[0.,0.,0.], eps=1e-12):
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
    r = 2*np.sqrt(r2)
    V = np.linalg.det(A)
    n = [int(np.ceil(norm(np.cross(A[:,(i+1)%3],A[:,(i+2)%3]))*r/V) + 1)
         for i in range(3)]

    ints = np.array(list(it.product(range(-n[0] + oi[0], n[0] + oi[0] + 1),
                   range(-n[1] + oi[1], n[1] + oi[1] + 1),
                   range(-n[2] + oi[2], n[2] + oi[2] + 1))))
    
    grid = np.dot(A, ints.T).T - offset
    norms = np.array([np.dot(p,p) for p in grid])
    return grid[np.where(norms < (r2 + eps))] + offset

def sphere_pts(A, r2, offset=[0.,0.,0.], eps=1e-12):
    """ Calculate all the points within a sphere that are
    given by an integer linear combination of the columns of 
    A.
    
    Args:
        A (numpy.ndarray): the columns representing basis vectors.
        r2 (float): the squared radius of the sphere.
        offset(list or numpy.ndarray): a vector that points to the center
            of the sphere in Cartesian coordinates.
        offset (numpy.ndarray): the center of the sphere.
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
    n = [int(np.ceil(norm(np.cross(A[:,(i+1)%3],A[:,(i+2)%3]))*r/V) + 10)
         for i in range(3)]

    ints = np.array(list(it.product(range(-n[0] + oi[0], n[0] + oi[0] + 1),
                   range(-n[1] + oi[1], n[1] + oi[1] + 1),
                   range(-n[2] + oi[2], n[2] + oi[2] + 1))))
    
    grid = np.dot(A, ints.T).T - offset
    norms = np.array([np.dot(p,p) for p in grid])
    
    return grid[np.where(norms < (r2 + eps))] + offset

def make_cell_points(lat_vecs, grid_vecs, offset=[0,0,0], cart=True, rtol=1e-5, atol=1e-8):
    """Sample within a parallelepiped using any regular grid. If the offset is such that
    the sampling points are moved outside the cell, they are translated back into the 
    unit cell.

    Args:
        lat_vecs (numpy.ndarray): the vectors defining the volume in which 
            to sample. The vectors are the columns of the matrix.
        grid_vecs (numpy.ndarray): the vectors that generate the grid as 
            columns of a matrix..
        offset (numpy.ndarray): the offset of the coordinate system in grid coordinates.
        cart (bool): if true, return the grid in Cartesian coordinates; other-
            wise, return the grid in cell coordinates. 
        rtol (float): the relative tolerance used for determining if the lattice and grid
            generating vectors are commensurate.
        atol (float): an absolute tolerance used for determining if the lattice and grid
            generating vectors are commensurate.

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

    # Check that the lattice and grid vectors are commensurate.
    check, N = check_commensurate(grid_vecs, lat_vecs, rtol=rtol, atol=atol)
    if not check:
        msg = "The lattice and grid vectors are incommensurate."
        raise ValueError(msg.format(grid_vecs))        

    H, U = HermiteNormalForm(N)
    D = np.diag(H).astype(int)    
    grid = []
    # Loop through the diagonal of the HNF matrix.
    for i,j,k in it.product(range(D[0]), range(D[1]), range(D[2])):
        # Find the point in Cartesian coordinates.
        pt = np.dot(grid_vecs, [i,j,k]) + car_offset
        
        # Put the point in cell coordinates and move it to the 
        # first unit cell.
        # pt = np.round(np.dot(pt, inv(lat_vecs)),12)%1
        pt = np.round(np.dot(inv(lat_vecs), pt),12)%1
        
        # Put the point back into Cartesian coordinates.
        if cart:
            pt = np.dot(lat_vecs, pt)
        grid.append(pt)
    return grid

def get_EPM_grid_energies(EPM, ndivs, neigvals, save_dir=None):
    """Create a grid in the Brillouin zone and get a list of 
    eigenvalue energies at the points on the grid.
    
    Args:
        EPM (object): an empirical pseudopotential object.
        ndivs (int): the number of divisions in constructing the grid.
            The size of the grid is ndivs**3.
        neigvals (int): the number of eigenvalues to save for each sampling
            point.
        file_name (str): if a string is provided, save the grid and energies
            at the location provided.
            
    Returns:
        grid (list): an approximately uniformly spaced grid in the BZ.
        all_energies (numpy.ndarray): a list of the eigenenergies at the
            positions in grid in the same order.
    """

    lat_vecs = EPM.lattice.vectors
    rlat_vecs = EPM.lattice.reciprocal_vectors

    grid_vecs = rlat_vecs/ndivs

    # This offset is probably unnecessary since the grid is going to be mapped to the 
    # Brillouin zone anyway.
    offset = np.dot(np.linalg.inv(grid_vecs), np.dot(rlat_vecs, [-0.5]*3)) + [0.5]*3
    grid = make_grid(rlat_vecs, grid_vecs, offset)
    
    # Plot the mesh in the origin-centered unit cell.
    # plot_offset = np.dot(rlat_vecs, [-0.5]*3)
    # plot_mesh(grid, rlat_vecs, plot_offset)    
    
    # Map grid to Brillouin zone
    bz_grid = just_map_to_bz(grid, rlat_vecs, eps=10)
        
    # Plot the grid in the Brilloun zone.
    # plot_all_bz(lat_vecs, grid=bz_grid, convention="angular")
    
    # Put all the energy eigenvalues in a list.
    all_energies = np.array([EPM.eval(pt, neigvals) for pt in bz_grid])

    if save_dir is not None:
        data = [bz_grid, all_energies]
        file_name = os.path.join(save_dir, EPM.material + ".p")
        with open(file_name, "wb") as file:
            pickle.dump(data, file)
    
    return bz_grid, all_energies
