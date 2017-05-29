"""Various methods of sampling the Brillouin zone.
"""
 
import numpy as np
from copy import deepcopy
from itertools import product
from math import ceil

from BZI.symmetry import make_ptvecs

# Find transformation to create HNF from integer matrix.
def get_minmax_indices(a):
    """Find the maximum and minimum elements of a list that aren't zero.

    Args:
        a (numpy.ndarray): a three element numpy array.

    Returns:
        minmax (list): the minimum and maximum values of array a with the
            minimum first and maximum second.
    """
    a = np.abs(a)
    maxi = 2 - np.argmax(a[::-1])
    min = 0
    i = 0
    while min == 0:
        min = a[i]
        i += 1
    mini = i-1
    for i,ai in enumerate(a):
        if ai > 0 and ai < min:
            min = ai
            mini = i
    return np.asarray([mini, maxi])

def swap_column(M, B, k):
    """Swap the column k with whichever column has the highest value (out of
    the columns to the right of k in row k). The swap is performed for both
    matrices M and B. 

    Args:
        M (numpy.ndarray): the matrix being transformed
        B (numpy.ndarray): a matrix to keep track of the transformation steps 
            on M. 
        k (int): the column to swap, as described in summary.
    """
    
    Ms = deepcopy(M)
    Bs = deepcopy(B)
    
    # Find the index of the non-zero element in row k.
    maxidx = np.argmax(np.abs(Ms[k,k:])) + k
    tmpCol = deepcopy(Bs[:,k]);
    Bs[:,k] = Bs[:,maxidx]
    Bs[:,maxidx] = tmpCol

    tmpCol = deepcopy(Ms[:,k])
    Ms[:,k] = Ms[:, maxidx]
    Ms[:,maxidx] = tmpCol

    return Ms, Bs

def swap_row(M, B, k):
    """Swap the row k with whichever row has the highest value (out of
    the rows below k in column k). The swap is performed for both matrices M and B.

    Args:
        M (numpy.ndarray): the matrix being transformed
        B (numpy.ndarray): a matrix to keep track of the transformation steps 
            on M. 
        k (int): the column to swap, as described in summary.
    """
    
    Ms = deepcopy(M)
    Bs = deepcopy(B)
    
    # Find the index of the non-zero element in row k.
    maxidx = np.argmax(np.abs(Ms[k:,k])) + k

    tmpCol = deepcopy(Bs[k,:]);
    Bs[k,:] = Bs[maxidx,:]
    Bs[maxidx,:] = tmpCol
    
    tmpRow = deepcopy(Ms[k,:])
    Ms[k,:] = Ms[maxidx,:]
    Ms[maxidx,:] = tmpRow

    return Ms, Bs

def LowerHermiteNormalForm(S):
    """Find the Hermite normal form (HNF) of a given integer matrix and the
    matrix that mediates the transformation.

    Args:
        S (numpy.ndarray): The 3x3 integer matrix describing the relationship 
            between two commensurate lattices.
    Returns:
        H (numpy.ndarray): The resulting HNF matrix.
        B (numpy.ndarray): The transformation matrix such that H = SB.
    """
    if np.linalg.det(S) == 0:
        raise ValueError("Singular matrix passed to HNF routine")
    B = np.identity(np.shape(S)[0]).astype(int)
    H = deepcopy(S)
    
    # Keep doing column operations until all elements in the first row are zero
    # except for the one on the diagonal.
    while np.count_nonzero(H[0,:]) > 1:
        # Divide the column with the smallest value into the largest.
        minidx, maxidx = get_minmax_indices(H[0,:])
        minm = H[0,minidx]
        # Subtract a multiple of the column containing the smallest element from
        # the row containing the largest element.
        multiple = int(H[0, maxidx]/minm)
        H[:, maxidx] = H[:, maxidx] - multiple*H[:, minidx]
        B[:, maxidx] = B[:, maxidx] - multiple*B[:, minidx]
        if np.allclose(np.dot(S, B), H) == False:
            raise ValueError("COLS: Transformation matrices didn't work.")
    if H[0,0] == 0:
        H, B = swap_column(H, B, 0) # Swap columns if (0,0) is zero.
    if H[0,0] < 0:
        H[:,0] = -H[:,0]
        B[:,0] = -B[:,0]

    if np.count_nonzero(H[0,:]) > 1:
        raise ValueError("Didn't zero out the rest of the row.")
    if np.allclose(np.dot(S, B), H) == False:
        raise ValueError("COLSWAP: Transformation matrices didn't work.")
    # Now work on element H[1,2].
    while H[1,2] != 0:
        if H[1,1] == 0:
            tempcol = deepcopy(H[:,1])
            H[:,1] = H[:,2]
            H[:,2] = tempcol

            tempcol = deepcopy(B[:,1])
            B[:,1] = B[:,2]
            B[:,2] = tempcol
            if H[1,2] == 0:
                break            
        if np.abs(H[1,2]) < np.abs(H[1,1]):
            maxidx = 1
            minidx = 2
        else:
            maxidx = 2
            minidx = 1

        multiple = int(H[1, maxidx]/H[1,minidx])
        H[:,maxidx] = H[:, maxidx] - multiple*H[:,minidx]
        B[:,maxidx] = B[:, maxidx] - multiple*B[:,minidx]
        
        if np.allclose(np.dot(S, B), H) == False:
            raise ValueError("COLS: Transformation matrices didn't work.")

    if H[1,1] == 0:
        tempcol = deepcopy(H[:,1])
        H[:,1] = H[:,2]
        H[:,2] = tempcol
    if H[1,1] < 0: # change signs
        H[:,1] = -H[:,1]
        B[:,1] = -B[:,1]
    if H[1,2] != 0:
        raise ValueError("Didn't zero out last element.")
    if np.allclose(np.dot(S,B), H) == False:
        raise ValueError("COLSWAP: Transformation matrices didn't work.")
    if H[2,2] < 0: # change signs
        H[:,2] = -H[:,2]
        B[:,2] = -B[:,2]
    check1 = (np.array([0,0,1]), np.array([1,2,2]))
    if np.count_nonzero(H[check1]) != 0:
        raise ValueError("Not lower triangular")
    if np.allclose(np.dot(S, B), H) == False:
        raise ValueError("End Part1: Transformation matrices didn't work.")
    
    # Now that the matrix is in lower triangular form, make sure the lower
    # off-diagonal elements are non-negative but less than the diagonal
    # elements.
    while H[1,1] <= H[1,0] or H[1,0] < 0:
        if H[1,1] <= H[1,0]:
            multiple = 1
        else:
            multiple = -1
        H[:,0] = H[:,0] - multiple*H[:,1]
        B[:,0] = B[:,0] - multiple*B[:,1]
    for j in [0,1]:
        while H[2,2] <= H[2,j] or H[2,j] < 0:
            if H[2,2] <= H[2,j]:
                multiple = 1
            else:
                multiple = -1
            H[:,j] = H[:,j] - multiple*H[:,2]
            B[:,j] = B[:,j] - multiple*B[:,2]

    if np.allclose(np.dot(S, B), H) == False:
        raise ValueError("End Part1: Transformation matrices didn't work.")
    if np.count_nonzero(H[check1]) != 0:
        raise ValueError("Not lower triangular")
    check2 = (np.asarray([0, 1, 1, 2, 2, 2]), np.asarray([0, 0, 1, 0, 1, 2]))
    if any(H[check2] < 0) == True:
        raise ValueError("Negative elements in lower triangle.")

    if H[1,0] > H[1,1] or H[2,0] > H[2,2] or H[2,1] > H[2,2]:
        raise ValueError("Lower triangular elements bigger than diagonal.")
    return H, B

def HermiteNormalForm(S):
    """Find the Hermite normal form (HNF) of a given integer matrix and the
    matrix that mediates the transformation.

    Args:
        S (numpy.ndarray): The 3x3 integer matrix describing the relationship 
            between two commensurate lattices.
    Returns:
        H (numpy.ndarray): The resulting HNF matrix.
        B (numpy.ndarray): The transformation matrix such that H = SB.
    """
    if np.linalg.det(S) == 0:
        raise ValueError("Singular matrix passed to HNF routine")
    B = np.identity(np.shape(S)[0]).astype(int)
    H = deepcopy(S)

    #    Keep doing row operations until all elements in the first column are zero
    #    except for the one on the diagonal.
    while np.count_nonzero(H[:,0]) > 1:
        # Divide the row with the smallest value into the largest.
        minidx, maxidx = get_minmax_indices(H[:,0])
        minm = H[minidx,0]
        # Subtract a multiple of the row containing the smallest element from
        # the row containing the largest element.
        multiple = int(H[maxidx,0]/minm)
        H[maxidx,:] = H[maxidx,:] - multiple*H[minidx,:]
        B[maxidx,:] = B[maxidx,:] - multiple*B[minidx,:]
        if np.allclose(np.dot(B, S), H) == False:
            raise ValueError("ROWS: Transformation matrices didn't work.")
    if H[0,0] == 0:
        H, B = swap_row(H, B, 0) # Swap rows if (0,0) is zero.
    if H[0,0] < 0:
        H[0,:] = -H[0,:]
        B[0,:] = -B[0,:]
    if np.count_nonzero(H[:,0]) > 1:
        raise ValueError("Didn't zero out the rest of the row.")
    if np.allclose(np.dot(B,S), H) == False:
        raise ValueError("ROWSWAP: Transformation matrices didn't work.")
    # Now work on element H[2,1].
    while H[2,1] != 0:
        if H[1,1] == 0:
            temprow = deepcopy(H[1,:])
            H[1,:] = H[2,:]
            H[2,:] = temprow

            temprow = deepcopy(B[1,:])
            B[1,:] = B[2,:]
            B[2,:] = temprow
            break         
        if np.abs(H[2,1]) < np.abs(H[1,1]):
            maxidx = 1
            minidx = 2
        else:
            maxidx = 2
            minidx = 1
        
        multiple = int(H[maxidx,1]/H[minidx,1])
        H[maxidx,:] = H[maxidx,:] - multiple*H[minidx,:]
        B[maxidx,:] = B[maxidx,:] - multiple*B[minidx,:]

        if np.allclose(np.dot(B,S), H) == False:
            raise ValueError("COLS: Transformation matrices didn't work.")

    if H[1,1] == 0:
        temprow = deepcopy(H[1,:])
        H[1,:] = H[0,:]
        H[0,:] = temprow
    if H[1,1] < 0: # change signs
        H[1,:] = -H[1,:]
        B[1,:] = -B[1,:]
    if H[1,0] != 0:
        raise ValueError("Didn't zero out last element.")
    if np.allclose(np.dot(B,S), H) == False:
        raise ValueError("COLSWAP: Transformation matrices didn't work.")
    if H[2,2] < 0: # change signs
        H[2,:] = -H[2,:]
        B[2,:] = -B[2,:]
    check1 = (np.array([2,2,1]), np.array([1,0,0]))

    if np.count_nonzero(H[check1]) != 0:
        raise ValueError("Not lower triangular")
    if np.allclose(np.dot(B,S), H) == False:
        raise ValueError("End Part1: Transformation matrices didn't work.")

    # Now that the matrix is in lower triangular form, make sure the lower
    # off-diagonal elements are non-negative but less than the diagonal
    # elements.    
    while H[1,1] <= H[0,1] or H[0,1] < 0:
        if H[1,1] <= H[0,1]:
            multiple = 1
        else:
            multiple = -1
        H[0,:] = H[0,:] - multiple*H[1,:]
        B[0,:] = B[0,:] - multiple*B[1,:]
    for j in [0,1]:
        while H[2,2] <= H[j,2] or H[j,2] < 0:
            if H[2,2] <= H[j,2]:
                multiple = 1
            else:
                multiple = -1
            H[j,:] = H[j,:] - multiple*H[2,:]
            B[j,:] = B[j,:] - multiple*B[2,:]

    if np.allclose(np.dot(B, S), H) == False:
        raise ValueError("End Part1: Transformation matrices didn't work.")
    if np.count_nonzero(H[check1]) != 0:
        raise ValueError("Not lower triangular")
    check2 = (np.asarray([0, 0, 0, 1, 1, 2]), np.asarray([0, 1, 2, 1, 2, 2]))
    if any(H[check2] < 0) == True:
        raise ValueError("Negative elements in lower triangle.")
    if H[0,1] > H[1,1] or H[0,2] > H[2,2] or H[1,2] > H[2,2]:
        raise ValueError("Lower triangular elements bigger than diagonal.")
    return H, B

def make_grid(cell_vecs, grid_vecs, offset, cart=True):
    """Sample within a parallelepiped using any regular grid.

    Args:
        cell_vecs (numpy.ndarray): the vectors defining the volume in which 
            to sample. The vectors are the columns of the matrix.
        grid_vecs (numpy.ndarray): the vectors that generate the grid as 
            columns of the matrix..
        offset: the offset of the coordinate system in Cartesian or lattice
            coordinates.
        cart (bool): if true, return the grid in Cartesian coordinates; other-
            wise, return the grid in lattice coordinates.

    Returns:
        grid (list): an array of sampling-point coordinates.

    Examples:
        >>> cell_centering = "face"
        >>> cell_consts = [1.]*3
        >>> cell_angles = [np.pi/2]*3
        >>> cell_vecs = make_ptvecs(cell_centering, cell_consts, cell_angles)
        >>> grid_centering = "base"
        >>> grid_consts = [cell_const/140]*3
        >>> grid_angles = [np.pi/2]*3
        >>> grid_vecs = make_ptvecs(grid_centering, grid_consts, grid_angles)
        >>> offset = .5*numpy.sum(cell_vecs,1)
        >>> grid = make_grid(cell_vecs, grid_vecs, offset)
    """
    
    # Integer matrix
    N = np.dot(np.linalg.inv(grid_vecs), cell_vecs)
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
    cell_const = np.linalg.norm(cell_vecs[:,0])

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
                    z = np.dot(np.linalg.inv(U), [z1p,z2p,z3p])
                    pt = np.dot(grid_vecs, z)
                    gpt = np.dot(np.linalg.inv(cell_vecs), pt)%1.
                    grid.append(np.dot(cell_vecs, gpt) - offset)
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
                    z = np.dot(np.linalg.inv(U), [z1p,z2p,z3p])
                    pt = np.dot(grid_vecs, z)
                    gpt = np.round(np.dot(np.linalg.inv(cell_vecs), pt), 12)%1
                    grid.append(gpt - offset)
        return grid
                    
def make_large_grid(cell_vectors, grid_vectors, offset, cart=True):
    """This function is similar to make_grid except it samples a volume
    that is larger and saves the points that are thrown away in make_grid.

    Args:
        cell_vecs (numpy.ndarray): the vectors defining the volume in which 
            to sample. The vectors are the columns of the matrix.
        grid_vecs (numpy.ndarray): the vectors that generate the grid as 
            columns of the matrix..
        offset: the origin of the coordinate system in Cartesian or lattice
            coordinates.
        cart (bool): if true, return the grid in Cartesian coordinates. Other-
            wise return the grid in lattice coordinates.        

    Returns:
        grid (numpy.ndarray): an array of sampling-point coordinates.
        null_grid (numpy.ndarray): an array of sampling-point coordinates outside
            volume.
    """
    
    grid = []
    null_grid = []
    cell_lengths = np.array([np.linalg.norm(cv) for cv in cell_vectors])
    cell_directions = [c/np.linalg.norm(c) for c in cell_vectors]
    # G is a matrix of lattice vector components along the cell vector
    # directions.
    G = np.asarray([[abs(np.dot(g, c)) for c in cell_directions] for g in grid_vectors])

    # Find the optimum integer values for creating the grid.
    n = [0,0,0]
    for i in range(len(n)):
        index = np.where(np.asarray(G) == np.max(G))
        # The factor 5./4 makes the sampling volume larger.
        n[index[1][0]] = int(cell_lengths[index[1][0]]/np.max(G)*(5./4))
        G[index[0][0],:] = 0.
        G[:,index[1][0]] = 0.
        
    for nv in product(range(-n[0], n[0]+1), range(-n[1], n[1]+1),
                      range(-n[2], n[2]+1)):
        # Sum across columns since we're adding up components of each vector.
        grid_pt = np.sum(grid_vectors*nv, 1) - offset
        projections = [np.dot(grid_pt,c) for c in cell_directions]
        projection_test = np.alltrue([abs(p) <= cl/2 for (p,cl) in zip(projections, cell_lengths)])
        if projection_test == True:
            grid.append(grid_pt)
        else:
            null_grid.append(grid_pt)
    return (np.asarray(grid), np.asarray(null_grid))

def sphere_pts(A,r2,offset=[0.,0.,0.]):
    """ Calculate all the points within a sphere that are 
    given by an integer linear combination of the columns of 
    A.
    
    Args:
        A (numpy.ndarray): the columns representing basis vectors.
        r2 (float): the squared radius of the sphere.
        offset(list or numpy.ndarray): a vector that points to the center
            of the sphere.
        
    Returns:
        grid (list): an array of grid coordinates in cartesian
            coordinates.
    """
    
    # This is a parameter that should help deal with rounding error.
    eps = 1e-9
    offset = np.asarray(offset)
    
    # Put the offset in lattice coordinates
    oi= np.round(np.dot(np.linalg.inv(A),offset))
    # Find a lattice point close to the offset
    oi = oi.astype(int)
    
    r = np.sqrt(r2)
    V = np.linalg.det(A)
    n = [0,0,0]
    for i in range(3):
        # Add 1 because the offset was rounded to the nearest lattice point.
        n[i] = int(np.ceil(np.linalg.norm(np.cross(A[:,(i+1)%3],A[:,(i+2)%3]))*r/V) + 1)

    grid = []
    for i,j,k in product(range(-n[0] + oi[0], n[0] + oi[0]),
                         range(-n[1] + oi[1], n[1] + oi[1]),
                         range(-n[2] + oi[2], n[2] + oi[2])):
        pt = np.dot(A,[i,j,k])
        if np.dot(pt-offset,pt-offset) <= r2 + eps:
            grid.append(np.array(pt))
        else:
            continue                
    return grid

def large_sphere_pts(A,r2,offset=[0.,0.,0.]):
    """ Calculate all the points within a sphere that are 
    given by an integer linear combination of the columns of 
    A.
    
    Args:
        A (numpy.ndarray): the grid basis with the columns 
            representing basis vectors.
        r2 (float): the squared radius of the sphere.
        offset(list or numpy.ndarray): a vector that points to the center
            of the sphere.
        
    Returns:
        grid (list): an array of grid coordinates in cartesian
            coordinates.
    """
    
    # This is a parameter that should help deal with rounding error.
    eps = 1e-9
    offset = np.asarray(offset)
    # Put the offset in lattice coordinates
    oi= np.round(np.dot(np.linalg.inv(A),offset))
    # Find a lattice point close to the offset
    oi = oi.astype(int)
    # scale the integers by about 100% to ensure all points are enclosed.
    scale = 2.
    imax,jmax,kmax = map(int,np.ceil(scale*np.sqrt(r2/np.sum(np.dot(A,A),0))))
    
    grid = []
    for i,j,k in product(range(-imax + oi[0],imax + oi[0]),
                         range(-jmax + oi[1],jmax + oi[1]),
                         range(-kmax + oi[2],kmax + oi[2])):
        pt = np.dot(A,[i,j,k])
        if np.dot(pt-offset,pt-offset) <= r2 + eps:
            grid.append(pt)
        else:
            continue                
    return grid

# Wiley's code.
def make_cell_points(lat_vecs, grid_vecs):
    """Makes the points of a grid inside the volume of the lattice vectors.
    
    Args:
        lat_vecs (numpy.ndarray): A matrix whose columns hold the basis vectors
            of the lattice in Cartesian coordinates.
        gridp_vecs (numpy.ndarray): A matrix whose columns hold the basis vectors
            of the grid in Cartesian coordinates.
            
    Returns:
        points (numpy.ndarray): The grid points inside the lattice expressed in
            cartesian coordinates.
    """
    
    diags = _get_HNF_diags(lat_vecs, grid_vecs)
    
    v1 = grid_vecs[:,0]
    v2 = grid_vecs[:,1]
    v3 = grid_vecs[:,2]
    
    points = []
    for i in range(diags[0]):
        for j in range(diags[1]):
            for k in range(diags[2]):
                points.append(np.dot(np.linalg.inv(lat_vecs),v1*i+v2*j+v3*k))
                
    points = np.array(points)%1
    return points

def _get_HNF_diags(lat_vecs, grid_vecs):
    """Find the diagonals of the HNF that makes the grid vector's supercell form
    the parent lattice.
    
    Args:
        lat_vecs (numpy.ndarray): The parent lattice vectors as columns of a 
            matrix.
        grid_vecs (numpy.ndarray): The grid vectors as columns of a matrix.
        
    Returns:
        HNF_diags (list): The diagonals of the HNF.
    """
    
    parent = lat_vecs
    super_cell = np.transpose(np.linalg.inv(grid_vecs))
    HNF = np.dot(np.linalg.inv(parent),super_cell)
    HNF_diags = np.diagonal(HNF)
    
    return [int(i) for i in HNF_diags]
