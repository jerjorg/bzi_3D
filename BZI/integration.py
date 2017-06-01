"""A variety of integration methods along with related quatities.
"""

import numpy as np
from math import ceil
from BZI.sampling import HermiteNormalForm

def rectangle_method(PP, cell_vecs, grid_vecs, offset, Fermi_level, neigvals):
    """Sample within a parallelepiped using any regular mesh and integrate with
    the rectangular method. The advantage of rectangle_method over
    rectangular_method is it saves memory.

    Args:
        PP (function): the pseudopotential function
        cell_vecs (numpy.ndarray): the vectors defining the volume in which 
            to sample. The vectors are the columns of the matrix.
        grid_vecs (numpy.ndarray): the vectors that generate the mesh as 
            columns of the matrix..
        offset (list): the offset of the grid (should be comparable to the size of the
            grid vectors.
        Fermi_level (float): the energy at which to cutoff the function.
        neigvals (int): the number of eigenvalues returned by the empirical
            pseudopotential.

    Returns:
        integral (float): the value from integrating the pseudopotential up
            to the Fermi level.

    Examples:
        >>> from BZI.pseudopots import W1
        >>> Fermi_level = 4.
        >>> cell_type = "fcc"
        >>> cell_const = 1.
        >>> cell_vecs = make_ptvecs(cell_type, cell_const)
        >>> mesh_type = "bcc"
        >>> mesh_const = cell_const/4
        >>> grid_vecs = make_ptvecs(mesh_type, mesh_const)
        >>> offset = [0.5, 0.5, 0.5]
        >>> integral = rectangle_method(W1, cell_vecs, grid_vecs, offset, Fermi_level)
    """

    # This function is almost an exact copy of make_grid in sampling.py.
    # The benefit of this function is every mesh point isn't saved.
    
    # Make sure vectors that define the grid are commensurate with the cell
    # vectors by checking that N is an integer matrix.
    N = np.dot(np.linalg.inv(grid_vecs), cell_vecs)
    for i in range(len(N[:,0])):
        for j in range(len(N[0,:])):
            if np.isclose(N[i,j]%1, 0) or np.isclose(N[i,j]%1, 1):
                N[i,j] = int(np.round(N[i,j]))
            else:
                raise ValueError("The cell and grid vectors are incommensurate.")

    # H is an HNF and U is the transform.
    L, U = HermiteNormalForm(N)
    a = L[0,0]
    b = L[0,1]
    c = L[0,2]
    d = L[1,1]
    e = L[1,2]
    f = L[2,2]
    cell_const = np.linalg.norm(cell_vecs[:,0])

    # Integration variables.
    integral = 0.
    npts = 0
    
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
                gpt = np.dot(np.linalg.inv(cell_vecs), pt)%1
                kpt = np.dot(cell_vecs, gpt) - offset                
                npts += 1
                integral += sum(filter(lambda x: x <= Fermi_level, PP(kpt, neigvals)))
    return npts, np.linalg.det(cell_vecs)/(npts)*integral

def rectangular_method(PP, neigvals, grid, cell_vecs, Fermi_level):
    """Integrate a pseudopotential within a cell below the Fermi level.
    
    Args:
        PP (function): the pseudopotential
        neigvals (int): the number of eigenvalues returned by the 
            pseudopotential.
        grid (list): a list of grid points
        cell_vecs (numpy.ndarray): a 3x3 numpy array with cell vectors as 
            columns
        Fermi_level (float): the cutoff energy or Fermi level
    """
    
    integral = 0
    for kpt in grid:
        integral += sum(filter(lambda x: x <= Fermi_level, PP(kpt, neigvals)))
    return np.linalg.det(cell_vecs)/len(grid)*integral

def simple_fl(PP, grid, neigvals, nvalence):
    """Find the energy at which the toy band structure it cut.
    
    Args:
        PP (function): the pseudopotential
        grid (list): the grid points
        neigvals (int): the number of eigenvalues returned by the pseudopotential
        nvalence (int): the number of valence electrons
    Return:
        (float) the Fermi level
    """
    
    C = ceil(nvalence*len(grid)/2)
    energies = np.array([])
    for g in grid:
        energies = np.concatenate((energies, PP(g, neigvals)))
    return np.sort(energies)[C-1] # C -1 since python is zero based

def monte_carlo(PP, cell_vecs, npts, Fermi_level):
    """Integrate a function using Monte Carlo sampling. Only works for integrations
    from 0 to 1 in all 3 directions.
    """
    
    integral = 0.
    for _ in range(npts):
        kpt = [np.random.random() - .5 for _ in range(3)]
        integral += sum(filter(lambda x: x <= Fermi_level, PP(kpt)))
    return np.linalg.det(cell_vecs)/(npts)*integral
