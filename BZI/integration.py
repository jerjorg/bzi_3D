"""A variety of integration methods along with related quatities.
"""

import numpy as np
from math import ceil
from BZI.sampling import HermiteNormalForm

def rectangular_method(PP, grid):
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
    neigvals = int(np.ceil(PP.nvalence_electrons/2.))
    for kpt in grid:
        integral += sum(filter(lambda x: x <= PP.fermi_level, PP.eval(kpt, neigvals)))
    return np.linalg.det(PP.lattice.reciprocal_vectors)/len(grid)*integral

def rectangular_fermi_level(PP, grid):
    """Find the energy at which the toy band structure it cut.
    
    Args:
        PP (function): the pseudopotential
        grid (list): the grid points
        neigvals (int): the number of eigenvalues returned by the pseudopotential
        nvalence (int): the number of valence electrons
    Return:
        (float) the Fermi level
    """
    
    C = int(ceil(PP.nvalence_electrons*len(grid)/2.))
    neigvals = int(np.ceil(PP.nvalence_electrons/2)+1)
    energies = np.array([])
    for g in grid:
        energies = np.concatenate((energies, PP.eval(g, neigvals)))
    return np.sort(energies)[C-1] # C -1 since python is zero based

def monte_carlo(PP, npts):
    """Integrate a function using Monte Carlo sampling. Only works for integrations
    from 0 to 1 in all 3 directions.
    """
    
    integral = 0.
    for _ in range(npts):
        kpt = [np.random.random() - .5 for _ in range(3)]
        integral += sum(filter(lambda x: x <= Fermi_level, PP(kpt)))
    return np.linalg.det(cell_vecs)/(npts)*integral
