"""A variety of integration methods along with related quatities.
"""

import numpy as np
from math import np.ceil
from BZI.sampling import HermiteNormalForm

def rectangular_method(PP, grid, weights):
    """Integrate a pseudopotential within a cell below the Fermi level.
    
    Args:
        PP (function): the empirical pseudopotential.
        grid (list): a list of grid points.
        weights(list): a list of k-point weights in the same order as grid.
    """

    integral = 0
    neigvals = np.ceil(np.round(PP.nvalence_electrons/2., 3)).astype(int)
    C = np.ceil(np.round(PP.nvalence_electrons*np.sum(weights)/2., 3)).astype(int)
    nstates = 0
    last_state_indices = []
    for i,kpt in enumerate(grid):
        
        filled_states = weights[i]*len(list(filter(lambda x: x <= PP.fermi_level,
                                          PP.eval(kpt, neigvals))))

        if any([np.isclose(PP.fermi_level, en) for en in filter( lambda x: x <= PP.fermi_level, PP.eval(kpt, neigvals) )]):
            last_state_indices.append(i)
            continue        
        nstates += filled_states
        integral += weights[i]*sum(filter(lambda x: x <= PP.fermi_level,
                                          PP.eval(kpt, neigvals)))
        
    # Loop over states that have energies near the Fermi level.
    for i in last_state_indices:
        filled_states = weights[i]*len(list(filter(lambda x: x <= PP.fermi_level,
                                                       PP.eval(grid[i], neigvals))))
        if filled_states + nstates < C:            
            nstates += filled_states
            integral += weights[i]*sum(filter(lambda x: x <= PP.fermi_level,
                                              PP.eval(grid[i], neigvals)))
        else:
            weight = C - nstates
            filled_states = weight*len(list(filter(lambda x: x <= PP.fermi_level,
                                                       PP.eval(grid[i], neigvals))))

            nstates += filled_states
            integral += weight*sum(filter(lambda x: x <= PP.fermi_level,
                                          PP.eval(grid[i], neigvals)))
            break
    return np.linalg.det(PP.lattice.reciprocal_vectors)/np.sum(weights)*integral
    
def rectangular_fermi_level(PP, grid, weights, eps=1e-9):
    """Find the energy at which the toy band structure it cut.
    
    Args:
        PP (function): the pseudopotential
        grid (list): the grid points
        neigvals (int): the number of eigenvalues returned by the pseudopotential
        nvalence (int): the number of valence electrons
    Return:
        (float) the Fermi level
    """
    
    C = np.ceil(np.round(PP.nvalence_electrons*np.sum(weights)/2., 3)).astype(int)
    neigvals = np.ceil(np.round(PP.nvalence_electrons/2+1, 3))).astype(int)
    energies = np.array([])
    for i,g in enumerate(grid):        
        energies = np.concatenate((energies, list(PP.eval(g, neigvals))*weights[i]))
    return np.sort(energies)[C-1] # + eps# C -1 since python is zero based

def monte_carlo(PP, npts):
    """Integrate a function using Monte Carlo sampling. Only works for integrations
    from 0 to 1 in all 3 directions.
    """
    
    integral = 0.
    for _ in range(npts):
        kpt = [np.random.random() - .5 for _ in range(3)]
        integral += sum(filter(lambda x: x <= Fermi_level, PP(kpt)))
    return np.linalg.det(cell_vecs)/(npts)*integral
