"""Module related to Blochl's improved linear tetrahedron method."""

import numpy as np
from numpy.linalg import norm
from itertools import product
from copy import deepcopy
import math

from BZI.symmetry import get_orbits, bring_into_cell

def find_tetrahedra(vertices):
    """Determine how a parallelepiped should be split into tetrahedra by
    finding its shortest diagonal, which is common to all the tetrahedra.

    Args:
        vertices (list or numpy.ndarray): an ordered list of vertices of the 
            parallelepiped. If the parallelepiped were a cube with edge length
            1, the vertices would be ordered as follows: [[0,0,0], [1,0,0],
            [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1]].
    
    Returns:
        tetrahedra (numpy.ndarray): a list of vertex indices. The points for the
            cube of edge length one are indexed [0,0,0] -> 1, [1,0,0] -> 2, 
            [0,1,0] -> 3, and so forth.
    """
    
    diagonal_indices = [[1,8],[2,7],[3,6],[4,5]]
    diagonal_lengths = [norm(vertices[d[1]-1] - vertices[d[0]-1]) for d in
                        diagonal_indices]
    shared_diagonal = diagonal_indices[np.where(diagonal_lengths == 
                                                min(diagonal_lengths))[0][0]]

    if shared_diagonal == [1,8]:
        tetrahedra = np.sort([[2,4,1,8],
                              [4,1,3,8],
                              [1,3,8,7],
                              [1,8,5,7],
                              [1,6,8,5],
                              [2,1,6,8]])
    elif shared_diagonal == [2,7]:
        tetrahedra = np.sort([[4,3,2,7],
                              [3,2,1,7],
                              [2,1,7,5],
                              [2,7,6,5],
                              [2,8,7,6],
                              [4,2,8,7]])
    elif shared_diagonal == [3,6]:
        tetrahedra = np.sort([[1,2,3,6],
                              [2,3,4,6],
                              [3,4,6,8],
                              [3,6,7,8],
                              [3,5,6,7],
                              [1,3,5,6]])
    elif shared_diagonal == [4,5]:
        tetrahedra = np.sort([[3,1,4,5],
                              [1,4,2,5],
                              [4,2,5,6],
                              [4,5,8,6],
                              [4,7,5,8],
                              [3,4,7,5]])
    else:
        msg = "Invalid indices for the diagonal shared among all tetrahedra."
        raise ValueError(msg.format())
    return tetrahedra

def number_of_states(VG, VT, energies, e):
    """Calculate the contribution of an individual tetrahedron to the total number
    of states. These weights differ from those in Blochl's paper by a factor of 2.
    This factor comes from spin degeneracy.
    
    Args:
        VG (float): the volume of the unit cell.
        VT (float): the volume of the tetrahedron.
        energies (list): the energies at the corners of the tetrahedron ordered from
            least to greatest.
        e (float): the energy at which the total number of states is being calculated.

    Returns:
        (float): the number of states from a given tetrahedron.
    """

    VG = float(VG)
    VT = float(VT)
    e = float(e)
    e1 = float(energies[0])
    e2 = float(energies[1])
    e3 = float(energies[2])
    e4 = float(energies[3])
    
    if e < e1:
        return 0.

    elif e1 <= e < e2:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1
        return 2.*(VT/VG)*(e - e1)**3/(e21*e31*e41)
        
    elif e2 <= e < e3:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1
        e32 = e3 - e2
        e42 = e4 - e2
        
        return 2.*(VT/VG)*1/(e31*e41)*(e21**2 + 3*e21*(e - e2) + 3*(e - e2)**2 
                                  - (e31 + e42)/(e32*e42)*(e - e2)**3) 
    elif e3 <= e < e4:
        e41 = e4 - e1
        e42 = e4 - e2
        e43 = e4 - e3
        return 2.*(VT/VG)*(1 - (e4 - e)**(3.)/(e41*e42*e43))
    
    else:
        return 2.*(VT/VG)

def find_adjacent_tetrahedra(tetrahedra_list, k):
    """For each k point in the grid, this function generates a list of
    tetrahedra indices for each tetrahedron containing the k point.

    Args:
        tetrahedra_list (list of lists of ints): a list of quadruples.
            There is exactly one quadruple for every tetrahedron. Each
            quadruple is a list of the grid_point indices at the corners of
            the tetrahedron.
        k (int): the k-point index.

    Returns:
        adjacent_tetrahedra (list): a list of tetrahedra containing the 
        k-point index.
    """

    adjacent_tetrahedra = []
    for tet in tetrahedra_list:
        for ki in tet:
            if ki == k:
                adjacent_tetrahedra.append(tet)
    adjacent_tetrahedra = [t.tolist() for t in adjacent_tetrahedra]

    return adjacent_tetrahedra


def convert_tet_index(tetind, ndiv0):
    """Convert the index of a k-point into its position in the array containing
    all the k-points in the grid. Only applicable to Blochl's tetrahedron method.
    
    Args:
        tetind (int): the index of the tetrahedron
        ndivs (list or numpy.ndarray): a array of the number of divisions made creating
            the grid.
    Return:
        tetind (int): the position of the k-point in the grid.
    """

    ndiv0 = np.array(ndiv0)
    ndiv1 = ndiv0 + 1
    npts = np.prod(ndiv1)
    ndiv2 = ndiv0 + 2
    ndiv3 = ndiv0 + 3
    if tetind < npts:
        page = 1
        column = 1
        while tetind >= ndiv1[0]*ndiv1[1]:
            page += 1
            tetind -= (ndiv1[0]*ndiv1[1])
        while tetind >= ndiv1[0]:
            column += 1
            tetind -= ndiv1[0]
        return page*ndiv3[0]*ndiv3[1] + column*ndiv3[0] + 1 + tetind
    else:
        return tetind - npts


def corrected_integration_weights(PP, grid, tetrahedra_list, tetrahedron,
                                  iband, ndiv0):
    """Determine the corrected integration weights of a single tetrahedron and 
    band.
    
    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object.
            tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        grid (numpy.ndarray): a Monkhorst-Pack grid.
        tetrahedra_list (list): a list of tetrahedra where each tetrahedra is a
            quadruple of integers.
        tetrahedron (list): the tetrahedron being considered.
        iband (int): the index of the band being considered.
        
    Returns:
        energies (list): the energies at the vertices of the provided
            tetrahedron and band index.
        weights (list): the integration weights of the corners of the 
            tetrahedron.
    """    
    
    tetrahedron = np.array(tetrahedron)
    
    # Convert the indices of the tetrahedron from 0-npts to the index of the
    # k-point in the extended grid.
    tet_grid_indices = np.array([0]*4)
    for i,ki in enumerate(tetrahedron):
        tet_grid_indices[i] = convert_tet_index(ki, ndiv0)
    
    npts = np.prod(ndiv0)
    ntets = npts*6
    
    VT = PP.lattice.reciprocal_volume/ntets
    VG = PP.lattice.reciprocal_volume
    neigvals = int(np.ceil(PP.nvalence_electrons/2)+1)

    # Find the correction to the integration weights. This is done according to
    # Eq. 22 in Blochl's improved tetrahedron paper.

    # We're going to calculate the weight correction for the k-points of
    # the tetrahedron being considered.
    corrections = [0.]*4
    energies = [PP.eval(grid[ki], iband+1)[iband] for ki in tet_grid_indices]
    tet_grid_indices = tet_grid_indices[np.argsort(energies)]
    tetrahedron = tetrahedron[np.argsort(energies)]
    energies = np.sort(energies)
    
    for i,ki in enumerate(tetrahedron):
        en = energies[i]
        adjacent_tetrahedra = find_adjacent_tetrahedra(tetrahedra_list, ki)        
        for tet in adjacent_tetrahedra:
            adj_tet_energies = []
            t_correction = 0
            for kj in tet:
                kindex = convert_tet_index(kj, ndiv0)
                enj = PP.eval(grid[kindex], neigvals)[iband]
                adj_tet_energies.append(enj)
                t_correction += enj - en
            adj_tet_energies = np.sort(adj_tet_energies)
            t_correction *= density_of_states(VG, VT, adj_tet_energies, PP.fermi_level)
            corrections[i] += t_correction
            
    corrections = np.array(corrections)/40
    uncorrected_weights = np.array(integration_weights(VT, energies, PP.fermi_level))
    
    return energies, uncorrected_weights + corrections


def integration_weights(VT, energies, eF):
    """Determine the integration weights of a single tetrahedron and band.
    
    Args:
        energies (list): a list of energies at the corners of the tetrahedron ordered
            from least to greatest.
        eF (float): the Fermi level or Fermi energy.

        
    Returns:
        (list): the integration weights of the corners of the tetrahedron.
    """

    # Find the correction to the integration weights. This is done according to
    # Eq. 22 in Blochl's improved tetrahedron paper.
    
    # In this equation there is a sum over adjacent tetrahedra. We find the
    # adjacent tetrahedra by looping over them and identifying the ones
    
    e1 = energies[0]
    e2 = energies[1]
    e3 = energies[2]
    e4 = energies[3]

    if eF < e1:
        return [0.]*4
    
    elif e1 <= eF < e2:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1
        C = VT/4.*(eF - e1)**3/(e21*e31*e41)
        
        w1 = C*(4 - (eF - e1)*(1./e21 + 1./e31 + 1/e41))
        w2 = C*(eF - e1)/e21
        w3 = C*(eF - e1)/e31
        w4 = C*(eF - e1)/e41
        return [w1, w2, w3, w4]
    
    elif e2 <= eF < e3:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1
        e32 = e3 - e2
        e42 = e4 - e2
        
        C1 = VT/4.*(eF - e1)**2/(e41*e31)
        C2 = VT/4.*((eF - e1)*(eF - e2)*(e3 - eF))/(e41*e32*e31)
        C3 = VT/4.*(eF - e2)**2*(e4 - eF)/(e42*e32*e41)

        w1 = C1 + (C1 + C2)*(e3 - eF)/e31 + (C1 + C2 + C3)*(e4 - eF)/e41
        w2 = C1 + C2 + C3 + (C2 + C3)*(e3 - eF)/e32 + C3*(e4 - eF)/e42
        w3 = (C1 + C2)*(eF - e1)/e31 + (C2 + C3)*(eF - e2)/e32
        w4 = (C1 + C2 + C3)*(eF - e1)/e41 + C3*(eF - e2)/e42
        return [w1, w2, w3, w4]
    
    elif e3 <= eF < e4:
        e41 = e4 - e1
        e42 = e4 - e2
        e43 = e4 - e3
        C = VT/4.*(e4 - eF)**3/(e41*e42*e43)

        w1 = VT/4. - C*(e4 - eF)/e41
        w2 = VT/4. - C*(e4 - eF)/e42
        w3 = VT/4. - C*(e4 - eF)/e43
        w4 = VT/4. - C*(4 - (1/e41 + 1/e42 + 1/e43)*(e4 - eF))
        return [w1, w2, w3, w4]
    
    else:
        return [VT/4.]*4

def density_of_states(VG, VT, energies, e):
    """Calculate the contribution to the density of states of a single tetrahedron
    and energy band. These weights differ from those in blochl's paper by a factor
    of 2. This factor comes from spin degeneracy.
    
    Args:
        VG (float): the volume of the unit cell.
        VT (float): the volume of the tetrahedron.
        energies (list): the energies at the corners of the tetrahedron ordered from
            least to greatest.
        e (float): the energy at which the total density of states is being calculated.

    Returns:
        (float): the density of states from a given tetrahedron.
    """
    
    e1 = energies[0]
    e2 = energies[1]
    e3 = energies[2]
    e4 = energies[3]

    if (e < e1 or e > e4):
        return 0.
    
    elif e1 <= e < e2:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1
        return 2.*3.*(VT/VG)/(e21*e31*e41)*(e - e1)**2.
    
    elif e2 <= e < e3:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1
        e32 = e3 - e2
        e42 = e4 - e2
        return (VT/VG)/(e31*e41)*(3*e21 + 6*(e - e2) - 
                                    3*(e31 + e42)*(e - e2)**2/(e32*e42))*2.
    
    else: # e3 <= e < e4:    
        e41 = e4 - e1
        e42 = e4 - e2
        e43 = e4 - e3        
        return 3*(VT/VG)*(e4 - e)**2/(e41*e42*e43)*2.


def make_extended_grid_indices(PP, ndivisions, lat_shift=[0,0,0], grid_shift=[0,0,0]):
    """Find the extended grid and indices for the non-extended grid for 
    the improved tetrahedron method.

    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object
        ndivisions (list): a list of integers that represent the number of 
            divisions along each basis vector to take when creating the grid.
        lat_shift (list or numpy.ndarray): a vector that shifts the grid in 
            fractions of the reciprocal lattice vectors.
        grid_shift (list of numpy.ndarray): the offset of the lattice in 
            fractions of the submesh translations.

    Returns:
        grid (numpy.ndarray): the extended grid in Cartesian coordinates.
        indices(numpy.ndaray): the indices of the points in the periodic case.
    """

    if type(ndivisions) == int:
        ndivisions = [ndivisions, ndivisions, ndivisions]
    ndiv0 = np.array(ndivisions)
    ndiv1 = ndiv0+1
    offset = np.dot(PP.lattice.reciprocal_vectors, lat_shift) - (
        np.dot(PP.lattice.reciprocal_vectors, grid_shift)/ndivisions)
    npts = np.prod(ndiv1)
    grid = np.empty(ndiv1, dtype=list)
    
    indices = np.empty(ndiv0, dtype=int)
    
    for k,j,i in product(range(ndiv1[0]), range(ndiv1[1]), range(ndiv1[2])):
        grid[k,j,i] = np.dot(PP.lattice.reciprocal_vectors, 
                             np.array([i,j,k], dtype=float)/ndiv0) + offset

    for k,j,i in product(range(ndiv0[0]), range(ndiv0[1]), range(ndiv0[2])):
        indices[k,j,i] = int(i + ndiv0[2]*(j + ndiv0[1]*k))

    # It goes z, y, x
    new_indices = np.zeros(np.array(ndiv1), dtype=int)
    new_indices[0:ndiv0[0], 0:ndiv0[1], 0:ndiv0[2]] = indices
    new_indices[:,:,ndiv0[2]] = new_indices[:,:,0]
    new_indices[:,ndiv0[1],:] = new_indices[:,0,:]
    new_indices[ndiv0[0],:,:] = new_indices[0,:,:]

    indices = np.array([i for i in new_indices.flatten()])
    grid = np.array([g for g in grid.flatten()])
    return grid, indices
    

def grid_and_tetrahedra(PP, ndivisions, lat_shift=[0,0,0], grid_shift=[0,0,0]):
    """Find the grid and tetrahedra for the improved tetrahedron method.

    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object
        ndivisions (list): a list of integers that represent the number of 
            divisions along each basis vector to take when creating the grid.
        lat_shift (list or numpy.ndarray): a vector that shifts the grid in 
            fractions of the reciprocal lattice vectors.
        grid_shift (list of numpy.ndarray): the offset of the lattice in 
            fractions of the submesh translations.

    Returns:
        grid (numpy.ndarray): the grid in Cartesian coordinates.
        tetrahedra (numpy.ndarray): lists of grid point indices that indicate
            the vertices of the tetrahedra.
    """

    if type(ndivisions) == int:
        ndivisions = [ndivisions, ndivisions, ndivisions]
    ndiv0 = np.array(ndivisions)
    ndiv1 = ndiv0+1
    offset = np.dot(PP.lattice.reciprocal_vectors, lat_shift) - (
        np.dot(PP.lattice.reciprocal_vectors, grid_shift)/ndivisions)
    npts = np.prod(ndiv1)
    grid = np.empty(ndiv1, dtype=list)
    indices = np.empty(ndiv1, dtype=int)
    
    for k,j,i in product(range(ndiv1[0]), range(ndiv1[1]), range(ndiv1[2])):
        index = int(i + ndiv1[2]*(j + ndiv1[1]*k))
        grid[k,j,i] = np.dot(PP.lattice.reciprocal_vectors, 
                             np.array([i,j,k], dtype=float)/ndiv0) + offset
        indices[k,j,i] = index

    tetrahedra = np.empty(np.prod(ndiv0)*6,dtype=list)
    
    for k,j,i in product(range(ndiv0[0]), range(ndiv0[1]), range(ndiv0[2])):
        submesh = np.empty([8], dtype=list)
        submesh_indices = np.empty([8], dtype=int)
        for kk,kj,ki in product(range(2),repeat=3):
            submesh[ki + 2*(kj + 2*kk)] = grid[k + kk, j + kj, i + ki]
            submesh_indices[ki + 2*(kj + 2*kk)] = indices[k + kk, j + kj, i + ki]
        # Find the tetrahedra with indexing 1-8.
        sub_tetrahedra = find_tetrahedra(submesh)
        
        # Replace 1-8 indices with sub_mesh indices.
        for m in range(6):
            # The index of the submeshcell
            ti = m + 6*(i + ndiv0[2]*(j + ndiv0[1]*k))
            tetrahedra[ti] = [0]*4
            
            for n in range(4):
                tetrahedra[ti][n] = submesh_indices[sub_tetrahedra[m][n]-1]
    
    tetrahedra = np.array([t for t in tetrahedra])
    grid = np.array([g.tolist() for g in grid.flatten()])
    return grid, tetrahedra


def get_grid_tetrahedra(PP, ndivisions, lat_shift=[0,0,0], grid_shift=[0,0,0]):
    """Find the grid and tetrahedra for the improved tetrahedron method.

    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object
        ndivisions (list): a list of integers that represent the number of 
            divisions along each basis vector to take when creating the grid.
        lat_shift (list or numpy.ndarray): a vector that shifts the grid in 
            fractions of the reciprocal lattice vectors.
        grid_shift (list of numpy.ndarray): the offset of the lattice in 
            fractions of the submesh translations.

    Returns:
        grid (numpy.ndarray): the grid in Cartesian coordinates.
        tetrahedra (numpy.ndarray): lists of grid point indices that indicate
            the vertices of the tetrahedra.
    """

    if type(ndivisions) == int:
        ndivisions = [ndivisions, ndivisions, ndivisions]
    ndiv0 = np.array(ndivisions)
    ndiv1 = ndiv0+1
    offset = np.dot(PP.lattice.reciprocal_vectors, lat_shift) - (
        np.dot(PP.lattice.reciprocal_vectors, grid_shift)/ndivisions)
    npts = np.prod(ndiv0)
    grid = np.empty(ndiv0, dtype=list)
    indices = np.empty(ndiv0, dtype=int)
    
    for k,j,i in product(range(ndiv0[0]), range(ndiv0[1]), range(ndiv0[2])):
        index = int(i + ndiv0[2]*(j + ndiv0[1]*k))
        grid[k,j,i] = np.dot(PP.lattice.reciprocal_vectors, 
                             np.array([i,j,k], dtype=float)/ndiv0) + offset
        indices[k,j,i] = index


    extended_grid = np.empty(ndiv1, dtype=list)
    for k,j,i in product(range(ndiv1[0]), range(ndiv1[1]), range(ndiv1[2])):
        extended_grid[k,j,i] = np.dot(PP.lattice.reciprocal_vectors, 
                                      np.array([i,j,k], dtype=float)/ndiv0) + offset
        
    # It goes z, y, x
    new_indices = np.zeros(np.array(ndiv0) + 1, dtype=int)
    new_indices[0:ndiv0[0], 0:ndiv0[1], 0:ndiv0[2]] = indices
    new_indices[:,:,ndiv0[2]] = new_indices[:,:,0]
    new_indices[:,ndiv0[1],:] = new_indices[:,0,:]
    new_indices[ndiv0[0],:,:] = new_indices[0,:,:]    
    indices = new_indices
    
    tetrahedra = np.empty(np.prod(ndiv0)*6,dtype=list)
    for k,j,i in product(range(ndiv0[0]), range(ndiv0[1]), range(ndiv0[2])):
        submesh = np.empty([8], dtype=list)
        submesh_indices = np.empty([8], dtype=int)
        for kk,kj,ki in product(range(2),repeat=3):
            submesh[ki + 2*(kj + 2*kk)] = extended_grid[k + kk, j + kj, i + ki]
            submesh_indices[ki + 2*(kj + 2*kk)] = indices[k + kk, j + kj, i + ki]
        # Find the tetrahedra with indexing 1-8.
        sub_tetrahedra = find_tetrahedra(submesh)
        
        # Replace 1-8 indices with sub_mesh indices.
        for m in range(6):
            # The index of the submeshcell
            ti = m + 6*(i + ndiv0[2]*(j + ndiv0[1]*k))
            tetrahedra[ti] = [0]*4
            
            for n in range(4):
                tetrahedra[ti][n] = submesh_indices[sub_tetrahedra[m][n]-1]

    tetrahedra = np.array([t for t in tetrahedra])
    grid = np.array([g.tolist() for g in grid.flatten()])
    
    return grid, tetrahedra


def calc_total_states(PP, tetrahedra, weights, grid, energy, nbands):
    """Calculate the total number of filled states.
    
    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object.
        tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        grid (numpy.ndarray): a grid of points in 3D.
        weights (list or numpy.ndarray): a list of tetrahedron weights.
        energy (float): the energy at which the total number of states is being
            calculated.
        nbands (int): the number of bands to include in calculation.
            
    Returns:
        total_states (float): the number of filled states.
    """

    Vg = PP.lattice.reciprocal_volume
    Vt = Vg/np.sum(weights)
    
    total_states = 0.
    for i,tet in enumerate(tetrahedra):
        energies = []
        for ind in tet:
            energies.append(PP.eval(grid[ind], nbands))
        # Reshape energies so that the energies of each band are grouped
        # together.
        energies = np.transpose(energies)
        for eband in energies:
            total_states += (weights[i]*
                             number_of_states(Vg, Vt, np.sort(eband), energy))
    return total_states

def calc_fermi_level(PP, tetrahedra, weights, grid, tol=1e-6):
    """Determine the Fermi level for a given pseudopotential using the 
    improved tetrahedron method.
    
    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object.
        tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        weights (list or numpy.ndarray): a list of tetrahedron weights. These
            should all be one if the tetrahedra are not symmetry reduced.
        grid (numpy.ndarray): an array of grid of points in 3D.
        tol (float): the tolerance on the error of the total number of states
            calculation.
        
    Returns:
        fermi_level (float): the calculated fermi level to within the provided tolerance
            on the total number of states.
    """

    # Gives the sign of a number (+/-) unless it is zero, in which case it
    # returns zero.
    sign = lambda x: x and (1, -1)[x<0]

    # number of filled states
    nfs = PP.nvalence_electrons# /2.    
    # highest filled band
    hfb = int(np.ceil(nfs))
    
    # number of bands included in the calculation
    nbands = hfb + 1 

    # Make a guess at where the Fermi level could be.
    if hfb <= 2:        
        # If there is only one filled band, have the guess be the energy value
        # at the origin of the highest filled band.
        fermi_level = PP.eval([0.]*3, nbands)[hfb-1]
    else:
        # If there are many occupied bands, take an average of the energies
        # of the band above and the band below.
        fermi_level = (PP.eval([0.]*3, nbands)[hfb] - 
                             PP.eval([0.]*3, nbands)[hfb-2])/2

    # Calculate the number of occupied states at the estimated Fermi level.
    total_states = calc_total_states(PP, tetrahedra, weights, grid,
                                     fermi_level, nbands)
    
    # Determine whether the initial guess overestimated (+) or underestimated
    # the Fermi level.
    initial_sign = sign(total_states - nfs)
    
    # We first need to find a window in which the Fermi level resides.
    # We know the upper/lower bound. Now we need to find the lower/upper bound
    # on the Fermi level. After this is obtained, we can perform a binary search
    # for the Fermi level on the obtained energy interval.
    # We do this by increasing or decreasing the estimated fermi level by the relatively
    # large energy step of 8 eV until we have too many or not enough filled states.

    # We know that we have a range of energies between in which the Fermi level is
    # located once we go from overestimating to underestimating the Fermi level.
    while sign(total_states - nfs) == initial_sign:
        fermi_level = fermi_level + (
                            sign(PP.nvalence_electrons - total_states))*8
        total_states = calc_total_states(PP, tetrahedra, weights, grid,
                                         fermi_level, nbands)

    # Adjust the bounds correctly. If overestimating, decrease the lower bound. If
    # underestimating, increase the upperbound. The Fermi level guess is the midpoint of
    # the upper and lower bound.
    if total_states > nfs:
        lower_bound = fermi_level - 8
        upper_bound = fermi_level
        fermi_level -= (upper_bound - lower_bound)/2
    else:
        lower_bound = fermi_level
        upper_bound = fermi_level + 8
        fermi_level += (upper_bound - lower_bound)/2

    # Now that we have an energy interval that contains the Fermi level, we can perform
    # a binary search within this interval until the Fermi level provides the correct
    # number of occupied states to within the provided tolerance.
    while abs(total_states - nfs) > tol:
        total_states = calc_total_states(PP, tetrahedra, weights, grid,
                                         fermi_level, nbands)
        if total_states > nfs:
            upper_bound = fermi_level
            fermi_level -= (upper_bound - lower_bound)/2.
        else:
            lower_bound = fermi_level
            fermi_level += (upper_bound - lower_bound)/2.
        
        if abs((upper_bound-lower_bound)/2.) < 1e-15:
            msg = ("Unable to determine Fermi level. Suggest using more k-points.")
            raise ValueError(msg.format(tol))

    return fermi_level

def find_irreducible_tetrahedra(PP, tetrahedra, grid, duplicates=True):
    """Find the irreducible tetrahedra and their weights.

    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object.
        tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        grid (numpy.ndarray): a grid of points in Cartesian coordinates.
    Returns:
        irreducible_tetrahedra (list): a list of irreducible tetrahedra vertices.
        weights (list): a list of tetrahedron weights.
    """
    
    if duplicates:
        orbits = get_orbits(grid, PP.lattice.reciprocal_vectors, duplicates=True)
    else:
        orbits = get_orbits(grid, PP.lattice.reciprocal_vectors)

    # Move all grid points into the first unit cell

    grid = [bring_into_cell(pt, PP.lattice.reciprocal_vectors) for pt in grid]

    # Each point in the grid is part of an orbit. The first point in each orbit
    # represents the orbit. The dictionary new_dict is a dictionary with keys as
    # the indices of all the grid points and values the index of the point that
    # represents its orbit.
    new_dict = {}
    for i,pt in enumerate(grid):
        # pt = bring_into_cell(pt, PP.lattice.reciprocal_vectors)
        new_dict[i] = None
        for k in orbits.keys():
            # Represntative k-point
            rpt = orbits[k][0]
            index = np.where([np.allclose(p,rpt) for p in grid])[0][0]
            for v in orbits[k]:
                if np.allclose(v,pt):
                    # index = np.where([np.allclose(p,v) for p in grid])[0][0]
                    new_dict[i] = index
                    
    tetrahedra_copy = deepcopy(tetrahedra)
    weights = []
    irreducible_tetrahedra = []
    for i,tet in enumerate(tetrahedra_copy):
        for j,ind in enumerate(tet):
            # Replace each index in tetrahedra with the one that represents its orbit.
            tetrahedra_copy[i][j] = new_dict[ind]
            
        # Sort the indices so they can be easily compared.
        tetrahedra_copy[i] = np.sort(tetrahedra_copy[i])
        # If the tetrahedra being considered is already apart of the list of 
        # irreducible tetrahedra, add 1 to its weight.
        if any([np.allclose(tetrahedra_copy[i],tet) for tet in
                irreducible_tetrahedra]):
            loc = np.where([np.allclose(tetrahedra_copy[i],tet) for tet in
                            irreducible_tetrahedra])[0][0]
            weights[loc] += 1.
            continue
        # If not add it to the list and give it weight 1.
        else:
            irreducible_tetrahedra.append(tetrahedra_copy[i])
            weights.append(1.)
            
    return irreducible_tetrahedra, weights

def calc_total_energy(PP, tetrahedra, weights, grid):
    """Calculate the total energy.
    
    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object.
        tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        weights (list or numpy.ndarray): a list of tetrahedron weights.
        grid (numpy.ndarray): a grid of points in 3D.
        fermi_level (float): the energy at which the total energy is being 
            calculated.
            
    Returns:
        total_states (float): the number of filled states.    
    """

    # The volume of the tetrahedra.
    VG = PP.lattice.reciprocal_volume
    VT = PP.lattice.reciprocal_volume/np.sum(weights)
    
    # The number of bands included in the calculation of the total energy.
    nbands = int(np.ceil(PP.nvalence_electrons/2))
    
    # Each contribution to the total energy will be stored in a list. The sum of these
    # contributions will be taken all at once to avoid numerical errors.
    total_energy = []
    for i,irr_tet in enumerate(tetrahedra):
        # Store the energies at the vertices of a tetrahedra in energies. Since the
        # function can be multivalued, this is a list of lists.
        energies = []
        for ind in irr_tet:
            energies.append(PP.eval(grid[ind], nbands))
            
        # Transpose the energies so that the energies of each band are group together.
        energies = np.transpose(energies)
        for eband in energies:
            eband = np.sort(eband)
            int_weights = integration_weights(VT, eband, PP.fermi_level)
            total_energy.append(weights[i]*np.dot(int_weights, eband))
    
    return math.fsum(total_energy)


def get_extended_tetrahedra(PP, ndivisions, lat_shift=[0,0,0],
                            grid_shift=[0,0,0]):
    """Generate the grid and tetrahedra required in calculating Blochl's 
    corrections.

    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object
        ndivisions (list): a list of integers that represent the number of 
            divisions along each basis vector to take when creating the grid.
        lat_shift (list or numpy.ndarray): a vector that shifts the grid in 
            fractions of the reciprocal lattice vectors.
        grid_shift (list of numpy.ndarray): the offset of the lattice in 
            fractions of the submesh translations.

    Returns:
        grid (numpy.ndarray): the extended grid in Cartesian coordinates.
        tetrahedra (numpy.ndarray): lists of grid point indices that indicate
            the vertices of the extended tetrahedra.
    """

    grid_shift = np.array(grid_shift)
    
    if type(ndivisions) == int:
        ndivisions = [ndivisions, ndivisions, ndivisions]
    ndiv0 = np.array(ndivisions)
    ndiv1 = ndiv0 + 1
    ndiv2 = ndiv0 + 2
    ndiv3 = ndiv0 + 3
    npts = np.prod(ndiv1)
    offset = np.dot(PP.lattice.reciprocal_vectors, lat_shift) - (
        np.dot(PP.lattice.reciprocal_vectors, grid_shift)/ndivisions)
    grid = np.empty(ndiv1, dtype=list)
    indices = np.empty(ndiv1, dtype=int)

    for k,j,i in product(range(ndiv1[0]), range(ndiv1[1]), range(ndiv1[2])):
        index = int(i + ndiv1[2]*(j + ndiv1[1]*k))
        grid[k,j,i] = np.dot(PP.lattice.reciprocal_vectors, 
                             np.array([i,j,k], dtype=float)/ndiv0) + offset
        indices[k,j,i] = index


    extended_indices = np.empty(np.array(ndiv3), dtype=int)
    for k,j,i in product(range(ndiv3[0]), range(ndiv3[1]), range(ndiv3[2])):
        if ((i > 0 and i < ndiv2[0]) and (j > 0 and j < ndiv2[1]) and
            (k > 0 and k < ndiv2[2])):
            extended_indices[k,j,i] = (i-1) + (j-1)*ndiv1[1] + (k-1)*ndiv1[0]*ndiv1[1]
        else:
            extended_indices[k,j,i] = i + j*ndiv3[1] + k*ndiv3[0]*ndiv3[1] + npts    

    grid_shift += 1
    offset = np.dot(PP.lattice.reciprocal_vectors, lat_shift) - (
        np.dot(PP.lattice.reciprocal_vectors, grid_shift)/ndivisions)

    extended_grid = np.empty(ndiv3, dtype=list)
    for k,j,i in product(range(ndiv3[0]), range(ndiv3[1]), range(ndiv3[2])):
        extended_grid[k,j,i] = np.dot(PP.lattice.reciprocal_vectors,
                                      np.array([i,j,k], dtype=float)/ndiv0) + offset            
    
    extended_tetrahedra = np.empty(np.prod(ndiv2)*6,dtype=list)
    # for k,j,i in product(range(ndiv1[0]), range(ndiv1[1]), range(ndiv1[2])):
    for k,j,i in product(range(ndiv2[0]), range(ndiv2[1]), range(ndiv2[2])):
        submesh = np.empty([8], dtype=list)
        submesh_indices = np.empty([8], dtype=int)
        for kk,kj,ki in product(range(2),repeat=3):        
            submesh[ki + 2*(kj + 2*kk)] = extended_grid[k + kk, j + kj, i + ki]
            submesh_indices[ki + 2*(kj + 2*kk)] = extended_indices[k + kk, j + kj, i + ki]
        # Find the tetrahedra with indexing 1-8.
        sub_tetrahedra = find_tetrahedra(submesh)

        # Replace 1-8 indices with sub_mesh indices.
        for m in range(6):
            # The index of the submeshcell
            # ti = m + 6*(i + ndiv1[2]*(j + ndiv1[1]*k))
            ti = m + 6*(i + ndiv2[2]*(j + ndiv2[1]*k))
            extended_tetrahedra[ti] = [0]*4

            for n in range(4):
                extended_tetrahedra[ti][n] = submesh_indices[sub_tetrahedra[m][n]-1]

    extended_tetrahedra = np.array([t for t in extended_tetrahedra])
    extended_grid = np.array([g.tolist() for g in extended_grid.flatten()])
    
    return extended_grid, extended_tetrahedra

def get_corrected_total_energy(PP, tetrahedra_list, extended_tetrahedra_list, grid,
                               extended_grid, ndiv0):
    """Calculate the corrected integration weights used in calculating the
    total energy.
    
    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object.
            tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        tetrahedra (list): a list of tetrahedra where each tetrahedra is a 
            quadruple of integers.
        extended_tetrahedra_list (list): a list of tetrahedra that includes those
            that surround the 
        grid (numpy.ndarray): a grid of points in 3D.
            
    Returns:
        total_states (float): the number of filled states.    
    """

    npts = len(grid)
    neigvals = int(np.ceil(PP.nvalence_electrons/2.))
    total_energy = 0.    
    for iband in range(neigvals):
        for tet in tetrahedra_list:
            energies, tweights = corrected_integration_weights(PP,
                                                extended_grid,
                                                extended_tetrahedra_list,
                                                tet, iband, ndiv0)
            total_energy += np.dot(tweights, energies)
    return total_energy



def tet_dos_nos(EPM, nbands, grid, energy_list, tetrahedra, weights):
    """Calculate the density of states and number of states using the 
    tetrahedron method.

    Args:
        EPM (object): a potential object.
        nbands (int): the number of bands included in the calculation.
        grid (list): a list of grid points.
        energy_list (list): a list of energies.
        tetrahedra (list): a list of quadruples of tetrahedra vectices.
        weights (list): a list of tetrahedron weights.

    Returns:
        energy_list (list): a list of energies, the same the argument energy_list.
        dos (list): a list of density of states values.
        nos (list): a list of number of states values.
    """
    
    VG = EPM.lattice.reciprocal_volume
    VT = VG/len(weights)
    dos = np.zeros(len(energy_list))
    nos = np.zeros(len(energy_list))

    # The input 'energy_list' is the same as the energies at which the
    # tetrahedron method calculates the DOS and NOS.
    for i,energy in enumerate(energy_list):
        for k, tet in enumerate(tetrahedra):
            for band in range(nbands):
                energies = np.sort([EPM.eval(grid[j], nbands)[band]
                                        for j in tet])
                dos[i] += weights[k]*density_of_states(VG, VT,
                                                           energies, energy)
                dos[i] += weights[k]*number_of_states(VG, VT,
                                                           energies, energy)

    return energy_list, dos, nos

