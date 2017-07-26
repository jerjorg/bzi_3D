"""Module related to Blochl's improved linear tetrahedron method."""

import numpy as np
from numpy.linalg import norm
from itertools import product
from copy import deepcopy
import math

from BZI.symmetry import find_orbitals

def find_tetrahedra(vertices):
    """Determine how a parallelepiped should be split into tetrahedra by
    finding its shortest diagonal, which is common to all the tetrahedra.

    Args:
        vertices (list or numpy.ndarray): a list of vertices of the 
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
    of states.
    
    Args:
        VG (float): the volume of the unit cell.
        VT (float): the volume of the tetrahedron.
        energies (list): the energies at the corners of the tetrahedron ordered from
            least to greatest.
        e (float): the energy at which the total number of states is being calculated.

    Returns:
        (float): the number of states from a given tetrahedron.
    """
    
    e1 = energies[0]
    e2 = energies[1]
    e3 = energies[2]
    e4 = energies[3]
    
    if e < e1:
        return 0.

    elif e1 <= e < e2:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1
        return VT/VG*(e - e1)**3/(e21*e31*e41)
        
    elif e2 <= e < e3:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1
        e32 = e3 - e2
        e42 = e4 - e2
        
        return (VT/VG)*1/(e31*e41)*(e21**2 + 3*e21*(e - e2) + 3*(e - e2)**2 
                                  - (e31 + e42)/(e32*e42)*(e - e2)**3)    
    elif e3 <= e < e4:
        e41 = e4 - e1
        e42 = e4 - e2
        e43 = e4 - e3
        return VT/VG*(1 - (e4 - e)**3/(e41*e42*e43))
    
    else:
        return VT/VG

def find_adjacent_tetrahedra(tetrahedra_list, k):
    """For each k point in the grid, this function generates a list of
    tetrahedra indices for each tetrahedron containing the k point.

    Args:
        tetrahedra_quadruples (list of lists of ints): a list of quadruples.
            There is exactly one quadruple for every tetrahedron. Each
            quadruple is a list of the grid_points indices for the corners of
            the tetrahedron.
        k (int): the total number of points in the grid.

    Returns:
        tetrahedra_by_point (list of list of ints): for each k point in the
            grid, a list of the indices of each tetrahedron containing that
            k point is given.
    """

    adjacent_tetrahedra = []
    """list of list of ints: for each k point in the grid, a list of the
    indices of each tetrahedron containing that k point is given."""

    # Find all tetrahedra containing the k-point.
    for tet in tetrahedra_list:
        for ki in tet:
            if ki == k:
                adjacent_tetrahedra.append(tet)

    return adjacent_tetrahedra

def corrected_integration_weights(PP, grid, tetrahedra_list, tetrahedron,
                                  iband):
    """Determine the corrected integration weights of a single tetrahedron and 
    band.
    
    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object.
            tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        grid (numpy.ndarray): a Monkhorst-Pack grid.
        tetrahedra_list (list): a list of tetrahedra where each tetrahedra is a
            quadruple of integers.
        tetrahedron(list): the tetrahedron being considered.
        iband (int): the index of the band being considered.
        
    Returns:
        energies (list): the energies at the vertices of the provided
            tetrahedron and band index.
        weights (list): the integration weights of the corners of the 
            tetrahedron.
    """
    
    tetrahedron = np.array(tetrahedron)
    VT = PP.lattice.reciprocal_volume/len(tetrahedra_list)
    VG = PP.lattice.reciprocal_volume
    neigvals = int(np.ceil(PP.nvalence_electrons/2)+1)

    # Find the correction to the integration weights. This is done according to
    # Eq. 22 in Blochl's improved tetrahedron paper.

    # We're going to calculate the weight correction for all the k-points in
    # the tetrahedron being considered.    
    corrections = [0.]*4
    energies = [PP.eval(grid[ki], iband)[iband] for ki in tetrahedron]
    tetrahedron = tetrahedron[np.argsort(energies)]
    energies = np.sort(energies)
    
    for i,ki in enumerate(tetrahedron):
        en = energies[i]
        kpt = grid[ki]
        adjacent_tetrahedra = find_adjacent_tetrahedra(tetrahedra_list, ki)
        for tet in adjacent_tetrahedra:
            for kj in tet:
                kp = grid[kj]
                enj = PP.eval(kp, neigvals)[iband]
                corrections[i] += enj - en
        corrections[i] *= 1/40.*density_of_states(VG, VT, energies, PP.fermi_level)

    c1 = corrections[0]
    c2 = corrections[1]
    c3 = corrections[2]
    c4 = corrections[3]

    e1 = energies[0]
    e2 = energies[1]
    e3 = energies[2]
    e4 = energies[3]

    eF = PP.fermi_level
    if eF < e1:
        return energies, [c1, c2, c3, c4]
    
    elif e1 <= eF < e2:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1    
        C = VT/4.*(eF - e1)**3/(e21*e31*e41)
        
        w1 = C*(4 - (eF - e1)*(1./e21 + 1./e31 + 1/e41))
        w2 = C*(eF - e1)/e21
        w3 = C*(eF - e1)/e31
        w4 = C*(eF - e1)/e41
        return energies, [w1 + c1, w2 + c1, w3 + c1, w4 + c1]
    
    elif e2 <= eF < e3:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1
        e32 = e3 - e2
        e42 = e4 - e2
        
        C1 = VT/4.*(eF - e1)**2/(e41*e31)
        C2 = VT/4.*(eF - e1)*(eF - e2)*(eF - e3)/(e41*e32*e31)
        C3 = VT/4.*(eF - e2)**2*(e4 - eF)/(e42*e32*e41)

        w1 = C1 + (C1 + C2)*(e3 - eF)/e31 + (C1 + C2 + C3)*(e4 - eF)/e41
        w2 = C1 + C2 + C3 + (C2 + C3)*(e3 - eF)/e32 + C3*(e4 - eF)/e42
        w3 = (C1 + C2)*(eF - e1)/e31 + (C2 + C3)*(eF - e2)/e32
        w4 = (C1 + C2 + C3)*(eF - e1)/e41 + C3*(eF - e2)/e42
        return energies, [w1 + c1, w2 + c2, w3 + c3, w4 + c4]
    
    elif e3 <= eF < e4:
        e41 = e4 - e1
        e42 = e4 - e2
        e43 = e4 - e3
        C = VT/4.*(e4 - eF)**3/(e41*e42*e43)

        w1 = VT/4. - C*(e4 - eF)/e41
        w2 = VT/4. - C*(e4 - eF)/e42
        w3 = VT/4. - C*(e4 - eF)/e43
        w4 = VT/4. - C*(4 - (1/e41 + 1/e42 + 1/e43)*(e4 - eF))
        return energies, [w1 + c1, w2 + c2, w3 + c3, w4 + c4]
    
    else:
        return energies, [VT/4. + c1, VT/4. + c2, VT/4. + c3, VT/4 + c4]
    

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
        C2 = VT/4.*(eF - e1)*(eF - e2)*(eF - e3)/(e41*e32*e31)
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
    and energy band.
    
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
        return 3*VT/VG*(e - e1)**2/(e21*e31*e41)
    
    elif e2 <= e < e3:
        e21 = e2 - e1
        e31 = e3 - e1
        e41 = e4 - e1
        e32 = e3 - e2
        e42 = e4 - e2
        return (VT/VG)*1./(e31*e41)*(3*e21 + 6*(e - e2) - 
                                    3*(e31 + e42)*(e - e2)**2/(e32*e42))
    
    else: # e3 <= e < e4:    
        e41 = e4 - e1
        e42 = e4 - e2
        e43 = e4 - e3        
        return 3*VT/VG*(e4 - e)**2/(e41*e42*e43)

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

    diagonal_indices = [[1,8],[2,7],[3,6],[4,5]]
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
    return grid.flatten(), tetrahedra

def calc_total_states(PP, tetrahedra, weights, grid, energy, nbands):
    """Calculate the total number of filled states.
    
    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object.
        tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        grid (numpy.ndarray): a grid of points in 3D.
        weights (list or numpy.ndarray): a list of tetrahedron weights.
        tol (float): the tolerance on the total number of states.    
        energy (float): the energy at which the total number of states is being
            calculated.
            
    Returns:
        total_states (float): the number of filled states.
    """
    
    Vg = PP.lattice.reciprocal_volume
    Vt = Vg/len(tetrahedra)
    
    total_states = 0.
    for i,tet in enumerate(tetrahedra):
        energies = []
        for ind in tet:
            energies.append(PP.eval(grid.flatten()[ind], nbands))
        # Reshape energies so that the energies of each band are grouped
        # together.
        energies = np.transpose(energies)
        for eband in energies:
            total_states += (weights[i]*
                             number_of_states(Vg, Vt, np.sort(eband), energy))
    return total_states

def calc_fermi_level(PP, tetrahedra, weights, grid, tol=1e-4):
    """Determine the Fermi level for a given pseudopotential using the 
    improved tetrahedron method.
    
    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object.
        tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        weights (list or numpy.ndarray): a list of tetrahedron weights. These
            should all be one if the tetrahedra are left unreduced.
        grid (numpy.ndarray): a grid of points in 3D.
        tol (float): the tolerance on the total number of states.
        
    Returns:
        fermi_level (float): the fermi level to within the prodived tolerance
            on the total number of states.
    """

    # Gives the sign of a number (+/-) unless it is zero, in which case it
    # returns zero.
    sign = lambda x: x and (1, -1)[x<0]
    
    nfs = PP.nvalence_electrons/2. # number of filled states
    hfb = int(np.ceil(nfs)) # highest filled band
    nbands = hfb + 1

    # Make a guess at where the Fermi level could be.
    if hfb <= 2:
        fermi_level = PP.eval([0.]*3, nbands)[hfb-1]
    else:
        fermi_level = (PP.eval([0.]*3, nbands)[hfb] - 
                             PP.eval([0.]*3, nbands)[hfb-2])/2    

    total_states = calc_total_states(PP, tetrahedra, weights, grid,
                                     fermi_level, nbands)
    initial_sign = sign(total_states - nfs)
    
    # We first need to find a window in which the Fermi level resides.
    # We do this by increasing or decreasing the estimated fermi level until
    # we have too many/not enough filled states.
    while sign(total_states - nfs) == initial_sign:
        fermi_level = fermi_level + (
                            sign(PP.nvalence_electrons/2. - total_states))*8
        total_states = calc_total_states(PP, tetrahedra, weights, grid,
                                         fermi_level, nbands)

    if total_states > nfs:
        lower_bound = fermi_level - 8
        upper_bound = fermi_level
        fermi_level -= (upper_bound - lower_bound)/2
    else:
        lower_bound = fermi_level
        upper_bound = fermi_level + 8
        fermi_level += (upper_bound - lower_bound)/2    

    # Now we can do something like a binary search.
    while abs(total_states - nfs) > tol:
        total_states = calc_total_states(PP, tetrahedra, weights, grid,
                                         fermi_level, nbands)
        if total_states > nfs:
            upper_bound = fermi_level
            fermi_level -= (upper_bound - lower_bound)/2.
        else:
            lower_bound = fermi_level
            fermi_level += (upper_bound - lower_bound)/2.

        # if abs((upper_bound-lower_bound)/2.) < 1e-14:
        #     msg = ("Unable to determine Fermi level. Suggest increasing"
        #            " tolerance value or using more k-points.")
        #     raise ValueError(msg.format(tol))

    return fermi_level

def find_irreducible_tetrahedra(free, tetrahedra, grid):
    """Find the irreducible tetrahedra and their weights.

    Args:
        free (:py:obj:`BZI.pseudopots.Empiricalfree`): a pseudopotential object.
        tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        grid (numpy.ndarray): a grid of points in Cartesian coordinates.
    Returns:
        irreducible_tetrahedra (list): a list of irreducible tetrahedra vertices.
        weights (list): a list of tetrahedron weights.
    """

    # grid_car = grid.flatten()
    grid_cell = np.round(np.array([np.dot(np.linalg.inv(
        free.lattice.reciprocal_vectors), gc) for
                                   gc in grid]), 9)%1
    orbitals = find_orbitals(grid, free.lattice.reciprocal_vectors,
                             coord="lat", duplicates=True)
    # Each point in the grid is part of an orbit. One point at random is chosen 
    # to represent each orbit. The dictionary new_dict is a dictionary with
    # keys as the indices of the grid points and values the index of the point
    # that represents its orbit.
    new_dict = {}
    for i,pt in enumerate(grid_cell):
        new_dict[i] = None
        for k in orbitals.keys():
            for v in orbitals[k]:
                if np.allclose(v,pt):
                    index = np.where([np.allclose(p,v) for p in grid_cell])[0][0]
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
        # irreducible tetrahedra, a 1 to its weight.
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
    
    Vt = PP.lattice.reciprocal_volume/len(tetrahedra)
    nbands = int(np.ceil(PP.nvalence_electrons/2))
    total_energy = []
    for i,irr_tet in enumerate(tetrahedra):
        energies = []
        for ind in irr_tet:
            energies.append(PP.eval(grid[ind], nbands))
        # Reshape energies so that the energies of each band are group together.
        energies = np.transpose(energies)
        for eband in energies:
            eband = np.sort(eband)
            int_weights = integration_weights(Vt, eband, PP.fermi_level)
            total_energy.append(weights[i]*np.dot(int_weights, eband))

    return math.fsum(total_energy)

def get_corrected_total_energy(PP, tetrahedra_list,grid):
    """Calculate the corrected integration weights used in calculating the
    total energy.
    
    Args:
        PP (:py:obj:`BZI.pseudopots.EmpiricalPP`): a pseudopotential object.
            tetrahedra (numpy.ndarray): lists of tetrahedra vertices.
        tetrahedra (list): a list of tetrahedra where each tetrahedra is a 
            quadruple of integers.
        grid (numpy.ndarray): a grid of points in 3D.
            
    Returns:
        total_states (float): the number of filled states.    
    """


    neigvals = int(np.ceil(PP.nvalence_electrons/2))
    print("number of eigenvalues ", neigvals)
    total_energy = 0.    
    for iband in range(neigvals):
        for tet in tetrahedra_list:
            energies, tweights = corrected_integration_weights(PP, grid, tetrahedra_list,
                                                    tet, iband)
            total_energy += np.dot(tweights, energies)
    return total_energy
