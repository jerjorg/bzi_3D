"""Generate quantities related to the symmetry of the lattice. This module
draws a lot from `Peter Hadley <http://lampx.tugraz.at/~hadley/ss1/bzones/>`_.
"""

import numpy as np
import itertools
import copy
from itertools import islice
from phenum.symmetry import _get_lattice_pointGroup

# Define the symmetry points for a fcc lattice in real space.
# Coordinates are in lattice coordinates.
fcc_sympts = {"G": [0., 0., 0.], # G is the gamma point.
               "X": [0., 1./2, 1./2],
               "L": [1./2, 1./2, 1./2],
               "W": [1./4, 3./4, 1./2],
               "U": [1./4, 5./8, 5./8],
               "K": [3./8, 3./4, 3./8],
               "G2":[1., 1., 1.]}

# Define the symmetry points for a bcc lattice in real space.
bcc_sympts = {"G": [0., 0., 0.],
               "H": [-1./2, 1./2, 1./2],
               "P": [1./4, 1./4, 1./4],
               "N": [0., 1./2, 0.]}

# Define the symmetry points for a sc lattice in real space.
sc_sympts = {"G": [0. ,0., 0.],
              "R": [1./2, 1./2, 1./2],
              "X": [0., 1./2, 0.],
              "M": [1./2, 1./2, 0.]}

def rsym_pts(lat_type, a, lat_coords=False):
    """Define the symmetry points in reciprocal space for a given real
    space lattice.

    Args:
        lat_type (string): the type of real space lattice, e.g., fcc or bcc.
        a (float): the lattice constant or characteristic spacing
        of atoms on the lattice.

    Return:
        (dict): A dictionary where a string of a lattice point is a
        key and the coordinate of the lattice point is the value.
    """

    if type(a) not in (float, int, np.float64):
        raise ValueError("The lattice constant must be an int or float.")
    
    if lat_coords == False:
        if lat_type == "fcc":    
            return {"G": [0,0,0],
                    "X": [0, 2*np.pi/a, 0],
                    "L": [np.pi/a, np.pi/a, np.pi/a],
                    "W": [np.pi/a, 2*np.pi/a, 0],
                    "U": [np.pi/(2*a), 2*np.pi/a, np.pi/(2*a)],
                    "K": [3*np.pi/(2*a), 3*np.pi/(2*a),0]}
        elif lat_type == "bcc":
            return {"G": [0,0,0],
                    "H": [0,0,2*np.pi/a],
                    "P": [np.pi/a,np.pi/a,np.pi/a],
                    "N": [0,np.pi/a,np.pi/a]}
        elif lat_type == "sc":
            return {"G": [0,0,0],
                    "R": [np.pi/a,np.pi/a,np.pi/a],
                    "X": [0,np.pi/a,0],
                    "M": [np.pi/a,np.pi/a,0]}
        else:
            raise KeyError("Please provide a cubic lattice type.")
    elif lat_coords == True:
        if lat_type == "fcc":    
            return {"G": [0,0,0],
                    "X": [0, 0.5, 0.5],
                    "L": [0.5, 0.5, 0.5],
                    "W": [0.25, 0.75, 0.5],
                    "U": [0.25, 0.625, 0.625],
                    "K": [0.375, 0.75, 0.375]}
        elif lat_type == "bcc":
            return {"G": [0,0,0],
                    "H": [-0.5, 0.5, 0.5],
                    "P": [0.25, 0.25, 0.25],
                    "N": [0, 0.5, 0]}
        elif lat_type == "sc":
            return {"G": [0, 0, 0],
                    "R": [0.5, 0.5, 0.5],
                    "X": [0, 0.5, 0],
                    "M": [0.5, 0.5, 0]}
        else:
            raise KeyError("Please provide a cubic lattice type.")
    else:
        raise KeyError("lat_coords can either be True or False.")
    
def make_ptvecs(lat_type, a, scale=[1,1,1]):
    """Provided the lattice type and constant, return the primitive translation
    vectors.

    Args:
        lat_type (str): the type of lattice of the material.
        a (float or int): the characteristic spacing of atoms in the 
            material.
        scale (list): a list of integers used for created custom primitive
            translation vectors.
    Returns:
        lattice_vectors (numpy.ndarray): returns the primitive translation vectors as
            the columns of a matrix.
    
    Example:
        >>> lattice_type = "fcc"
        >>> a = 1.2
        >>> vectors = primitive_translation_vectors(lattice_type, a)
    """

    if type(a) not in (float, int, np.float64):
        raise ValueError("The lattice constant must be an int or float.")

    if lat_type == "sc":
        a1 = a*np.array([1, 0, 0])
        a2 = a*np.array([0, 1, 0])
        a3 = a*np.array([0, 0, 1])

        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lat_type == "bcc":    
        a1 = a/2.*np.array([1, 1, -1])
        a2 = a/2.*np.array([-1, 1, 1])
        a3 = a/2.*np.array([1, -1, 1])

        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lat_type == "fcc":
        a1 = a/2.*np.array([1, 0, 1])
        a2 = a/2.*np.array([1, 1, 0])
        a3 = a/2.*np.array([0, 1, 1])

        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lat_type == "custom_sc":
        a1 = a*np.array([1, 0, 0])
        a2 = a*np.array([0, 1, 0])
        a3 = a*np.array([0, 0, 1])

        lattice_vectors = scale*np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lat_type == "custom_bcc":    
        a1 = a/2.*np.array([1, 1, -1])
        a2 = a/2.*np.array([-1, 1, 1])
        a3 = a/2.*np.array([1, -1, 1])

        lattice_vectors = scale*np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lat_type == "custom_fcc":
        a1 = a/2.*np.array([1, 0, 1])
        a2 = a/2.*np.array([1, 1, 0])
        a3 = a/2.*np.array([0, 1, 1])

        lattice_vectors = scale*np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors    
    else:
        msg = "Please provide a sc, bcc, fcc or custom lattice type."
        raise ValueError(msg.format(lat_type))

    
def make_rptvecs(lat_type, a, scale=[1., 1., 1.]):
    """Provided the lattice type and constant, return the primitive translation
    vectors in reciprocal space.

    Args:
        lat_type (str): the type of lattice of the material.
        a (float or int): the characteristic spacing of atoms in the 
            material.
    Return:
        B (numpy.ndarray): returns the primitive translation vectors in 
        reciprocal space as the columns of a matrix.
    
    Example:
        >>> lat_type = "fcc"
        >>> a = 1.2
        >>> vectors = primitive_translation_vectors(lat_type, a)
    """
    
    ndims = 3
    A = make_ptvecs(lat_type,a)
    V = np.linalg.det(A) # volume of unit cell
    B = np.empty(np.shape(A))
    for i in range(ndims):
        B[:,i] = 2*np.pi*np.cross(A[:,np.mod(i+1, ndims)],
                                  A[:, np.mod(i+2, ndims)])/V
    return scale*B

def sym_path(lat_type, npts, sym_pairs):
    """Create an array of coordinates between the provided symmetry points in
    reciprocal space in lattice coordinates.

    Args:
        lat_type (str): the lattice type in real space
        npts (int): the number of coordinates to create between the symmetry
            points.
        sym_pair (numpy.array): an array of point coordinates.

    Return:
        (numpy.array): an array of lattice coordinates along a line connecting
            two symmetry points.
    """
    
    if lat_type is "bcc":
        dict = bcc_sympts
    elif lat_type is "fcc":
        dict = fcc_sympts
    elif lat_type is "sc":
        dict = sc_sympts
    else:
        raise ValueError("Invalid lattice type")
    paths = []
    for i,sym_pair in enumerate(sym_pairs):
        sym_pti = dict[sym_pair[0]]
        sym_ptf = dict[sym_pair[1]]

        pxi = sym_pti[0]
        pxf = sym_ptf[0]
        pyi = sym_pti[1]
        pyf = sym_ptf[1]
        pzi = sym_pti[2]
        pzf = sym_ptf[2]
        px = np.linspace(pxi,pxf,npts)
        py = np.linspace(pyi,pyf,npts)
        pz = np.linspace(pzi,pzf,npts)
        ipath = [[px[i],py[i],pz[i]] for i in range(len(px))]
        if i is 0:
            paths += ipath
        else:
            del ipath[0]
            paths += ipath
    return paths

def point_group(lat_vecs):
    """Return the point group of a lattice.

    Args:
        lat_vecs (numpy.ndarray or list): the vectors as the columns of a matrix.

    Returns
        pg (list): A list of the operators in the point group.
    """
    # _get_lattice_pointGroup has the vectors as rows instead of columns.
    lat_vecs = np.transpose(lat_vecs)
    return _get_lattice_pointGroup(lat_vecs)

def shells(vector, lat_vecs):
    """Find the vectors that are equivalent to another vector by symmetry
    
    Args:
        vector (list or numpy.ndarray): a vector in cartesian coordinates.
        lat_vecs (numpy.ndarray or list): a matrix with the lattice vectors as
            columns.
    
    Returns:
        unique_shells (list): a list of vectors expressed as numpy arrays.
    
    Example:
        >>> from bzi.symmetry import shells
        >>> from bzi.symmetry import make_ptvecs
        >>> lat_type = "sc"
        >>> lat_const = 1.
        >>> lat_vecs = make_ptvecs(lat_type, lat_const)
        >>> vector = [1.,0.,0.]
        >>> sc_shells(vector, lat_vecs)
    """

    pointgroup = point_group(lat_vecs)
    all_shells = [np.dot(ptg,vector).tolist() for ptg in pointgroup]
    unique_shells = []
    for sh in all_shells:  
        if any([np.allclose(sh, us) for us in unique_shells]) == True:
            continue
        else:
            unique_shells.append(np.array(sh))
                
    tol = 1.e-15
    for (i,us) in enumerate(unique_shells):
        for (j,elem) in enumerate(us):
            if np.abs(elem) < tol:
                unique_shells[i][j] = 0.
    return unique_shells

def shells_list(vectors, lat_vecs):
    """Returns a list of several shells useful for constructing 
    pseudo-potentials.

    Args:
        vector (list or numpy.ndarray): a vector in cartesian coordinates.
    
    Returns:
        unique_shells (list): a list of vectors expressed as numpy arrays.
    
    Example:
        >>> from bzi.symmetry import sc_shells
        >>> vectors = [[0.,0.,0.], [1.,0.,0.]]
        >>> sc_shells_list(vector)
    """
    nested_shells = [shells(i, lat_vecs) for i in vectors]
    return np.array(list(itertools.chain(*nested_shells)))

def find_orbitals(mesh_car, cell_vecs, coord = "cart"):
    """ Find the partial orbitals of the points in a mesh, including only the
    points that are in the mesh.

    Args:
        mesh_car (list): a list of mesh point positions.
        cell_vecs (numpy.ndarray): the vectors that define the integration cell
        coord (str): a string that indicatese coordinate system of the points. It
            can be in catesian ("cart") or reciprocal lattice ("cell").

    Returns:
        mp_orbitals (dict): the orbitals of the mesh points in a dictionary. 
            The keys of the dictionary are integer labels and the values are the
            mesh points in the orbital.
    """

    mesh_cell = [np.dot(np.linalg.inv(cell_vecs), mp) for mp in mesh_car]
    mp_orbitals = {}
    nirr_kpts = 0
    mesh_copy = copy.deepcopy(mesh_cell)
    pointgroup = point_group(cell_vecs)        
    while mesh_copy != []:
        # Grap a point and build its orbit but only include points from the mesh.
        mp = mesh_copy.pop()
        nirr_kpts += 1
        mp_orbitals[nirr_kpts] = [mp]
        for pg in pointgroup:
            new_mp = np.dot(pg, mp)
            if any([np.allclose(new_mp, mc) for mc in mesh_copy]):
                ind = np.where(np.array([np.allclose(new_mp, mc) for mc in mesh_copy]) == True)[0][0]
                del mesh_copy[ind]
                mp_orbitals[nirr_kpts].append(new_mp)
            else:
                continue

    if coord == "cart":
        for i in range(1, len(mp_orbitals.keys()) + 1):
            for j in range(len(mp_orbitals[i])):
                mp_orbitals[i][j] = np.dot(cell_vecs, mp_orbitals[i][j])
        return mp_orbitals
    elif coord == "cell":
        return mp_orbitals
    else:
        raise ValueError("There is no method for the coordinate system provided yet.")
        
    return mp_orbitals

def find_full_orbitals(mesh_car, cell_vecs, coord = "cart"):
    """ Find the complete orbitals of the points in a mesh.

    Args:
        mesh_car (list): a list of mesh point positions in cartesian coordinates.
        cell_vecs (numpy.ndarray): the vectors that define the integration cell

    Returns:
        mp_orbitals (dict): the orbitals of the mesh points in a dictionary. 
            The keys of the dictionary are integer labels and the values are the
            mesh points in the orbital.
    """

    mesh_cell = [np.dot(np.linalg.inv(cell_vecs), mp) for mp in mesh_car]
    mp_orbitals = {}
    nirr_kpts = 0
    mesh_copy = copy.deepcopy(mesh_cell)
    pointgroup = _get_lattice_pointGroup(cell_vecs)
    while mesh_copy != []:
        # Grap a point and build its orbit but only include points from the mesh.
        mp = mesh_copy.pop()
        nirr_kpts += 1
        mp_orbitals[nirr_kpts] = []
        for pg in pointgroup:
            new_mp = np.dot(pg, mp)
            if any([np.allclose(new_mp, mc) for mc in mesh_copy]):
                ind = np.where(np.array([np.allclose(new_mp, mc) for mc in mesh_copy]) == True)[0][0]
                del mesh_copy[ind]
                mp_orbitals[nirr_kpts].append(new_mp)
            else:
                mp_orbitals[nirr_kpts].append(new_mp)                
                continue

    if coord == "cart":
        for i in range(1, len(mp_orbitals.keys()) + 1):
            for j in range(len(mp_orbitals[i])):
                mp_orbitals[i][j] = np.dot(cell_vecs, mp_orbitals[i][j])
        return mp_orbitals
    elif coord == "cell":
        return mp_orbitals
    else:
        raise ValueError("There is no method for the coordinate system provided yet.")

    return mp_orbitals
