"""Generate quantities related to the symmetry of the lattice. This module
draws heavily from Setyawan, Wahyu, and Stefano Curtarolo. "High-throughput 
electronic band structure calculations: Challenges and tools." Computational 
Materials Science 49.2 (2010): 299-312.
"""

import numpy as np
import itertools
import copy
from itertools import islice
from phenum.symmetry import _get_lattice_pointGroup

# Define the symmetry points for a fcc lattice in lattice coordinates.
# Coordinates are in lattice coordinates.
fcc_sympts = {"G": [0., 0., 0.], # G is the gamma point.
               "X": [1./2, 0., 1./2],
               "L": [1./2, 1./2, 1./2],
               "W": [1./2, 1./4, 3./4],
               "U": [5./8, 1./4, 5./8],
               "K": [3./8, 3./8, 3./4],
               "G2":[1., 1., 1.]}

# Define the symmetry points for a bcc lattice in lattice coordinates
bcc_sympts = {"G": [0., 0., 0.],
               "H": [1./2, 1./2, -1./2],
               "P": [1./4, 1./4, 1./4],
               "N": [1./2, 0., 0.]}

# Define the symmetry points for a sc lattice in lattice coordinates.
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
            return {"G": [0., 0., 0.],
                    "X": [0, 0.5, 0.5],
                    "L": [0.5, 0.5, 0.5],
                    "W": [0.25, 0.75, 0.5],
                    "U": [0.25, 0.625, 0.625],
                    "K": [0.375, 0.75, 0.375]}
        elif lat_type == "bcc":
            return {"G": [0., 0., 0.],
                    "H": [-0.5, 0.5, 0.5],
                    "P": [0.25, 0.25, 0.25],
                    "N": [0, 0.5, 0]}
        elif lat_type == "sc":
            return {"G": [0., 0., 0.],
                    "R": [0.5, 0.5, 0.5],
                    "X": [0, 0.5, 0],
                    "M": [0.5, 0.5, 0]}
        else:
            raise KeyError("Please provide a cubic lattice type.")
    else:
        raise KeyError("lat_coords can either be True or False.")
    
def make_ptvecs(center_type, lat_consts, lat_angles):
    """Provided the lattice type, constants, and angles, return the primitive 
    translation vectors.

    Args:
        center_type (str): identifies the location of the atoms in the cell.
        lat_consts (float or int): the characteristic spacing of atoms in the
            material with a first, b second, and c third in the list. These
            should be ordered such that a < b < c.
        angles (list): a list of angles between the primitive translation vectors,
            in radians, with alpha the first entry, beta the second, and gamma the 
            third in the list.

    Returns:
        lattice_vectors (numpy.ndarray): returns the primitive translation vectors as
            the columns of a matrix.

    Example:
        >>> center_type = "prim"
        >>> lat_consts = [1.2]*3
        >>> angles = [np.pi/2]*3
        >>> vectors = make_ptvecs(lattice_type, lat_consts, angles)
    """

    if type(lat_consts) not in (list, np.ndarray):
        raise ValueError("The lattice constants must be in a list or numpy "
                         "array.")
    if type(lat_angles) not in (list, np.ndarray):
        raise ValueError("The lattice angles must be in a list or numpy array.")

    for i in range(len(lat_consts)-1):
        if lat_consts[i+1] < lat_consts[i]:
            msg = ("The lattice constants should be ordered from least to "
                   "greatest.")
            raise ValueError(msg.format(lat_consts))
    
    alpha = float(lat_angles[0])
    beta = float(lat_angles[1])
    gamma = float(lat_angles[2])
    a = float(lat_consts[0])
    b = float(lat_consts[1])
    c = float(lat_consts[2])
    
    avec = np.array([a, 0., 0.])
    bvec = np.array([b*np.cos(gamma), b*np.sin(gamma), 0])
    # I had to round the argument of the sqrt function in order to avoid
    # numerical errors.
    cvec = np.array([c*np.cos(beta),
                c/np.sin(gamma)*(np.cos(alpha) -
                                 np.cos(beta)*np.cos(gamma)),
                np.sqrt(np.round(c**2 - (c*np.cos(beta))**2 -
                                 (c/np.sin(gamma)*(np.cos(alpha) -
                                  np.cos(beta)*np.cos(gamma)))**2, 9))])
    
    if center_type == "prim":
        pt_vecs = np.transpose(np.array([avec, bvec, cvec], dtype=float))
        return pt_vecs
    
    elif center_type == "base":
        av = .5*(avec - bvec)
        bv = .5*(avec + bvec)
        cv = cvec
        # I have to rotate two of my vectors in the xy-plane so that they
        # agree with Stefano's paper for base-centered monoclinic.
        if (alpha < np.pi/2 and np.isclose(beta, np.pi/2)
            and np.isclose(gamma, np.pi/2) and a <= c and b <= c):
            rotatex = [[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]]
            av = np.dot(rotatex, av)
            bv = np.dot(rotatex, bv)
        pt_vecs  = np.transpose(np.array([av, bv, cv], dtype=float))
        return pt_vecs
    
    elif center_type == "body":
        av = .5*(-avec + bvec + cvec)
        bv = .5*(avec - bvec + cvec)
        cv = .5*(avec + bvec - cvec)
        pt_vecs = np.transpose(np.array([av, bv, cv], dtype=float))
        return pt_vecs

    elif center_type == "face":
        av = .5*(bvec + cvec)
        bv = .5*(avec + cvec)
        cv = .5*(avec + bvec)
        pt_vecs = np.transpose(np.array([av, bv, cv], dtype=float))
        return pt_vecs

    elif center_type == "hex":
        # The vectors in Stefano's paper are mine rotated 60 degrees.
        rotate = [[np.cos(gamma/2), np.sin(gamma/2), 0],
                    [-np.sin(gamma/2), np.cos(gamma/2), 0],
                    [0, 0, 1]]
        av = np.dot(rotate, avec)
        bv = np.dot(rotate, bvec)
        cv = np.dot(rotate, cvec)
        pt_vecs = np.transpose(np.array([av, bv, cv], dtype=float))
        return pt_vecs
    elif center_type == "rhom":
        # The vectors in Stefano's paper are mine rotated 60 degrees.
        rotate = [[np.cos(alpha/2), np.sin(alpha/2), 0],
                  [-np.sin(alpha/2), np.cos(alpha/2), 0],
                  [0, 0, 1]]
        av = np.dot(rotate, avec)
        bv = np.dot(rotate, bvec)
        cv = np.dot(rotate, cvec)
        pt_vecs = np.transpose(np.array([av, bv, cv], dtype=float))
        return pt_vecs
    else:
        msg = "Please provide a valid centering type."
        raise ValueError(msg.format(center_type))

def make_rptvecs(A):
    """Return the reciprocal primitive translation vectors of the provided
    vectors.

    Args:
        A (list or numpy.ndarray): the primitive translation vectors in real space 
            as the columns of a nested list or numpy array.
    Return:
        B (numpy.ndarray): return the primitive translation vectors in 
            reciprocal space as the columns of a matrix.    
    """
    
    ndims = 3
    V = np.linalg.det(A) # volume of unit cell
    B = np.empty(np.shape(A))
    for i in range(ndims):
        B[:,i] = 2*np.pi*np.cross(A[:,np.mod(i+1, ndims)],
                                  A[:, np.mod(i+2, ndims)])/V
    return B

def make_reciprocal_lattice_vectors(A):
    """Return the reciprocal lattice vectors of the provided lattice vectors.

    Args:
        A (list or numpy.ndarray): the lattice vectors in real space ase columns
            of a nested list or numpy array.
    Return:
        (numpy.ndarray): the reciprocal lattice vectors.    
    """

    return np.transpose(np.linalg.inv(A))*2*np.pi

def make_lattice_vectors(lattice_type, lattice_constants, lattice_angles):
    """Create the vectors that generate a lattice.

    Args:
        lattice_type (str): the lattice type.
        lattice_constants (list or numpy.ndarray): the axial lengths of the
            conventional lattice vectors.
        lattice_angles (list or numpy.ndarray): the interaxial angles of the
            conventional lattice vectors.

    Returns:
        lattice_vectors (numpy.ndarray): the vectors that generate the lattice
            as columns of an array [a1, a2, a3] where a1, a2, and a3 are column
            vectors.

    Example:
        >>> lattice_type = "face_centered_cubic"
        >>> lattice_constants = [1]*3
        >>> lattice_angles = [numpy.pi/2]*3
        >>> lattice_vectors = make_lattice_vectors(lattice_type, 
                                                   lattice_constants, 
                                                   lattice_angles)
    """
    
    # Extract parameters.
    a = float(lattice_constants[0])
    b = float(lattice_constants[1])
    c = float(lattice_constants[2])
    alpha = float(lattice_angles[0])
    beta = float(lattice_angles[1])
    gamma = float(lattice_angles[2])
    
    if lattice_type == "simple_cubic":
        if not ((np.isclose(a, b) and np.isclose(b, c))):
            msg = ("The lattice constants should all be the same for a simple-"
                    "cubic lattice")
            raise ValueError(msg.format(lattice_constants))
        
        if not (np.isclose(alpha, np.pi/2) and np.isclose(beta, np.pi/2)
                and np.isclose(gamma, np.pi/2)):
            msg = ("The lattice angles should all be the same and equal to pi/2"
                    " for a simple-cubic lattice.")
            raise ValueError(msg.format(lattice_angles))
        
        a1 = [a, 0, 0]
        a2 = [0, a, 0]
        a3 = [0, 0, a]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors
    
    elif lattice_type == "face_centered_cubic":
        if not ((np.isclose(a, b) and np.isclose(b, c))):
            msg = ("The lattice constants should all be the same for a face-"
                   "centered, cubic lattice.")
            raise ValueError(msg.format(lattice_constants))
        if not (np.isclose(alpha, np.pi/2) and np.isclose(beta, np.pi/2)
                and np.isclose(gamma, np.pi/2)):
            msg = ("The lattice angles should all be the same and equal to pi/2"
                   " for a face-centered, cubic lattice.")
            raise ValueError(msg.format(lattice_angles))

        a1 = [  0, a/2, a/2]
        a2 = [a/2,   0, a/2]
        a3 = [a/2, a/2,   0]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors
    
    elif lattice_type == "body_centered_cubic":
        if not ((np.isclose(a, b) and np.isclose(b, c))):
            msg = ("The lattice constants should all be the same for a body-"
                   "centered, cubic lattice.")
            raise ValueError(msg.format(lattice_constants))

        if not (np.isclose(alpha, np.pi/2) and np.isclose(beta, np.pi/2)
                and np.isclose(gamma, np.pi/2)):
            msg = ("The lattice angles should all be the same and equal to pi/2"
                   " for a body-centered, cubic lattice.")
            raise ValueError(msg.format(lattice_angles))
        
        a1 = [-a/2,  a/2,  a/2]
        a2 = [ a/2, -a/2,  a/2]
        a3 = [ a/2,  a/2, -a/2]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors
    
    elif lattice_type == "tetragonal":
        if not (np.isclose(a, b) and
                not np.isclose(b, c)):
            msg = ("For tetragonal lattice, a = b != c where a, b, and c are "
                   "the first, second, and third entries in lattice_constants, "
                   "respectively.")
            raise ValueError(msg.format(lattice_constants))
        if not (np.isclose(alpha, np.pi/2) and np.isclose(beta, np.pi/2)
                and np.isclose(gamma, np.pi/2)):
            msg = ("The lattice angles should all be the same and equal to pi/2"
                   " for a tetragonal lattice.")
            raise ValueError(msg.format(lattice_angles))
        
        a1 = [a, 0, 0]
        a2 = [0, a, 0]
        a3 = [0, 0, c]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors
    
    elif lattice_type == "body_centered_tetragonal":
        if not (np.isclose(a, b) and
                not np.isclose(b, c)):
            msg = ("For a body-centered, tetragonal lattice, a = b != c where "
                   "a, b, and c are the first, second, and third entries in "
                   "lattice_constants, respectively.")
            raise ValueError(msg.format(lattice_constants))
        if not (np.isclose(alpha, np.pi/2) and np.isclose(beta, np.pi/2)
                and np.isclose(gamma, np.pi/2)):
            msg = ("The lattice angles should all be the same and equal to pi/2"
                   " for a body-centered, tetragonal lattice.")
            raise ValueError(msg.format(lattice_angles))

        a1 = [-a/2,  a/2,  c/2]
        a2 = [ a/2, -a/2,  c/2]
        a3 = [ a/2,  a/2, -c/2]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lattice_type == "orthorhombic":
        if (np.isclose(a, b) or np.isclose(b, c) or np.isclose(a, c)):
            msg = ("The lattice constants should all be different for an "
                   "orthorhombic lattice.")
            raise ValueError(msg.format(lattice_constants))
        if not (np.isclose(alpha, np.pi/2) and np.isclose(beta, np.pi/2)
                and np.isclose(gamma, np.pi/2)):
            msg = ("The lattice angles should all be the same and equal to pi/2 "
                   "for an orthorhombic lattice.")
            raise ValueError(msg.format(lattice_angles))
        if not (a < b < c):
            msg = ("The lattice constants should in ascending order for an "
                   "orthorhombic lattice.")
            raise ValueError(msg.format(lattice_constants))

        a1 = [a, 0, 0]
        a2 = [0, b, 0]
        a3 = [0, 0, c]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lattice_type == "face_centered_orthorhombic":
        if (np.isclose(a, b) or np.isclose(b, c) or np.isclose(a, c)):
            msg = ("The lattice constants should all be different for a "
                   "face-centered, orthorhombic lattice.")
            raise ValueError(msg.format(lattice_constants))
        if not (np.isclose(alpha, np.pi/2) and np.isclose(beta, np.pi/2)
                and np.isclose(gamma, np.pi/2)):
            msg = ("The lattice angles should all be the same and equal to pi/2"
                   " for a face-centered, orthorhombic lattice.")
            raise ValueError(msg.format(lattice_angles))
        if not (a < b < c):
            msg = ("The lattice constants should in ascending order for a ."
                   "face-centered, orthorhombic lattice.")
            raise ValueError(msg.format(lattice_constants))

        a1 = [  0, b/2, c/2]
        a2 = [a/2,   0, c/2]
        a3 = [a/2, b/2,   0]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lattice_type == "body_centered_orthorhombic":
        if (np.isclose(a, b) or np.isclose(b, c) or np.isclose(a, c)):
            msg = ("The lattice constants should all be different for a "
                   "body-centered, orthorhombic lattice.")
            raise ValueError(msg.format(lattice_constants))
        if not (np.isclose(alpha, np.pi/2) and np.isclose(beta, np.pi/2)
                and np.isclose(gamma, np.pi/2)):
            msg = ("The lattice angles should all be the same and equal to pi/2"
                   " for a body-centered, orthorhombic lattice.")
            raise ValueError(msg.format(lattice_angles))
        if not (a < b < c):
            msg = ("The lattice constants should in ascending order for a ."
                   "body-centered, orthorhombic lattice.")
            raise ValueError(msg.format(lattice_constants))

        a1 = [-a/2,  b/2,  c/2]
        a2 = [ a/2, -b/2,  c/2]
        a3 = [ a/2,  b/2, -c/2]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors
 
    elif lattice_type == "base_centered_orthorhombic":
        if (np.isclose(a, b) or np.isclose(b, c) or np.isclose(a, c)):
            msg = ("The lattice constants should all be different for a "
                   "base-centered, orthorhombic lattice.")
            raise ValueError(msg.format(lattice_constants))
        if not (np.isclose(alpha, np.pi/2) and np.isclose(beta, np.pi/2)
                and np.isclose(gamma, np.pi/2)):
            msg = ("The lattice angles should all be the same and equal to pi/2"
                   " for a base-centered, orthorhombic lattice.")
            raise ValueError(msg.format(lattice_angles))
        if not (a < b < c):
            msg = ("The lattice constants should in ascending order for a ."
                   "base-centered, orthorhombic lattice.")
            raise ValueError(msg.format(lattice_constants))

        a1 = [a/2, -b/2, 0]
        a2 = [a/2,  b/2, 0]
        a3 = [  0,    0, c]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lattice_type == "hexagonal":
        if not (np.isclose(a, b) and
                not np.isclose(b, c)):
            msg = ("For a hexagonal lattice, a = b != c where "
                   "a, b, and c are the first, second, and third entries in "
                   "lattice_constants, respectively.")
            raise ValueError(msg.format(lattice_constants))
        if not (np.isclose(alpha, beta) and np.isclose(beta, np.pi/2) and
                np.isclose(gamma, 2*np.pi/3)):
            msg = ("The first two lattice angles, alpha and beta, should be the "
                   "same and equal to pi/2 while the third gamma should be "
                   "2pi/3 radians for a hexagonal lattice.")
            raise ValueError(msg.format(lattice_angles))
        
        a1 = [a/2, -a*np.sqrt(3)/2, 0]
        a2 = [a/2, a*np.sqrt(3)/2, 0]
        a3 = [0, 0, c]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lattice_type == "rhombohedral":
        if not (np.isclose(a, b) and np.isclose(b,c) and np.isclose(a, c)):
            msg = ("For a rhombohedral lattice, a = b = c where "
                   "a, b, and c are the first, second, and third entries in "
                   "lattice_constants, respectively.")
            raise ValueError(msg.format(lattice_constants))
        if not (np.isclose(alpha, beta) and np.isclose(beta, gamma) and
                np.isclose(alpha, gamma)):
            msg = ("All lattice angles should be the same for a rhombohedral "
                   "lattice.")
            raise ValueError(msg.format(lattice_angles))
        if (np.isclose(alpha, np.pi/2) or np.isclose(beta, np.pi/2) or
            np.isclose(gamma, np.pi/2)):
            msg = ("No lattice angle should be equal to pi/2 radians for a "
                   "rhombohedral lattice.")
            raise ValueError(msg.format(lattice_angles))
        
        a1 = [a*np.cos(alpha/2), -a*np.sin(alpha/2), 0]
        a2 = [a*np.cos(alpha/2),  a*np.sin(alpha/2), 0]
        a3 = [a*np.cos(alpha)/np.cos(alpha/2), 0,
              a*np.sqrt(1 - np.cos(alpha)**2/np.cos(alpha/2)**2)]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lattice_type == "monoclinic":
        if (a > c or b > c):
            msg = ("The first and second lattice constants, a and b, should "
                   "both be less than or equal to the last lattice constant, c,"
                   " for a monoclinic lattice.")
            raise ValueError(msg.format(lattice_constants))
        if alpha >= np.pi/2:
            msg = ("The first lattice angle, alpha, should be less than pi/2 "
                   "radians for a monoclinic lattice.")
            raise ValueError(msg.format(lattice_angles))
        if not (np.isclose(beta, np.pi/2) and np.isclose(gamma, np.pi/2)):
            msg = ("The second and third lattice angles, beta and gamma, "
                   "should both be pi/2 radians for a monoclinic lattice.")
            raise ValueError(msg.format(lattice_angles))
        
        a1 = [a, 0, 0]
        a2 = [0, b, 0]
        a3 = [0, c*np.cos(alpha), c*np.sin(alpha)]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lattice_type == "base_centered_monoclinic":
        if (a > c or b > c):
            msg = ("The first and second lattice constants, a and b, should "
                   "both be less than or equal to the last lattice constant c. "
                   " for a monoclinic lattice.")
            raise ValueError(msg.format(lattice_constants))
        if alpha >= np.pi/2:
            msg = ("The first lattice angle, alpha, should be less than pi/2 "
                   "radians for a monoclinic lattice.")
            raise ValueError(msg.format(lattice_angles))
        if not (np.isclose(beta, np.pi/2) and np.isclose(gamma, np.pi/2)):
            msg = ("The second and third lattice angles, beta and gamma, "
                   "should both be pi/2 radians for a monoclinic lattice.")
            raise ValueError(msg.format(lattice_angles))
        
        a1 = [ a/2, b/2, 0]
        a2 = [-a/2, b/2, 0]
        a3 = [0, c*np.cos(alpha), c*np.sin(alpha)]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors
                
    elif lattice_type == "triclinic":
        if (np.isclose(a, b) or np.isclose(b, c) or np.isclose(a, c)):
            msg = ("The lattice constants should all be different for a "
                   "triclinic lattice.")
            raise ValueError(msg.format(lattice_constants))
        if (np.isclose(alpha, beta) or np.isclose(beta, gamma) or
            np.isclose(alpha, gamma)):
            msg = ("The lattice angles should all be different for a "
                   "triclinic lattice.")
            raise ValueError(msg.format(lattice_angles))
        
        a1 = [a, 0, 0]
        a2 = [b*np.cos(gamma), b*np.sin(gamma), 0]
        a3 = [c*np.cos(beta), c/np.sin(gamma)*(np.cos(alpha) -
                                               np.cos(beta)*np.cos(gamma)),
              c/np.sin(gamma)*np.sqrt(np.sin(gamma)**2 - np.cos(alpha)**2 - 
                                       np.cos(beta)**2 + 2*np.cos(alpha)*
                                       np.cos(beta)*np.cos(gamma))]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors
    else:
        msg = "Please provide a valid lattice type."
        raise ValueError(msg.format(lattice_type))

    
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
        mesh_car (list): a list of mesh point positions in Cartesian 
            coordinates.
        cell_vecs (numpy.ndarray): the vectors that define the integration cell.
        coord (str): a string that indicatese coordinate system of the points.
            It can be in Cartesian ("cart") or lattice ("cell").

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
            # If the group operation moves the point outside the cell, %1 moves
            # it back in.
            # I ran into floating point precision problems the last time I ran
            # %1. Just to be safe it's included here.
            new_mp = np.round(np.dot(pg, mp), 12)%1.
            if any([np.allclose(new_mp, mc) for mc in mesh_copy]):
                ind = np.where(np.array([np.allclose(new_mp, mc)
                                         for mc in mesh_copy]) == True)[0][0]
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
