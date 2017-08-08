"""Generate quantities related to the symmetry of the lattice. This module
draws heavily from Setyawan, Wahyu, and Stefano Curtarolo. "High-throughput 
electronic band structure calculations: Challenges and tools." Computational 
Materials Science 49.2 (2010): 299-312.
"""

import numpy as np
from numpy.linalg import norm, inv, det
import math
import itertools
from copy import deepcopy
from itertools import islice, product
from BZI.msg import err

from phenum.symmetry import get_lattice_pointGroup
from phenum.grouptheory import SmithNormalForm
from phenum.vector_utils import _minkowski_reduce_basis

class Lattice(object):
    """Create a lattice.

    Args:
        centering_type (str): identifies the position of lattice points in
            the conventional unit cell. Option include 'prim', 'base', 'body',
            and 'center'.
        lattice_constants (list): a list of constants that correspond to the
            lengths of the lattice vectors in the conventional unit cell ordered
            as [a,b,c].
        lattice_angles (list): a list of angles in radians that correspond to
            the angles between lattice vectors in the conventional unit cell
            ordered as [alpha, beta, gamma] where alpha is the angle between bc,
            beta is the angle between ac, and gamma is the angle between ab.

    Attributes:
        centering (str): the type of lattice point centering in the 
            conventional unit cell. Option include 'prim', 'base', 'body', and
            'center'.
        constants (list): a list of constants that correspond to the
            lengths of the lattice vectors in the conventional unit cell ordered
            as [a,b,c].
        angles (list): a list of angles in radians that correspond to
            the angles between lattice vectors in the conventional unit cell
            ordered as [alpha, beta, gamma] where alpha is the angle between bc,
            beta is the angle between ac, and gamma is the angle between ab.
        vectors (numpy.ndarray): an array of primitive lattice vectors
            as columns of a 3x3 matrix.
        reciprocal_vectors (numpy.ndarray): the reciprocal primitive 
            translation vectors as columns of a 3x3 matrix.
        symmetry_group (numpy.ndarray): the group of transformations under which
            the lattice in invariant.
        symmetry_points (dict): a dictionary of high symmetry points with the
            keys as letters and values as lattice coordinates.
        symmetry_paths (list): a list of symmetry point pairs used when creating
            a band structure plot.
        type (str): the Bravais lattice type.
        volume (float): the volume of the parallelepiped given by the three
            lattice vectors
        reciprocal_volume (float): the volume of the parallelepiped given by the three
            reciprocal lattice vectors

    """
    
    def __init__(self, centering_type, lattice_constants, lattice_angles):
        self.centering = centering_type
        self.constants = lattice_constants
        self.angles = lattice_angles
        self.type = find_lattice_type(centering_type, lattice_constants,
                                      lattice_angles)        
        self.vectors = make_lattice_vectors(self.type, lattice_constants,
                                            lattice_angles)
        self.reciprocal_vectors = make_rptvecs(self.vectors)
        self.symmetry_group = point_group(self.vectors)
        self.symmetry_points = get_sympts(centering_type, lattice_constants,
                                          lattice_angles)
        self.symmetry_paths = get_sympaths(centering_type, lattice_constants,
                                           lattice_angles)
        self.volume = det(self.vectors)
        self.reciprocal_volume = det(self.reciprocal_vectors)

# Define the symmetry points for a simple-cubic lattice in lattice coordinates.
sc_sympts = {"G": [0. ,0., 0.],
              "R": [1./2, 1./2, 1./2],
              "X": [0., 1./2, 0.],
              "M": [1./2, 1./2, 0.]}

# Define the symmetry points for a fcc lattice in lattice coordinates.
# Coordinates are in lattice coordinates.
fcc_sympts = {"G": [0., 0., 0.], # G is the gamma point.
              "K": [3./8, 3./8, 3./4],
              "L": [1./2, 1./2, 1./2],
              "U": [5./8, 1./4, 5./8],              
              "W": [1./2, 1./4, 3./4],
              "X": [1./2, 0., 1./2],
              "G2":[1., 1., 1.]}

# Define the symmetry points for a bcc lattice in lattice coordinates
bcc_sympts = {"G": [0., 0., 0.],
               "H": [1./2, -1./2, 1./2],
               "P": [1./4, 1./4, 1./4],
               "N": [0., 0., 1./2]}

# Tetragonal high symmetry points
tet_sympts = {"G": [0., 0., 0.],
              "A": [1./2, 1./2, 1./2],
              "M": [1./2, 1./2, 0.],
              "R": [0., 1./2, 1./2],
              "X": [0., 1./2, 0.],
              "Z": [0., 0., 1./2]}

def bct1_sympts(a, c):
    """Return the body-centered tetragonal high symmetry points for c < a as a 
    dictionary.
    """
    
    eta = (1. + c**2/a**2)/4.
    return {"G": [0., 0., 0.],
            "M": [-1./2, 1./2, 1./2],
            "N": [0., 1./2, 0.],
            "P": [1./4, 1./4, 1./4],
            "X": [0., 0., 1./2],
            "Z": [eta, eta, -eta],
            "Z1": [-eta, 1-eta, eta]}

def bct2_sympts(a, c):
    """Return the body-centered tetragonal high symmetry points for a < c
    as a dictionary.
    """
    
    eta = (1. + a**2/c**2)/4.
    zeta = a**2/(2*c**2)
    return {"G": [0., 0., 0.],
            "N": [0., 1./2, 0.],
            "P": [1./4, 1./4, 1./4],
            "S": [-eta, eta, eta], # Sigma
            "S1": [eta, 1-eta, -eta], # Sigma_1
            "X": [0., 0., 1./2],
            "Y": [-zeta, zeta, 1./2],
            "Y1": [1./2, 1./2, -zeta],
            "Z": [1./2, 1./2, -1./2]}

# Orthorhombic high symmetry points
orc_sympts = {"G": [0., 0., 0.],
              "R": [1./2, 1./2, 1./2],
              "S": [1./2, 1./2, 0.],
              "T": [0., 1./2, 1./2],
              "U": [1./2, 0., 1./2],
              "X": [1./2, 0., 0.],
              "Y": [0., 1./2, 0.],
              "Z": [0., 0., 1./2]}

def orcf13_sympts(a, b, c):
    """Return the face-centered orthorhombic high symmetry points for
     1/a**2 > 1/b**2 +1/c**2 and 1/a**2 = 1/b**2 +1/c**2 as a dictionary.
    """
    
    a = float(a)
    b = float(b)
    c = float(c)
    zeta = (1 + (a/b)**2 - (a/c)**2)/4.
    eta = (1 + (a/b)**2 + (a/c)**2)/4.
    
    return {"G": [0., 0., 0.],
            "A": [1./2, 1./2+zeta, zeta],
            "A1": [1./2, 1./2 - zeta, 1 - zeta],
            "L": [1./2, 1./2, 1./2],
            "T": [1., 1./2, 1./2],
            "X": [0., eta, eta],
            "X1": [1., 1-eta, 1-eta],
            "Y": [1./2, 0., 1./2],
            "Z": [1./2, 1./2, 0.]}

def orcf2_sympts(a, b, c):
    """Return the face-centered orthorhombic high symmetry points for
     1/a**2 < 1/b**2 +1/c**2 as a dictionary.
    """

    a = float(a)
    b = float(b)
    c = float(c)
    eta = (1 + a**2/b**2 - a**2/c**2)/4
    phi = (1 + c**2/b**2 - c**2/a**2)/4
    delta = (1 + b**2/a**2 - b**2/c**2)/4
    
    return {"G": [0., 0., 0.],
            "C": [1./2, 1./2 - eta, 1. - eta],
            "C1": [1./2, 1./2 + eta, eta],
            "D": [1./2 - delta, 1./2, 1. - delta],
            "D1": [1./2 + delta, 1./2, delta],
            "L": [1./2, 1./2, 1./2],
            "H": [1 - phi, 1./2 - phi, 1./2],
            "H1": [phi, 1./2 + phi, 1./2],
            "X": [0., 1./2, 1./2],
            "Y": [1./2, 0., 1./2],
            "Z": [1./2, 1./2, 0.]}

def orci_sympts(a, b, c):
    """Return the body-centered orthorhombic high symmetry points.
    """
    
    a = float(a)
    b = float(b)
    c = float(c)
    zeta = (1 + a**2/c**2)/4
    eta = (1 + b**2/c**2)/4
    delta = (b**2 - a**2)/(4*c**2)
    mu = (a**2 + b**2)/(4*c**2)
    
    return {"G": [0., 0., 0.],
            "L": [-mu, mu, 1./2 - delta],
            "L1": [mu, -mu, 1./2 + delta],
            "L2": [1./2 - delta, 1./2 + delta, -mu],
            "R": [0., 1./2, 0.],
            "S": [1./2, 0., 0.],
            "T": [0., 0., 1./2],
            "W": [1./4, 1./4, 1./4],
            "X": [-zeta, zeta, zeta],
            "X1": [zeta, 1-zeta, -zeta],
            "Y": [eta, -eta, eta],
            "Y1": [1-eta, eta, -eta],
            "Z": [1./2, 1./2, -1./2]}

def orcc_sympts(a, b):
    """Return the base-centered orthorhombic high symmetry points.
    """
    
    a = float(a)
    b = float(b)
    zeta = (1 + a**2/b**2)/4
    
    return {"G": [0., 0., 0.],
            "A": [zeta, zeta, 1./2],
            "A1": [-zeta, 1-zeta, 1./2],
            "R": [0., 1./2, 1./2],
            "S": [0., 1./2, 0.],
            "T": [-1./2, 1./2, 1./2],
            "X": [zeta, zeta, 0],
            "X1": [-zeta, 1-zeta, 0],
            "Y": [-1./2, 1./2, 0.],
            "Z": [0., 0., 1./2]}

# High symmetry points for a hexagonal lattice.
hex_sympts = {"G": [0., 0., 0.],
              "A": [0., 0., 1./2],
              "H": [1./3, 1./3, 1./2],
              "K": [1./3, 1./3, 0.],
              "L": [1./2, 0., 1./2],
              "M": [1./2, 0., 0.]}

def rhl1_sympts(alpha):
    """Return the rhombohedral lattice points for alpha < pi/2 radians.
    """
    alpha = float(alpha)
    eta = (1 + 4*np.cos(alpha))/(2 + 4*np.cos(alpha))
    nu = 3./4 - eta/2
    
    return {"G": [0., 0., 0.],
            "B": [eta, 1./2, 1-eta],
            "B1": [1./2, 1-eta, eta-1],
            "F": [1./2, 1./2, 0.],
            "L": [1./2, 0.,  0.],
            "L1": [0., 0., -1./2],
            "P": [eta, nu, nu],
            "P1": [1-nu, 1-nu, 1-eta],
            "P2": [nu, nu, eta-1],
            "Q": [1-nu, nu, 0],
            "X": [nu, 0, -nu],
            "Z": [1./2, 1./2, 1./2]}

def rhl2_sympts(alpha):
    """Return the rhombohedral lattice points for alpha > pi/2 radians.
    """
    
    alpha = float(alpha)
    eta = 1/(2*np.tan(alpha/2)**2)
    nu = 3./4 - eta/2
    return {"G": [0., 0., 0.],
            "F": [1./2, -1./2, 0.],
            "L": [1./2, 0., 0.],
            "P": [1-nu, -nu, 1-nu],
            "P1": [nu, nu-1, nu-1],
            "Q": [eta, eta, eta],
            "Q1": [1-eta, -eta, -eta],
            "Z": [1./2, -1./2, 1./2]}

def mcl_sympts(b, c, alpha):
    """Return the high symmetry points for the monoclinic lattice as a 
    dictionary where the keys are strings the values are the lattice coordinates
    of the high symmetry points.
    """

    b = float(b)
    c = float(c)
    alpha = float(alpha)
    
    eta = (1 - b*np.cos(alpha)/c)/(2*np.sin(alpha)**2)
    nu = 1./2 - eta*c*np.cos(alpha)/b
    return {"G": [0., 0., 0.],
            "A": [1./2, 1./2, 0.],
            "C": [0., 1./2, 1./2],
            "D": [1./2, 0., 1./2],
            "D1": [1./2, 0., -1./2],
            "E": [1./2, 1./2, 1./2],
            "H": [0., eta, 1-nu],
            "H1": [0., 1-eta, nu],
            "H2": [0, eta, -nu],
            "M": [1./2, eta, 1-nu],
            "M1": [1./2, 1-eta, nu],
            "M2": [1./2, eta, -nu],
            "X": [0., 1./2, 0.],
            "Y": [0., 0., 1./2],
            "Y1": [0., 0., -1./2],
            "Z": [1./2, 0., 0.]}

def mclc12_sympts(a, b, c, alpha):
    """Return the high symmetry points for a base-centered monoclinic lattice 
    with kgamma > pi/2 and kgamma = pi/2 as a dictionary where the keys are 
    strings the values are the lattice coordinates of the high symmetry points.
    """
    
    a = float(a)
    b = float(b)
    c = float(c)
    alpha = float(alpha)
    
    zeta = (2 - b*np.cos(alpha)/c)/(4*np.sin(alpha)**2)
    eta = 1./2 + 2*zeta*c*np.cos(alpha)/b
    psi = 3./4 - a**2/(4*b**2*np.sin(alpha)**2)
    phi = psi + (3./4 - psi)*b*np.cos(alpha)/c

    return {"G": [0., 0., 0.],
            "N": [1./2, 0., 0.],
            "N1": [0., -1./2, 0.],
            "F": [1-zeta, 1-zeta, 1-eta],
            "F1": [zeta, zeta, eta],
            "F2": [-zeta, -zeta, 1-eta],
            "F3": [1-zeta, -zeta, 1-eta],
            "I": [phi, 1-phi, 1./2],
            "I1": [1-phi, phi-1, 1./2],
            "L": [1./2, 1./2, 1./2],
            "M": [1./2, 0., 1./2],
            "X": [1-psi, psi-1, 0.],
            "X1": [psi, 1-psi, 0.],
            "X2": [psi-1, -psi, 0.],
            "Y": [1./2, 1./2, 0.],
            "Y1": [-1./2, -1./2, 0.],
            "Z": [0., 0., 1./2]}

def mclc34_sympts(a, b, c, alpha):
    """Return the high symmetry points for a base-centered monoclinic lattice
    with gamma < pi/2 and b*cos(alpha/c) + b**2*sin(alpha/a**2)**2 <= 1 (3 is < 1, 4 = 1) as
    a dictionary where the keys are strings the values are the lattice
    coordinates of the high symmetry points. 
    """
    
    a = float(a)
    b = float(b)
    c = float(c)
    alpha = float(alpha)

    mu = (1 + b**2/a**2)/4
    delta = b*c*np.cos(alpha)/(2*a**2)
    zeta = mu - 1./4 + (1 - b*np.cos(alpha)/c)/(4*np.sin(alpha)**2)
    eta = 1./2 + 2*zeta*c*np.cos(alpha)/b
    phi = 1 + zeta - 2*mu
    psi = eta - 2*delta
                
    return {"G": [0., 0., 0.],
            "F": [1-phi, 1-phi, 1-psi],
            "F1": [phi, phi-1, psi],
            "F2": [1-phi, -phi, 1-psi],
            "H": [zeta, zeta, eta],
            "H1": [1-zeta, -zeta, 1-eta],
            "H2": [-zeta, -zeta, 1-eta],
            "I": [1./2, -1./2, 1./2],
            "M": [1./2, 0., 1./2],
            "N": [1./2, 0., 0.],
            "N1": [0., -1./2, 0.],
            "X": [1./2, -1./2, 0.],
            "Y": [mu, mu, delta],
            "Y1": [1-mu, -mu, -delta],
            "Y2": [-mu, -mu, -delta],
            "Y3": [mu, mu-1, delta],
            "Z": [0., 0., 1./2]}

def mclc5_sympts(a, b, c, alpha):
    """Return the high symmetry points for a base-centered monoclinic lattice
    with gamma < pi/2 and b*cos(alpha/c) + b**2*sin(alpha/a**2)**2 > 1 as
    a dictionary where the keys are strings the values are the lattice
    coordinates of the high symmetry points. 
    """

    a = float(a)
    b = float(b)
    c = float(c)
    alpha = float(alpha)

    zeta = (b**2/a**2 + (1 - b*np.cos(alpha)/c)/np.sin(alpha)**2)/4
    eta = 1./2 + 2*zeta*c*np.cos(alpha)/b
    mu = eta/2 + b**2/(4*a**2) - b*c*np.cos(alpha)/(2*a**2)
    nu = 2*mu - zeta
    omega = (4*nu - 1 - b**2*np.sin(alpha)**2/a**2)*c/(2*b*np.cos(alpha))
    delta = zeta*c*np.cos(alpha)/b + omega/2 - 1./4
    rho = 1 - zeta*a**2/b**2

    return {"G": [0., 0., 0.],
            "F": [nu, nu, omega],
            "F1": [1-nu, 1-nu, 1-omega],
            "F2": [nu, nu-1, omega],
            "H": [zeta, zeta, eta],
            "H1": [1-zeta, -zeta, 1-eta],
            "H2": [-zeta, -zeta, 1-eta],
            "I": [rho, 1-rho, 1./2],
            "I1": [1-rho, rho-1, 1./2],
            "L": [1./2, 1./2, 1./2],
            "M": [1./2, 0., 1./2],
            "N": [1./2, 0., 0.],
            "N1": [0., -1./2, 0.],
            "X": [1./2, -1./2, 0.],
            "Y": [mu, mu, delta],
            "Y1": [1-mu, -mu, -delta],
            "Y2": [-mu, -mu, -delta],
            "Y3": [mu, mu-1, delta],
            "Z": [0., 0., 1./2]}

# Triclinic symmatry points with lattice parameters that satisfy

## tri1a ##
# k_alpha > pi/2
# k_beta > pi/2
# k_gamma > pi/2 where k_gamma = min(k_alpha, k_beta, k_gamma)

## tri2a ##
# k_alpha > pi/2
# k_beta > pi/2
# k_gamma = pi/2
tri1a2a_sympts = {"G": [0., 0., 0.],
                  "L": [1./2, 1./2, 0.],
                  "M": [0., 1./2, 1./2],
                  "N": [1./2, 0., 1./2],
                  "R": [1./2, 1./2, 1./2],
                  "X": [1./2, 0., 0.],
                  "Y": [0., 1./2, 0.],
                  "Z": [0., 0., 1./2]}

# Triclinic symmatry points with lattice parameters that satisfy

## tri1b ##
# k_alpha < pi/2
# k_beta < pi/2
# k_gamma < pi/2 where k_gamma = max(k_alpha, k_beta, k_gamma)

## tri2b ##
# k_alpha < pi/2
# k_beta < pi/2
# k_gamma = pi/2
tr1b2b_sympts = {"G": [0., 0., 0.],
                 "L": [1./2, -1./2, 0.],
                 "M": [0., 0., 1./2],
                 "N": [-1./2, -1./2, 1./2],
                 "R": [0., -1./2, 1./2],
                 "X": [0., -1./2, 0.],
                 "Y": [1./2, 0., 0.],
                 "Z": [-1./2, 0., 1./2]}

def get_sympts(centering_type, lattice_constants, lattice_angles):
    """Find the symmetry points for the provided lattice.

    Args:
        centering_type (str): the centering type for the lattice. Vaild
            options include 'prim', 'base', 'body', and 'face'.
        lattice_constants (list): a list of lattice constants [a, b, c].
        lattice_angles (list): a list of lattice angles [alpha, beta, gamma].

    Returns:
        (dict): a dictionary with a string of letters as the keys and lattice 
            coordinates of the symmetry points ase values.

    Example:
        >>> lattice_constants = [4.05]*3
        >>> lattice_angles = [numpy.pi/2]*3
        >>> symmetry_points = get_sympts(lattice_constants, lattice_angles)
    """
    
    a = float(lattice_constants[0])
    b = float(lattice_constants[1])
    c = float(lattice_constants[2])
    
    alpha = float(lattice_angles[0])
    beta = float(lattice_angles[1])
    gamma = float(lattice_angles[2])
    
    lattice_vectors = make_ptvecs(centering_type, lattice_constants,
                                  lattice_angles)
    reciprocal_lattice_vectors = make_rptvecs(lattice_vectors)
    
    rlat_veca = reciprocal_lattice_vectors[:,0] # individual reciprocal lattice vectors
    rlat_vecb = reciprocal_lattice_vectors[:,1]
    rlat_vecc = reciprocal_lattice_vectors[:,2]
    
    ka = norm(rlat_veca) # lengths of primitive reciprocal lattice vectors
    kb = norm(rlat_vecb)
    kc = norm(rlat_vecc)

    # These are the angles between reciprocal lattice vectors.
    kalpha = np.arccos(np.dot(rlat_vecb, rlat_vecc)/(kb*kc))
    kbeta = np.arccos(np.dot(rlat_veca, rlat_vecc)/(ka*kc))
    kgamma = np.arccos(np.dot(rlat_veca, rlat_vecb)/(ka*kb))
    
    # Start with the cubic lattices, which have all angles equal to pi/2 radians.
    if (np.isclose(alpha, np.pi/2) and
        np.isclose(beta, np.pi/2) and
        np.isclose(gamma, np.pi/2)):
        if (np.isclose(a, b) and
            np.isclose(b, c)):
            if centering_type == "prim":
                return sc_sympts
            elif centering_type == "body":
                return bcc_sympts
            elif centering_type == "face":
                return fcc_sympts
            else:
                msg = ("Valid lattice centerings for cubic latices include "
                       "'prim', 'body', and 'face'.")
                raise ValueError(msg.format(centering_type))
            
        # Tetragonal.
        elif (np.isclose(a,b) and not np.isclose(b,c)):
            if centering_type == "prim":
                return tet_sympts
            elif centering_type == "body":
                if c < a:
                    return bct1_sympts(a, c)
                else:
                    return bct2_sympts(a, c)
            else:
                msg = ("Valid lattice centerings for tetragonal lattices "
                       "include 'prim' and 'body'.")
                raise ValueError(msg.format(centering_type))
            
        # Last of the lattices with all angles equal to pi/2 is orthorhombic.
        else:
            if centering_type == "prim":
                return orc_sympts
            
            elif centering_type == "base":
                return orcc_sympts(a, b)
            
            elif centering_type == "body":
                return orci_sympts(a, b, c)
            
            elif centering_type == "face":
                if  (1/a**2 >= 1/b**2 +1/c**2):
                    return orcf13_sympts(a, b, c)
                else:
                    return orcf2_sympts(a, b, c)
                
            else:
                msg = ("Valid lattice centerings for orthorhombic lattices "
                       "include 'prim', 'base', 'body', and 'face'.")
                raise ValueError(msg.format(centering_type))
            
    # Hexagonal has alpha = beta = pi/2, gamma = 2pi/3, a = b != c.
    if (np.isclose(alpha, beta) and np.isclose(beta, np.pi/2) and
        np.isclose(gamma, 2*np.pi/3) and np.isclose(a, b) and not
        np.isclose(b, c)):
        return hex_sympts

    # Rhombohedral has equal angles and constants.
    elif (np.isclose(alpha, beta) and np.isclose(beta, gamma) and 
          np.isclose(a, b) and np.isclose(b, c)):
            if alpha < np.pi/2:
                return rhl1_sympts(alpha)
            else:
                return rhl2_sympts(alpha)

    # Monoclinic a,b <= c, alpha < pi/2, beta = gamma = pi/2, a != b != c
    elif (not (a > c or b > c) and np.isclose(beta, gamma) and
          np.isclose(beta, np.pi/2) and alpha < np.pi/2):
        if centering_type == "prim":
            return mcl_sympts(b, c, alpha)
        elif centering_type == "base":
            if kgamma > np.pi/2 or np.isclose(kgamma, np.pi/2):
                return mclc12_sympts(a, b, c, alpha)
            
            elif (kgamma < np.pi/2
                  and ((b*np.cos(alpha)/c + (b*np.sin(alpha)/a)**2) < 1.
                       or np.isclose(b*np.cos(alpha)/c + (b*np.sin(alpha)/a)**2, 1))):
                return mclc34_sympts(a, b, c, alpha)
            
            elif (kgamma < np.pi/2 and
                  (b*np.cos(alpha)/c + (b*np.sin(alpha)/a)**2) > 1.):
                return mclc5_sympts(a, b, c, alpha)
            
            else:
                msg = "Something is wrong with the monoclinic lattice provided."
                raise ValueError(msg.format(reciprocal_lattice_vectors))
        else:
            msg = ("Valid lattice centerings for monoclinic lattices "
                   "include 'prim' and 'base'")
            raise ValueError(msg.format(centering_type))
        
    # Triclinic a != b != c, alpha != beta != gamma
    elif not (np.isclose(a,b) and np.isclose(b,c) and np.isclose(alpha,beta) and
              np.isclose(beta, gamma)):
        if ((kalpha > np.pi/2 and kbeta > np.pi/2 and kgamma > np.pi/2) or
            (kalpha > np.pi/2 and kbeta > np.pi/2 and np.isclose(kgamma, np.pi/2))):
            return tri1a2a_sympts
        elif ((kalpha < np.pi/2 and kbeta < np.pi/2 and kgamma < np.pi/2) or
              (kalpha < np.pi/2 and kbeta < np.pi/2 and np.isclose(kgamma, np.pi/2))):
            return tr1b2b_sympts
        else:
            msg = "Something is wrong with the triclinic lattice provided."
            raise ValueError(msg.format(reciprocal_lattice_vectors))
    else:
        msg = ("The lattice parameters provided don't correspond to a valid "
               "3D Bravais lattice.")
        raise ValueError(msg.format())
    
def get_sympaths(centering_type, lattice_constants, lattice_angles):
    """Find the symmetry paths for the provided lattice.

    Args:
        centering_type (str): the centering type for the lattice. Vaild
            options include 'prim', 'base', 'body', and 'face'.
        lattice_constants (list): a list of lattice constants [a, b, c].
        lattice_angles (list): a list of lattice angles [alpha, beta, gamma].

    Returns:
        (dict): a dictionary with a string of letters as the keys and lattice 
            coordinates of the symmetry points ase values.

    Example:
        >>> lattice_constants = [4.05]*3
        >>> lattice_angles = [numpy.pi/2]*3
        >>> symmetry_points = get_sympts(lattice_constants, lattice_angles)
    """
    
    a = float(lattice_constants[0])
    b = float(lattice_constants[1])
    c = float(lattice_constants[2])
    
    alpha = float(lattice_angles[0])
    beta = float(lattice_angles[1])
    gamma = float(lattice_angles[2])
    
    lattice_vectors = make_ptvecs(centering_type, lattice_constants,
                                  lattice_angles)
    reciprocal_lattice_vectors = make_rptvecs(lattice_vectors)
    
    rlat_veca = reciprocal_lattice_vectors[:,0] # individual reciprocal lattice vectors
    rlat_vecb = reciprocal_lattice_vectors[:,1]
    rlat_vecc = reciprocal_lattice_vectors[:,2]
    
    ka = norm(rlat_veca) # lengths of primitive reciprocal lattice vectors
    kb = norm(rlat_vecb)
    kc = norm(rlat_vecc)

    # These are the angles between reciprocal lattice vectors.
    kalpha = np.arccos(np.dot(rlat_vecb, rlat_vecc)/(kb*kc))
    kbeta = np.arccos(np.dot(rlat_veca, rlat_vecc)/(ka*kc))
    kgamma = np.arccos(np.dot(rlat_veca, rlat_vecb)/(ka*kb))

    # Start with the cubic lattices, which have all angles equal to pi/2 radians.
    if (np.isclose(alpha, np.pi/2) and
        np.isclose(beta, np.pi/2) and
        np.isclose(gamma, np.pi/2)):
        if (np.isclose(a, b) and
            np.isclose(b, c)):
            if centering_type == "prim":
                return [["G", "X"], ["X", "M"], ["M", "G"], ["G", "R"],
                        ["R", "X"], ["M", "R"]]
            elif centering_type == "body":
                return [["G", "H"], ["H", "N"], ["N", "G"], ["G", "P"],
                        ["P", "H"], ["P", "N"]]
            elif centering_type == "face":
                return [["G", "X"], ["X", "W"], ["W", "K"], ["K", "G"],
                        ["G", "L"], ["L", "U"], ["U", "W"], ["W", "L"],
                        ["L", "K"], ["U", "X"]]
            else:
                msg = ("Valid lattice centerings for cubic latices include "
                       "'prim', 'body', and 'face'.")
                raise ValueError(msg.format(centering_type))
            
        # Tetragonal.
        elif (np.isclose(a,b) and not np.isclose(b,c)):
            if centering_type == "prim":
                return [["G", "X"], ["X", "M"], ["M", "G"], ["G", "Z"],
                        ["Z", "R"], ["R", "A"], ["A", "Z"], ["X", "R"],
                        ["M", "A"]]
            elif centering_type == "body":
                if c < a:
                    return [["G", "X"], ["X", "M"], ["M", "G"], ["G", "Z"],
                            ["Z", "P"], ["P", "N"], ["N", "Z1"], ["Z1", "M"],
                            ["X", "P"]]
                else:
                    return [["G", "X"], ["X", "Y"], ["Y", "S"], ["S", "G"],
                            ["G", "Z"], ["Z", "S1"], ["S1", "N"], ["N", "P"],
                            ["P", "Y1"], ["Y1", "Z"], ["X", "P"]]
            else:
                msg = ("Valid lattice centerings for tetragonal lattices "
                       "include 'prim' and 'body'.")
                raise ValueError(msg.format(centering_type))
            
        # Last of the lattices with all angles equal to pi/2 is orthorhombic.
        else:
            if centering_type == "prim": # orc
                return [["G", "X"], ["X", "S"], ["S", "Y"], ["Y", "G"],
                        ["G", "Z"], ["Z", "U"], ["U", "R"], ["R", "T"],
                        ["T", "Z"], ["Y", "T"], ["U", "X"], ["S", "R"]]
            elif centering_type == "base": # orcc
                return [["G", "X"], ["X", "S"], ["S", "R"], ["R", "A"],
                        ["A", "Z"], ["Z", "G"], ["G", "Y"], ["Y", "X1"],
                        ["X1", "A1"], ["A1", "T"], ["T", "Y"], ["Z", "T"]]
            elif centering_type == "body": # orci
                return [["G", "X"], ["X", "L"], ["L", "T"], ["T", "W"],
                        ["W", "R"], ["R", "X1"], ["X1", "Z"], ["Z", "G"],
                        ["G", "Y"], ["Y", "S"], ["S", "W"], ["L1", "Y"],
                        ["Y1", "Z"]]
            elif centering_type == "face":
                if (1/a**2 > 1/b**2 +1/c**2): # orcf1
                    return[["G", "Y"], ["Y", "T"], ["T", "Z"], ["Z", "G"],
                           ["G", "X"], ["X", "A1"], ["A1", "Y"], ["T", "X1"],
                           ["X", "A"], ["A", "Z"], ["L", "G"]]
                elif np.isclose(1/a**2, 1/b**2 +1/c**2): # orcf3
                    return [["G", "Y"], ["Y", "T"], ["T", "Z"], ["Z", "G"],
                            ["G", "X"], ["X", "A1"], ["A1", "Y"], ["X", "A"],
                            ["A", "Z"], ["L", "G"]]                    
                else: #orcf2
                    return [["G", "Y"], ["Y", "C"], ["C", "D"], ["D", "X"],
                            ["X", "G"], ["G", "Z"], ["Z", "D1"], ["D1", "H"],
                            ["H", "C"], ["C1", "Z"], ["X", "H1"], ["H", "Y"],
                            ["L", "G"]]            
            else:
                msg = ("Valid lattice centerings for orthorhombic lattices "
                       "include 'prim', 'base', 'body', and 'face'.")
                raise ValueError(msg.format(centering_type))
            
    # Hexagonal has alpha = beta = pi/2, gamma = 2pi/3, a = b != c.
    if (np.isclose(alpha, beta) and np.isclose(beta, np.pi/2) and
        np.isclose(gamma, 2*np.pi/3) and np.isclose(a, b) and not
        np.isclose(b, c)):
        return [["G", "M"], ["M", "K"], ["K", "G"], ["G", "A"], ["A", "L"],
                ["L", "H"], ["H", "A"], ["L", "M"], ["K", "H"]]

    # Rhombohedral has equal angles and constants.
    elif (np.isclose(alpha, beta) and np.isclose(beta, gamma) and 
          np.isclose(a, b) and np.isclose(b, c)):
            if alpha < np.pi/2: # RHL1
                return [["G", "L"], ["L", "B1"], ["B", "Z"], ["Z", "G"],
                        ["G", "X"], ["Q", "F"], ["F", "P1"], ["P1", "Z"],
                        ["L", "P"]]
            else: #RHL2
                return [["G", "P"], ["P", "Z"], ["Z", "Q"], ["Q", "G"],
                        ["G", "F"], ["F", "P1"], ["P1", "Q1"], ["Q1", "L"],
                        ["L", "Z"]]

    # Monoclinic a,b <= c, alpha < pi/2, beta = gamma = pi/2, a != b != c
    elif (not (a > c or b > c) and np.isclose(beta, gamma) and
          np.isclose(beta, np.pi/2) and alpha < np.pi/2):
        if centering_type == "prim":
            return [["G", "Y"], ["Y", "H"], ["H", "C"], ["C", "E"],
                    ["E", "M1"], ["M1", "A"], ["A", "X"], ["X", "H1"],
                    ["M", "D"], ["D", "Z"], ["Y", "D"]]
        elif centering_type == "base": # MCLC1
            if kgamma > np.pi/2:
                return [["G", "Y"], ["Y", "F"], ["F", "L"], ["L", "I"],
                        ["I1", "Z"], ["Z", "F1"], ["Y", "X1"], ["X", "G"],
                        ["G", "N"], ["M", "G"]]
            elif np.isclose(kgamma, np.pi/2): # MCLC2
                return [["G", "Y"], ["Y", "F"], ["F", "L"], ["L", "I"],
                        ["I1", "Z"], ["Z", "F1"], ["N", "G"], ["G", "M"]]
            elif (kgamma < np.pi/2 # MCLC3
                  and ((b*np.cos(alpha)/c + (b*np.sin(alpha)/a)**2) < 1)):
                return [["G", "Y"], ["Y", "F"], ["F", "H"], ["H", "Z"],
                        ["Z", "I"], ["I", "F1"], ["H1", "Y1"], ["Y1", "X"],
                        ["X", "G"], ["G", "N"], ["M", "G"]]
            elif (kgamma < np.pi/2 and # MCLC4
                  np.isclose(b*np.cos(alpha)/c + (b*np.sin(alpha)/a)**2, 1)):
                return [["G", "Y"], ["Y", "F"], ["F", "H"], ["H", "Z"], 
                        ["Z", "I"], ["H1", "Y1"], ["Y1", "X"], ["X", "G"],
                        ["G", "N"], ["M", "G"]]
            elif (kgamma < np.pi/2 and # MCLC5
                  (b*np.cos(alpha)/c + (b*np.sin(alpha)/a)**2) > 1.):
                return [["G", "Y"], ["Y", "F"], ["F", "L"], ["L", "I"],
                        ["I1", "Z"], ["Z", "H"], ["H", "F1"], ["H1", "Y1"],
                        ["Y1", "X"], ["X", "G"], ["G", "N"], ["M", "G"]]
            else:
                msg = "Something is wrong with the monoclinic lattice provided."
                raise ValueError(msg.format(reciprocal_lattice_vectors))
        else:
            msg = ("Valid lattice centerings for monoclinic lattices "
                   "include 'prim' and 'base'")
            raise ValueError(msg.format(centering_type))
        
    # Triclinic a != b != c, alpha != beta != gamma
    elif not (np.isclose(a,b) and np.isclose(b,c) and np.isclose(a,c) and 
              np.isclose(alpha,beta) and np.isclose(beta, gamma) and
              np.isclose(alpha, gamma)):
        kangles = np.sort([kalpha, kbeta, kgamma])
        if kangles[0] > np.pi/2: # TRI1a
            return [["X", "G"], ["G", "Y"], ["L", "G"], ["G", "Z"], ["N", "G"],
                    ["G", "M"], ["R", "G"]]
        elif kangles[2] < np.pi/2: #TRI1b
            return [["X", "G"], ["G", "Y"], ["L", "G"], ["G", "Z"],
                    ["N", "G"], ["G", "M"], ["R", "G"]]
        elif (np.isclose(kangles[0], np.pi/2) and (kangles[1] > np.pi/2) and
              (kangles[2] > np.pi/2)): #TRI2a
            return [["X", "G"], ["G", "Y"], ["L", "G"], ["G", "Z"], ["N", "G"],
                    ["G", "M"], ["R", "G"]]
        elif (np.isclose(kangles[2], np.pi/2) and (kangles[0] < np.pi/2) and
              (kangles[1] < np.pi/2)): #TRI2b
            return [["X", "G"], ["G", "Y"], ["L", "G"], ["G", "Z"],
                    ["N", "G"], ["G", "M"], ["R", "G"]]
        else:
            msg = "Something is wrong with the triclinic lattice provided."
            raise ValueError(msg.format(reciprocal_lattice_vectors))
    else:
        msg = ("The lattice parameters provided don't correspond to a valid "
               "3D Bravais lattice.")
        raise ValueError(msg.format())
    
def make_ptvecs(center_type, lat_consts, lat_angles):
    """Provided the lattice type, constants, and angles, return the primitive 
    translation vectors.

    Args:
        center_type (str): identifies the location of the atoms in the cell.
        lat_consts (float or int): the characteristic spacing of atoms in the
            material with a first, b second, and c third in the list. These
            are typically ordered such that a < b < c.
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

    if (np.sum(np.sort(lat_angles)[:2]) < max(lat_angles) or
        np.isclose(np.sum(np.sort(lat_angles)[:2]), max(lat_angles))):
        msg = ("The sum of the two smallest lattice angles must be greater than "
               "the largest lattice angle for the lattice vectors to be "
               "linearly independent.")
        raise ValueError(msg.format(lat_angles))

    # Extract the angles
    alpha = float(lat_angles[0])
    beta = float(lat_angles[1])
    gamma = float(lat_angles[2])

    if (np.isclose(alpha, beta) and np.isclose(beta, gamma) and
        np.isclose(beta, 2*np.pi/3)):
        msg = ("The lattice vectors are linearly dependent with all angles "
               "equal to 2pi/3.")
        raise ValueError(msg.format(lat_angles))
    
    # Extract the lattice constants for the conventional lattice.
    a = float(lat_consts[0])
    b = float(lat_consts[1])
    c = float(lat_consts[2])
    
    # avec is chosen to lie along the x-direction.
    avec = np.array([a, 0., 0.])
    # bvec is chosen to lie in the xy-plane.
    bvec = np.array([b*np.cos(gamma), b*np.sin(gamma), 0])
    # I had to round the argument of the sqrt function in order to avoid
    # numerical errors in cvec.
    cvec = np.array([c*np.cos(beta),
                c/np.sin(gamma)*(np.cos(alpha) -
                                 np.cos(beta)*np.cos(gamma)),
                np.sqrt(np.round(c**2 - (c*np.cos(beta))**2 -
                                 (c/np.sin(gamma)*(np.cos(alpha) -
                                                   np.cos(beta)*np.cos(gamma)))**2, 15))])
    
    if center_type == "prim":
        # I have to take care that a hexagonal grid is rotated 60 degrees so
        # it matches what was obtained in Stefano's paper.
        if ((np.isclose(a, b) and not np.isclose(b,c)) and
            np.isclose(alpha, beta) and np.isclose(beta, np.pi/2) and
            np.isclose(gamma, 2*np.pi/3)):
            rotate = [[np.cos(gamma/2), np.sin(gamma/2), 0],
                        [-np.sin(gamma/2), np.cos(gamma/2), 0],
                        [0, 0, 1]]
            av = np.dot(rotate, avec)
            bv = np.dot(rotate, bvec)
            cv = np.dot(rotate, cvec)
            pt_vecs = np.transpose(np.array([av, bv, cv], dtype=float))
            return pt_vecs
        
        # The rhombohedral lattice vectors also need to be rotated to match
        # those of Stefano.
        elif (np.isclose(alpha, beta) and np.isclose(beta, gamma) and
              not np.isclose(beta, np.pi/2) and np.isclose(a, b) and
              np.isclose(b,c)):
            
            # The vectors in Stefano's paper are mine rotated 60 degrees.
            rotate = [[np.cos(gamma/2), np.sin(gamma/2), 0],
                      [-np.sin(gamma/2), np.cos(gamma/2), 0],
                      [0, 0, 1]]
            av = np.dot(rotate, avec)
            bv = np.dot(rotate, bvec)
            cv = np.dot(rotate, cvec)
            pt_vecs = np.transpose(np.array([av, bv, cv], dtype=float))
            return pt_vecs
        else:
            pt_vecs = np.transpose(np.array([avec, bvec, cvec], dtype=float))
            return pt_vecs
    
    elif center_type == "base":
        av = .5*(avec - bvec)
        bv = .5*(avec + bvec)
        cv = cvec
        
        # The vectors defined in Stefano's paper are defined
        # differently for base-centered, monoclinic lattices.
        if (alpha < np.pi/2 and np.isclose(beta, np.pi/2)
            and np.isclose(gamma, np.pi/2) and a <= c and b <= c
            and not (np.isclose(a,b) or np.isclose(b,c) or np.isclose(a,c))):            
            av = .5*(avec + bvec)
            bv = .5*(-avec + bvec)
            cv = cvec
        pt_vecs  = np.transpose(np.array([av, bv, cv], dtype=float))
        return pt_vecs
        
    elif (not (a > c or b > c) and np.isclose(beta, gamma) and
          np.isclose(beta, np.pi/2) and alpha < np.pi/2):
        av = .5*(avec + bvec)
        bv = .5*(-avec + bvec)
        cv = cvec
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
        >>> lattice_type = "face-centered cubic"
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
    
    if lattice_type == "simple cubic":
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
    
    elif lattice_type == "face-centered cubic":
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
    
    elif lattice_type == "body-centered cubic":
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
    
    elif lattice_type == "body-centered tetragonal":
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
        if (b < a) or (c < b):
            msg = ("The lattice constants should in ascending order for an "
                   "orthorhombic lattice.")
            raise ValueError(msg.format(lattice_constants))

        a1 = [a, 0, 0]
        a2 = [0, b, 0]
        a3 = [0, 0, c]
        lattice_vectors = np.transpose(np.array([a1, a2, a3], dtype=float))
        return lattice_vectors

    elif lattice_type == "face-centered orthorhombic":
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

    elif lattice_type == "body-centered orthorhombic":
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
 
    elif lattice_type == "base-centered orthorhombic":
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
        if (np.isclose(a,b) or np.isclose(b,c) or np.isclose(a,c)):
            msg = ("No two lattice constants are the same for a monoclinic "
                   "lattice.")
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

    elif lattice_type == "base-centered monoclinic":
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

def sym_path(lattice, npts):
    """Create an array of lattice coordinates along the symmetry paths of 
    the lattice.

    Args:
        lattice (:py:obj:`BZI.symmetry.Lattice`): an instance of the Lattice
            class.
        npts (int): the number of points on each symmetry path.
    Return:
        (numpy.array): an array of lattice coordinates along the symmetry
            paths.
    """    
    
    paths = []
    for i,sym_pair in enumerate(lattice.symmetry_paths):
        sym_pti = lattice.symmetry_points[sym_pair[0]]
        sym_ptf = lattice.symmetry_points[sym_pair[1]]

        pxi = sym_pti[0]
        pxf = sym_ptf[0]
        pyi = sym_pti[1]
        pyf = sym_ptf[1]
        pzi = sym_pti[2]
        pzf = sym_ptf[2]
        px = np.linspace(pxi,pxf,npts)
        py = np.linspace(pyi,pyf,npts)
        pz = np.linspace(pzi,pzf,npts)
        ipath = [[px[j],py[j],pz[j]] for j in range(len(px))]
        if i == 0:
            paths += ipath
        else:
            del ipath[-1]
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
    return get_lattice_pointGroup(lat_vecs)

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
            
def find_orbitals(grid_car, lat_vecs, coord = "cart", duplicates=False,
                  eps=10):
    """ Find the partial orbitals of the points in a grid, including only the
    points that are in the grid.

    Args:
        grid_car (numpy.ndarray): a list of grid point positions in Cartesian 
            coordinates.
        lat_vecs (numpy.ndarray): the vectors that define the integration cell.
        coord (str): a string that indicatese coordinate system of the points.
            It can be in Cartesian ("cart") or lattice ("lat").
        duplicates (bool): if there are points in the grid outside the first
            unit cell, duplicates should be true.
        eps (int): a finite precision parameter that determines the number of
            decimals kepts when rounding.

    Returns:
        gp_orbitals (dict): the orbitals of the grid points in a dictionary. 
            The keys of the dictionary are integer labels and the values are the
            grid points in the orbital.
    """
        
    # Put the grid in lattice coordinates and move it into the first unit cell.
    # grid_cell = list(np.round([np.dot(inv(lat_vecs), g) for g in grid_car], 15)%1)
    # I removed the mod 1, put it back in on monday (maybe)
    # Remove duplicates if necessary.
    grid_lat = list(np.round([np.dot(inv(lat_vecs), g) for g in grid_car], eps)%1)
        
    if duplicates:
        grid_copy = list(deepcopy(grid_lat))
        grid_lat = []
        while len(grid_copy) != 0:
            gp = grid_copy.pop()
            if any([np.allclose(gp, gc) for gc in grid_copy]):
                continue
            else:
                grid_lat.append(gp)


    gp_orbitals = {}
    nirr_kpts = 0
    grid_copy = list(deepcopy(grid_lat))
    pointgroup = point_group(lat_vecs)
    pointgroup = [np.dot(np.dot(inv(lat_vecs), pg), lat_vecs) for pg in pointgroup]

    while len(grid_copy) != 0:
        # Grab a point and build its orbit but only include points from the grid.
        g = grid_copy.pop()
        nirr_kpts += 1
        gp_orbitals[nirr_kpts] = [g]
        for pg in pointgroup:
            # If the group operation moves the point outside the cell, mod moves
            # it back in.
            # I ran into floating point precision problems the last time I ran
            # mod, rounding fixed these.
            new_gps = [np.round(np.dot(pg, g), 15)%1, np.round(np.dot(pg, g), 15)]
                  
            # If the new grid point is in the grid, remove it and add it to the
            # orbital of grid point (g).
            for new_gp in new_gps:
                if any([np.allclose(new_gp, gc) for gc in grid_copy]):
                    ind = np.where(np.array([np.allclose(new_gp, gc)
                                             for gc in grid_copy]))[0][0]
                    gp_orbitals[nirr_kpts].append(new_gp)
                    del grid_copy[ind]
                else:
                    continue
    if coord == "cart":
        for i in range(1, len(gp_orbitals.keys()) + 1):
            for j in range(len(gp_orbitals[i])):
                gp_orbitals[i][j] = np.dot(lat_vecs, gp_orbitals[i][j])
        return gp_orbitals# , pointgroup
    elif coord == "lat":
        return gp_orbitals
    else:
        raise ValueError("Coordinate options are 'cell' and 'lat'.")

def nfind_orbitals(grid_car, lat_vecs, HNF, coord = "cart", duplicates=False):
    """ Find the partial orbitals of the points in a grid, including only the
    points that are in the grid.

    Args:
        grid_car (numpy.ndarray): a list of grid point positions in Cartesian 
            coordinates.
        lat_vecs (numpy.ndarray): the vectors that define the integration cell.
        coord (str): a string that indicatese coordinate system of the points.
            It can be in Cartesian ("cart") or lattice ("lat").
        duplicates (bool): if there are points in the grid outside the first
            unit cell, duplicates should be true.

    Returns:
        gp_orbitals (dict): the orbitals of the grid points in a dictionary. 
            The keys of the dictionary are integer labels and the values are the
            grid points in the orbital.
    """
    
    # Put the grid in lattice coordinates.
    grid_cell = np.dot(np.linalg.inv(lattice.vectors), np.array(grid_car).T).T
    # grid_cell = list(np.round([np.dot(inv(lat_vecs), g) for g in grid_car], 9)%1)

    # Put the grid in group coordinates
    S,L,R = SmithNormalForm(HNF)
    grid_group = np.dot(L, grid_cell.T).T

    # Remove duplicates if necessary.
    if duplicates:
        grid_copy = list(deepcopy(grid_cell))
        grid_cell = []
        while len(grid_copy) != 0:
            gp = grid_copy.pop()
            if any([np.allclose(gp, gc) for gc in grid_copy]):
                continue
            else:
                grid_cell.append(gp)
        
    gp_orbitals = {}
    nirr_kpts = 0
    grid_copy = list(deepcopy(grid_cell))
    pointgroup = point_group(lat_vecs)
    pointgroup = [np.dot(np.dot(inv(lat_vecs), pg), lat_vecs) for pg in pointgroup]



    return None
    # if coord == "cart":
    #     for i in range(1, len(gp_orbitals.keys()) + 1):
    #         for j in range(len(gp_orbitals[i])):
    #             gp_orbitals[i][j] = np.dot(lat_vecs, gp_orbitals[i][j])
    #     return gp_orbitals# , pointgroup
    # elif coord == "lat":
    #     return gp_orbitals
    # else:
    #     raise ValueError("Coordinate options are 'cell' and 'lat'.")

def find_full_orbitals(grid_car, lat_vecs, coord = "cart"):
    """ Find the complete orbitals of the points in a grid, including points
    not contained in the grid.

    Args:
        grid_car (list): a list of grid point positions in cartesian coordinates.
        lat_vecs (numpy.ndarray): the vectors that define the integration cell

    Returns:
        gp_orbitals (dict): the orbitals of the grid points in a dictionary. 
            The keys of the dictionary are integer labels and the values are the
            grid points in the orbital.
    """

    grid_lat = [np.dot(np.linalg.inv(lat_vecs), gp) for gp in grid_car]
    gp_orbitals = {}
    nirr_kpts = 0
    grid_copy = deepcopy(grid_lat)
    pointgroup = point_group(lat_vecs)
    pointgroup = [np.dot(np.dot(inv(lat_vecs), pg), lat_vecs) for pg in pointgroup]    
    while grid_copy != []:
        # Grap a point and build its orbit but only include points from the grid.
        gp = grid_copy.pop()
        nirr_kpts += 1
        gp_orbitals[nirr_kpts] = []
        for pg in pointgroup:
            # If the group operation moves the point outside the cell, %1 moves
            # it back in.
            # I ran into floating point precision problems the last time I ran
            # %1. Just to be safe it's included here.
            new_gp = np.round(np.dot(pg, gp), 15)%1
            if any([np.allclose(new_gp, gc) for gc in grid_copy]):
                ind = np.where(np.array([np.allclose(new_gp, gc) for gc in grid_copy]) == True)[0][0]
                del grid_copy[ind]
                gp_orbitals[nirr_kpts].append(new_gp)
            else:
                gp_orbitals[nirr_kpts].append(new_gp)                
                continue

    if coord == "cart":
        for i in range(1, len(gp_orbitals.keys()) + 1):
            for j in range(len(gp_orbitals[i])):
                gp_orbitals[i][j] = np.dot(lat_vecs, gp_orbitals[i][j])
        return gp_orbitals
    elif coord == "cell":
        return gp_orbitals
    else:
        raise ValueError("Coordinate options are 'cell' and 'lat'.")

def find_lattice_type(centering_type, lattice_constants, lattice_angles):
    """Find the Bravais lattice type of the lattice.

    Args:
        centering_type (str): how points are centered in the conventional
            unit cell of the lattice. Options include 'prim', 'base', 'body',
            and 'face'.
        lattice_constants (list or numpy.ndarray): the axial lengths of the
            conventional lattice vectors.
        lattice_angles (list or numpy.ndarray): the interaxial angles of the
            conventional lattice vectors.

    Returns:
        (str): the Bravais lattice type.
    Example:
        >>> centering_type = "prim
        >>> lattice_constants = [1]*3
        >>> lattice_angles = [numpy.pi/2]*3
        >>> lattice_type = find_lattice_type(centering_type,
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

    # Lattices with all angles = pi/2.
    if (np.isclose(alpha, beta) and np.isclose(beta, gamma) and
        np.isclose(gamma, np.pi/2)):
        # Check if it is a cubic lattice.
        if (np.isclose(a,b) and np.isclose(b,c)):
            if centering_type == "body":
                return "body-centered cubic"
            elif centering_type == "prim":
                return "simple cubic"
            elif centering_type == "face":
                return "face-centered cubic"
            else:
                msg = ("Valid centering types for cubic lattices include "
                       "'prim', 'body', and 'face'.")
                raise ValueError(msg.format(centering_type))
                
        # Check if it is tetragonal.
        elif (np.isclose(a,b) and not np.isclose(b,c)):
            if centering_type == "prim":
                return "tetragonal"
            elif centering_type == "body":
                return "body-centered tetragonal"
            else:
                msg = ("Valid centering types for tetragonal lattices include "
                       "'prim' and 'body'.")
                raise ValueError(msg.format(centering_type))
            
        # Check if it is orthorhombic
        elif not (np.isclose(a,b) and np.isclose(b,c) and np.isclose(a,c)):
            if centering_type == "body":
                return "body-centered orthorhombic"
            elif centering_type == "prim":
                return "orthorhombic"
            elif centering_type == "face":
                return "face-centered orthorhombic"
            elif centering_type == "base":
                return "base-centered orthorhombic"
            else:
                msg = ("Valid centering types for orthorhombic lattices include "
                       "'prim', 'base', 'body', and 'face'.")
                raise ValueError(msg.format(centering_type))
        else:
            msg = ("The lattice constants provided do not appear to correspond "
                   "to a Bravais lattice. They almost represent a cubic, "
                   "tetragonal, or orthorhombic lattice.")
            raise ValueError(msg.format(lattice_constants))
        
    # Check if it is rhombohedral.
    elif (np.isclose(alpha, beta) and np.isclose(beta, gamma)):
        if (np.isclose(a, b) and np.isclose(b,c)):
            if centering_type == "prim":
                return "rhombohedral"
            else:
                msg = ("The only valid centering type for rhombohedral lattices "
                       "is 'prim'.")
                raise ValueError(msg.format(centering_type))
        else:
            msg = ("None of the lattice constants should have the same value "
                   "for a rhombohedral lattice")
            raise ValueError(msg.format(lattice_constants))
        
    # Check if it is hexagonal.
    elif (np.isclose(alpha, beta) and np.isclose(beta, np.pi/2) and
          np.isclose(gamma, 2*np.pi/3)):
          if (np.isclose(a, b) and not np.isclose(b, c)):
            if centering_type == "prim":
                return "hexagonal"
            else:
                msg = ("The only valid centering type for hexagonal lattices "
                       "is 'prim'.")
                raise ValueError(msg.format(centering_type))
          else:
              msg = ("For a hexagonal lattice, a = b != c.")
              raise ValueError(msg.format(lattice_constants))
          
    # Check if it is monoclinic
    # Monoclinic a,b <= c, alpha < pi/2, beta = gamma = pi/2, a != b != c
    elif (np.isclose(beta, gamma) and np.isclose(beta, np.pi/2) and
          (alpha < np.pi/2)):
        if ((a < c or np.isclose(a, c)) and (b < c or np.isclose(b,c))):
            if centering_type == "prim":
                return "monoclinic"
            elif centering_type == "base":
                return "base-centered monoclinic"
            else:
                msg = ("Valid centering types for monoclinic lattices include "
                       "'prim' and 'base'.")
                raise ValueError(msg.format(centering_type))
        else:
            msg = ("The lattice constants of a monoclinic lattice should be "
                   "arranged such that a, b <= c.")
            raise ValueError(msg.format(lattice_constants))
            
    # Check if the lattice is triclinic.
    elif not (np.isclose(alpha, beta) and np.isclose(beta, gamma) and
          np.isclose(alpha, gamma)):
        if not (np.isclose(a, b) and np.isclose(b, c) and np.isclose(a, c)):
            if centering_type == "prim":
                return "triclinic"
            else:
                msg = ("The onld valid centering type for triclinic "
                       "lattices is 'prim'.")
                raise ValueError(msg.format(centering_type))
        else:
            msg = ("None of the lattice constants are equivalent for a "
                   "triclinic lattice.")
            raise ValueError(msg.format(lattice_constants))
    else:
        msg = ("The lattice angles provided do not correspond to any Bravais "
               "lattice type.")
        raise ValueError(msg.format(lattice_angles))

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

def HermiteNormalForm(S, eps=10):
    """Find the Hermite normal form (HNF) of a given integer matrix and the
    matrix that mediates the transformation.

    Args:
        S (numpy.ndarray): The 3x3 integer matrix describing the relationship 
            between two commensurate lattices.
        eps (int): finite precision parameter that determines number of decimals
            kept when rounding.
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
    H = np.round(H, eps).astype(int)
    return H, B

def UpperHermiteNormalForm(S):
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


def find_kpt_index(kpt, invK, L, D, eps=9):
    """This function takes a k-point in Cartesian coordinates and "hashes" it 
    into a single number, corresponding to its place in the k-point list.

    Args:
    kpt (list or numpy.ndarray): the k-point in Cartesian coordinates
    invK(list or numpy.ndarray): the inverse of the k-point grid generating
        vectors
    L (list or numpy.ndarray): the left transform for the SNF conversion
    D (list or numpy.ndarray): the diagonal of the SNF
    eps (float): a finite-precision parameter

    Returns:
    """

    gpt = np.dot(invK, kpt)
    gpt = np.dot(L, np.round(gpt).astype(int))%D
    
    # Convert from group coordinates to a singe base-10 number between 1 and
    # the number of k-points in the unreduced grid.
    return gpt[0]*D[1]*D[2] + gpt[1]*D[2] + gpt[2]


def bring_into_cell(kpt, lat_vecs, eps=10):
    """Bring a k-point into the first unit cell.
    """    
    k = np.round(np.dot(inv(lat_vecs), kpt), eps)%1
    return np.dot(lat_vecs, k)

def reduce_kpoint_list(kpoint_list, lattice_vectors, grid_vectors, shift,
                       eps=9):
    """Use the point group symmetry of the lattice vectors to reduce a list of
    k-points.
    
    Args:
        kpoint_list (list or numpy.ndarray): a list of k-point positions in
            Cartesian coordinates.
        lattice_vectors (list or numpy.ndarray): the vectors that generate the
            reciprocal lattice in a 3x3 array with the vectors as columns.
        grid_vectors (list or numpy.ndarray): the vectors that generate the
            k-point grid in a 3x3 array with the vectors as columns in 
            Cartesian coordinates.
        shift (list or numpy.ndarray): the offset of the k-point grid in grid
            coordinates.

    Returns:
        reduced_kpoints (list): an ordered list of irreducible k-points
        kpoint_weights (list): an ordered list of irreducible k-point weights.
    """
    
    try:
        inv(lattice_vectors)
    except np.linalg.linalg.LinAlgError:
        msg = "The lattice generating vectors are linearly dependent."
        raise ValueError(msg.format(lattice_vectors))
    
    try:
        inv(grid_vectors)
    except np.linalg.linalg.LinAlgError:
        msg = "The grid generating vectors are linearly dependent."
        raise ValueError(msg.format(lattice_vectors))
        
    if abs(det(lattice_vectors)) < abs(det(grid_vectors)):
        msg = """The k-point generating vectors define a grid with a unit cell 
        larger than the reciprocal lattice unit cell."""
        raise ValueError(msg.format(grid_vectors))

    # Put the shift in Cartesian coordinates.
    shift = np.dot(grid_vectors, shift)

    # Integer matrix
    N = np.dot(inv(grid_vectors), lattice_vectors)
    # Check that N is an integer matrix. In other words, verify that the grid
    # and lattice are commensurate.
    for i in range(len(N[:,0])):
        for j in range(len(N[0,:])):
            if np.isclose(N[i,j]%1, 0) or np.isclose(N[i,j]%1, 1):
                N[i,j] = int(np.round(N[i,j]))
            else:
                msg = "The lattice and grid vectors are incommensurate."
                raise ValueError(msg.format(grid_vectors))

    # Find the HNF of N. B is the transformation matrix (BN = H).
    H,B = HermiteNormalForm(N)
    H = [list(H[i]) for i in range(3)]
    # Find the SNF of H. L and R are the left and right transformation matrices
    # (LHR = S).
    S,L,R = SmithNormalForm(H)

    # Get the diagonal of SNF.
    D = np.round(np.diag(S), eps).astype(int)
    
    cOrbit = 0 # unique orbit counter
    pointgroup = point_group(lattice_vectors) # a list of point group operators
    nSymOps = len(pointgroup) # the number of symmetry operations
    nUR = len(kpoint_list) # the number of unreduced k-points
    
    # A dictionary to keep track of the number of symmetrically-equivalent
    # k-points.
    hashtable = dict.fromkeys(range(nUR))

    # A dictionary to keep track of the k-points that represent each orbital.
    iFirst = {}

    # A dictionary to keep track of the number of symmetrically-equivalent
    # k-points in each orbit.
    iWt = {}
    invK = inv(grid_vectors)

    # Loop over unreduced k-points.
    for i in range(nUR):
        ur_kpt = kpoint_list[i]
        idx = find_kpt_index(ur_kpt - shift, invK, L, D, eps)
        if hashtable[idx] != None:
            continue
        cOrbit += 1        
        hashtable[idx] = cOrbit
        iFirst[cOrbit] = i
        iWt[cOrbit] = 1
        for pg in pointgroup:
            # Rotate the k-point.
            rot_kpt = np.dot(pg, ur_kpt)
            # Bring it back into the first unit cell.
            rot_kpt = bring_into_cell(rot_kpt, lattice_vectors)
            if not np.allclose(np.dot(invK, rot_kpt-shift),
                               np.round(np.dot(invK, rot_kpt-shift))):
                continue
            idx = find_kpt_index(rot_kpt - shift, invK, L, D, eps)
            if hashtable[idx] == None:
                hashtable[idx] = cOrbit
                iWt[cOrbit] += 1
    sum = 0
    kpoint_weights = list(iWt.values())[:cOrbit]
    reduced_kpoints = [[0,0,0] for _ in range(cOrbit)]
    for i in range(cOrbit):
        sum += kpoint_weights[i]
        reduced_kpoints[i] = kpoint_list[iFirst[i+1]]

    return reduced_kpoints, kpoint_weights

def find_orbits(kpoint_list, lattice_vectors, grid_vectors, shift,
                eps=9):
    """Use the point group symmetry of the lattice vectors to reduce a list of
    k-points.
    
    Args:
        kpoint_list (list or numpy.ndarray): a list of k-point positions in
            Cartesian coordinates.
        lattice_vectors (list or numpy.ndarray): the vectors that generate the
            reciprocal lattice in a 3x3 array with the vectors as columns.
        grid_vectors (list or numpy.ndarray): the vectors that generate the
            k-point grid in a 3x3 array with the vectors as columns in 
            Cartesian coordinates.
        shift (list or numpy.ndarray): the offset of the k-point grid in grid
            coordinates.

    Returns:
        reduced_kpoints (list): an ordered list of irreducible k-points
        kpoint_weights (list): an ordered list of irreducible k-point weights.
    """
    
    try:
        inv(lattice_vectors)
    except np.linalg.linalg.LinAlgError:
        msg = "The lattice generating vectors are linearly dependent."
        raise ValueError(msg.format(lattice_vectors))
    
    try:
        inv(grid_vectors)
    except np.linalg.linalg.LinAlgError:
        msg = "The grid generating vectors are linearly dependent."
        raise ValueError(msg.format(lattice_vectors))
        
    if abs(det(lattice_vectors)) < abs(det(grid_vectors)):
        msg = """The k-point generating vectors define a grid with a unit cell 
        larger than the reciprocal lattice unit cell."""
        raise ValueError(msg.format(grid_vectors))

    # Put the shift in Cartesian coordinates.
    shift = np.dot(grid_vectors, shift)

    # Integer matrix
    N = np.dot(inv(grid_vectors), lattice_vectors)
    # Check that N is an integer matrix. In other words, verify that the grid
    # and lattice are commensurate.
    for i in range(len(N[:,0])):
        for j in range(len(N[0,:])):
            if np.isclose(N[i,j]%1, 0) or np.isclose(N[i,j]%1, 1):
                N[i,j] = int(np.round(N[i,j]))
            else:
                msg = "The lattice and grid vectors are incommensurate."
                raise ValueError(msg.format(grid_vectors))

    # Find the HNF of N. B is the transformation matrix (BN = H).
    H,B = HermiteNormalForm(N)
    H = [list(H[i]) for i in range(3)]
    # Find the SNF of H. L and R are the left and right transformation matrices
    # (LHR = S).
    S,L,R = SmithNormalForm(H)

    # Get the diagonal of SNF.
    D = np.round(np.diag(S), eps).astype(int)
    
    cOrbit = 0 # unique orbit counter
    pointgroup = point_group(lattice_vectors) # a list of point group operators
    nSymOps = len(pointgroup) # the number of symmetry operations
    nUR = len(kpoint_list) # the number of unreduced k-points
    # A dictionary to keep track of the number of symmetrically-equivalent
    # k-points. It goes from the k-point index in the grid to the label of that
    # k-point's orbit.
    hashtable = dict.fromkeys(range(nUR))

    # A dictionary to keep track of the k-points that represent each orbital.
    # It goes from orbit label to the index of the k-point that represents
    # the orbit.
    iFirst = {}

    # A dictionary to keep track of the number of symmetrically-equivalent
    # k-points in each orbit.
    iWt = {}
    invK = inv(grid_vectors)

    # Loop over unreduced k-points.
    for i in range(nUR):
        ur_kpt = kpoint_list[i]
        idx = find_kpt_index(ur_kpt - shift, invK, L, D, eps)
        if hashtable[idx] != None:
            continue
        cOrbit += 1
        hashtable[idx] = cOrbit
        iFirst[cOrbit] = i
        iWt[cOrbit] = 1
        for pg in pointgroup:
            # Rotate the k-point.
            rot_kpt = np.dot(pg, ur_kpt)
            # Bring it back into the first unit cell.
            rot_kpt = bring_into_cell(rot_kpt, lattice_vectors)
            if not np.allclose(np.dot(invK, rot_kpt-shift),
                               np.round(np.dot(invK, rot_kpt-shift))):
                continue
            idx = find_kpt_index(rot_kpt - shift, invK, L, D, eps)
            if hashtable[idx] == None:
                hashtable[idx] = cOrbit
                iWt[cOrbit] += 1

    hashtable = dict((k, v) for k, v in hashtable.items() if v)
    return hashtable, iFirst

def minkowski_reduce_basis(basis, eps):
    """Find the Minkowski representation of a basis.

    Args:
        basis(numpy.ndarray): a matrix with the generating vectors as columns.
        eps (int): a finite precision parameter in 10**(-eps).
    """
    return _minkowski_reduce_basis(basis.T, 10**(-eps)).T

def brillouin_zone_mapper(grid, rlattice_vectors, grid_vectors, shift, eps=15):
    """Map a grid into the first Brillouin zone in Minkowski space.
    
    Args:
        grid (list): a list of grid points in Cartesian coordinates.
        rlattice_vectors (numpy.ndarray): a matrix whose columes are the
            reciprocal lattice generating vectors.
        grid_vectors (numpy.ndarray): the grid generating vectors.
        eps (int): finite precision parameter that is the integer exponent in
            10**(-eps).
    Returns:
        mgrid (numpy.ndarray): a numpy array of grid points in the first 
            Brillouin zone in Minkowski space.
        weights (numpy.ndarray): the k-point weights
    """

    # Reduce the grid and move into the unit cell.
    reduced_grid, weights = reduce_kpoint_list(grid, rlattice_vectors, grid_vectors,
                                      shift, eps)
    
    reduced_grid = np.array(reduced_grid)
    print("reduced_grid\n", np.round(reduced_grid, 3))
    
    # Find the Minkowski basis.
    mink_basis = minkowski_reduce_basis(rlattice_vectors, eps)
    print("mink basis\n", mink_basis)
    
    # Find the transformation matrix that gets us to Minkowski space.
    M = np.dot(mink_basis, inv(rlattice_vectors))
    print("M\n", np.round(M, 3))

    # Move the reduced grid points into Minkowski space.
    print(reduced_grid[:,0])
    print(np.dot(M, reduced_grid[:,0]))
    mreduced_grid = np.dot(M, reduced_grid.T).T

    print("transformed grid\n", mreduced_grid)

    for i, pt1 in enumerate(mreduced_grid):
        norm_pt1 = np.dot(pt1, pt1)
        for n in product([-1,0,1], repeat=3):
            pt2 = pt1 + np.dot(mink_basis, n)
            norm_pt2 = np.dot(pt2, pt2)
            if (norm_pt2 + 10**(-eps)) < norm_pt1:
                norm_pt1 = norm_pt2
                mreduced_grid[i] = pt2

    

    return mreduced_grid, weights
