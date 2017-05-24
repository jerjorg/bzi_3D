"""Unit tests for symmetry.py module
"""

import pytest
import numpy as np
from numpy.linalg import norm
from itertools import product
from itertools import combinations

from BZI.symmetry import *

def test_rsym_pts():
    """Verify all the keys return the appropriate values and errors are raised
    for ill-suited ones."""
    
    ndims = 3
    # Raise an error for ill-defined inputs.
    lat_angles = [np.pi/2]*3
    center_list = ["prim", "face", "body"]
                 
    # Verify the reciprocal lattice points are correct by finding them
    # another way.
    lat_consts = [1]*3
    lat_const = 1
    fccvs = ["G", "X", "L", "W", "U", "K"]
    bccvs = ["G", "H", "P", "N"]
    scvs = ["G", "R", "X", "M"]
    sym_pt_dict = {"fcc": 1, "bcc": 2, "sc": 2}
    pnames = {"fcc": fccvs, "bcc": bccvs, "sc": scvs}
    sym_pt_dict = {"fcc": fcc_sympts, "bcc": bcc_sympts, "sc":sc_sympts}
    
    for center_type, lat_type in zip(center_list, sym_pt_dict.keys()):
        lat_vecs = make_ptvecs(center_type, lat_consts, lat_angles)
        rlat_vecs = make_rptvecs(lat_vecs)
        sym_dict = sym_pt_dict[lat_type]
    
        for sym_pt in pnames[lat_type]: # pnames = symmetry point names
            # symmetry point in lattice coordinates
            pt = sym_dict[sym_pt]
            # symmetry point in Cartesian coordinates
            rpt = rsym_pts(lat_type, lat_const)[sym_pt]
            rpt_new = np.array([0,0,0], dtype=float)
            for j in range(ndims):
                rpt_new += rlat_vecs[:,j]*pt[j]
            assert np.allclose(rpt, rpt_new) == True
            
def test_make_ptvecs():
    """Verify the primitive translation vectors are correct.
    """
    center_list = ["prim", "base", "body", "face", "hex", "rhom"]
    consts_list = [[1]*3, [1.2]*3, [np.pi]*3, [1, 1, 2],
                   [1, 2, 3],[1.2, 1.5, 3], [np.pi, np.pi, 4],
                   [100,110.3,210.3]]
    angles_list = [[np.pi/2]*3, [1]*3,
                   [np.pi/3, np.pi/3, np.pi/4],
                   [2, 2, 1], [1, 2, 3],
                   [np.pi/2, np.pi/3, np.pi/3],
                   [np.pi/2, np.pi/2, np.pi/4]]

    for center_type in center_list:
        for lat_consts in consts_list:
            for lat_angles in angles_list:
                print(center_type)
                vecs = make_ptvecs(center_type,
                                   lat_consts,
                                   lat_angles)
                a = float(lat_consts[0])
                b = float(lat_consts[1])
                c = float(lat_consts[2])
                
                alpha = float(lat_angles[0])
                beta = float(lat_angles[1])
                gamma = float(lat_angles[2])
                
                v0 = vecs[:,0]
                v1 = vecs[:,1]
                v2 = vecs[:,2]
                
                # Primitive centering type
                if center_type == "prim":
                    # Verify the vectors have the correct lengths.
                    assert np.isclose(norm(v0), a)
                    assert np.isclose(norm(v1), b)
                    assert np.isclose(norm(v2), c)                                        
                    
                    # Verify the angles between the vectors are correct.
                    assert np.isclose(np.dot(v0,v1), a*b*np.cos(gamma))
                    assert np.isclose(np.dot(v1,v2), b*c*np.cos(alpha))
                    assert np.isclose(np.dot(v2,v0), a*c*np.cos(beta))
                # Base centering type
                elif center_type == "base":
                    # Verify the vectors have the correct lengths.
                    assert np.isclose(norm(v0),
                                      1./2*np.sqrt(a**2 + b**2 -
                                                  2*a*b*np.cos(gamma)))
                    assert np.isclose(norm(v1),
                                      1./2*np.sqrt(a**2 + b**2 +
                                                  2*a*b*np.cos(gamma)))
                    assert np.isclose(norm(v2), c)

                    # Verify the angles between the vectors are correct.
                    assert np.isclose(np.dot(v0, v1), 1./4*(a**2 - b**2))
                    assert np.isclose(np.dot(v1, v2), 1./2*a*c*np.cos(beta)
                                      + 1./2*b*c*np.cos(alpha))
                    assert np.isclose(np.dot(v0, v2), 1./2*a*c*np.cos(beta)
                                      - 1./2*b*c*np.cos(alpha))

                # Body centering type
                elif center_type == "body":
                    # Verify the vectors have the correct lengths.
                    assert np.isclose(norm(v0), 1./2*np.sqrt(
                        a**2 + b**2 + c**2
                        - 2*a*b*np.cos(gamma)
                        + 2*b*c*np.cos(alpha)
                        - 2*a*c*np.cos(beta)))                    
                    assert np.isclose(norm(v1), 1./2*np.sqrt(
                        a**2 + b**2 + c**2
                        - 2*a*b*np.cos(gamma)
                        - 2*b*c*np.cos(alpha)
                        + 2*a*c*np.cos(beta)))
                    assert np.isclose(norm(v2), 1./2*np.sqrt(
                        a**2 + b**2 + c**2
                        + 2*a*b*np.cos(gamma)
                        - 2*b*c*np.cos(alpha)
                        - 2*a*c*np.cos(beta)))
                    
                    # Verify the angles between the vectors are correct.
                    assert np.isclose(gamma, (np.arccos((4*np.dot(v0,v1)
                                              + a**2 + b**2 - c**2)/(2*a*b))))
                    assert np.isclose(beta, (np.arccos((4*np.dot(v0,v2)
                                              + a**2 - b**2 + c**2)/(2*a*c))))
                    assert np.isclose(alpha, (np.arccos((4*np.dot(v1,v2)
                                              - a**2 + b**2 + c**2)/(2*b*c))))
                # Face centering type
                elif center_type == "face":
                    # Verify the vectors have the correct lengths.
                    assert np.isclose(norm(v0), 1./2*np.sqrt(b**2 + c**2 +
                                                        2*b*c*np.cos(alpha)))
                    assert np.isclose(norm(v1), 1./2*np.sqrt(a**2 + c**2 +
                                                        2*a*c*np.cos(beta)))
                    assert np.isclose(norm(v2), 1./2*np.sqrt(a**2 + b**2 +
                                                        2*a*b*np.cos(gamma)))

                    # Verify the angles between the vectors are correct.
                    common = (a*b*np.cos(gamma) + a*c*np.cos(beta) +
                              b*c*np.cos(alpha))
                    assert np.isclose(np.dot(v0,v1), 1./4*(common + c**2))
                    assert np.isclose(np.dot(v0,v2), 1./4*(common + b**2))
                    assert np.isclose(np.dot(v1,v2), 1./4*(common + a**2))

                # Primitive centering type
                if center_type == "hex":
                    # Verify the vectors have the correct lengths.
                    assert np.isclose(norm(v0), a)
                    assert np.isclose(norm(v1), b)
                    assert np.isclose(norm(v2), c)                                        
                    
                    # Verify the angles between the vectors are correct.
                    assert np.isclose(np.dot(v0,v1), a*b*np.cos(gamma))
                    assert np.isclose(np.dot(v1,v2), b*c*np.cos(alpha))
                    assert np.isclose(np.dot(v2,v0), a*c*np.cos(beta))                    
                    
                # Rhombohedral centering type
                elif center_type == "rhom":
                    # Verify the vectors have the correct lengths.
                    assert np.isclose(norm(v0), a)
                    assert np.isclose(norm(v1), b)
                    assert np.isclose(norm(v2), c)                                        
                    
                    # Verify the angles between the vectors are correct.
                    assert np.isclose(np.dot(v0,v1), a*b*np.cos(gamma))
                    assert np.isclose(np.dot(v1,v2), b*c*np.cos(alpha))
                    assert np.isclose(np.dot(v2,v0), a*c*np.cos(beta))

def test_make_lattice_vectors():
    """Check that make_lattice_vectors agrees with what is obtained with
    make_ptvecs."""
    
    lat_type = "simple_cubic"
    lat_consts = [1]*3
    lat_angles = [np.pi/2]*3
    lat_centering = "prim"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "body_centered_cubic"
    lat_consts = [1]*3
    lat_angles = [np.pi/2]*3
    lat_centering = "body"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "face_centered_cubic"
    lat_consts = [1]*3
    lat_angles = [np.pi/2]*3
    lat_centering = "face"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "tetragonal"
    lat_consts = [1, 1, 2]
    lat_angles = [np.pi/2]*3
    lat_centering = "prim"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "body_centered_tetragonal"
    lat_consts = [1, 1, 2]
    lat_angles = [np.pi/2]*3
    lat_centering = "body"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "orthorhombic"
    lat_consts = [1, 2, 3]
    lat_angles = [np.pi/2]*3
    lat_centering = "prim"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "face_centered_orthorhombic"
    lat_consts = [1, 2, 3]
    lat_angles = [np.pi/2]*3
    lat_centering = "face"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "body_centered_orthorhombic"
    lat_consts = [1, 2, 3]
    lat_angles = [np.pi/2]*3
    lat_centering = "body"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "base_centered_orthorhombic"
    lat_consts = [1, 2, 3]
    lat_angles = [np.pi/2]*3
    lat_centering = "base"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "hexagonal"
    lat_consts = [1., 1., 3.]
    lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
    lat_centering = "hex"
    
    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)
    
    print(lat_vecs1)
    print(lat_vecs2)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "rhombohedral"
    lat_consts = [1., 1., 1.]
    lat_angles1 = [np.pi/3]*3
    lat_angles2 = [np.pi/3]*3
    lat_centering = "rhom"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles1)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles2)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "monoclinic"
    lat_consts = [1, 1, 2]
    lat_angles = [np.pi/3, np.pi/2, np.pi/2]
    lat_centering = "prim"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "base_centered_monoclinic"
    lat_consts = [1, 1, 3]
    lat_angles = [np.pi/4, np.pi/2, np.pi/2]
    lat_centering = "base"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)

    lat_type = "triclinic"
    lat_consts = [1, 2, 3]
    lat_angles = [np.pi/6, np.pi/4, np.pi/3]
    lat_centering = "prim"

    lat_vecs1 = make_ptvecs(lat_centering, lat_consts, lat_angles)
    lat_vecs2 = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    assert np.allclose(lat_vecs1, lat_vecs2)    

    # Verify that an error gets raised for poor input parameters.
    
    # Simple cubic
    lat_type = "simple_cubic"
    lat_consts = [1, 2, 3]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "simple_cubic"
    lat_consts = [1, 1, 1]
    lat_angles = [np.pi/2, np.pi/2, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "simple_cubic"
    lat_consts = [1, 1, 2]
    lat_angles = [np.pi/3, np.pi/2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    # Body-centered cubic

    lat_type = "body_centered_cubic"
    lat_consts = [1, 2, 1]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "body_centered_cubic"
    lat_consts = [1, 1, 1]
    lat_angles = [np.pi/2, np.pi/3, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "body_centered_cubic"
    lat_consts = [2, 1, 1]
    lat_angles = [np.pi/2, np.pi/3, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    # Face-centered cubic
    lat_type = "face_centered_cubic"
    lat_consts = [3.3, 1, 1]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "face_centered_cubic"
    lat_consts = [np.pi, np.pi, np.pi]
    lat_angles = [np.pi/2, np.pi/2, np.pi/5]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "face_centered_cubic"
    lat_consts = [np.pi, np.pi, np.pi]
    lat_angles = [np.pi/2, np.pi/5, np.pi/5]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    # Tetragonal

    lat_type = "tetragonal"
    lat_consts = [1, 1, 1]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "tetragonal"
    lat_consts = [1, 1, 3]
    lat_angles = [np.pi/2, np.pi/2, np.pi/8]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "tetragonal"
    lat_consts = [1, 1, 3]
    lat_angles = [np.pi/8, np.pi/8, np.pi/8]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    # Body-centered tetragonal

    lat_type = "body_centered_tetragonal"
    lat_consts = [np.pi/3, np.pi/3, np.pi/3]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "body_centered_tetragonal"
    lat_consts = [1, 1, 2]
    lat_angles = [np.pi/2, np.pi/3, np.pi/4]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "body_centered_tetragonal"
    lat_consts = [1.1, 1.1, 2.2]
    lat_angles = [np.pi/2, 1, 1]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    # Orthorhombic
        
    lat_type = "orthorhombic"
    lat_consts = [2, 2, 3]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "orthorhombic"
    lat_consts = [2.2, 2.2, 2.2]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "orthorhombic"
    lat_consts = [1, 2, 3]
    lat_angles = [np.pi/2, np.pi/2, np.pi/10]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "orthorhombic"
    lat_consts = [1, 2, 3]
    lat_angles = [np.pi/10, np.pi/10, np.pi/10]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "orthorhombic"
    lat_consts = [1, 2, 3]
    lat_angles = [np.pi/3, np.pi/2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    # Face-centered orthorhombic

    lat_type = "face_centered_orthorhombic"
    lat_consts = [1, 1, 3]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "face_centered_orthorhombic"
    lat_consts = [1, 1, 1]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "face_centered_orthorhombic"
    lat_consts = [1, 1, 1]
    lat_angles = [np.pi/2, np.pi/2, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "face_centered_orthorhombic"
    lat_consts = [1, 1, 1]
    lat_angles = [np.pi/2, np.pi/4, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "face_centered_orthorhombic"
    lat_consts = [1, 1, 1]
    lat_angles = [np.pi/5, np.pi/4, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    # Body-centered orthorhombic    
    lat_type = "body_centered_orthorhombic"
    lat_consts = [2.2, 2.2, 5.5]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "body_centered_orthorhombic"
    lat_consts = [2.2, 5.5, 5.5]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "body_centered_orthorhombic"
    lat_consts = [5.5, 5.5, 5.5]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
    
    lat_type = "body_centered_orthorhombic"
    lat_consts = [1.1, 1.2, 1.3]
    lat_angles = [np.pi/2, np.pi/2, np.pi/7]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "body_centered_orthorhombic"
    lat_consts = [1.1, 1.2, 1.3]
    lat_angles = [np.pi/2, np.pi/7, np.pi/7]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "body_centered_orthorhombic"
    lat_consts = [1.1, 1.2, 1.3]
    lat_angles = [np.pi/7, np.pi/7, np.pi/7]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    # Base-centered orthorhombic
    
    lat_type = "base_centered_orthorhombic"
    lat_consts = [1, 2, 2]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "base_centered_orthorhombic"
    lat_consts = [2, 2, 2]
    lat_angles = [np.pi/2]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "base_centered_orthorhombic"
    lat_consts = [2, 2, 2]
    lat_angles = [np.pi/2, np.pi/2, 1]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "base_centered_orthorhombic"
    lat_consts = [2, 2, 2]
    lat_angles = [np.pi/2, 1, 1]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "base_centered_orthorhombic"
    lat_consts = [2, 2, 2]
    lat_angles = [1, 1, 1]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    # Hexagonal
        
    lat_type = "hexagonal"
    lat_consts = [1., 1., 3.]
    lat_angles = [np.pi/2, np.pi/2, 2*np.pi/4]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "hexagonal"
    lat_consts = [1., 1., 3.]
    lat_angles = [np.pi/2, np.pi/3, 2*np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "hexagonal"
    lat_consts = [1., 1., 3.]
    lat_angles = [np.pi/3, np.pi/2, 2*np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "hexagonal"
    lat_consts = [1., 1., 3.]
    lat_angles = [np.pi/3, np.pi/3, 2*np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "hexagonal"
    lat_consts = [1., 2., 3.]
    lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "hexagonal"
    lat_consts = [1., 3., 3.]
    lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "hexagonal"
    lat_consts = [3., 2., 3.]
    lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "hexagonal"
    lat_consts = [3.1, 3.1, 3.1]
    lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    # Rhombohedral
    
    lat_type = "rhombohedral"
    lat_consts = [1., 1., 2.]
    lat_angles = [np.pi/3]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "rhombohedral"
    lat_consts = [1., 1.1, 1.]
    lat_angles = [np.pi/3]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "rhombohedral"
    lat_consts = [np.pi/3, 1., 1.]
    lat_angles = [np.pi/3]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "rhombohedral"
    lat_consts = [np.pi/3, np.pi/3, 1.]
    lat_angles = [np.pi/3]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "rhombohedral"
    lat_consts = [np.pi/3, np.pi/2, np.pi/3]
    lat_angles = [np.pi/3]*3
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "rhombohedral"
    lat_consts = [np.pi/3, np.pi/3, np.pi/3]
    lat_angles = [1, 1, 2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "rhombohedral"
    lat_consts = [np.pi/3, np.pi/3, np.pi/3]
    lat_angles = [1, 2, 2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "rhombohedral"
    lat_consts = [np.pi/3, np.pi/3, np.pi/3]
    lat_angles = [2, 1, 2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    # Monoclinic

    lat_type = "monoclinic"
    lat_consts = [1, 3, 2]
    lat_angles = [np.pi/3, np.pi/2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "monoclinic"
    lat_consts = [3, 1, 2]
    lat_angles = [np.pi/3, np.pi/2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "monoclinic"
    lat_consts = [1, 2.00001, 2]
    lat_angles = [np.pi/3, np.pi/2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "monoclinic"
    lat_consts = [1, 1, 2]
    lat_angles = [np.pi/3, np.pi/2, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "monoclinic"
    lat_consts = [1, 1, 2]
    lat_angles = [np.pi/3, np.pi/3, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "monoclinic"
    lat_consts = [1, 1, 2]
    lat_angles = [np.pi, np.pi/3, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "monoclinic"
    lat_consts = [1, 1, 2]
    lat_angles = [np.pi, np.pi, np.pi]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "monoclinic"
    lat_consts = [1, 1, 2]
    lat_angles = [np.pi, np.pi/2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    # Base-centered monoclinic

        
    lat_type = "base_centered_monoclinic"
    lat_consts = [1, 3.1, 3]
    lat_angles = [np.pi/4, np.pi/2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "base_centered_monoclinic"
    lat_consts = [3+1e-6, 3, 3]
    lat_angles = [np.pi/4, np.pi/2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "base_centered_monoclinic"
    lat_consts = [3+1e-6, 3+1e-6, 3]
    lat_angles = [np.pi/4, np.pi/2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "base_centered_monoclinic"
    lat_consts = [1, 1, 3]
    lat_angles = [np.pi, np.pi/2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
    
    lat_type = "base_centered_monoclinic"
    lat_consts = [3, 3, 3]
    lat_angles = [np.pi, np.pi, np.pi]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "base_centered_monoclinic"
    lat_consts = [3, 2, 3]
    lat_angles = [np.pi/3, np.pi, np.pi]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "base_centered_monoclinic"
    lat_consts = [2, 2, 3]
    lat_angles = [np.pi/3, np.pi/2+1e-2, np.pi/2]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    # Triclinic
    
    lat_type = "triclinic"
    lat_consts = [1, 2, 2+1e-14]
    lat_angles = [np.pi/6, np.pi/4, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "triclinic"
    lat_consts = [1, 2, 2]
    lat_angles = [np.pi/6, np.pi/4, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "triclinic"
    lat_consts = [2, 2, 2]
    lat_angles = [np.pi/6, np.pi/4, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "triclinic"
    lat_consts = [8, 2, 2]
    lat_angles = [np.pi/6, np.pi/4, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "triclinic"
    lat_consts = [8, 2, 2]
    lat_angles = [np.pi/4, np.pi/4, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
    lat_type = "triclinic"
    lat_consts = [8, 2, 2]
    lat_angles = [np.pi/3, np.pi/4, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
    
    lat_type = "triclinic"
    lat_consts = [8, 2, 2]
    lat_angles = [np.pi/3, np.pi/3, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)

    lat_type = "triclinic"
    lat_consts = [8.1, 8.1, 8.1]
    lat_angles = [np.pi/3, np.pi/3, np.pi/3]
    with pytest.raises(ValueError) as error:
        lat_vecs = make_lattice_vectors(lat_type, lat_consts, lat_angles)
        
def test_sym_path():
    """Veryify the vector created from two points along a symmetry path all
    point in the same direction.
    """
    
    lat_types = ["fcc", "bcc", "sc"]
    fccvs = ["G", "X", "L", "W", "U", "K"]
    bccvs = ["G", "H", "P", "N"]
    scvs = ["G", "R", "X", "M"]

    lat_const = 1
    lat_consts = [lat_const]*3
    lat_angles = [np.pi/2]*3
    npts = 100
    
    pnames = {"fcc": fccvs, "bcc": bccvs, "sc": scvs}
    sym_pt_dict = {"fcc": fcc_sympts, "bcc": bcc_sympts, "sc":sc_sympts}    
    center_list = ["face", "body", "prim"]
    
    for i in range(len(center_list)):
        center_type = center_list[i]
        lat_vecs = make_ptvecs(center_type, lat_consts, lat_angles)
        rlat_vecs = make_rptvecs(lat_vecs)
        lat_type = sym_pt_dict.keys()[i]
        sym_dict = sym_pt_dict[lat_type]
        
        for sym_pt1, sym_pt2 in combinations(pnames[lat_type], 2):
            spath = sym_path(lat_type, npts, [[sym_pt1,sym_pt2]])
            for i in range(len(spath)-2):
                # Make sure the distance between all points in the path is the same.
                assert np.allclose(np.linalg.norm(np.array(spath[i+2]) - np.array(spath[i+1])),
                                   np.linalg.norm(np.array(spath[i+1]) - np.array(spath[i])))
                      
