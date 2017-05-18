"""Unit tests for symmetry.py module.
"""

import pytest
import numpy as np
from itertools import product
from itertools import combinations

# from BZI.symmetry import rsym_pts, make_ptvecs, make_rptvecs
# from BZI.symmetry import fcc_sympts, bcc_sympts, sc_sympts
from BZI.symmetry import *

def test_rsym_pts():
    """Verify all the keys return the appropriate values and errors are raised
    for ill-suited ones."""
    
    ndims = 3
    # Raise an error for ill-defined inputs.
    lat_types = ["fcc", "bcc", "sc"]
    a_s = ["a", [1,2,3], {3:2, 2:3}, (8,3)]
    
    for lat_type in lat_types:
        for a in a_s:
            with pytest.raises(ValueError):
                rsym_pts(lat_type, a)
                
    # Verify the reciprocal lattice points are correct by finding them
    # another way.
    lat_const = 2*np.pi

    fccvs = ["G", "X", "L", "W", "U", "K"]
    bccvs = ["G", "H", "P", "N"]
    scvs = ["G", "R", "X", "M"]
    
    pnames = {"fcc": fccvs, "bcc": bccvs, "sc": scvs}
    sym_pt_dict = {"fcc": fcc_sympts, "bcc": bcc_sympts, "sc":sc_sympts}
    
    for lat_type in lat_types:
        print(lat_type)
        lat_vecs = make_ptvecs(lat_type, lat_const)
        rlat_vecs = make_rptvecs(lat_type, lat_const)
        sym_dict = sym_pt_dict[lat_type]
        print("sym_dict \n", sym_dict)
    
        for sym_pt in pnames[lat_type]: # pnames = symmetry point names
            print("symbol ", sym_pt)
            # symmetry point in lattice coordinates
            pt = sym_dict[sym_pt] 
            # symmetry point in Cartesian coordinates
            rpt = rsym_pts(lat_type, lat_const)[sym_pt]
            
            rpt_new = np.array([0,0,0], dtype=float)
            for i in range(ndims):
                rpt_new += rlat_vecs[:,i]*pt[i]
            print("rpt ", rpt)
            print("rpt_new ", rpt_new)
            assert np.allclose(rpt, rpt_new) == True
            
def test_make_ptvecs():
    """Verify the primitive translation vectors are correct.
    """

    lattice_types = ["sc", "bcc", "fcc"]
    lattice_constants = [1.2, np.pi, 3.3457, 2]

    # Test that an error is raised for invalid lattice types.
    with pytest.raises(ValueError):
        make_ptvecs("lat_type", 1)
    with pytest.raises(ValueError):
        make_ptvecs(3.2, 1)
        
    
    for lattice_type in lattice_types:
        for lattice_constant in lattice_constants:
            ptvs = make_ptvecs(lattice_type, lattice_constant)

            # Extract the primitive translation vectors.
            a1 =  ptvs[:,0]
            a2 =  ptvs[:,1]
            a3 =  ptvs[:,2]
                
            if lattice_type == "sc":
                # Verify the vectors have the correct length.                
                assert np.isclose(np.linalg.norm(a1), lattice_constant) == True
                assert np.isclose(np.linalg.norm(a2), lattice_constant) == True
                assert np.isclose(np.linalg.norm(a3), lattice_constant) == True

                # Check that the vectors are orthogonal.
                assert np.isclose(np.dot(a1, a2), 0) == True
                assert np.isclose(np.dot(a2, a3), 0) == True
                assert np.isclose(np.dot(a3, a1), 0) == True
                
                # Check that the cell defined by these vectors has the correct
                # volume.
                assert np.isclose(abs(np.dot(np.cross(a1, a2), a3)),
                                  np.abs(np.linalg.det(ptvs))) == True
                
            elif lattice_type == "bcc":
                # Verify the vectors have the correct length.                
                assert np.isclose(np.linalg.norm(a1),
                                  lattice_constant*np.sqrt(3)/2) == True
                assert np.isclose(np.linalg.norm(a2),
                                  lattice_constant*np.sqrt(3)/2) == True
                assert np.isclose(np.linalg.norm(a3),
                                  lattice_constant*np.sqrt(3)/2) == True


                # Check that the vector dot product is correct.
                assert np.isclose(np.dot(a1, a2), -lattice_constant**2/4) == True
                assert np.isclose(np.dot(a2, a3), -lattice_constant**2/4) == True
                assert np.isclose(np.dot(a3, a1), -lattice_constant**2/4) == True
                
                # Check that the cell defined by these vectors has the correct
                # volume.
                assert np.isclose(abs(np.dot(np.cross(a1, a2), a3)),
                                  np.abs(np.linalg.det(ptvs))) == True
                
            elif lattice_type == "fcc":
                # Verify the vectors have the correct length.                
                assert np.isclose(np.linalg.norm(a1),
                                  lattice_constant*np.sqrt(2)/2) == True
                assert np.isclose(np.linalg.norm(a2),
                                  lattice_constant*np.sqrt(2)/2) == True
                assert np.isclose(np.linalg.norm(a3),
                                  lattice_constant*np.sqrt(2)/2) == True

                # Check that the vectors are orthogonal.
                assert np.isclose(np.dot(a1, a2), lattice_constant**2/4) == True
                assert np.isclose(np.dot(a2, a3), lattice_constant**2/4) == True
                assert np.isclose(np.dot(a3, a1), lattice_constant**2/4) == True
                
                # Check that the cell defined by these vectors has the correct
                # volume.
                assert np.isclose(abs(np.dot(np.cross(a1, a2), a3)),
                                  np.abs(np.linalg.det(ptvs))) == True

            else:
                msg = "Please provide a valid lattice type"
                raise ValueError(msg.format(lattice_type))

def test_sym_path():
    """Veryify the vector created from two points along a symmetry path all
    point in the same direction.
    """

    lat_types = ["fcc", "bcc", "sc"]
    fccvs = ["G", "X", "L", "W", "U", "K"]
    bccvs = ["G", "H", "P", "N"]
    scvs = ["G", "R", "X", "M"]

    lat_const = 1
    npts = 100
    
    pnames = {"fcc": fccvs, "bcc": bccvs, "sc": scvs}
    sym_pt_dict = {"fcc": fcc_sympts, "bcc": bcc_sympts, "sc":sc_sympts}    
    
    for lat_type in lat_types:
        lat_vecs = make_rptvecs(lat_type, lat_const)
        rlat_vecs = make_rptvecs(lat_type, lat_const)
        sym_dict = sym_pt_dict[lat_type]

        for sym_pt1, sym_pt2 in combinations(pnames[lat_type], 2):
            spath = sym_path(lat_type, npts, [[sym_pt1,sym_pt2]])
            for i in range(len(spath)-2):
                # Make sure the distance between all points in the path is the same.
                assert np.allclose(np.linalg.norm(np.array(spath[i+2]) - np.array(spath[i+1])),
                                   np.linalg.norm(np.array(spath[i+1]) - np.array(spath[i])))
