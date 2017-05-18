"""Unit tests for pseudopots.py module.
"""

import pytest
import numpy as np
from BZI.pseudopots import find_intvecs, customSi_PP, Si_lat_const
from BZI.symmetry import sym_path, make_rptvecs

def test_find_intvecs():
    
    a_list = [3,4,8,11]
    test3 = [ [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1],
                         [-1,1,-1], [-1,-1,1], [-1,-1,-1] ]

    test4 =[ [0,0,2], [0,0,-2], [0,2,0], [0,-2,0], [2,0,0], [-2,0,0] ]
    
    test8 = [ [0,2,2], [0,-2,2], [0,2,-2], [0,-2,-2], [2,0,2], [-2,0,2],
              [2,0,-2], [-2,0,-2], [2,2,0], [-2,2,0], [2,-2,0], [-2,-2,0] ]

    test11 = [ [3,1,1], [-3,1,1], [3,-1,1], [-3,-1,1], [3,1,-1], [-3,1,-1],
               [3,-1,-1], [-3,-1,-1], [1,3,1], [-1,3,1], [1,-3,1], [-1,-3,1],
               [1,3,-1], [-1,3,-1], [1,-3,-1], [-1,-3,-1], [1,1,3], [-1,1,3],
               [1,-1,3], [-1,-1,3], [1,1,-3], [-1,1,-3], [1,-1,-3], [-1,-1,-3] ]
    tests = [test3, test4, test8, test11]

    for i, a in enumerate(a_list):
        intvecs = find_intvecs(a)
        assert len(intvecs) == len(tests[i])        
        for ivec in intvecs:
            contained = False
            for at in tests[i]:
                if np.allclose(ivec, at):
                    contained = True
            assert contained is True

def test_Si_PP():
    # Test that the Hamiltonian matrix is Hermitian along various symmetry
    # paths.
    rlat_pts = find_intvecs(0) + find_intvecs(3) + find_intvecs(8) + find_intvecs(11)
    rlat_pts = [np.array(r) for r in rlat_pts]
    sympt_pairs = [("L","G"),("G","X"),("X","K"),("K","G")]
    lat_type = "fcc"
    npts = 3

    # The k-points between symmetry points in reciprocal lattice coordinates.
    lat_kpoints = sym_path(lat_type,npts,sympt_pairs)
    rlat_vecs = make_rptvecs(lat_type, Si_lat_const)

    # The k-points between symmetry points in cartesian coordinates.
    car_kpoints = [np.dot(rlat_vecs,k) for k in lat_kpoints]
    t = Si_lat_const/8*np.array([1,1,1])
    for kpt in car_kpoints:
        # Verify the Hamiltonian matrix is Hermitian.
        H = customSi_PP(kpt, 10, 11*(2*np.pi/Si_lat_const)**2, shift=True, matrix=True)
        assert np.allclose(H, H.conj().T)
