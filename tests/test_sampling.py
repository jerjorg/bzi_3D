"""Test the Brillouin zone sampling methods.
"""
import os
import pytest
import numpy as np
from itertools import product
import csv

from BZI.sampling import (make_grid, get_minmax_indices,
                          swap_column, HermiteNormalForm, swap_row,
                          LowerHermiteNormalForm, make_large_grid,
                          sphere_pts, large_sphere_pts)


from BZI.symmetry import make_ptvecs, make_rptvecs
from BZI.plots import PlotMesh

def test_make_grid():
    """Verify the mesh satisfies various properties, such as verifying
    the neighbors of each point are withing the mesh as long as the 
    neighbors lie within the unit cell. Also verify none of the points
    lie outside the unit cell.
    """
    
    mesh_types = ["sc", "fcc", "bcc"]
    mesh_constants = [1, 1.1, 12./11, .7]
    # For the time being, the code only works for simple cubic cells.
    cell_types = ["sc", "fcc", "bcc"]
    cell_constants = [2*np.sqrt(2)] 
    offsets = [[0., 0., 0.], [1./2, 1./2, 1./2]]
    
    for mesh_constant in mesh_constants:
        for mesh_type in mesh_types:
            mesh_vectors = make_ptvecs(mesh_type, mesh_constant)
            mesh_lengths = [np.linalg.norm(lv) for lv in mesh_vectors]
            for cell_type in cell_types:
                for cell_constant in cell_constants:
                    cell_vectors = make_ptvecs(cell_type,
                                               cell_constant)
                    cell_lengths = [np.linalg.norm(cv) for cv in
                                        cell_vectors]
                    for offset in offsets:
                        mesh = make_grid(cell_vectors, mesh_vectors, offset)
                        large_mesh = make_large_mesh(cell_vectors, mesh_type,
                                                  mesh_vectors, offset)
                        
                        # Verify all the points in the cell for the large mesh
                        # are contained in mesh.
                        for lg in large_mesh[0]:
                            included = False
                            for g in mesh:
                                if np.allclose(lg,g) == True:
                                    included = True
                            assert included == True

def test_get_minmax_indices():
    """Various tests taxen from symlib."""
    ntests =50
    
    vecs = []
    mins = []
    maxs = []
    for n in range(1,ntests+1):
        vec_filename = os.path.join(os.path.dirname(__file__), "sampling_testfiles/get_minmax_indices_invec.in.%s" %n)
        max_filename = os.path.join(os.path.dirname(__file__), "./sampling_testfiles/get_minmax_indices_max.out.%s" %n)
        min_filename = os.path.join(os.path.dirname(__file__), "./sampling_testfiles/get_minmax_indices_min.out.%s" %n)

        print("file ", vec_filename)
        
        with open(vec_filename,'r') as f:
            next(f) # skip headings
            vec = np.asarray([int(elem) for elem in next(f).strip().split()])
        
        with open(max_filename,'r') as f:
            next(f) # skip headings
            max = int(next(f).strip())

        with open(min_filename,'r') as f:
            next(f) # skip headings
            min = int(next(f).strip())

        assert np.allclose(get_minmax_indices(np.array(vec)), np.asarray([min-1, max-1])) == True

def test_swap_columns():
    """Various tests taxen from symlib."""    
    import csv
    ntests =50

    vecs = []
    mins = []
    maxs = []
    for n in range(1,ntests+1):
        Bin_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/swap_column_B.in.%s" %n))
        Bout_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/swap_column_B.out.%s" %n))
        Min_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/swap_column_M.in.%s" %n))
        Mout_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/swap_column_M.out.%s" %n))
        kin_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/swap_column_k.in.%s" %n))

        Bin = []
        Bin_file.readline() # skip headings 
        for line in Bin_file.readlines():
            Bin.append([int(x) for x in line.split()])
        Bin = np.asarray(Bin)
        Bout = []
        Bout_file.readline() # skip headings 
        for line in Bout_file.readlines():
            Bout.append([int(x) for x in line.split()])
        Bout = np.asarray(Bout)
        Min = []
        Min_file.readline() # skip headings 
        for line in Min_file.readlines():
            Min.append([int(x) for x in line.split()])
        Min = np.asarray(Min)

        Mout = []
        Mout_file.readline() # skip headings 
        for line in Mout_file.readlines():
            Mout.append([int(x) for x in line.split()])
        Mout = np.asarray(Mout)
        
        kin = 0
        kin_file.readline() # skip headings
        for line in kin_file.readlines():
            kin = int(line)
                
        assert np.allclose(swap_column(Min,Bin,kin-1), (Mout, Bout)) == True

def test_swap_rows():
    """Various tests taxen from symlib."""    
    import csv
    ntests =50

    vecs = []
    mins = []
    maxs = []
    for n in range(1,ntests+1):
        Bin_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/swap_column_B.in.%s" %n))
        Bout_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/swap_column_B.out.%s" %n))
        Min_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/swap_column_M.in.%s" %n))
        Mout_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/swap_column_M.out.%s" %n))
        kin_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/swap_column_k.in.%s" %n))
        
        Bin = []
        Bin_file.readline() # skip headings 
        for line in Bin_file.readlines():
            Bin.append([int(x) for x in line.split()])
        Bin = np.asarray(Bin)
        Bout = []
        Bout_file.readline() # skip headings 
        for line in Bout_file.readlines():
            Bout.append([int(x) for x in line.split()])
        Bout = np.asarray(Bout)
        Min = []
        Min_file.readline() # skip headings 
        for line in Min_file.readlines():
            Min.append([int(x) for x in line.split()])
        Min = np.asarray(Min)

        Mout = []
        Mout_file.readline() # skip headings 
        for line in Mout_file.readlines():
            Mout.append([int(x) for x in line.split()])
        Mout = np.asarray(Mout)
        
        kin = 0
        kin_file.readline() # skip headings
        for line in kin_file.readlines():
            kin = int(line)
            
        Min = np.transpose(Min)
        Bin = np.transpose(Bin)
        Min, Bin = swap_row(Min,Bin,kin-1)
        Min = np.transpose(Min)
        Bin = np.transpose(Bin)
        
        assert np.allclose((Min, Bin), (Mout, Bout)) == True

        
def test_LowerHermiteNormalForm():
    """Various tests taxen from symlib."""
    import csv
    ntests =50

    vecs = []
    mins = []
    maxs = []
    for n in range(1,ntests+1):
        Sin_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/HermiteNormalForm_S.in.%s" %n))
        Bout_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/HermiteNormalForm_B.out.%s" %n))
        Hout_file = open(os.path.join(os.path.dirname(__file__),
                        "sampling_testfiles/HermiteNormalForm_H.out.%s" %n))
        
        Sin = []
        Sin_file.readline() # skip headings 
        for line in Sin_file.readlines():
            Sin.append([int(x) for x in line.split()])
        Sin = np.asarray(Sin)
        
        Bout = []
        Bout_file.readline() # skip headings 
        for line in Bout_file.readlines():
            Bout.append([int(x) for x in line.split()])
        Bout = np.asarray(Bout)
        
        Hout = []
        Hout_file.readline() # skip headings 
        for line in Hout_file.readlines():
            Hout.append([int(x) for x in line.split()])
        Hout = np.asarray(Hout)

        assert np.allclose(LowerHermiteNormalForm(Sin), (Hout, Bout)) == True

def test_HermiteNormalForm():
    """Various tests taxen from symlib."""
    import csv
    ntests =100

    vecs = []
    mins = []
    maxs = []
    for n in range(1,ntests+1):
        S = np.random.randint(-100,100, size=(3,3))
        H, B = HermiteNormalForm(S)

        assert np.isclose(abs(np.linalg.det(B)), 1) == True
        assert np.allclose(np.dot(B, S), H) == True
        check1 = (np.array([2,2,1]), np.array([1,0,0]))
        assert np.count_nonzero(H[check1]) == 0
        check2 = (np.asarray([0, 0, 0, 1, 1, 2]), np.asarray([0, 1, 2, 1, 2, 2]))
        assert any(H[check2] < 0) == False
        assert (H[0,1] < H[1,1]) == True
        assert (H[0,2] < H[2,2]) == True
        assert (H[1,2] < H[2,2]) == True


def test_make_grid():
    mesh_pts1 = [[0,0,0],
                 [1,1,1],
                 [0,0,2],
                 [0,2,0],
                 [0,2,2],
                 [2,0,0],
                 [0,2,2],
                 [2,0,2],
                 [2,2,0],
                 [2,2,2],
                 [1,3,3],
                 [3,1,3],
                 [3,1,1],
                 [1,1,3],
                 [1,3,1],
                 [3,1,1]]
    mesh_pts1 = np.asarray(mesh_pts1)*1./4

    cell_type = "sc"
    cell_const = 1.
    cell_vecs = make_ptvecs(cell_type, cell_const)
    
    mesh_type = "bcc"
    mesh_const = cell_const/2
    mesh_vecs = make_ptvecs(mesh_type, mesh_const)
    offset = np.asarray([0.,0.,0.])
    mesh = make_grid(cell_vecs, mesh_vecs, offset)
    
    for g1 in mesh_pts1:
        check = False
        for g2 in mesh:
            if np.allclose(g1,g2) == True:
                check = True
        assert check == True
    
    lat_type_list = ["fcc","bcc", "sc"]
    lat_const_list = [10,10.1, 3*np.pi]
    offset_list = [[1.3, 1.1,1.7],[11,9,8],[np.pi,np.pi,np.pi]]
    r_list = [1, 2.3, np.pi]

    for lat_type in lat_type_list:
        for lat_const in lat_const_list:
            lat_vecs = make_rptvecs(lat_type,lat_const)
            for offset in offset_list:
                offset = np.asarray(offset)
                for r in r_list:
                    total_grid = large_sphere_pts(lat_vecs,r,offset)
                    grid = sphere_pts(lat_vecs,r,offset)
                    contained = False
                    for tg in total_grid:
                        if np.dot(tg-offset,tg-offset) <= r:
                            contained = False
                            for g in grid:
                                if np.allclose(g,tg):
                                    contained = True
                            assert contained == True
