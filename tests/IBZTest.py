#!/usr/bin/env python
'''    Tests routine for finding best mesh for each structure in dir
'''
import sys,os,subprocess
from numpy import zeros,transpose,array,sum,float64,rint
from numpy.linalg import norm
# sys.path.append('/bluehome2/bch/pythonscripts/cluster_expansion/ceflashscripts/')
# import meshConstruct5
import makeIBZ
from kmeshroutines import nstrip, readposcar,create_poscar

################# script #######################
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATA11000/test101x/'
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATA500/AlIr/'
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATA11000/test/'

#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATA11000/AlIr/'
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATAf1_50e/test2/'
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATAf1_50e/test2.10xNk/'

#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATAf1_50e/test.10xNk/'
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATAf1_50e/test.noshift/'
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATAf1_50e/test10^3/'
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATAf1_50e/f3varyN/'
# maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATAf1_50e/test/'
maindir = '/fslhome/bch/cluster_expansion/vcmesh/AlAl'
# maindir = '/fslhome/bch/cluster_expansion/vcmesh/cu.pt.ntest/cubicTest/'
method = 0  #0,1 for now.
            #0: exact: use vertices of mesh voronoi cell that are closest/farthest 
            #         from the IBZ center origin to check if the point's volume is cut. 
            #         Cut the VC to determine the volume contribution    
            #0.5 approx 1.5.  If cut volume is less than 50%, distribute weight to neighbors of equivalent points.  
            #1: approx 1. Use sphere around mesh point to test whether it is near a surface.  
            #         For a 1-plane cut, use the spherical section that is inside. 
            #         For 2 or 3 plane cut, we use the exact method. 
            #2: approx 2. For spheres with their centers beyond the cell boundaries
                      # but inside the expanded boundaries, add their weights to 
                      # nearest neighbors that are inside. 
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATAf1_50e/testSi/'
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATAf1_50e/testMP/'
#maindir = '/fslhome/bch/cluster_expansion/alir/AFLOWDATAf1_50e/AlIr34-50/'
#maindir = '/fslhome/bch/cluster_expansion/sisi/test10^3/'
#maindir = '/fslhome/bch/cluster_expansion/sisi/test10^4/'

meshc = makeIBZ.dynamicPack() #instance

testfile = 'POSCAR'
# Nkppra = 10000#*10  


Nkppra = 18000
meshtype = 'fcc'  #cub, fcc, bcc  

#reallatt = zeros((3,3))
os.chdir(maindir)
dirs = sorted([d for d in os.listdir(os.getcwd()) if os.path.isdir(d)])
if os.path.exists('meshsummary.csv'):
    file1 = open('meshsummary.csv','a')
else: 
    file1 = open('meshsummary.csv','w')
file1.write('Structure,Lattice,amax/amin,pfB,pf_orth,pf_orth2fcc,pf_maxpf, pf_pf2fcc, pfmax, meshtype' + ',' \
             + 'Improvement,fcc compatibility,Nmesh,TargetNmesh,Nmesh/Target,cbest' + '\n')
#for i,dir in enumerate(dirs):    

for dir in dirs:
    path = maindir+'/'+dir
    if testfile in os.listdir(path):        
        print 
        print dir + '=========================================================='
        os.chdir(path)
#        print readposcar('POSCAR',path)
        [descriptor, scale, latticevecs, reciplatt, natoms, postype, positions] = readposcar('POSCAR',path) #
        create_poscar('POSCAR',descriptor, scale, latticevecs, natoms, postype, positions, path) #just to remove the scale problem
        os.chdir(maindir)
#        print 'reciprocal lattice vectors (rows)';print reciplatt
        totatoms = sum(natoms)
        targetNmesh = Nkppra/totatoms
        atype = 1
        aTypes = []
        for natom in natoms:
            for i in range(natom):
                aTypes.append(atype)
            atype += 1
        aTypes = array(aTypes)

        meshc.mainRoutine(latticevecs,reciplatt,totatoms,aTypes,postype,transpose(positions),targetNmesh,meshtype,path,method)
        sys.exit('stop')


file1.close()
        
print 'Done'