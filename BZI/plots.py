"""Plot various quantities related to electronic stucture integration.
"""

import numpy as np
from numpy.linalg import norm, inv, det
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from itertools import product, chain
from scipy.spatial import ConvexHull
from copy import deepcopy
import time, pickle, os

from BZI.symmetry import (bcc_sympts, fcc_sympts, sc_sympts, make_ptvecs,
                          make_rptvecs, sym_path, number_of_point_operators,
                          find_orbits)
from BZI.sampling import make_cell_points
# from BZI.integration import (rectangular_fermi_level, rectangular_method,
#                              rec_dos_nos)
from BZI.integration import rectangular_method, rec_dos_nos
from BZI.tetrahedron import (grid_and_tetrahedra, calc_fermi_level,
                             calc_total_energy, get_extended_tetrahedra,
                             get_corrected_total_energy, density_of_states,
                             number_of_states, tet_dos_nos, find_irreducible_tetrahedra)
from BZI.make_IBZ import find_bz, orderAngle, planar3dTo2d
from BZI.utilities import remove_points, find_point_indices, check_contained

def ScatterPlotMultiple(func, states, ndivisions, cutoff=None):
    """Plot the energy states of a multivalued toy pseudo-potential using 
    matplotlib's function scatter. 
    
    Args:
        nstates (int): the number of states to plot.
        ndivisions (int): the number of divisions along each coordinate direction.
        cutoff (float): the value at which the states get cutoff.
        
    Returns:
        None
        
    Example:
        >>> from BZI import ScatterPlotMultiple
        >>> nstates = 2
        >>> ndivisions = 11
        >>> ScatterPlotMultiple(nstates, ndivisions)
    """
    
    kxs = np.linspace(-1./2, 1./2, ndivisions)
    kys = np.linspace(-1./2, 1./2, ndivisions)
    kzs = [0.]
    
    kxlist = [kxs[np.mod(i,len(kxs))] for i in range(len(kxs)*len(kys))]
    kylist = [kxs[int(i/len(kxs))] for i in range(len(kxs)*len(kys))]

    kpts = [[kx, ky, kz] for kx in kxs for ky in kys for kz in kzs]
    all_estates = [func(kpt) for kpt in kpts]

    prows = int(np.sqrt(len(states)))
    pcols = int(np.ceil(len(states)/prows))
    
    p = 0
    if cutoff == None:
        for n in states:
            p += 1
            estates = np.array([], dtype=complex)
            for es in all_estates:
                estates = np.append(np.real(estates), es[n])
            ax = plt.subplot(prows,pcols,p,projection="3d")
            ax.scatter(kxlist, kylist, estates,s=.5);
    else:
        for n in states:
            p += 1
            estates = np.array([], dtype=complex)
            for es in all_estates:
                if es[n] > cutoff:
                    estates = np.append(np.real(estates), 0.)
                else:
                    estates = np.append(np.real(estates), es[n])
            ax = plt.subplot(prows,pcols,p,projection="3d")
            ax.scatter(kxlist, kylist, estates,s=.5);
    plt.show()

    
def scatter_plot_pp(EPM, states, nbands, ndivisions, grid_vectors,
                    plane_value, offset, cutoff=None):
    """Plot the energy states of a multivalued toy pseudo-potential using 
    matplotlib's function scatter.
    
    Args:
        EPM (:py:obj:`BZI.pseudopots.EmpiricalPseudopotential`): a pseudopotential object.
        states (list): a list of states to plot.
        nbands (int): the number of bands to include. No value in states can be 
            greater than nbands.
        ndivisions (int): the number of divisions along each coordinate direction.
        grid_vectors(list): two integers that indicate which reciprocal lattice
            vectors to take as the plane over which to plot the band structure.       
        plane_value (float): the value along the axis perpendicular to the plane
            at which to plot. It is given in reciprocal lattic coordinates.
        offset (numpy.ndarray): the offset of the grid in lattice coordinates.
        cutoff (float): the value at which the states get cutoff.
        
    Returns:
        None
        
    Example:
        >>> from BZI import ScatterPlotMultiple
        >>> nstates = 2
        >>> ndivisions = 11
        >>> ScatterPlotMultiple(nstates, ndivisions)
    """

    g1 = EPM.lattice.reciprocal_vectors[:, grid_vectors[0]]/ndivisions
    g2 = EPM.lattice.reciprocal_vectors[:, grid_vectors[1]]/ndivisions

    orth_vec = np.setdiff1d([0, 1, 2], grid_vectors)[0]    
    orthogonal_vector = EPM.lattice.reciprocal_vectors[orth_vec]

    # Distance in the direction orthogonal to the plane.
    d = plane_value*orthogonal_vector

    grid = []
    kxlist = []
    kylist = []
    for i,j in product(range(ndivisions+1), repeat=2):
        grid.append(i*g1 + j*g2 + d + offset)
        kxlist.append(grid[-1][grid_vectors[0]])
        kylist.append(grid[-1][grid_vectors[1]])
    
    all_estates = [EPM.eval(kpt, nbands) for kpt in grid]
    prows = int(np.sqrt(len(states)))
    pcols = int(np.ceil(len(states)/prows))
    
    p = 0
    if cutoff == None:
        for n in states:
            p += 1
            estates = np.array([], dtype=complex)
            for es in all_estates:
                estates = np.append(np.real(estates), es[n])
            ax = plt.subplot(prows,pcols,p,projection="3d")
            ax.scatter(kxlist, kylist, estates,s=.5);
    else:
        for n in states:
            p += 1
            estates = np.array([], dtype=complex)
            for es in all_estates:
                if es[n] > cutoff:
                    estates = np.append(np.real(estates), 0.)
                else:
                    estates = np.append(np.real(estates), es[n])
            ax = plt.subplot(prows,pcols,p,projection="3d")
            ax.scatter(kxlist, kylist, estates,s=.5);
    plt.show()
    return grid
    
def ScatterPlotSingle(func, ndivisions, cutoff=None):
    """Plot the energy states of a single valued toy pseudo-potential using 
    matplotlib's function scatter.
    
    Args:
        func (function): one of the single valued functions from pseudopots.
        ndivisions (int): the number of divisions along each coordinate direction.
        cutoff (float): the value at which function is gets cutoff.
        
    Returns:
        None
        
    Example:
        >>> from BZI.pseudopots import W1
        >>> from BZI.plots import ScatterPlotSingle
        >>> nstates = 2
        >>> ndivisions = 11
        >>> ScatterPlotMultiple(W1, nstates, ndivisions)
    """
    
    kxs = np.linspace(0, 1, ndivisions)
    kys = np.linspace(0, 1, ndivisions)
    kzs = [0.]

    kxlist = [kxs[np.mod(i,len(kxs))] for i in range(len(kxs)*len(kys))]
    kylist = [kxs[int(i/len(kxs))] for i in range(len(kxs)*len(kys))]

    kpts = [[kx, ky, kz] for kx in kxs for ky in kys for kz in kzs]
    estates = [func(kpt)[0] for kpt in kpts]

    if cutoff == None:
        ax = plt.subplot(1,1,1,projection="3d")
        ax.scatter(kxlist, kylist, estates,s=.5);
    else:
        estates, kxlist, kylist = zip(*filter(lambda x: x[0]
                                              <= cutoff, zip(estates,
                                                             kxlist, kylist)))
        ax = plt.subplot(1,1,1,projection="3d")
        ax.scatter(kxlist, kylist, estates,s=.5);
    plt.show()

def plot_just_points(mesh_points, ax=None):
    """Plot just the points in a mesh.

    Args:
        mesh_points (numpy.ndarray or list): a list of points
    """

    ngpts = len(mesh_points)
    kxlist = [mesh_points[i][0] for i in range(ngpts)]
    kylist = [mesh_points[i][1] for i in range(ngpts)]
    kzlist = [mesh_points[i][2] for i in range(ngpts)]
    if not ax:
        ax = plt.subplot(1,1,1,projection="3d")        
    ax.scatter(kxlist, kylist, kzlist, c="black", s=10)

    
def plot_mesh(mesh_points, cell_vecs, offset = np.asarray([0.,0.,0.]),
              indices=None, show=True, save=False, file_name=None):
    """Create a 3D scatter plot of a set of mesh points inside a cell.
    
    Args:
        mesh_points (list or numpy.ndarray): a list of mesh points in Cartesian
            coordinates.
        cell_vecs (list or numpy.ndarray): a list vectors that define a cell.
        offset (list or numpy.ndarray): the offset of the unit cell, which is
           also plotted, in Cartesian coordinates.
        indices (list or numpy.ndarray): the indices of the points. If
            provided, they will be plotted with the mesh points.
        show (bool): if true, the plot will be shown.
        save (bool): if true, the plot will be saved.
        file_name (str): the file name under which the plot is saved. If not 
            provided the plot is saved as "mesh.pdf".
    Returns:
        None
    """
    
    ngpts = len(mesh_points)
    kxlist = [mesh_points[i][0] for i in range(ngpts)]
    kylist = [mesh_points[i][1] for i in range(ngpts)]
    kzlist = [mesh_points[i][2] for i in range(ngpts)]

    ax = plt.subplot(1,1,1,projection="3d")
    ax.scatter(kxlist, kylist, kzlist, c="red")

    # Give the points labels if provided.
    if (type(indices) == list or type(indices) == np.ndarray):
        for x,y,z,i in zip(kxlist,kylist,kzlist,indices):
            ax.text(x,y,z,i)    
    
    c1 = cell_vecs[:,0] 
    c2 = cell_vecs[:,1] 
    c3 = cell_vecs[:,2] 
    O = np.asarray([0.,0.,0.]) 

    l1 = zip(O + offset, c1 + offset)
    l2 = zip(c2 + offset, c1 + c2 + offset)
    l3 = zip(c3 + offset, c1 + c3 + offset)
    l4 = zip(c2 + c3 + offset, c1 + c2 + c3 + offset)
    l5 = zip(O + offset, c3 + offset)
    l6 = zip(c1 + offset, c1 + c3 + offset)
    l7 = zip(c2 + offset, c2 + c3 + offset)
    l8 = zip(c1 + c2 + offset, c1 + c2 + c3 + offset)
    l9 = zip(O + offset, c2 + offset)
    l10 = zip(c1 + offset, c1 + c2 + offset)
    l11 = zip(c3 + offset, c2 + c3 + offset)
    l12 = zip(c1 + c3 + offset, c1 + c2 + c3 + offset)

    ls = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]

    for l in ls:
        ax.plot3D(*l, c="blue")
    if show:
        plt.show()
    elif save:
        if file_name:
            plt.savefig(file_name + ".pdf")
        else:
            plt.savefig("mesh.pdf")
    return None

def plot_bz_mesh(mesh_points, lat_vecs):
    """Plot the irreducible k-points inside the Wigner-Seitz construction of the first
    Brillouin zone.

    Args:
        mesh_points (numpy.ndarray): an 2D array of mesh points.
        lat_vecs (numpy.ndarray): an array of lattice vectors as columns of a 3x3 array.
    """

    ngpts = len(mesh_points)
    kxlist = [mesh_points[i][0] for i in range(ngpts)]
    kylist = [mesh_points[i][1] for i in range(ngpts)]
    kzlist = [mesh_points[i][2] for i in range(ngpts)]

    ax = plt.subplot(1,1,1,projection="3d")
    ax.scatter(kxlist, kylist, kzlist, c="red")

    BZ = find_bz(lat_vecs)
    for simplex in BZ.simplices:
        # We're going to plot lines between the vertices of the simplex.
        # To make sure we make it all the way around, append the first element
        # to the end of the simplex.
        simplex = np.append(simplex, simplex[0])
        simplex_pts = [BZ.points[i] for i in simplex]
        plot_simplex_edges(simplex_pts, ax)


def PlotMeshes(mesh_points_list, cell_vecs, atoms, offset = np.asarray([0.,0.,0.])):
    """Create a 3D scatter plot of a set of mesh points inside a cell.
    
    Args:
        mesh_points (list or np.ndarray): a list of mesh points.
        cell_vecs (list or np.ndarray): a list vectors that define a cell.
        
    Returns:
        None
    """
    
    ax = plt.subplot(1,1,1,projection="3d")
    colors = ["red", "blue", "green", "black"]
    for i,mesh_points in enumerate(mesh_points_list):
        ngpts = len(mesh_points)
        kxlist = [mesh_points[i][0] for i in range(ngpts)]
        kylist = [mesh_points[i][1] for i in range(ngpts)]
        kzlist = [mesh_points[i][2] for i in range(ngpts)]
        
        ax.scatter(kxlist, kylist, kzlist, c=colors[atoms[i]])
    
    c1 = cell_vecs[:,0] 
    c2 = cell_vecs[:,1] 
    c3 = cell_vecs[:,2] 
    O = np.asarray([0.,0.,0.]) 

    l1 = zip(O + offset, c1 + offset)
    l2 = zip(c2 + offset, c1 + c2 + offset)
    l3 = zip(c3 + offset, c1 + c3 + offset)
    l4 = zip(c2 + c3 + offset, c1 + c2 + c3 + offset)
    l5 = zip(O + offset, c3 + offset)
    l6 = zip(c1 + offset, c1 + c3 + offset)
    l7 = zip(c2 + offset, c2 + c3 + offset)
    l8 = zip(c1 + c2 + offset, c1 + c2 + c3 + offset)
    l9 = zip(O + offset, c2 + offset)
    l10 = zip(c1 + offset, c1 + c2 + offset)
    l11 = zip(c3 + offset, c2 + c3 + offset)
    l12 = zip(c1 + c3 + offset, c1 + c2 + c3 + offset)

    ls = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]

    for l in ls:
        ax.plot3D(*l, c="black")
    plt.show()
    return None

def PlotSphereMesh(mesh_points,r2, offset = np.asarray([0.,0.,0.]),
                   save=False, show=True):
    """Create a 3D scatter plot of a set of points inside a sphere.
    
    Args:
        mesh_points (list or np.ndarray): a list of mesh points.
        r2 (float): the squared radius of the sphere
        cell_vecs (list or np.ndarray): a list vectors that define a cell.
        save (bool): if true, the plot is saved as sphere_mesh.png.
        show (bool): if true, the plot is displayed.
        
    Returns:
        None
    Example:
        >>> from BZI.sampling import sphere_pts
        >>> from BZI.symmetry import make_rptvecs
        >>> from BZI.plots import PlotSphereMesh
        >>> import numpy as np
        >>> lat_type = "fcc"
        >>> lat_const = 10.26
        >>> lat_vecs = make_rptvecs(lat_type, lat_const)
        >>> r2 = 3.*(2*np.pi/lat_const)**2
        >>> offset = [0.,0.,0.]
        >>> grid = sphere_pts(lat_vecs,r2,offset)
        >>> PlotSphereMesh(grid,r2,offset)
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    r = np.sqrt(r2)
    x = r * np.outer(np.cos(u), np.sin(v)) + offset[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + offset[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + offset[2]
    
    ax.scatter(x,y,z,s=0.001)

    # Plot the points within the sphere.
    ngpts = len(mesh_points)
    kxlist = [mesh_points[i][0] for i in range(ngpts)]
    kylist = [mesh_points[i][1] for i in range(ngpts)]
    kzlist = [mesh_points[i][2] for i in range(ngpts)]
    
    ax.set_aspect('equal')
    ax.scatter(kxlist, kylist, kzlist, c="black",s=1)

    lim = np.sqrt(r2)*1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    if save:
        plt.savefig("sphere_mesh.png")
    if show:
        plt.show()
    return None

def plot_band_structure(materials_list, EPMlist, EPMargs_list, lattice, npts,
                        neigvals, energy_shift=0.0, energy_limits=False,
                        fermi_level=False, save=False, show=True, sum_bands=None,
                        plot_below=False, ax=None):
    """Plot the band structure of a pseudopotential along symmetry paths.
    
    Args:
        materials_list (str): a list of materials whose bandstructures will be 
        plotted. The first string will label figures and files.
        EPMlist (function): a list of pseudopotenial functions.
        EPMargs_list (list): a list of pseudopotential arguments as dictionaries.
        lattice
        npts (int): the number of points to plot along each symmetry line.
        neigvals (int): the number of lower-most eigenvalues to include 
        in the plot.
        energy_shift (float): energy shift for band structure
        energy_limits (list): the energy window for the plot.
        fermi_level (float): if provided, the Fermi level will be included
        in the plot.
        save (bool): if true, the band structure will be saved.

    Returns:
        Display or save the band structure.
    """

    if ax is None:
        fig, ax = plt.subplots()
    
    # k-points between symmetry point pairs in Cartesian coordinates.
    car_kpoints = sym_path(lattice, npts, cart=True)
    
    # Find the distance of each symmetry path by putting the symmetry point pairs 
    # that make up a path in lattice coordinates, converting to Cartesian, and then
    # taking the norm of the difference of the pairs.
    lat_symmetry_paths = np.empty_like(lattice.symmetry_paths, dtype=list)
    car_symmetry_paths = np.empty_like(lattice.symmetry_paths, dtype=list)
    distances = []
    
    for i,path in enumerate(lattice.symmetry_paths):
        for j,sympt in enumerate(path):
            lat_symmetry_paths[i][j] = lattice.symmetry_points[sympt]
            car_symmetry_paths[i][j] = np.dot(lattice.reciprocal_vectors,
                                              lat_symmetry_paths[i][j])
        distances.append(norm(car_symmetry_paths[i][1] - car_symmetry_paths[i][0]))

    # Create coordinates for plotting.
    lines = []
    for i in range(len(distances)):
        start = np.sum(distances[:i])
        stop = np.sum(distances[:i+1])
        if i == (len(distances) - 1):
            lines += list(np.linspace(start, stop, npts))
        else:
            lines += list(np.delete(np.linspace(start, stop, npts),-1))
            
    # Store the energy eigenvalues in an nested array.
    nEPM = len(EPMlist)
    energies = [[] for i in range(nEPM)]
    for i in range(nEPM):
        EPM = EPMlist[i]
        EPMargs = EPMargs_list[i]
        EPMargs["neigvals"] = neigvals
        for kpt in car_kpoints:
            EPMargs["kpoint"] = kpt
            energies[i].append(EPM.eval(**EPMargs) - energy_shift)
            # energies[i].append(EPM.eval(**EPMargs))


    colors = ["blue", "green", "red", "violet", "orange", "cyan", "black"]            
    energies = np.array(energies)
    if plot_below:
        colors = ["red", "blue", "green", "violet", "orange", "cyan", "black"]        
        for i in range(nEPM):
            energies[i][energies[i] > EPMlist[i].fermi_level] = np.nan
            
    # Find the x-axis labels and label locations.
    plot_xlabels = [lattice.symmetry_paths[0][0]]
    plot_xlabel_pos = [0.]
    for i in range(len(lattice.symmetry_paths) - 1):
        if (lattice.symmetry_paths[i][1] == lattice.symmetry_paths[i+1][0]):
            plot_xlabels.append(lattice.symmetry_paths[i][1])
            plot_xlabel_pos.append(np.sum(distances[:i+1]))
        else:
            plot_xlabels.append(lattice.symmetry_paths[i][1] + "|" + 
                                lattice.symmetry_paths[i+1][0])
            plot_xlabel_pos.append(np.sum(distances[:i+1]))
    plot_xlabels.append(lattice.symmetry_paths[-1][1])
    plot_xlabel_pos.append(np.sum(distances))    

    # Plot the energy dispersion curves one at a time.
    for i in range(nEPM):
        if sum_bands is not None:
            ienergy = []
            for nk in range(len(car_kpoints)):
                tmp_energies = np.array(energies[i][nk][:neigvals])
                # print("tmp energies: ", tmp_energies)
                # print("all included bands: ", tmp_energies[np.where(tmp_energies < sum_bands)])
                ienergy.append(np.sum(tmp_energies[np.where(tmp_energies < sum_bands)]))
            ax.plot(lines, ienergy, color=colors[i], label="%s"%materials_list[i])
        else:
            for ne in range(neigvals):
                ienergy = []
                for nk in range(len(car_kpoints)):
                    ienergy.append(energies[i][nk][ne])                    
                if ne == 0:
                    ax.plot(lines, ienergy, color=colors[i], label="%s"%materials_list[i])
                else:
                    ax.plot(lines, ienergy, color=colors[i])

    # Plot the Fermi level if provided.
    if fermi_level:
        ax.axhline(y = EPMlist[0].fermi_level, c="yellow", label="Fermi level")

    # Plot a vertical line at the symmetry points with proper labels.
    for pos in plot_xlabel_pos:
        ax.axvline(x = pos, c="gray")
    plt.xticks(plot_xlabel_pos, plot_xlabels, fontsize=14)

    # Adjust the energy range if one was provided.
    if energy_limits:
        ax.set_ylim(energy_limits)
    
    # Adjust the legend.
    # lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim([0,np.sum(distances)])
    ax.set_xlabel("Symmetry points", fontsize=16)
    ax.set_ylabel("Energy (eV)", fontsize=16)
    ax.set_title("%s Band Structure" %materials_list[0], fontsize=16)
    ax.grid(linestyle="dotted")
    if save:
        # ax.savefig("%s_band_structure.pdf" %materials_list[0],
        #             bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig("%s_band_structure.pdf" %materials_list[0],
                    bbox_inches='tight')
        
    if show:
        plt.show()
    return None

def PlotVaspBandStructure(file_loc, material, lat_type, lat_consts, lat_angles,
                          energy_shift=0.0, fermi_level=False, elimits=False,
                          save=False, show=True):
    """Plot the band structure from a VASP INCAR, KPOINT, and OUTCAR file.

    Args:
        file_loc (str): the location of the directory with the VASP output files.
        The KPOINTS file MUST be in a very particular format: there must be
        4 introductory lines, each k-point must be on its own line and there
        must only be one space between pairs of k-points. There mustn't be 
        any space after the last entered k-point.
        material (str): the material whose band structure is being plotted.
        lat_type (str): the lattice type
        energy_shift (float): energy shift for band structure
        fermi_level (bool): if true, plot the Fermi level.
        elimits (list): the energy window for the plot.
        save (bool): if true, the band structure will be saved.
        show (bool): if false, return none. This is useful for showing multiple 
            plots.

    Returns:
        Display or save the band structure.
    """

    # Get the correct symmetry point dictionary.
    if lat_type == "fcc":
        sympt_dict = fcc_sympts
        lat_centering = "face"
    elif lat_type == "bcc":
        sympt_dict = bcc_sympts
        lat_centering = "body"
    elif lat_type == "sc":
        sympt_dict = sc_sympts
        lat_centering = "prim"
    else:
        raise ValueError("Invalid lattice type")


    # Extract the lattice constant from the POSCAR file.
    sympt_list = []
    with open(file_loc + "POSCAR","r") as file:
        lat_const = ""    
        f = file.readlines()
        for c in f[1]:
            try:
                int(c)
                lat_const += c
            except:
                if c == ".":
                    lat_const += c
                if c == "!":
                    break
                continue
    lat_const = float(lat_const)
    angstrom_to_Bohr = 1.889725989
    lat_const *= angstrom_to_Bohr
    lat_vecs = make_ptvecs(lat_centering, lat_consts, lat_angles)
    rlat_vecs = make_rptvecs(lat_vecs)
    
    nbands = ""
    with open(file_loc + "INCAR","r") as file:
        f = file.readlines()
        for line in f:
            if "NBANDS" in line:
                for l in line:
                    try:
                        int(l)
                        nbands += l
                    except ValueError:
                        continue
        # nbands = int(nbands)

    nbands = 10
    # Extract the total number of k-points, number of k-points per path,
    # the number of paths and the symmetry points from the KPOINTs file.
    with open(file_loc + "KPOINTS","r") as file:
        f = file.readlines()
        npts_path = int(f[1].split()[0])
        npaths = (len(f)-3)/3
        nkpoints = int(npts_path*npaths)
        
        sympt_list = []
        f = f[4:]
        for line in f:
            spt = ""
            sline = line.strip()
            for l in sline:
                if (l == " " or
                    l == "!" or 
                    l == "." or 
                    l == "\t"):
                    continue
                else:
                    try:
                        int(l)
                    except:
                        spt += l
            if spt != "":
                sympt_list.append(spt)

    for i,sympt in enumerate(sympt_list):
        if sympt == "gamma" or sympt == "Gamma":
            sympt_list[i] = "G"
            
    # Remove all duplicate symmetry points
    unique_sympts = [sympt_list[i] for i in range(0, len(sympt_list), 2)] + [sympt_list[-1]]
    
    # Replace symbols representing points with their lattice coordinates.
    lat_sympt_coords = [sympt_dict[sp] for sp in unique_sympts]
    car_sympt_coords = [np.dot(rlat_vecs,k) for k in lat_sympt_coords]
    
    with open(file_loc + "OUTCAR", "r") as file:
        f = file.readlines()
        EFERMI = ""
        for line in f:
            sline = line.strip()
            if "EFERMI" in sline:
                for c in sline:
                    try:
                        int(c)
                        EFERMI += c
                    except:
                        if c == ".":
                            EFERMI += c
        EFERMI = float(EFERMI)    
        
    id_line = "  band No.  band energies     occupation \n"        
    with open(file_loc + "OUTCAR", "r") as file:
        f = file.readlines()
        energies = []
        occupancies = []
        en_occ = []
        lat_kpoints = []
        nkpt = 0
        nkpts_dr = 0 # number of kpoints with duplicates removed
        for i,line in enumerate(f):
            if line == id_line:
                nkpt += 1
                if nkpt % npts_path == 0 and nkpt != nkpoints:
                    continue
                else:
                    nkpts_dr += 1
                    energies.append([])
                    occupancies.append([])
                    en_occ.append([])
                    lat_kpoints.append(list(map(float,f[i-1].split()[3:6])))
                    for j in range(1,nbands+1):
                        energies[nkpts_dr-1].append(float(f[i+j].split()[1]) - energy_shift)
                        occupancies[nkpts_dr-1].append(float(f[i+j].split()[2]))
                        en_occ[nkpts_dr-1].append(energies[nkpts_dr-1][-1]*(
                            occupancies[nkpts_dr-1][-1]/2))
    car_kpoints = [np.dot(rlat_vecs,k) for k in lat_kpoints]

    # Find the distances between symmetry points.
    nsympts = len(unique_sympts)
    sympt_dist = [0] + [norm(car_sympt_coords[i+1]
                             - car_sympt_coords[i])
                        for i in range(nsympts - 1)]

    # Create coordinates for plotting
    lines = []
    for i in range(nsympts - 1):
        start = np.sum(sympt_dist[:i+1])
        stop = np.sum(sympt_dist[:i+2])
        if i == (nsympts - 2):
            lines += list(np.linspace(start, stop, npts_path))
        else:
            lines += list(np.delete(np.linspace(start, stop, npts_path),-1))

    for nb in range(nbands):
        ienergy = []
        for nk in range(len(car_kpoints)):
            ienergy.append(energies[nk][nb])
        if nb == 0:
            plt.plot(lines,ienergy, label="VASP Band structure",color="blue")
        else:
            plt.plot(lines,ienergy,color="blue")

    # Plot a vertical line at the symmetry points with proper labels.
    for i in range(nsympts):
        pos = np.sum(sympt_dist[:i+1])
        plt.axvline(x = pos, c="gray")        
    tick_labels = unique_sympts
    tick_locs = [np.sum(sympt_dist[:i+1]) for i in range(nsympts)]
    plt.xticks(tick_locs,tick_labels)

    # Adjust the energy range if one was provided.
    if elimits:
        plt.ylim(elimits)
    if fermi_level:
        plt.axhline(y = EFERMI, c="yellow", label="Fermi level")

    # Adjust the legend.
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xlim([0,np.sum(sympt_dist)])
    plt.xlabel("Symmetry points")
    plt.ylabel("Energy (eV)")
    plt.title("%s Band Structure" %material)
    plt.grid(linestyle="dotted")
    if save:
        plt.savefig("%s_band_struct.pdf" %material,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
    elif show:
        plt.show()
    else:
        return None


def plot_paths(EPM, npts, save=False):
    """Plot the path along which the band structure is plotted.
    """
    
    # k-points between symmetry point pairs in Cartesian coordinates.
    car_kpoints = sym_path(EPM.lattice, npts, cart=True)

    # Plot the paths.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = [ckp[0] for ckp in car_kpoints]
    y = [ckp[1] for ckp in car_kpoints]
    z = [ckp[2] for ckp in car_kpoints]

    ax.plot(x,y,z)

    # Label the paths.
    sympt_labels = list(EPM.lattice.symmetry_points.keys())
    sympts = [np.dot(EPM.lattice.reciprocal_vectors, p) for p in
              list(EPM.lattice.symmetry_points.values())]

    x_list = [sp[0] for sp in sympts]
    y_list = [sp[1] for sp in sympts]
    z_list = [sp[2] for sp in sympts]
    
    for x,y,z,i in zip(x_list, y_list, z_list, sympt_labels):
        ax.text(x,y,z,i)
    
    plt.show()


def create_convergence_plot(EPM, ndivisions, exact_fl, improved, symmetry,
                            file_names, location, err_correlation=False,
                            convention="ordinary", degree=None):
    """Create a convergence plot of the total energy fermi level convergence for the
    free elecetron model.
    
    Args:
        ndivisions (list): a list of integers that gives the size of the grid.
        degree (int): the degree of the free electron dispersion relation.
        exact_fl (bool): if true fix the Fermi level at the exact value.
        improved (bool): if true include the improved tetrahedron method.
        symmetry (bool): if true, use symmetry reduction with tetrahedron method.
        EPM_name (str): the name of the pseudopotential, and the folder in which
            it is saved.
        file_names (list): a list of file name strings. The first corresponds to 
            the name of the ferme level plot; the second the total energy.
        location (str): the file path to where the plots are saved.
        err_correlation (bool): if true, generate a plot of Fermi level error against
            total energy error, and must include three strings in file names.
    """
    
    if err_correlation:
        if len(file_names) != 3:
            msg = "There must be three file names when error correlation is included."
            raise ValueError(msg.format(err_correlation))

    # Lists for storing errors
    rec_fl_err = []
    rec_te_err = []
    tet_fl_err = []
    tet_te_err = []
    ctet_te_err = []
    sym_rec_fl_err = []
    sym_rec_te_err = []
    sym_tet_fl_err = []
    sym_tet_te_err = []
    
    # Figures
    fermi_fig, fermi_axes = plt.subplots()
    energy_fig, energy_axes = plt.subplots()

    # Offsets for the tetrahedron method.
    lat_shift = [-1./2]*3
    grid_shift = [0,0,0]

    # Change the degree for the free electron model.
    if degree is not None:
        EPM.set_degree(degree)

    # Make the Fermi level exact if selected.
    if exact_fl:
        EPM.fermi_level = EPM.fermi_level_ans
    
    tot_timei = time.time()
    for ndivs in ndivisions:
        print("Divisions ", ndivs)
        t0 = time.time()
        
        # Create the grid for rectangles
        grid_consts = np.array(EPM.lattice.constants)*ndivs
        grid_angles = [np.pi/2]*3
        grid_centering = "prim"
        grid_vecs = make_ptvecs(grid_centering, grid_consts, grid_angles)
        rgrid_vecs = make_rptvecs(grid_vecs, convention)
        offset = np.dot(inv(rgrid_vecs), -np.sum(EPM.lattice.reciprocal_vectors, 1)/2) + (
                                                 [0.5]*3)
        # Calculate the Fermi level, if applicable, and total energy for the rectangular method.
        # Calculate percent error for each.
        grid = make_cell_points(EPM.lattice.reciprocal_vectors, rgrid_vecs, offset)
        weights = np.ones(len(grid), dtype=int)
        
        # Calculate errors using rectangle method with symmetry.
        if symmetry:
            sym_grid, sym_weights = find_orbits(grid, EPM.lattice.reciprocal_vectors,
                                                rgrid_vecs, offset)
            if not exact_fl:
                EPM.fermi_level = rectangular_fermi_level(EPM, sym_grid, sym_weights)
                sym_rec_fl_err.append( abs(EPM.fermi_level - 
                                           EPM.fermi_level_ans)/EPM.fermi_level_ans*100)
            EPM.total_energy = rectangular_method(EPM, sym_grid, list(sym_weights))
            sym_rec_te_err.append( abs(EPM.total_energy - 
                                       EPM.total_energy_ans)/EPM.total_energy_ans*100)

        # Rectangle Fermi level
        if not exact_fl:
            EPM.fermi_level = rectangular_fermi_level(EPM, grid, weights)
            print("rectangles: ", EPM.fermi_level)
            rec_fl_err.append( abs(EPM.fermi_level - EPM.fermi_level_ans)/EPM.fermi_level_ans*100)
            
        # Rectangle Total energy
        EPM.total_energy = rectangular_method(EPM, grid, weights)
        print("rectangles te: ", EPM.total_energy)
        # rec_te.append(EPM.total_energy)
        rec_te_err.append( abs(EPM.total_energy - EPM.total_energy_ans)/EPM.total_energy_ans*100)

        # Calculate grid, tetrahedra, and weights for tetrahedron method.
        grid, tetrahedra = grid_and_tetrahedra(EPM, ndivs, lat_shift, grid_shift)
        weights = np.ones(len(tetrahedra), dtype=int)

        # Calculate errors using tetrahedron method and symmetry reduction.
        if symmetry:
            irr_tet, tet_weights = find_irreducible_tetrahedra(EPM, tetrahedra, grid)
            if not exact_fl:
                EPM.fermi_level = calc_fermi_level(EPM, irr_tet, tet_weights, grid, tol=1e-8)
                sym_tet_fl_err.append( abs(EPM.fermi_level - EPM.fermi_level_ans)/EPM.fermi_level_ans*100)
        
            EPM.total_energy = calc_total_energy(EPM, irr_tet, tet_weights, grid)
            sym_tet_te_err.append( abs(EPM.total_energy - EPM.total_energy_ans)/EPM.total_energy_ans*100)

        if not exact_fl:
            EPM.fermi_level = calc_fermi_level(EPM, tetrahedra, weights, grid, tol=1e-8)
            print("tetrahedra: ", EPM.fermi_level)
            tet_fl_err.append( abs(EPM.fermi_level - EPM.fermi_level_ans)/EPM.fermi_level_ans*100)

        # Calculate the error for the tetrahedron method.
        EPM.total_energy = calc_total_energy(EPM, tetrahedra, weights, grid)
        print("tetrahedra te ", EPM.total_energy)
        tet_te_err.append( abs(EPM.total_energy - EPM.total_energy_ans)/EPM.total_energy_ans*100)

        # Calculate the error for the corrected tetrahedron method.
        if improved:
            ndiv0 = [ndivs]*3
            extended_grid, extended_tetrahedra = get_extended_tetrahedra(EPM, ndivs, lat_shift, grid_shift)
            EPM.total_energy = get_corrected_total_energy(EPM, tetrahedra, extended_tetrahedra,
                                                         grid, extended_grid, ndiv0)
            ctet_te_err.append( abs(EPM.total_energy - EPM.total_energy_ans)/EPM.total_energy_ans*100)
        print("run time", time.time() - t0)            
    
    # Location where plots are saved.
    loc = os.path.join(location, EPM.material)
    
    # Plot errors.
    if not exact_fl:
        fermi_axes.loglog(np.array(ndivisions)**3, rec_fl_err,label="Rectangles")
        fermi_axes.loglog(np.array(ndivisions)**3, tet_fl_err,label="Tetrahedra")
        if symmetry:
            fermi_axes.loglog(np.array(ndivisions)**3, sym_rec_fl_err,label="Reduced Rectangles")
            fermi_axes.loglog(np.array(ndivisions)**3, sym_tet_fl_err,label="Reduced Tetrahedra")
        
        fermi_axes.set_title(EPM.material + " Fermi Level Convergence")
        fermi_axes.set_xlabel("Number of k-points")
        fermi_axes.set_ylabel("Percent Error")
        fermi_axes.legend(loc="best")    
        fermi_fig_name = os.path.join(loc, file_names[0] + ".pdf")
        fermi_fig.savefig(fermi_fig_name)

    print("rectangles ", rec_te_err)
    print("tetrahedra ", tet_te_err)
    energy_axes.loglog(np.array(ndivisions)**3, rec_te_err,label="Rectangles")
    energy_axes.loglog(np.array(ndivisions)**3, tet_te_err,label="Tetrahedra")
    
    if improved:
        energy_axes.loglog(np.array(ndivisions)**3, ctet_te_err,label="Improved Tetrahedra")
    
    if symmetry:
        energy_axes.loglog(np.array(ndivisions)**3, sym_rec_te_err,label="Reduced Rectangles")
        energy_axes.loglog(np.array(ndivisions)**3, sym_tet_te_err,label="Reduced Tetrahedra")
    
    energy_axes.set_title(EPM.material + " Total Energy Convergence")
    energy_axes.set_xlabel("Number of k-points")
    energy_axes.set_ylabel("Percent Error")
    energy_axes.legend(loc="best")
    energy_fig_name = os.path.join(loc, file_names[1] + ".pdf")
    energy_fig.savefig(energy_fig_name)
    
    if (not exact_fl) and err_correlation:
        corr_fig = plt.figure()
        corr_axes = corr_fig.add_subplot(1,1,1)
        
        corr_axes.scatter(rec_fl_err, rec_te_err, label="Rectangles")
        corr_axes.scatter(tet_fl_err, tet_te_err, label="Tetrahedra")

        corr_axes.set_xscale("log")
        corr_axes.set_yscale("log")
        corr_axes.set_title(EPM.material + " Error Correlation")
        corr_axes.set_xlabel("Percent Error Fermi Level")
        corr_axes.set_ylabel("Percent Error Total Energy")
        corr_axes.legend(loc="best")
        corr_fig_name = os.path.join(loc, file_names[2] + ".pdf")
        corr_fig.savefig(corr_fig_name)

    tot_timef = time.time()
    print("total elapsed time: ", (tot_timef - tot_timei)/60, " minutes")


def plot_states(EPM, grid, tetrahedra, weights, method, energy_list, quantity, nbands, answer,
                           title, xlimits, ylimits, labels, bin_size=0.1, show=True, save=False,
                           root_dir=None):
    """Plot the density of states and the correct density of states.

    Args:
        EPM (:py:obj:`BZI.pseudopots.EmpiricalPseudopotential`): an instance of a pseudopotential class.
        grid (numpy.ndarray): a list of grid point over which the density of states is calculated.
        tetrahedra (list or numpy.ndarray): a list of tetrahedra given district labels by a quadruple
            of grid indices
        weights (numpy.ndarray or list): a list of tetrahedron weights.
        method (str): the method used to calculate the density of states.
        energy_list (list): a list of energies at which to calculate the density of states.
        quantity (str): the quantity to plot. Can be density or number of states.
        nbands (int): the number of bands included in the calculation.
        answer (function): a function of energy that returns the exact density of states.
        title (str): the title of the plot.
        xlimits (tuple): the x-axis limits.
        ylimits (tuple): the y-axis limits.
        labels (tuple): the labels for the plots, first comes the calculated DOS label.
        bin_size (float): the size of the energy bins. Only applicable to rectangles.
        show (bool): display the density of states.
        save (str): save the plot with this file name. If not provided, the plot isn't saved. The
            string must include the file format, such as .pdf or .png.
        root_dir (str): the root directory where the plot is saved.
    """

    if method == "rectangles":        
        energies = np.sort(np.array([EPM.eval(g, nbands) for g in grid]).flatten())
        dE = bin_size
        dV = EPM.lattice.reciprocal_volume/len(grid)
        V = EPM.lattice.reciprocal_volume
        Ei = 0
        Ef = 0
        dos = [] # density of states
        nos = [] # number of states
        dos_energies = [] # energies
        while max(energies) > Ef:
            Ef += dE
            dos_energies.append(Ei + (Ef-Ei)/2.)
            dos.append( len(energies[(Ei <= energies) & (energies < Ef)])/(len(grid)*dE)*2)
            nos.append(np.sum(dos)*dE)
            Ei += dE
        if EPM.degree:
            answer_list = [answer(en, EPM.degree) for en in energy_list]
        else:
            answer_list = [answer(en) for en in energy_list]

        if quantity == "dos":
            plt.scatter(dos_energies, dos, label=labels[0], c="blue")
            plt.ylabel("Density of States")
        elif quantity == "nos":
            plt.scatter(dos_energies, nos, label=labels[0], c="blue")
            plt.ylabel("Number of States")
        else:
            msg = "The supported quantities are dos and nos."
            raise ValueError(msg.format(quantity))
            
        plt.plot(energy_list, answer_list, label=labels[1], c="black")
        plt.xlabel("Energy (eV)")
        plt.title(title)
        plt.xlim(xlimits[0], xlimits[1])
        plt.ylim(ylimits[0], ylimits[1])
        plt.legend(loc="best")
        if save:
            plt.savefig(root_dir + save)
        elif show:
            plt.show()

    elif method == "tetrahedra":
        VG = EPM.lattice.reciprocal_volume
        VT = VG/len(weights)
        dos = np.zeros(len(energy_list))
        nos = np.zeros(len(energy_list))


        if quantity == "dos":
            for i,energy in enumerate(energy_list):
                for tet in tetrahedra:
                    for band in range(nbands):
                        tet_energies = np.sort([EPM.eval(grid[j], nbands)[band] for j in tet])
                        dos[i] += density_of_states(VG, VT, tet_energies, energy)

        elif quantity == "nos":
            for i,energy in enumerate(energy_list):
                for tet in tetrahedra:
                    for band in range(nbands):
                        tet_energies = np.sort([EPM.eval(grid[j], nbands)[band] for j in tet])
                        nos[i] += number_of_states(VG, VT, tet_energies, energy)
        else:
            msg = "The supported quantities are dos and nos."
            raise ValueError(msg.format(quantity))
                                
        if EPM.degree:
            answer_list = [answer(en, EPM.degree) for en in energy_list]
        else:
            answer_list = [answer(en) for en in energy_list]
            
        if quantity == "dos":
            plt.plot(energy_list, dos, label=labels[0], c="blue")
            plt.ylabel("Density of States")
        else:
            plt.plot(energy_list, nos, label=labels[0], c="blue")
            plt.ylabel("Number of States")            
        plt.plot(energy_list, answer_list, label=labels[1], c="black")
        plt.xlabel("Energy (eV)")
        plt.xlim(xlimits[0], xlimits[1])
        plt.ylim(ylimits[0], ylimits[1])
        plt.title(title)
        plt.legend(loc="best")
        if save:
            plt.savefig(root_dir + save)
        elif show:
            plt.show()
        else:
            None
    else:
        msg = "The supported methods are rectangles and tetrahedra."
        raise ValueError(msg.format(methods))


def generate_states_data(EPM, nbands, grid, methods_list, weights_list,
                         energy_list, file_location, file_number,
                         bin_size=0.1, tetrahedra=None):

    """Generate data needed to plot the density of states and number of states
    of a pseudopotential.

    Args:
        EPM (obj): an instance of a pseudopotential class.
        nbands (int): the number of bands included in the calculation.
        grid (list or numpy.ndarray): a list of grid points over which the quantity
            provided is calculated.
        methods_list (str): a list of methods used to calculate the provided quantity.
        weights_list (list): a list of symmetry reduction weights. Must be in the same
            order as methods_list.
        energy_list (list): a list of energies at which the provided quantity is
            calculated.
        file_location (str): the file path to where the data is saved. Exclude the last
            backslash.
        file_number (str): the number of the file. This should avoid overwriting 
            previous data.
        quantity (str): the quantity whose data is saved.
        function_list (list): a list of functions that provide the exact value of the
            quantities provide. It must be in the same order as quantity_list.
        bin_size (float): the bin size for the rectangular method.
        tetrahedra (list): a list of tetrahedra vertices.
        weights (list): a list of tetrahedron weights.
    """
    
    # The file structure is as follows: potential/potential_variations/valency_#
    # There may not be any potential variations, such as different degrees for the
    # free electron model.
    file_prefix = file_location + "/data"
    # If the data folder doesn't exist, make one.
    if not os.path.isdir(file_prefix):
        os.mkdir(file_prefix)
    
    file_prefix += "/" + EPM.name
    # If the pseudopotential folder doesn't exist, make one.
    if not os.path.isdir(file_prefix):
        os.mkdir(file_prefix)
    
    # If the variation of the potential folder doesn't exist, make one.
    if EPM.degree:
        file_prefix += "/degree_" + str(EPM.degree)
        if not os.path.isdir(file_prefix):
            os.mkdir(file_prefix)
    
    # If the considered valency folder doesn't exist, make one.
    file_prefix += "/" + "valency_" + str(EPM.nvalence_electrons)
    if not os.path.isdir(file_prefix):
        os.mkdir(file_prefix)
        
    # Generate the energies, density of states, and number of states with
    # the rectangular method if it is one of the methods provided.
    if "rectangles" in methods_list:
        rec_weights = weights_list[0]
        # Remove rectangles from methods_list.
        del methods_list[methods_list == "rectangles"]
        
        # Make sure that the method folder exists. If not, make one.
        rec_prefix = file_prefix + "/rectangles"
        if not os.path.isdir(rec_prefix):
            os.mkdir(rec_prefix)
        
        # The energies of the potential at each point in the grid.
        rec_energies = np.array(list(chain(*[EPM.eval(grid[i], nbands)*rec_weights[i]
                                             for i in range(len(grid))])))
        energies, dos, nos = rec_dos_nos(rec_energies, nbands, bin_size)

        # Generate and save the exact values of the quantities provided at the energies
        # provided.
        exact_dos_list = [EPM.density_of_states(en) for en in energy_list]
        exact_nos_list = [EPM.number_of_states(en) for en in energy_list]

        # Add all these quantities to the data dictionary and pickle it.
        data = {}
        data["energies"] = energy_list
        data["binned energies"] = energies
        data["density of states"] = dos
        data["number of states"] = nos
        data["analytic density of states"] = exact_dos_list
        data["analytic number of states"] = exact_nos_list
        with open(rec_prefix + "/run_" + file_number + ".p", "w") as file:
            pickle.dump(data, file)

    # Generate the energies, density of states, and number of states with
    # the rectangular method if it is one of the methods provided.
    if "tetrahedra" in methods_list:        
        # Remove rectangles from methods_list.
        del methods_list[methods_list == "tetrahedra"]

        # The tetrahedral weights should always come second in the weights list.
        if len(weights_list) > 1:
            tet_weights = weights_list[1]
        else:
            tet_weights = weights_list[0]
        
        # Make sure that the method folder exists. If not, make one.
        tet_prefix = file_prefix + "/tetrahedra"
        if not os.path.isdir(tet_prefix):
            os.mkdir(tet_prefix)

        # Generate and save the exact values of the quantities provided at the energies
        # provided, as well as the numerical values.
        energies, dos, nos = tet_dos_nos(EPM, nbands, grid, energy_list, tetrahedra,
                                         tet_weights)        
        exact_dos_list = [EPM.density_of_states(en) for en in energy_list]
        exact_nos_list = [EPM.number_of_states(en) for en in energy_list]
                
        data = {}
        data["energies"] = energy_list
        data["density of states"] = dos
        data["number of states"] = nos
        data["analytic density of states"] = exact_dos_list
        data["analytic number of states"] = exact_nos_list
        with open(tet_prefix + "/run_" + file_number + ".p", "w") as file:
            pickle.dump(data, file)

    if methods_list != []:
        msg = "The allowed methods are 'rectangles' and 'tetrahedra'."
        raise ValueError(msg.format(methods_list))
                
    
def plot_states_data(EPM, file_location, file_number, file_name, quantity, method,
                     title, xlimits, ylimits, labels):
    """Plot the density of states or number of states of a pseudopotential with
    data retrieved from file.

    Args:
        file_location (str): the file path to where the data is saved.
        file_number (str): the number of the file. This should avoid overwriting
            previous data.
        quantity (str): the quantity whose data is plotted.
        method (str): the method used to generate the data.
        title (str): the title of the plot.
        xlimits (tuple): the x-axis limits.
        ylimits (tuple): the y-axis limits.
        labels (tuple): the labels for the plots, first comes the analytic label, and then
            numeric label.
        save_location (str): the location where the plot is saved.
        save_name (str): the file name of the plot.
    """
    
    
    if EPM.degree:
        data_file = (file_location + "/data/" + EPM.name + "/degree_" + str(EPM.degree) +
                         "/" "valency_" + str(EPM.nvalence_electrons) + "/" + method +
                     "/run_" + str(file_number) + ".p")
    else:
        data_file = (file_location + "/data/" + EPM.name + "/degree_" + "valency_" +
                         str(EPM.nvalence_electrons) + "/" + method + "/run_" +
                     str(file_number) + ".p")
        
    file_prefix = file_location + "/plots"
    # If the plots directory doesn't exist, make one.
    if not os.path.isdir(file_prefix):
        os.mkdir(file_prefix)

    file_prefix += "/" + EPM.name    
    # If the pseudopotential folder doesn't exist, make one.
    if not os.path.isdir(file_prefix):
        os.mkdir(file_prefix)

    # If the variation of the potential folder doesn't exist, make one.
    if EPM.degree:
        file_prefix += "/degree_" + str(EPM.degree)
        if not os.path.isdir(file_prefix):
            os.mkdir(file_prefix)

    # If the considered valency folder doesn't exist, make one.
    file_prefix += "/" + "valency_" + str(EPM.nvalence_electrons)
    if not os.path.isdir(file_prefix):
        os.mkdir(file_prefix)

    # Get the dictionary with the data.
    print("data", data_file)
    data = pickle.load(open(data_file, "r"))
    
    
    if method == "rectangles":
        plt.scatter(data["binned energies"], data[quantity], label=labels[0], c="blue")
    else:
        plt.scatter(data["energies"], data[quantity], label = labels[0], c="blue")

    plt.plot(data["energies"], data["analytic " + quantity], label=labels[1], c="black")
    
    plt.xlabel("Energy (eV)")


    ylabel_dict = {"density of states": "Density of States",
                   "number of states": "Number of States"}
    plt.ylabel(ylabel_dict[quantity])
    plt.title(title)
    plt.xlim(xlimits[0], xlimits[1])
    plt.ylim(ylimits[0], ylimits[1])
    plt.legend(loc="best")
    plt.savefig(file_prefix + "/" + file_name + ".pdf")


def plot_simplex_edges(vertices, axes, color="blue"):
    """Plot the edges of a 3-simplex.

    Args:
        vertices (numpy.ndarray): a list of simplex vertices.
        axes (matplotlib.axes)
    """

    if len(vertices) == 3:
        vertices = np.append(vertices, [vertices[0]], axis=0)
    
    for i in range(len(vertices)-1):
        xstart = vertices[i][0]
        xfinish = vertices[i+1][0]
        xs = np.linspace(xstart, xfinish, 100)
        ystart = vertices[i][1]
        yfinish = vertices[i+1][1]
        ys = np.linspace(ystart, yfinish, 100)
        zstart = vertices[i][2]
        zfinish = vertices[i+1][2]
        zs = np.linspace(zstart, zfinish, 100)
        axes.plot(xs, ys, zs, c=color)


def plot_bz(bz, symmetry_points=None, remove=True, ax=None, color="blue"):
    """Plot a Brillouin zone
    
    Args:
        BZ (scipy.spatial.ConvexHull): a convex hull object.
        symmetry_points (dict): a dictionary of symmetry points in Cartesian coordinates.
        remove (bool): if True, plot the facets instead of the simplices that make up the
            boundary of the Brillouin zone or irreducible Brilloun zone.
        ax (matplotlib.axes): an axes object.
    """

    fig = plt.figure()
    if ax == None:
        ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    
    if symmetry_points != None:
        eps = np.average(list(symmetry_points.values()))*0.05
        for spt in symmetry_points.keys():
            coords = symmetry_points[spt]
            ax.scatter(coords[0], coords[1], coords[2], c="black")
            ax.text(coords[0] + eps, coords[1] + eps, coords[2] + eps, spt, size=14)

    if remove:
        facet_list = []
        equations = list(deepcopy(bz.equations))
        simplices = list(deepcopy(bz.points[bz.simplices]))
        while len(equations) != 0:
            equation, equations = equations[-1], equations[:-1]            
            facet, simplices = simplices[-1], simplices[:-1]
            indices = find_point_indices(equation, equations)
            
            if len(indices) > 0:
                # Remove duplicate equations from the list of equations
                equations_to_remove = [equations[i] for i in indices]
                for eq in equations_to_remove:
                    equations = remove_points(equation, equations)
                
                # Find all simplices that lie on the same plane to get the facet.
                simplices_to_remove = []
                for index in indices:
                    facet = np.append(facet, simplices[index], axis=0)                    
                    simplices_to_remove.append(simplices[index])
                
                for s in simplices_to_remove:
                    simplices = remove_points(s, simplices)
                
                # Remove duplicate points on facet.
                unique_facet = []
                for pt in facet:
                    if not check_contained(pt, unique_facet):
                        unique_facet.append(pt)
                facet_list.append(orderAngle(unique_facet))
            else:
                facet_list.append(orderAngle(facet))

        
        for facet in facet_list:
            # We want to plot all the edges, so we append the last vertex to the
            # beginning of the facet.
            facet = np.append(facet, [facet[0]], axis=0)
            plot_simplex_edges(facet, ax, color=color)
    else:
        for simplex in bz.simplices:
            # We're going to plot lines between the vertices of the simplex.
            # To make sure we make it all the way around, append the first element
            # to the end of the simplex.
            simplex = np.append(simplex, simplex[0])
            simplex_pts = [bz.points[i] for i in simplex]
            plot_simplex_edges(simplex_pts, ax, color=color)
    plt.close()


def plot_all_bz(lat_type, lat_vecs, grid=None, sympts=None, ax=None, convention="ordinary"):
    """Plot the Brillouin zone and optionally the irreducible Brillouin zone and points
    within the Brillouin zone.

    Args:
        lat_type (str): the lattice type, such as 'simple cubic'
        lat_vecs (numpy.ndarray): a 3x3 array with lattice vectors as columns.
        grid (list or numpy.ndarray): a list of list or 2D array of points to plot.
        sympts (list or numpy.ndarray): a dictionary whose key is the Greek or Roman
            letter representing the symmetry point. The corresponding value is the 
            coordinate of the point in lattice coordinates.
        ax (matplotlib.axes): an axes object.
        convention (str): the convention for finding the reciprocal lattice vectors.
            Options include 'ordinary' and 'angular'.
    """
    
    rlat_vecs = make_rptvecs(lat_vecs, convention=convention)
    bz = find_bz(rlat_vecs)
    plot_bz(bz, ax=ax)    
    
    if sympts is not None:
        # Get the vertices of the IBZ.
        ibz_vertices = list(sympts.values())
        ibz_vertices = [np.dot(rlat_vecs, v) for v in ibz_vertices]
        
        # Get the symmetry points in Cartesian coordinates.
        sympts_cart = {}
        for spt in sympts.keys():
            sympts_cart[spt] = np.dot(rlat_vecs, sympts[spt])

        ibz = ConvexHull(ibz_vertices)
        plot_bz(ibz, sympts_cart, ax=ax, color="red")

    if grid is not None:
        plot_just_points(grid, ax)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)            

            
def plot_vecs(vecs, colors, labels):
    """Plot a list of vectors."""

    xmin = min([vecs[i][0] for i in range(len(vecs))])
    xmax = max([vecs[i][0] for i in range(len(vecs))])
    ymin = min([vecs[i][1] for i in range(len(vecs))])
    ymax = max([vecs[i][1] for i in range(len(vecs))])
    zmin = min([vecs[i][2] for i in range(len(vecs))])
    zmax = max([vecs[i][2] for i in range(len(vecs))])
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    for i,v in enumerate(vecs):
        arrow = Arrow3D([0,v[0]], [0,v[1]], [0,v[2]],
                        mutation_scale=20, lw=3,
                        arrowstyle="-|>", color=colors[i])

        ax.text(v[0], v[1], v[2], labels[i])
        ax.add_artist(arrow)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_zlim(zmin,zmax)
    plt.legend()
    plt.show()
