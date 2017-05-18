"""Plot various quantities related to electronic stucture integration.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
from BZI.pseudopots import ToyPP, W1
from BZI.symmetry import (bcc_sympts, fcc_sympts, sc_sympts, make_rptvecs,
                          sym_path)

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
        >>> from BZI import ScatterPlotToyPP
        >>> nstates = 2
        >>> ndivisions = 11
        >>> ScatterPlotToyPP(nstates, ndivisions)
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
        estates, kxlist, kylist = zip(*filter(lambda x: x[0] <= cutoff, zip(estates, kxlist, kylist)))
        ax = plt.subplot(1,1,1,projection="3d")
        ax.scatter(kxlist, kylist, estates,s=.5);
    plt.show()
    
def PlotMesh(mesh_points, cell_vecs, offset = np.asarray([0.,0.,0.])):
    """Create a 3D scatter plot of a set of mesh points inside a cell.
    
    Args:
        mesh_points (list or np.ndarray): a list of mesh points.
        cell_vecs (list or np.ndarray): a list vectors that define a cell.
        
    Returns:
        None
    """
    
    ngpts = len(mesh_points)
    kxlist = [mesh_points[i][0] for i in range(ngpts)]
    kylist = [mesh_points[i][1] for i in range(ngpts)]
    kzlist = [mesh_points[i][2] for i in range(ngpts)]

    ax = plt.subplot(1,1,1,projection="3d")
    ax.scatter(kxlist, kylist, kzlist, c="red")
    
    c1 = cell_vecs[:,0] 
    c2 = cell_vecs[:,1] 
    c3 = cell_vecs[:,2] 
    O = np.asarray([0.,0.,0.]) 

    l1 = zip(O - offset, c1 - offset)
    l2 = zip(c2 - offset, c1 + c2 - offset)
    l3 = zip(c3 - offset, c1 + c3 - offset)
    l4 = zip(c2 + c3 - offset, c1 + c2 + c3 - offset)
    l5 = zip(O - offset, c3 - offset)
    l6 = zip(c1 - offset, c1 + c3 - offset)
    l7 = zip(c2 - offset, c2 + c3 - offset)
    l8 = zip(c1 + c2 - offset, c1 + c2 + c3 - offset)
    l9 = zip(O - offset, c2 - offset)
    l10 = zip(c1 - offset, c1 + c2 - offset)
    l11 = zip(c3 - offset, c2 + c3 - offset)
    l12 = zip(c1 + c3 - offset, c1 + c2 + c3 - offset)

    ls = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]

    for l in ls:
        ax.plot3D(*l, c="blue")
    plt.show()
    return None

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

    l1 = zip(O - offset, c1 - offset)
    l2 = zip(c2 - offset, c1 + c2 - offset)
    l3 = zip(c3 - offset, c1 + c3 - offset)
    l4 = zip(c2 + c3 - offset, c1 + c2 + c3 - offset)
    l5 = zip(O - offset, c3 - offset)
    l6 = zip(c1 - offset, c1 + c3 - offset)
    l7 = zip(c2 - offset, c2 + c3 - offset)
    l8 = zip(c1 + c2 - offset, c1 + c2 + c3 - offset)
    l9 = zip(O - offset, c2 - offset)
    l10 = zip(c1 - offset, c1 + c2 - offset)
    l11 = zip(c3 - offset, c2 + c3 - offset)
    l12 = zip(c1 + c3 - offset, c1 + c2 + c3 - offset)

    ls = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]

    for l in ls:
        ax.plot3D(*l, c="black")
    plt.show()
    return None

def PlotBandStructure(materials_list, PPlist, PPargs_list, lat_type, lat_const,
                      sympt_pairs, npts, neigvals, energy_shift, elimits=False,
                      fermi_level=False, save=False, show=True):
    """Plot the band structure of a pseudopotential along symmetry paths.
    
    Args:
        materials_list (str): a list of materials whose bandstructures will be 
            plotted. The first string will label figures and files.
        PPlist (function): a list of pseudopotenial functions.
        PPargs_list (list): a list of pseudopotential arguments as dictionaries.
        lat_type (str): the lattice type.
        lat_const (float): the lattice constant.
        sympt_pairs (list or numpy.ndarray): a list of symmetry point pairs.
        npts (int): the number of points to plot along each symmetry line.
        neigvals (int): the number of lower-most eigenvalues to include 
            in the plot.
        energy_shift (float): energy shift for band structure
        elimits (list): the energy window for the plot.
        fermi_level (float): if provided, the Fermi level will be included
            in the plot.
        save (bool): if true, the band structure will be saved.
    Returns:
        Display or save the band structure.
    """
    
    rlat_vecs = make_rptvecs(lat_type, lat_const)

    # k-points between symmetry point pairs in lattice coordinates.
    lat_kpoints = sym_path(lat_type, npts, sympt_pairs)

    # k-points between symmetry point pairs in cartesian coordinates.
    car_kpoints = [np.dot(rlat_vecs,k) for k in lat_kpoints]

    # Store the energy eigenvalues in an nested array.
    nPP = len(PPlist)
    energies = [[] for i in range(nPP)]
    for i in range(nPP):
        PP = PPlist[i]
        PPargs = PPargs_list[i]
        PPargs["neigvals"] = neigvals
        for kpt in car_kpoints:
            PPargs["kpoint"] = kpt
            energies[i].append(PP(**PPargs) - energy_shift)

    # Grab the symmetry points.
    if lat_type == "fcc":
        sympt_dict = fcc_sympts
    elif lat_type == "bcc":
        sympt_dict = bcc_sympts
    elif lat_type == "sc":
        sympt_dict = sc_sympts
    else:
        raise ValueError("Invalid lattice type")

    # Flatten the list of symmetry point pairs.
    sympts = [item for sublist in sympt_pairs for item in sublist]
    # Remove all duplicate symmetry points
    unique_sympts = [sympts[i] for i in range(0, len(sympts), 2)] + [sympts[-1]]
    # Replace symbols representing points with their lattice coordinates.
    lat_sympt_coords = [sympt_dict[sp] for sp in unique_sympts]
    # Put the symmetry points in cartesian coordinates
    car_sympt_coords = [np.dot(rlat_vecs,k) for k in lat_sympt_coords]

    # Find the distances between symmetry points.
    nsympts = len(unique_sympts)
    sympt_dist = [0] + [np.linalg.norm(car_sympt_coords[i+1]
                                       - car_sympt_coords[i])
                        for i in range(nsympts - 1)]

    # Create coordinates for plotting.
    lines = []
    for i in range(nsympts - 1):
        start = np.sum(sympt_dist[:i+1])
        stop = np.sum(sympt_dist[:i+2])
        if i == (nsympts - 2):
            lines += list(np.linspace(start, stop, npts))            
        else:
            lines += list(np.delete(np.linspace(start, stop, npts),-1))

    # Plot the energy dispersion curves one at a time.
    colors = ["green", "blue", "red", "violet", "orange", "cyan", "black"]
    for i in range(nPP):        
        for ne in range(neigvals):
            ienergy = []
            for nk in range(len(car_kpoints)):
                ienergy.append(energies[i][nk][ne])
            if ne == 0:
                plt.plot(lines, ienergy, color=colors[i], label="%s"%materials_list[i])
            else:
                plt.plot(lines, ienergy, color=colors[i])

    # Plot the Fermi level if provided.
    if type(fermi_level) != bool:
        plt.axhline(y = fermi_level, c="yellow", label="Fermi level")

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
    
    # Adjust the legend.
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim([0,np.sum(sympt_dist)])
    plt.xlabel("Symmetry points")
    plt.ylabel("Energy (eV)")
    plt.title("%s Band Structure" %materials_list[0])
    plt.grid(linestyle="dotted")
    if save:
        plt.savefig("%s_band_struct.pdf" %materials_list[0],
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
    if show:
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
    
    # Plot the points within the sphere.
    ngpts = len(mesh_points)
    kxlist = [mesh_points[i][0] for i in range(ngpts)]
    kylist = [mesh_points[i][1] for i in range(ngpts)]
    kzlist = [mesh_points[i][2] for i in range(ngpts)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.scatter(kxlist, kylist, kzlist, c="black",s=1)
    
    # Plot the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    r = np.sqrt(r2)
    x = r * np.outer(np.cos(u), np.sin(v)) + offset[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + offset[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + offset[2]
    
    ax.scatter(x,y,z,s=0.001)
    if save:
        plt.savefig("sphere_mesh.png")
    if show:
        plt.show()
    return None

def PlotVaspBandStructure(file_loc, material, lat_type, energy_shift=0.0,
                          fermi_level=False, elimits=False, save=False,
                          show=True):
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
    elif lat_type == "bcc":
        sympt_dict = bcc_sympts
    elif lat_type == "sc":
        sympt_dict = sc_sympts
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
    rlat_vecs = make_rptvecs(lat_type, lat_const)
    
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
        nbands = int(nbands)

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
                        en_occ[nkpts_dr-1].append(energies[nkpts_dr-1][-1]*occupancies[nkpts_dr-1][-1]/2)
    car_kpoints = [np.dot(rlat_vecs,k) for k in lat_kpoints]

    # Find the distances between symmetry points.
    nsympts = len(unique_sympts)
    sympt_dist = [0] + [np.linalg.norm(car_sympt_coords[i+1]
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
            plt.plot(lines,ienergy, label="VASP Band structure",color="black")
        else:
            plt.plot(lines,ienergy,color="black")

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
        plt.savefig("%s_band_struct.pdf" %materials,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
    elif show:
        plt.show()
    else:
        return None
