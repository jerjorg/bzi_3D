"""Methods to for calculating and visualizing lattices and Brillouin zones in 2D."""

from numpy.linalg import norm, inv, det
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
from copy import deepcopy
from scipy.spatial import ConvexHull

from BZI.symmetry import get_minmax_indices, bring_into_cell
from BZI.utilities import remove_points
plt.style.use('seaborn-colorblind')

def make2D_lattice_basis(lattice_constants, lattice_angle):
    """Create the basis vectors that generate a lattice in 2D.
    
    Args;
        lattice_constants (numpy.ndarray): the lengths of the basis vectors.
        lattice_angle (float): the angle between the lattice vectors in radians.
        
    Returns:
        basis (numpy.ndarray): the basis vectors of the lattice as columns of a 
            2x2 array.
    """
    
    v1 = lattice_constants[0]*np.array([1, 0])
    v2 = np.array([lattice_constants[1]*np.cos(lattice_angle),
                   lattice_constants[1]*np.sin(lattice_angle)])
    basis = np.transpose([v1, v2])
    return basis


def get_2Dlattice_type(basis, rtol=1e-5, atol=1e-8):
    """Find the lattice type from the lattice generating vectors in 2D.
    
    Args:
        basis (numpy.ndarray): the lattice generating vectors as columns of a
            2D array.
    Returns
        (str): the lattice type
    """
    
    v1 = basis[:,0]
    v2 = basis[:,1]
    
    lattice_constants = [norm(v1), norm(v2)]
    lattice_angle = np.arccos(np.dot(v1, v2)/np.prod(lattice_constants))
    
    # If the length of the lattice vectors is the same
    if np.isclose(lattice_constants[0], lattice_constants[1], rtol=rtol, atol=atol):
        # If the lattice angle is pi/2
        if np.isclose(lattice_angle, np.pi/2, rtol=rtol, atol=atol):
            return "square"
        # If the lattice angle is pi/3 or 2pi/3
        elif (np.isclose(lattice_angle, np.pi/3, rtol=rtol, atol=atol) or
              np.isclose(lattice_angle, 2*np.pi/3, rtol=rtol, atol=atol)):
            return "hexagonal"
        else:
            return "rhombic"
    # If the lattice vectors are different lengths, and the lattice
    # angle is pi/2.
    elif np.isclose(lattice_angle, np.pi/2, rtol=rtol, atol=atol):
        return "rectangular"
    else:
        if lattice_constants[0] < lattice_constants[1]:
            if np.isclose(np.dot(v1, v2/lattice_constants[1]), lattice_constants[1]/2,
                          rtol=rtol, atol=atol):
                return "centered rectangular"
            else:
                return "oblique"
        else:
            if np.isclose(np.dot(v2, v1/lattice_constants[0]), lattice_constants[0]/2,
                          rtol=rtol, atol=atol):
                return "centered rectangular"
            else:
                return "oblique"

        
def HermiteNormalForm2D(S, rtol=1e-5, atol=1e-6):
    """Find the Hermite normal form of a 2x2 matrix as well as the unimodular
    matrix that mediates the transform.
    
    Args:
        S (numpy.ndarray): a 2x2 array of integers
        eps (int): finite precision parameter
        
    Returns:
        B (numpy.ndarray): the unimodular transformation matrix
        H (numpy.ndarray): the Hermite normal form of the integer matrix
    """
        
    # Make sure the input is integer.
    if not (np.allclose(np.max(S%1), 0, rtol=rtol, atol=atol) or
            np.allclose(np.min(S%1), 0, rtol=rotl, atol=atol)):
        msg = "Please provide an integer 2x2 array."
        raise ValueError(msg.format(S))
    
    H = deepcopy(S)
    B = np.eye(2,2)
    
    # Make sure the elements in the first row are positive.
    if H[0,1] < 0:
        H[:,1] *= -1
        B[:,1] *= -1
    if H[0,0] < 0:
        H[:,0] *= -1
        B[:,0] *= -1        
        
    # Subract to two columns from each other until one element in
    # the first row becomes zero.
    while np.count_nonzero(H[0,:]) > 1:
        if H[0,1] > H[0,0]:
            H[:,1] -= H[:,0]
            B[:,1] -= B[:,0]
        else:
            H[:,0] -= H[:,1]
            B[:,0] -= B[:,1]
            
    if not np.allclose(np.dot(S,B), H, rtol=rtol, atol=atol):
        msg = "The transformation isn't working"
        raise ValueError(msg.format(S))                    
    
    # If the zero ends up in the wrong place, swap columns.
    if np.isclose(H[0,0], 0, rtol=rtol, atol=atol):
        tmp_column = deepcopy(H[:,0])
        H[:,0] = H[:,1]
        H[:,1] = tmp_column            
    
        tmp_column = deepcopy(B[:,0])
        B[:,0] = B[:,1]
        B[:,1] = tmp_column            
    
    if not np.allclose(np.dot(S,B), H, rtol=rtol, atol=atol):
        msg = "The transformation isn't working"
        raise ValueError(msg.format(S))

    if H[1,1] < 0:
        H[:,1] *= -1
        B[:,1] *= -1
        
    while H[1,0] < 0 or (H[1,0] >= H[1,1]):
        if H[1,0] < 0:
            H[:,0] = H[:,0] + H[:,1]
            B[:,0] = B[:,0] + B[:,1]
        else:
            H[:,0] = H[:,0] - H[:,1]
            B[:,0] = B[:,0] - B[:,1]
            
    if not np.allclose(np.dot(S,B), H, rtol=rtol, atol=atol):
        msg = "The transformation isn't working"
        raise ValueError(msg.format(S))
        
    return H, B
        

def make_cell_points2D(lat_vecs, grid_vecs, offset=[0,0], cart=True, rtol=1e-5, atol=1e-8):
    """Sample within a parallelogram using any regular grid.

    Args:
        lat_vecs (numpy.ndarray): the vectors defining the area in which 
            to sample. The vectors are the columns of the matrix.
        grid_vecs (numpy.ndarray): the vectors that generate the grid as 
            columns of a matrix..
        offset (numpy.ndarray): the offset of the coordinate system in grid coordinates.
        cart (bool): if true, return the grid in Cartesian coordinates; otherwise, 
            return the grid in lattice coordinates. 

    Returns:
        grid (list): an array of sampling-point coordinates.
    """

    # Offset in Cartesian coordinates
    car_offset = np.dot(grid_vecs, offset)
    
    # Offset in lattice coordinates.
    lat_offset = np.dot(inv(lat_vecs), car_offset)

    # Integer matrix
    N = np.dot(inv(grid_vecs), lat_vecs)

    # Check that N is an integer matrix.
    for i,j in it.product(range(len(N[:,0])), repeat=2):
        if np.isclose(N[i,j]%1, 0) or np.isclose(N[i,j]%1, 1):
            N[i,j] = int(np.round(N[i,j]))
        else:
            raise ValueError("The cell and grid vectors are incommensurate.")

    H, U = HermiteNormalForm2D(N)
    D = np.diag(H).astype(int)    
    grid = []
    if cart == True:
        # Loop through the diagonal of the HNF matrix.
        for i,j in it.product(range(D[0]), range(D[1])):
            
            # Find the point in Cartesian coordinates.
            pt = np.dot(grid_vecs, [i,j])

            # Bring the point into the unit cell. The offset moves the entire unit cell.
            pt = bring_into_cell(pt, lat_vecs, rtol=rtol, atol=atol) + car_offset
            
            grid.append(pt)
        return grid
    else:
        for i,j in it.product(range(D[0]), range(D[1])):
            # Find the point in cartesian coordinates.
            pt = np.dot(grid_vecs, [i,j])
            grid.append(bring_into_cell(pt, lat_vecs, rtol=rtol, atol=atol))
            
            # Put the point in cell coordinates and move it to the 
            # first unit cell.
            # pt = np.round(np.dot(inv(lat_vecs), pt),12)%1 + offset
            # pt = np.round(np.dot(inv(lat_vecs), pt) + offset, 12)%1
            # pt = np.dot(inv(lat_vecs, pt)) + lat_offset
            # grid.append(pt)
        return grid


def plot_mesh2D(grid, lattice_basis, offset = np.array([0,0]), ax=None):
    """Plot points and the unit cell of a lattice in 2D.

    Args:
        grid (numpy.ndarray): a list of points two plot in 2D.
        lattice_basis (numpy.ndarray): the generating vectors of the lattice as columns
            of a 2x2 array.
        offset (list or numpy.ndarray): the offset of the unit cell in Cartesian 
            coordinates. 
    """
    ngpts = len(grid)
    kxlist = [grid[i][0] for i in range(ngpts)]
    kylist = [grid[i][1] for i in range(ngpts)]
    kzlist = [0 for i in range(ngpts)]

    if ax is None:
        ax = plt.subplot(1,1,1)
    ax.scatter(kxlist, kylist, c="red")
    ax.set_aspect('equal')            

    c1 = lattice_basis[:,0] 
    c2 = lattice_basis[:,1] 
    O = np.asarray([0.,0.]) 

    l1 = zip(O + offset, c1 + offset)
    l2 = zip(c2 + offset, c1 + c2 + offset)
    l3 = zip(O + offset, c2 + offset)
    l4 = zip(c1 + offset, c1 + c2 + offset)
    ls = [l1, l2, l3, l4]
    for l in ls:
        ax.plot(*l, c="blue")    


def get_circle_pts(A, r2, offset=[0.,0.], eps=1e-12):
    """ Calculate all the points within a circle that are
    given by an integer linear combination of the columns of 
    A.
    
    Args:
        A (numpy.ndarray): the columns representing basis vectors.
        r2 (float): the squared radius of the circle.
        offset(list or numpy.ndarray): a vector that points to the center
            of the circle in Cartesian coordinates.
        offset (numpy.ndarray): the center of the circle.
    Returns:
        grid (list): an array of grid coordinates in cartesian
            coordinates.
    """
    
    offset = np.asarray(offset)
    
    # Put the offset in cell coordinates and find a cell point close to the
    # offset.
    oi= np.round(np.dot(inv(A),offset)).astype(int)
    r = np.sqrt(r2)
    V = np.linalg.det(A)
    n = [int(np.ceil(norm(np.cross(A[:,(i+1)%2],A[:,(i+2)%2]))*r/V) + 10)
         for i in range(2)]

    ints = np.array(list(it.product(range(-n[0] + oi[0], n[0] + oi[0] + 1),
                   range(-n[1] + oi[1], n[1] + oi[1] + 1))))
    
    grid = np.dot(A, ints.T).T - offset
    norms = np.array([np.dot(p,p) for p in grid])
    
    return grid[np.where(norms < (r2 + eps))] + offset


def plot_circle_mesh(mesh_points, r2, offset = np.asarray([0.,0.])):
    """Create a scatter plot of a set of points inside a circle.
    
    Args:
        mesh_points (list or np.ndarray): a list of mesh points.
        r2 (float): the squared radius of the circle
        cell_vecs (list or np.ndarray): a list of vectors that define a cell.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the sphere
    u = np.linspace(0, 2 * np.pi, 1000)
    r = np.sqrt(r2)
    x = r*np.cos(u)
    y = r*np.sin(u)
    ax.scatter(x,y,s=0.01)
    
    # Plot the points within the sphere.
    ngpts = len(mesh_points)
    kxlist = [mesh_points[i][0] for i in range(ngpts)]
    kylist = [mesh_points[i][1] for i in range(ngpts)]
    
    ax.set_aspect('equal')
    ax.scatter(kxlist, kylist, c="black",s=10)
    
    lim = np.sqrt(r2)*1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def point_line_location(point, line):
    """Determine if a point is inside the line, outside the line, or lies on the line. 

    Inside is the side of the line opposite that in which the vector normal to the 
    line points.

    Args:
        point (numpy.ndarray): a point in Cartesian coordinates.
        plane (numpy.ndarray): an array with two elements. The first provides
            a vector normal to the line. The second element is the distance of the
            line from the origin in the direction of the vector normal to the line.

    Returns:
        (str): a string that indicates where the point is located. Options include
            "inside", "outside", and "on".
    """
    
    n = np.array(line[0])
    d = line[1]
    loc = np.dot(point, n) - d
    
    if np.isclose(loc, 0):
        return "on"
    elif loc > 0:
        return "outside"
    else:
        return "inside"

def find_2Dbz(lattice_basis):
    """Find the Brillouin zone of a 2D lattice

    Args:
        lattice_basis (numpy.ndarray): the lattice generating vectors as
            columns of a 2x2 array.

    Returns:
        convex_hull (scipy.spatial.ConvexHull): the Brillouin zone
    """
    
    # Find all the lattice points near the origin within a circle of 
    # radius of two times the longest lattice vector.
    r2 = (2*np.max(norm(lattice_basis, axis=0)))**2
    circle_pts = get_circle_pts(lattice_basis, r2)
    
    # Sort these points by distance from the origin.
    indices = np.argsort(norm(circle_pts, axis=1))
    circle_pts = circle_pts[indices]
    
    # Find the index of the circle points where the distance of the 
    # points increases.
    indices = [0]
    circle_pts = remove_points([0,0], circle_pts)
    n = norm(circle_pts[0])
    for i,pt in enumerate(circle_pts):
        if not np.isclose(n, norm(pt)):
            n = norm(pt)
            indices.append(i)
    
    # Find the Bragg lines. These are a list with the vector normal to the
    # line as the first element and the shortest distance from the origin as
    # the second.
    bragg_lines = []
    for pt in circle_pts:
        d = norm(pt)/2
        n = pt/(2*d)
        bragg_lines.append([n, d])

    # Find the Brillouin zone. You know you have it when it the convex hull has
    # the correct volume.
    volume = 0
    ind = 1
    while not np.isclose(volume, det(lattice_basis)):
        intersections = []
        lines = bragg_lines[:indices[ind]]
        for line1, line2 in it.product(lines, repeat=2):
            a1 = line1[0][0]
            b1 = line1[0][1]
            d1 = line1[1]
            a2 = line2[0][0]
            b2 = line2[0][1]
            d2 = line2[1]

            if np.isclose(a1*b2 - a2*b1, 0) or np.isclose(a2*b1 - a1*b2, 0):
                continue
            x = (b2*d1 - b1*d2)/(a1*b2 - a2*b1)
            y = (a2*d1 - a1*d2)/(a2*b1 - a1*b2)
            pt = [x,y]

            if any(np.array([point_line_location(pt, line) for line in lines]) == "outside"):
                continue
            else:
                intersections.append(pt)
        convex_hull = ConvexHull(intersections)
        volume = convex_hull.volume
        return convex_hull


def plot_all2D_bz(mesh_points, bz):
    """Plot the Brillouin zone and lattice points.
    
    Args:
        mesh_points (numpy.ndarray): a list of lattice points
        bz (scipy.spatial.ConvexHull): the Brillouin zone
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the points within the sphere.
    ngpts = len(mesh_points)
    kxlist = [mesh_points[i][0] for i in range(ngpts)]
    kylist = [mesh_points[i][1] for i in range(ngpts)]    
    ax.set_aspect('equal')
    ax.scatter(kxlist, kylist, c="black",s=1)

    for simplex in bz.simplices:
        pt1 = bz.points[simplex[0]]
        pt2 = bz.points[simplex[1]]

        xstart = pt1[0]
        xfinish = pt2[0]
        xs = np.linspace(xstart, xfinish, 100)
        ystart = pt1[1]
        yfinish = pt2[1]
        ys = np.linspace(ystart, yfinish, 100)
        ax.plot(xs, ys, c="blue")

        
class FreeElectron2D():
    """This is the popular free electron model. In this model the potential is
    zero everywhere. It is useful for testing.
    
    Args:
        lattice_basis (numpy.ndarray): the lattice basis vectors as columns of
            a 2x2 array.
        nvalence_electrons (int): the number of valence electrons.
        degree (int): the degree of the radial dispersion relation.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        total_energy(float): the total energy
        prefactor (float): the prefactor to the dispersion relation. It takes the
            form E(k) = Ak**n where A is the prefactor and n is the degree.
    
    Attributes:
        lattice_basis (numpy.ndarray): the lattice basis vectors as columns of
            a 2x2 array.
        nvalence_electrons (int): the number of valence electrons is 1.
        degree (int): the degree of the radial dispersion relation.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        fermi_level_ans (float): the exact, analytical value for the Fermi 
            level.
        total_enery (float): the total energy
        total_enery_ans (float): the exact, analytical value for the total
            energy.
        material (str): the material or model name.
        prefactor (float): the prefactor to the dispersion relation. It takes the
            form E(k) = Ak**n where A is the prefactor and n is the degree.
    """
    
    def __init__(self, lattice_basis, degree, nvalence_electrons, energy_shift=None,
                 fermi_level=None, band_energy=None, prefactor=1):
        self.material = "2D free electron model"
        self.lattice_basis = lattice_basis
        self.prefactor = prefactor
        if degree == 3:
            msg = "In 2D, the dispersion relation cannot be of degree three."
            raise ValueError(msg.format(degree))
        self.degree = degree
        self.nvalence_electrons = nvalence_electrons
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.
        self.fermi_level_ans = self.prefactor*(self.nvalence_electrons*det(self.lattice_basis)/
                                               (2*np.pi))**(self.degree/2)        
        self.band_energy = band_energy or 0.
        self.band_energy_ans = 2*np.pi*self.prefactor/(self.degree + 2)*(
            self.fermi_level_ans/self.prefactor)**(1 + 2/self.degree)
        nfilled_states = self.nvalence_electrons/2.
        
    def eval(self, kpoint, neigvals, sigma=False):
        kpoint = np.array(kpoint)
        l0 = np.linalg.norm(self.lattice_basis[:,0])
        l1 = np.linalg.norm(self.lattice_basis[:,1])

        # offset = np.dot(self.lattice_basis, [0.5]*2)
        pts = [np.dot(self.lattice_basis, [i,j]) for i,j in it.product(range(-2,3),
                                                                       repeat=2)]
        if sigma:
            return np.sum(np.sort([np.linalg.norm(kpoint - pt)**self.degree
                                   for pt in pts])[:neigvals])
        else:
            return np.sort([np.linalg.norm(kpoint - pt)**self.degree
                            for pt in pts])[:neigvals]
        
    def change_potential(self, prefactor, degree):
        if degree == 3:
            msg = "In 2D, the dispersion relation cannot be of degree three."
            raise ValueError(msg.format(degree))
        self.degree = degree
        self.prefactor = prefactor
        self.fermi_level_ans = self.prefactor*(self.nvalence_electrons*det(self.lattice_basis)/
                                               2*np.pi)**(self.degree/2)        
        self.band_energy_ans = 2*np.pi*self.prefactor*self.degree*(
            (self.fermi_level_ans/self.prefactor)**(1-1/self.degree))
        
        
def plot_2Dbands(EPM, neigvals):
    """Plot the bands of a 2D empirical pseudopotential
    
    Args:
        EPM (class): an empirical pseudopotential object.
        neigvals (int): the number of eigenvalues to plot.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    grid_basis = EPM.lattice_basis/100
    offset = np.dot(inv(grid_basis), np.dot(EPM.lattice_basis, [-.5]*2)) + [.5]*2
    # offset = [0, 0]
    grid = make_cell_points2D(EPM.lattice_basis, grid_basis, offset)
    
    kx = [grid[i][0] for i in range(len(grid))]
    ky = [grid[i][1] for i in range(len(grid))]        
    ax = plt.subplot(1,1,1,projection="3d")
    fig = plt.gca()
    
    all_states = np.full((len(grid), neigvals), np.nan)
    
    for i,pt in enumerate(grid):
        all_states[i,:] = EPM.eval(pt, neigvals)

    for n in range(neigvals):
        kz = all_states[:,n]
        kz[np.where(kz > EPM.fermi_level_ans)] = np.nan
        ax.scatter(kx, ky, kz, s=0.5)


def plot_sigma_band2D(EPM, neigvals):
    """Plot the bands of a 2D empirical pseudopotential
    
    Args:
        EPM (class): an empirical pseudopotential object.
        neigvals (int): the number of eigenvalues to plot.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    grid_basis = EPM.lattice_basis/100
    offset = np.dot(inv(grid_basis), np.dot(EPM.lattice_basis, [-.5]*2)) + [.5]*2
    # offset = [0, 0]
    
    grid = make_cell_points2D(EPM.lattice_basis, grid_basis, offset)
    
    kx = [grid[i][0] for i in range(len(grid))]
    ky = [grid[i][1] for i in range(len(grid))]        
    ax = plt.subplot(1,1,1,projection="3d")
    fig = plt.gca()
    
    all_states = [np.sum(EPM.eval(pt,neigvals)) for pt in grid]
    
    ax.scatter(kx, ky, all_states, s=0.5)
        
    
# def plot_2Dbands(EPM, neigvals):
#     """Plot the bands of a 2D empirical pseudopotential
    
#     Args:
#         EPM (class): an empirical pseudopotential object.
#         neigvals (int): the number of eigenvalues to plot.
#     """
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     grid_basis = EPM.lattice_basis/100
#     offset = np.dot(inv(grid_basis), np.dot(EPM.lattice_basis, [-.5]*2)) + [.5]*2
#     grid = make_cell_points2D(EPM.lattice_basis, grid_basis, offset)
    
#     kx = [grid[i][0] for i in range(len(grid))]
#     ky = [grid[i][1] for i in range(len(grid))]        
#     ax = plt.subplot(1,1,1,projection="3d")
#     fig = plt.gca()
    
#     all_states = np.full((len(grid), neigvals), np.nan)
    
#     for i,pt in enumerate(grid):
#         all_states[i,:] = EPM.eval(pt, neigvals)
    
#     for n in range(neigvals):
#         kz = all_states[:,n]
#         kz[np.where(kz > EPM.fermi_level)] = np.nan
#         ax.scatter(kx, ky, kz, s=0.5)
        
        
def plot_2Dfermi_curve(EPM, neigvals, ndivs, atol=1e-2):
    """Plot the bands of a 2D empirical pseudopotential
    
    Args:
        EPM (class): an empirical pseudopotential object.
        neigvals (int): the number of eigenvalues to plot.
        ndivs (int): the number of divisions for the grid.
        atol (float): the tolerance for the Fermi level.
    """
    
    fig,ax = plt.subplots()
    grid_basis = EPM.lattice_basis/ndivs
    offset = np.dot(inv(grid_basis), np.dot(EPM.lattice_basis, [-.5]*2)) + [.5]*2
    grid = make_cell_points2D(EPM.lattice_basis, grid_basis, offset)

    kx = np.array([grid[i][0] for i in range(len(grid))])
    ky = np.array([grid[i][1] for i in range(len(grid))])
    fermi_levelx = np.full(np.shape(kx), np.nan)
    fermi_levely = np.full(np.shape(ky), np.nan)
    all_states = np.full((len(grid), neigvals), np.nan)
    for i,pt in enumerate(grid):
        all_states[i,:] = EPM.eval(pt, neigvals)
    
    for n in range(neigvals):
        kz = all_states[:,n]
        fermi_levelx[np.isclose(kz, EPM.fermi_level, atol=atol)] = (
            kx[np.isclose(kz, EPM.fermi_level, atol=atol)])
        fermi_levely[np.isclose(kz, EPM.fermi_level, atol=atol)] = (
            ky[np.isclose(kz, EPM.fermi_level, atol=atol)])
        
    ax.scatter(fermi_levelx, fermi_levely, s=0.5, c="black")

    offset = np.dot(EPM.lattice_basis, [-.5]*2)
    c1 = EPM.lattice_basis[:,0]
    c2 = EPM.lattice_basis[:,1] 
    O = np.asarray([0.,0.])

    l1 = zip(O + offset, c1 + offset)
    l2 = zip(c2 + offset, c1 + c2 + offset)
    l3 = zip(O + offset, c2 + offset)
    l4 = zip(c1 + offset, c1 + c2 + offset)
    ls = [l1, l2, l3, l4]
    for l in ls:
        ax.plot(*l, c="blue")

        
def rectangular_integration2D(EPM, grid, weights):
    """Integrate an empirical pseudopotential in 2D using the rectangular method to
    find the Fermi level and total energy.
    
    Args:
        EPM (class): an empirical pseudopotential model
        grid (numpy.ndarray): an array of grid points
        weights (numpy.ndarray): an array of grid points weights in the same order as grid.
        
    Returns:
        fermi_level (float): the energy of the highest occupied state
        band_energy (float): the band energy
    """
    
    # Find the index of the highest occupied state.
    C = np.ceil(np.round(EPM.nvalence_electrons*np.sum(weights)/2., 3)).astype(int)
    
    # An estimate of the number of eigenvalues to keep at each k-point.
    neigvals = np.ceil(np.round(EPM.nvalence_electrons/2+1, 3)).astype(int) + 4
    energies = np.array([])
    for i,g in enumerate(grid):
        energies = np.concatenate((energies, list(EPM.eval(g, neigvals))*
                                   int(np.round(weights[i]))))
    energies = np.sort(energies)[:C]
    fermi_level = energies[-1]
    band_energy = np.sum(energies)*np.linalg.det(EPM.lattice_basis)/(
                   np.sum(weights))
    return fermi_level, band_energy        
