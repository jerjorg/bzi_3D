"""Methods to for calculating and visualizing lattices and Brillouin zones in 2D."""

from numpy.linalg import norm, inv, det
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
from copy import deepcopy, copy
from scipy.spatial import ConvexHull

from BZI.symmetry import get_minmax_indices, bring_into_cell
from BZI.utilities import remove_points, rprint, check_inside, check_contained
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
        

def make_cell_points2D(lat_vecs, grid_vecs, offset=[0,0], coords="Cart",
                       grid_type="open", rtol=1e-5, atol=1e-8):
    """Sample within a parallelogram using any regular grid.

    Args:
        lat_vecs (numpy.ndarray): the vectors defining the area in which 
            to sample. The vectors are the columns of the matrix.
        grid_vecs (numpy.ndarray): the vectors that generate the grid as 
            columns of a matrix..
        offset (numpy.ndarray): the offset of the coordinate system in grid coordinates.
        coords (str): a string that determines the coordinate of the returen k-points.
            Options include "Cart" for Cartesian and "lat" for lattice coordinates.
        grid_type (str): if "closed" the grid will include points along both 
            boundaries. If open, only points on one boundary are included.       

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

    if grid_type == "closed":
        D += 1
    grid = []
    if coords == "Cart":
        # Loop through the diagonal of the HNF matrix.
        for i,j in it.product(range(D[0]), range(D[1])):
            
            # Find the point in Cartesian coordinates.
            pt = np.dot(grid_vecs, [i,j]) + car_offset

            # Bring the point into the unit cell. The offset moves the entire unit cell.
            # pt = bring_into_cell(pt, lat_vecs, rtol=rtol, atol=atol) + car_offset
            
            grid.append(pt)
        return grid
    elif coords == "lat":
        for i,j in it.product(range(D[0]), range(D[1])):
            # Find the point in cartesian coordinates.
            pt = np.dot(grid_vecs, [i,j])
            # grid.append(bring_into_cell(pt, lat_vecs, rtol=rtol, atol=atol))
            
            # Put the point in cell coordinates and move it to the 
            # first unit cell.
            # pt = np.round(np.dot(inv(lat_vecs), pt),12)%1 + offset
            # pt = np.round(np.dot(inv(lat_vecs), pt) + offset, 12)%1
            pt = np.dot(inv(lat_vecs, pt)) + lat_offset
            grid.append(pt)
        return grid            
    else:
        msg = "Coordinate options include 'Cart' and 'lat'."
        raise ValueError(msg)


def plot_mesh2D(grid, lattice_basis, offset = np.array([0,0]), ax=None, color="black"):
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
        fig,ax = plt.subplots()
    ax.scatter(kxlist, kylist, c=color)
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


def get_perpendicular_vector2D(vector, atol=1e-8, rtol=1e-5):
    """Find a unit vector perpendicular to the input vector in 2D.
    
    Args:
        vector (list or numpy.ndarray): a vector in 2D.
        
    Returns:
        perp_vector (numpy.ndarray): a vector perpendicular to the input
            vector.
    """
    
    # If the vector is the origin, return a unit vector in the x-direction.
    if np.allclose(vector, [0,0], atol=atol, rtol=rtol):
        return np.array([1, 0])
    
    perp_vector = np.array([0., 0.])

    # Find the location of an element of the vector that isn't zero and 
    # any other element.
    i = np.where(np.invert(np.isclose(vector, 0, atol=atol, rtol=rtol)))[0][0]
    j = (i + 1)%2

    # Create the vector perpendicular.
    perp_vector[i], perp_vector[j] = -vector[j], vector[i]

    if np.cross(vector, perp_vector) < 0:
        perp_vector *= -1

    return perp_vector/norm(perp_vector)


def get_line_equation2D(pt1, pt2):
    """Find the equation of a line in normal form given two points
    that lie one line.
    
    Args:
        pt1 (list or numpy.ndarray): a point in 2D
        pt2 (list or numpy.ndarray): a point in 2D
        
    Returns:
        _ (list): the equation of a line. The first element is a 
            vector normal to the line. The second is the closest distance from the
            origin to the line along the direction of the normal vector.
    """
    
    # Find the vector that points from the first point to the second.
    r12 = np.array(pt2) - np.array(pt1)
    r_perp = get_perpendicular_vector2D(r12)
    
    # Find the distance from the origin to the line.
    d = np.dot(pt1, r_perp)
    
    return [r_perp, d]
    

def point_line_location(point, line, rtol=1e-5, atol=1e-8):
    """Determine if a point is inside the line, outside the line, or lies on the line. 

    Inside is the side of the line opposite that in which the vector normal to the 
    line points.

    Args:
        point (numpy.ndarray): a point in Cartesian coordinates.
        line (numpy.ndarray): an array with two elements. The first provides
            a vector normal to the line. The second element is the distance of the
            line from the origin in the direction of the vector normal to the line.

    Returns:
        (str): a string that indicates where the point is located. Options include
            "inside", "outside", and "on".
    """
    
    n = np.array(line[0])
    d = line[1]
    loc = np.dot(point, n) - d
    
    if np.isclose(loc, 0, atol=atol, rtol=rtol):
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
        nsheets (int): the number of bands included when evaluating the EPM.
    
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
        nsheets (int): the number of bands included when evaluating the EPM.
    """
    
    def __init__(self, lattice_basis, degree, nvalence_electrons, energy_shift=None,
                 fermi_level=None, band_energy=None, prefactor=1, nsheets=3):
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
        self.nsheets = nsheets        
        
    def eval(self, kpoint, nsheets=None, sigma=False, refine=False):

        if nsheets is None:
            nsheets = self.nsheets        
        
        kpoint = np.array(kpoint)
        
        # offset = np.dot(self.lattice_basis, [0.5]*2)
        pts = [np.dot(self.lattice_basis, [i,j]) for i,j in it.product(range(-2,3),
                                                                       repeat=2)]
        if sigma:                        
            # return np.sum(np.sort([np.linalg.norm(kpoint - pt)**self.degree
            #                        for pt in pts])[:nsheets])
            values = np.array([np.linalg.norm(kpoint - pt)**self.degree for pt in pts])
            return np.sum(values[values < self.fermi_level])
            
        elif refine:
            return np.sum(np.sort([np.linalg.norm(kpoint - pt)**self.degree
                                   for pt in pts])[:nsheets])
        else:
            return np.sort([np.linalg.norm(kpoint - pt)**self.degree
                            for pt in pts])[:nsheets]
        
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
        
        
def plot_2Dbands(EPM, sigma=False):
    """Plot the bands of a 2D empirical pseudopotential
    
    Args:
        EPM (class): an empirical pseudopotential object.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    grid_basis = EPM.lattice_basis/100
    offset = np.dot(inv(grid_basis), np.dot(EPM.lattice_basis, [-.5]*2)) # + [.5]*2
    # offset = [0, 0]
    grid = make_cell_points2D(EPM.lattice_basis, grid_basis, offset, grid_type="closed")
    
    kx = [grid[i][0] for i in range(len(grid))]
    ky = [grid[i][1] for i in range(len(grid))]        

    if sigma:
        all_states = np.full((len(grid)), np.nan)

        for i,pt in enumerate(grid):
            all_states[i] = EPM.eval(pt, sigma=sigma)

        kz = all_states[:]
        ax.scatter(kx, ky, kz, s=0.5)            

    else:    
        all_states = np.full((len(grid), EPM.nsheets), np.nan)
        
        for i,pt in enumerate(grid):
            all_states[i,:] = EPM.eval(pt)

        for n in range(EPM.nsheets):
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
        
        
def plot_2Dfermi_curve(EPM, neigvals, ndivs, atol=1e-2, ax=None):
    """Plot the bands of a 2D empirical pseudopotential
    
    Args:
        EPM (class): an empirical pseudopotential object.
        neigvals (int): the number of eigenvalues to plot.
        ndivs (int): the number of divisions for the grid.
        atol (float): the tolerance for the Fermi level.
    """

    if ax is None:
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

        
def rectangular_integration2D(EPM, grid, weights, areas=None):
    """Integrate an empirical pseudopotential in 2D using the rectangular method to
    find the Fermi level and total energy.
    
    Args:
        EPM (class): an empirical pseudopotential model
        grid (numpy.ndarray): an array of grid points
        weights (numpy.ndarray): an array of grid points weights in the same order as 
            grid. These weights are the result of symmetry reducing the grid.
        areas (list or numpy.ndarray): the subgrid areas. This is only used when the
            grid isn't uniform.        

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
        energies = np.concatenate((energies, list(EPM.eval(g, nsheets=neigvals))*
                                   int(np.round(weights[i]))))        
    energies = np.sort(energies)[:C]
    fermi_level = energies[-1]
    band_energy = np.sum(energies)*np.linalg.det(EPM.lattice_basis)/(
        np.sum(weights))
    
    return fermi_level, band_energy


def square_tesselation(grid, atol=1e-5, rtol=1e-8):
    """Create a tesselation of squares from a grid.
    
    Args:
        grid (list or numpy.ndarray): a list of points in 2D.

    Returns:
        tesselation (numpy.ndarray): the grid grouped with points grouped in subsquares.
    """
    
    grid = np.array(grid)
    grid_copy = deepcopy(grid)
    tesselation = []

    while(len(grid_copy)) > 0:
        
        # Grab a point from the grid.

        grid_pt = grid_copy[-1]
        grid_copy = grid_copy[:-1]
        # rprint("grid point:", grid_pt)
        
        # Find the distance between this point and all other points in the grid.
        distances = np.array([norm(grid_pt - gpt) for gpt in grid])
        
        # Sort the distances in ascending order. Sort the grid in the same manner.
        indexing = np.argsort(distances)
        sorted_grid = grid[indexing]
        
        # The points with the shortest distances but also greater x- and y-components
        # will form the subsquare.
        
        subsquare = [grid_pt]
        for nearby_pt in sorted_grid:
            
            # If the nearby point is the same as the grid point, skip it.
            if np.allclose(nearby_pt, grid_pt, atol=atol, rtol=rtol):
                continue
            
            # If the x-component of the nearby point is less than the that
            # of the grid point, skip it.
            if (not np.isclose(nearby_pt[0], grid_pt[0])) and (nearby_pt[0] < grid_pt[0]):
                continue
            
            # If the y-component of the nearby point is less than that of the grid point,
            # skip it.
            if (not np.isclose(nearby_pt[1], grid_pt[1])) and (nearby_pt[1] < grid_pt[1]):
                continue            
            if len(subsquare) > 1:
                
                # If this nearby point lies on the same line as the other two points,
                # skip it.
                line_eq = get_line_equation2D(subsquare[0], subsquare[1])
                
                if point_line_location(nearby_pt, line_eq, rtol=rtol, atol=atol) == "on":
                    continue
                    
            # If all those tests fail, add the point to the subsquare.
            subsquare.append(nearby_pt)
            
            # Once we have 4 points, we have all we need.
            if len(subsquare) == 4:
                break
                
        # For points on the far left and top of the grid, there won't be any nearby points
        # with greater x- and y-components. These we will skip.
        if len(subsquare) < 4:
            continue                
        else:
            # Sort the points in the tesselation in clockwise order starting with the lower
            # left point.            
            subsquare = np.array(subsquare)

            # The indices of the points that need to be sorted.
            indices = {0, 1, 2, 3}            
            sorted_indices = [0]*4

            # Find the index of point one. It is the point with the smallest y-coordinate.
            sorted_yindices = np.argsort(subsquare[:,1])
            
            if sorted_yindices[0] == 0:
                pt1_index = sorted_yindices[1]
            else:
                pt1_index = sorted_yindices[0]

            sorted_indices[1] = pt1_index

            # Find the index of point three. It is the point, of the remaining two, that
            # has the smallest x-coordinate.
            sorted_xindices = np.argsort(subsquare[:,0])

            if sorted_xindices[0] == 0:
                pt3_index = sorted_xindices[1]
            else:
                pt3_index = sorted_xindices[0]

            sorted_indices[3] = pt3_index

            pt2_index = list(indices.symmetric_difference(sorted_indices))[0]
            sorted_indices[2] = pt2_index            

            subsquare = subsquare[sorted_indices]

            tesselation.append(subsquare)
            
    return np.array(tesselation)


def refine_square(square_pts, EPM, method="derivative",
                  ndivisions=2, derivative_tol=1e3,
                  fermilevel_tol=1e-1, integration_tol = 1e-1,
                  atol=1e-8, rtol=1e-5):
    """Refine a square given a refinement method and sampling.
    
    Args:
        square_pts (list or numpy.ndarray): A list with four x,y pairs. These must
            be ordered starting with the lower left corner and moving counter-clockwise
            to the other corners.
        EPM (func): a function of x and y.
        method (str): the refinement method. This determines if points will be
            added to the square.
        ndivisions (int): the number of times the square is divided.
        derivative_tol (float): the size of the derivative which merits refinement.
        fermi_level_tol (float): how close the function values need to be to the Fermi
            level to merit refinement of the square.
        atol (float): an absolute tolerance used when comparing the equivalency of two
            points.
        tol (float): a relative tolerance used when comparing the equivalency of two
            points.
    
    Returns:
        new_squares (list): a list of new squares.
        area (float): the area of the new squares.
    """

    EPM_values = [EPM.eval(pt, sigma=True) for pt in square_pts]

    # Calculate the area of the input square
    v0 = square_pts[1] - square_pts[0]
    v1 = square_pts[3] - square_pts[0]

    square_basis = np.transpose([v0, v1])
    area = det(square_basis)
    
    # Refine will determine if the square is refined.
    refine = False

    if method == "interpolate":

        zeroth_order = np.mean(EPM_values)*area

        first_order = integrate_bilinear(square_pts, EPM_values, square_pts[0])
        
        refine = abs(first_order - zeroth_order) > integration_tol

    
    elif method == "derivative":

        # Find the points along each edge. Corners and edges are labeled as follows:
        #    3 --- 2 --- 2
        #    |           |
        #    3           1
        #    |           |
        #    0 --- 0 --- 1
        edge_list = [[square_pts[i], square_pts[ (i+1)%4 ]] for i in range(4)]

        # Create a list of the values at the edges.
        values_list = [[EPM_values[i], EPM_values[ (i+1)%4 ]] for i in range(4)]

        # Add the diagonals the the list of edges. These are included so we can calculate
        # derivatives along those directions.    
        edge_list.append([square_pts[0], square_pts[2]])
        edge_list.append([square_pts[1], square_pts[3]])

        # Add the values at the diagonals
        values_list.append([EPM_values[0], EPM_values[2]])
        values_list.append([EPM_values[1], EPM_values[3]])    

        edge_lengths = [norm(edge[0] - edge[1]) for edge in edge_list]        
        
        # Calculate the numerical derivative of the function along the edges and diagonals
        # of the square using finite differences.
        derivatives = np.sort([abs(np.diff(values_list[i])[0])/edge_lengths[i]
                               for i in range(len(values_list))])
        
        # print("high/low: ", np.mean(derivatives[3:])/np.mean(derivatives[:2]))

        compare = 10# np.min(EPM_values)/np.max(edge_lengths)
        
        # refine = np.mean(derivatives[:2])*compare < np.mean(derivatives[3:])

        refine = 5*np.min(derivatives) < np.max(derivatives)
                        
    # Refine the square regardless.
    elif method == "refine":
        refine = True
        
    else:
        msg = "Invalid refinement method provided"
        raise ValueError(msg)                
    
    # If the square needs refinement, divide the square.
    if refine:
        
        new_squares = []
                
        # Let's calculate a grid basis for these squares and their area.
        grid_basis = square_basis/ndivisions
        area /= ndivisions**2

        # Get the bottom left point of the square. This is the same as the offset of
        # the grid. Put it in grid coordinates.
        grid_offset = np.dot(inv(grid_basis), square_pts[0])

        new_points = make_cell_points2D(square_basis, grid_basis, offset=grid_offset,
                                        grid_type="closed")

        # Sort the new points into squares.
        new_squares = square_tesselation(new_points, atol=atol, rtol=rtol)
        
        return new_squares, area
    else:
        return square_pts, area


def get_bilin_coeffs(points, values):
    """Find the coefficients for a bilinear interpolation between four points
    in 2D.
    
    Args:
        points (list or numpy.ndarray): a list of four x- and y-coordinates.
        values (list or numpy.ndarray): a list of four function values
    
    Returns:
        _ (numpy.ndarray): a list of coefficients for the bilinear intepolation
            ordered as [c0, c1, c2, c3] where the interpolation is written as 
            f[x,y] = c0 + c1*x + c2*y + c3*x*y.
    """
    
    points = np.array(points)

    xs = points[:,0]
    ys = points[:,1]

    x0, x1, x2, x3 = xs[0], xs[1], xs[2], xs[3]
    y0, y1, y2, y3 = ys[0], ys[1], ys[2], ys[3]    

    M = np.array([[1, x0, y0, x0*y0],
                  [1, x1, y1, x1*y1],
                  [1, x2, y2, x2*y2],
                  [1, x3, y3, x3*y3]])
    
    return np.dot(inv(M), values)


def eval_bilin(coeffs, point):
    """Evaluate the bilinear interpolation at a point
    
    Args:
        coeffs (list): a list of four coefficients.
        point (list): a list of x- and y-coordinates.    
    """
    
    x,y = point[0],point[1]    
    values = [1, x, y, x*y]
    return np.dot(coeffs, values)


def integrate_bilinear(vertices, values, offset):
    """Integrate a bilinear interpolation within a parallelogram.

    Args:
        vertices (list or numpy.ndarray)): the coordinates of the vertices of the 
            parallelogram. These must be arranged in counterclockwise order starting with
            lower-left point.

    Returns:
        _ (float): the integral of the bilinear interpolation over the parallelogram.
    """
    
    vertices = np.array(vertices)
    
    v0 = vertices[1] - vertices[0]
    v1 = vertices[3] - vertices[0]
    
    a0 = v0[0]
    a1 = v0[1]
    a2 = v1[0]
    a3 = v1[1]
    
    # values = [EPM.eval(v, sigma=True) for v in vertices]
    coeffs = get_bilin_coeffs(vertices, values)

    c0, c1, c2, c3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

    k0,k1 = offset[0], offset[1]
    
    return 0.5*(2*c0 + a0*c1 + a2*c1 + a1*c2 + a3*c2 + 2*c3 + 2*c1*k0 +
                2*c2*k1)*abs(a0*a3 - a1*a2)


def integrate_tess(EPM, tesselation, areas_list):
    """Integrate the tesselation of a square with square tiles.
    
    Args:
        EPM (function): an empirical pseudopotential function.
        tesselation (list or numpy.ndarray): a list of squares. A square is
            given by the four points at its corners starting at the lower, left
            point and continuing in counterclockwise order.
        areas_list (list or numpy.ndarray): the areas of the tiles in the tesselation.
    
    Returns:
        integral (float): the integral of the band structure within the Fermi level.
    """
    
    integral = 0
    for area, tess in zip(areas_list, tesselation):
        value = np.mean([EPM.eval(pt, sigma=True) for pt in tess])
        
        if value < EPM.fermi_level:
            integral += value*area
    
    return integral

def find_param_intersect(square_pts, coeffs, isovalue, atol=1e-8, rtol=1e-5):
    """Calculate the two values of the parameter where the constant energy curve 
    of a bilinear interpolation intersects the boundaries of the parallelogram.
    
    Args:
        square_pts (list): A list of coordinates of the corners of the parallelogram.
        coeffs (list): a list of coefficients for the bilinear interpolation.
        isovalue (float): the value of the function on the isocontour. 
        atol (float): the absolute tolerance used when comparing the parameter to 0 and 1.
        rtol (float): the relative tolerance used when comparing the parameter to 0 and 1.
        
    Returns:
        unique_param_edge (list): a list of parameter values for the parametric equations
            on the boundaries of the parallelogram.
        unique_param_isocurve (list): a list of parameter values for the isocurve 
            parametric equation.
        unique_intersecting_edges (list): a list of edges intersected by the iscurve.
        unique_xy (list): the xy-coordinates where the isocurve intersects the boundary of
            the paralellogram.
    """
    
    param_edge = []
    param_isocurve = []
    intersecting_edges = []
    edge_list = [[square_pts[i], square_pts[ (i+1)%4 ]] for i in range(4)]
    
    c0, c1, c2, c3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    Ec = isovalue
    
    for edge in edge_list:
        # print("\n")
        # rprint("edge", edge)
        
        x0, x1, y0, y1 = edge[0][0], edge[1][0], edge[0][1], edge[1][1]
        
        # Have to consider c1 == c2 == c3 == 0 as a separate case.
        if (np.isclose(c1, 0, rtol=rtol, atol=atol) and
            np.isclose(c2, 0, rtol=rtol, atol=atol) and
            np.isclose(c3, 0, rtol=rtol, atol=atol)):
        
            # When the bilinear is a constant, the curve of constant energy becomes a
            # sheet of constant energy. This sheet could intersect every point along the
            # boundary and the interior or not intersect at all. Hopefully this is a
            # pathological case. Since it intersects everywhere, we can prtovide any path
            # through the parallelogram. Technically this isn't correct but it's work for
            # our purposes.
            if np.isclose(c0, Ec, rtol=rtol, atol=atol):
                unique_param_edge = [0, 1]
                unique_param_isocurve = [x0, x1]
                unique_intersecting_edges = [edge, edge]
                return unique_param_edge, unique_param_isocurve, unique_intersecting_edges
            else:
                continue
        
        # Have to consider c2 == c3 == 0 as a separate case.
        elif (np.isclose(c2, 0, rtol=rtol, atol=atol) and
              np.isclose(c3, 0, rtol=rtol, atol=atol)):
            
            # If the x-coordinates are the same.
            if np.isclose(x0, x1, rtol=rtol, atol=atol):

                if np.isclose(x0, (Ec-c0)/c1, rtol=rtol, atol=atol):
                    
                    unique_param_edge = [0, 1]
                    unique_param_isocurve = [y0, y1]
                    unique_intersecting_edges = [edge, edge]
                    return (unique_param_edge, unique_param_isocurve,
                            unique_intersecting_edges)                
                else:
                    continue
                
            # If the y-coordinates are the same.
            elif np.isclose(y0, y1, rtol=rtol, atol=atol):
                te = check_inside((c0 - Ec + c1*x0)/(c1*x0 - c1*x1), rtol=rtol, atol=atol)

                if te is not None:                    
                    param_edge.append(te)
                    param_isocurve.append(y0)
                    intersecting_edges.append(edge)

            # If both the x- and y-coordinates are different.
            else:
                te = check_inside((c0 - Ec + c1*x0)/(c1*x0 - c1*x1), rtol=rtol, atol=atol)
                if te is not None:
                    x,tc = get_param_xy(te, edge)
                    param_edge.append(te)
                    param_isocurve.append(tc)
                    intersecting_edges.append(edge)                    
                else:
                    continue
        
        # Have to consider c1 == c3 == 0 as a separate case.
        elif (np.isclose(c1, 0, rtol=rtol, atol=atol) and
              np.isclose(c3, 0, rtol=rtol, atol=atol)):
            
            # If the y-coordinates are the same.
            if np.isclose(y0, y1, rtol=rtol, atol=atol):

                if np.isclose(y0, (Ec-c0)/c2, rtol=rtol, atol=atol):
                    
                    param_edge = [0, 1]
                    param_isocurve = [x0, x1]
                    intersecting_edges = [edge, edge]
                    return param_edge, param_isocurve, intersecting_edges
                
                else:
                    continue
                
            # If the x-coordinates are the same.
            elif np.isclose(x0, x1, rtol=rtol, atol=atol):
                te = check_inside((c0 - Ec + c2*y0)/(c2*y0 - c2*y1), rtol=rtol, atol=atol)
                
                if te is not None:                    
                    param_edge.append(te)
                    param_isocurve.append(x0)
                    intersecting_edges.append(edge)

            # If both the x- and y-coordinates are different.
            else:
                te = check_inside((c0 - Ec + c2*y0)/(c2*y0 - c2*y1), rtol=rtol, atol=atol)

                if te is not None:
                    tc,y = get_param_xy(te, edge)
                    param_edge.append(te)
                    param_isocurve.append(tc)
                    intersecting_edges.append(edge)
                    
                else:
                    continue

        # General case with x0 == x1 and y0 != y1
        elif np.isclose(x0, x1, atol=atol, rtol=rtol):
            # print("x0 == x1")


            te = (c0 - Ec + c1*x0 + c2*y0 + c3*x0*y0)/((c2 + c3*x0)*(y0 - y1))
            # print("te: ", te)
            te = check_inside(te, rtol=rtol, atol=atol)
            # print("te: ", te)            
            tc =  x0
            # print("tc: ", tc)
            if te is not None:
                # print("appended")
                param_edge.append(te)
                param_isocurve.append(tc)
                intersecting_edges.append(edge)
            else:
                continue

        # General case with x0 != x1 and y0 == y1
        elif np.isclose(y0, y1, atol=atol, rtol=rtol):
            
            # print("y0 == y1")
            te = (c0 - Ec + c1*x0 + c2*y0 + c3*x0*y0)/((x0 - x1)*(c1 + c3*y0))
            # print("te: ", te)
            te = check_inside(te, rtol=rtol, atol=atol)
            # print("te: ", te)
            tc = -((c0 - Ec + c2*y0)/(c1 + c3*y0))
            # print("tc: ", tc)
            
            if te is not None:
                param_edge.append(te)
                param_isocurve.append(tc)
                intersecting_edges.append(edge)
            else:
                continue                
        
        # General case with x0 != x1 and y0 != y1
        else:

            te1 = -(1/(
                2*c3*(x0 - x1)*(y0 - y1)))*(-c1*x0 + c1*x1 - c2*y0 - 2*c3*x0*y0 + 
                c3*x1*y0 + c2*y1 + c3*x0*y1 + np.sqrt(-4*c3*(x0 - x1)*(c0 - Ec +
                c1*x0 + c2*y0 + c3*x0*y0)*(y0 - y1) + (c1*(-x0 + x1) + c2*(-y0 + y1) +
                                                       c3 (-2*x0*y0 + x1*y0 + x0*y1))**2))

            tc1 =  (1/(
                2*c3*(y0 - y1)))*(-c1*x0 + c1*x1 - c2*y0 + c3*x1*y0 + c2*y1 - 
                c3*x0*y1 + np.sqrt(-4*c3*(x0 - x1)*(c0 - Ec + c1*x0 + c2*y0 + c3*x0*y0)*(y0 -
                y1) + (c1*(-x0 + x1) + c2*(-y0 + y1) + c3*(-2*x0*y0 + x1*y0 + x0*y1))**2))


            te2 = (1/(
                2*c3*(x0 - x1)*(y0 - y1)))*(c1*x0 - c1*x1 + c2*y0 + 2*c3*x0*y0 -
                c3*x1*y0 - c2*y1 - c3*x0*y1 + np.sqrt(-4*c3*(x0 - x1)*(c0 -
                Ec + c1*x0 + c2*y0 + c3*x0*y0)*(y0 -
                y1) + (c1*(-x0 + x1) + c2*(-y0 + y1) + c3*(-2*x0*y0 + x1*y0 + x0*y1))**2))
 
            tc2 = -(1/(
                2*c3 (y0 - y1)))*(c1*x0 - c1*x1 + c2*y0 - c3*x1*y0 - c2*y1 + c3*x0*y1 + 
                np.sqrt(-4*c3*(x0 - x1)*(c0 - Ec + c1*x0 + c2*y0 + c3*x0*y0) (y0 - 
                y1) + (c1*(-x0 + x1) + c2*(-y0 + y1) + 
                c3*(-2*x0*y0 + x1*y0 + x0*y1))**2))
            
            te1 = check_inside(te1, rtol=rtol, atol=atol)
            te2 = check_inside(te2, rtol=rtol, atol=atol)
            
            if te1 is not None:
                param_edge.append(te1)
                param_isocurve.append(tc1)
                intersecting_edges.append(edge)
            elif te2 is not None:
                param_edge.append(te2)
                param_isocurve.append(tc2)
                intersecting_edges.append(edge)
            else:
                continue

    # When the intersection occurs at a corner, there may be duplicate intersections.
    # Let's remove duplicate points

    # Find the points of intersections.
    xy = []
    for p,e in zip(param_edge, intersecting_edges):
        xy.append(get_param_xy(p, e))

    # print("xy", xy)

    unique_xy = []
    unique_param_edge = []
    unique_param_isocurve = []
    unique_intersecting_edges = []

    # Only keep the unique intersections.
    for i in range(len(xy)):
        if not check_contained([xy[i]], unique_xy, rtol=rtol, atol=atol):
            unique_xy.append(xy[i])
            unique_param_edge.append(param_edge[i])
            unique_param_isocurve.append(param_isocurve[i])
            unique_intersecting_edges.append(intersecting_edges[i])

    # print("unique: ", unique_xy)
                    
    return unique_param_edge, unique_param_isocurve, unique_intersecting_edges, unique_xy


def get_param_xy(param, edge):
    """Get the x- and y-values where an isocurve intersects an edge of a parallelogram.
        
    Args:
        param (float): the value of the parameter.
        edge (list or numpy.ndarray): a list of two xy-coordinates that define an edge.
    
    Returns:
        _ (list): the xy-values of the intersection corresponding to the parameter value.
    """
    
    x0, x1, y0, y1 = edge[0][0], edge[1][0], edge[0][1], edge[1][1]
        
    # Find the x- and y-coordinates that coorespond to this value of the parameter.
    x = (1 - param)*x0 + param*x1
    y = (1 - param)*y0 + param*y1
    
    return [x, y]

def bilin_density_of_states(square_pts, coeffs, isovalue, atol=1e-8, rtol=1e-5, eps=1e-9):
    """Calculate the density of states of a bilinear interpolation
    within a parallelogram.
    
    Args:
        square_pts (list): A list of coordinates of the corners of the parallelogram.
            This list should start with the lower left point and traverse the 
            parallelogram in counterclockwise order.
        coeffs (list or numpy.ndarray): the coefficients of the bilinear
            interpolation.
        isovalue (float): the energy at which to calculate the density of states.
        atol (float): the absolute tolerance used in "find_param_intersect".
        rtol (float): the relative tolerance used in "find_param_intersect".
        eps (float): the analytic solution to the density of states for a bilinear 
            interpolation is proportional to a log function and divergencs if the 
            integration domain covers any interval including 0. This value is an 
            approximation of zero when the integration domains includes 0.
            
    Returns:
        dos (float): the density of states
    """

    # rprint("square_pts", square_pts)
    # rprint("coeffs", coeffs)
    # rprint("isovalue", isovalue)
                                                
    (edge_params, isocurve_params,
     edges, intersect_pts) = find_param_intersect(square_pts, coeffs, isovalue, atol=atol,
                                                 rtol=rtol)
    
    if len(isocurve_params) < 2:
        return 0

    # rprint("edge_params", edge_params)
    # rprint("isocurve_params", isocurve_params)
    # rprint("edges", edges)
    

    # Find the locations where the isocurve intersects the parallelogram boundaries.
    # intersect_pts = []
    # for p,e in zip(edge_params, edges):
    #     intersect_pts.append(get_param_xy(p, e))    

    # Get the coefficients.
    c0, c1, c2, c3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    
    # Initialize the density of states.
    dos = 0
    
    grouped_intersect_pts, grouped_edge_params, grouped_isocurve_params, grouped_edges = (
        group_by_quad(intersect_pts, edge_params, isocurve_params, edges))

    for i in range(len(grouped_intersect_pts)):
        # print("dos", dos)
        edge_params = grouped_edge_params[i]
        isocurve_params = grouped_isocurve_params[i]
        edges = grouped_edges[i]
            
        # rprint("loop edge_params", edge_params)
        # rprint("loop isocurve_params", isocurve_params)
        # rprint("loop edges", edges)        
        
        # Get the limits of the integration parametric variable and the
        # coefficients in the bilinear expansion.
        ti, tf = isocurve_params[0], isocurve_params[1]
        
        # Sort the bounds from least to greatest (this is the similar
        # to taking the absolute value.
        if ti > tf:
            ti, tf = tf, copy(ti)
        
        # print("ti: ", ti)
        # print("tf: ", tf)
        
        # Take care of the case when c1 == c2 == c3 == 0.
        if (np.isclose(c1, 0, rtol=rtol, atol=atol) and
            np.isclose(c2, 0, rtol=rtol, atol=atol) and
            np.isclose(c3, 0, rtol=rtol, atol=atol)):
            
            if np.isclose(c0, isovalue, rtol=rtol, atol=atol):
                print("singular1")
                dos += 1/eps
                continue
            else:
                dos += 0.
                continue
            
        # Take care of the case when c2 == c3 == 0.
        if (np.isclose(c2 ,0, rtol=rtol, atol=atol) and
            np.isclose(c3, 0, rtol=rtol, atol=atol)):
            dos += 1/abs(c1)*abs(tf - ti)
            continue
        
        # Take care of the case when c3 == 0.
        # (this is only a special case because of the change of integration variables)
        if np.isclose(c3, 0, rtol=rtol, atol=atol):
            dos += 1/abs(c2)*abs(tf - ti)
            continue
        
        # Change of variables.
        ti_p = c2 + c3*ti
        tf_p = c2 + c3*tf    
        
        # Sort the bounds from least to greatest (this is the similar
        # to taking the absolute value.
        if ti_p > tf_p:
            ti_p, tf_p = tf_p, copy(ti_p)

        # print("ti_p: ", ti_p)
        # print("tf_p: ", tf_p)
        
        # If both bounds are greater than zero.
        if (0 < ti_p) and (0 < tf_p):

            dos += 1/abs(c3)*np.log(tf_p/ti_p)
            continue
        
        # If both bounds are less than zero.
        elif (ti_p < 0) and (tf_p < 0):
            dos += -1/abs(c3)*np.log(tf_p/ti_p)
            continue
        
        # If the lower bound is zero.
        elif np.isclose(ti_p, 0, rtol=rtol, atol=atol):
            print("singular2")                        
            dos += -1/abs(c3)*np.log(tf_p/eps)
            continue
        
        # If the upper bound is zero.
        elif np.isclose(tf_p, 0, rtol=rtol, atol=atol):
            print("singular3")                        
            # rprint("2", -1/c3*np.log(-eps/ti_p))
            dos += -1/abs(c3)*np.log(-eps/ti_p)
            continue
        
        # If the boundaries straddle zero.
        elif (ti_p < 0) and (0 < tf_p ):
            # print("singular4")            
            dos += -1/abs(c3)*np.log(-eps/ti_p) + 1/abs(c3)*np.log(tf_p/eps)
            continue
        
    return dos


def group_by_quad(pts, param_edge, param_isocurve, intersecting_edges):
    """Group a list of points by quadrant, starting with the first and ending with the 
    fourth. Group the other arguments in the same manner.
    
    Args:
        pts (list): a list of points in 2D Cartesian space.
        param_edge (list): a list of parametric variables.
        param_isocurve (list): a list of parametric variables.
        intersecting_edges (list): a list of edges given by two points at the ends of the
            edge.
        
    Returns:
        grouped_pts (list): a list of groups of points.
        grouped_param_edge (list): a list of grouped edge parameters.
        grouped_param_isocurve (list): a list of grouped curve parameters.
        grouped_intersecting_edges (list): a list of grouped edges that intersect the
            the isocurve
    """
    
    pts = np.array(pts)
    param_edge = np.array(param_edge)
    param_isocurve = np.array(param_isocurve)
    intersecting_edges = np.array(intersecting_edges)
    
    # Get the x- and y-coordinates.
    xs = pts[:,0]
    ys = pts[:,1]    
    
    # Find the coordinates that are greater than zero and less than zero.
    x_positive = xs > 0
    x_negative = xs < 0

    y_positive = ys > 0
    y_negative = ys < 0    
    
    # Find the points in each quadrant.
    q1 = pts[x_positive*y_positive]
    q2 = pts[x_negative*y_positive]
    q3 = pts[x_negative*y_negative]
    q4 = pts[x_positive*y_negative]
    
    grouped_pts = np.array([q1, q2, q3, q4])
    
    # Remove quadrants that don't contain any points.    
    grouped_pts = grouped_pts[[len(g) > 1 for g in grouped_pts]]

    if len(grouped_pts) == 0:
        
        return (np.array([pts]), np.array([param_edge]), np.array([param_isocurve]),
                np.array([intersecting_edges]))
    
    else:

        q1 = param_edge[x_positive*y_positive]
        q2 = param_edge[x_negative*y_positive]
        q3 = param_edge[x_negative*y_negative]
        q4 = param_edge[x_positive*y_negative]
        
        grouped_param_edge = np.array([q1, q2, q3, q4])
        
        # Remove quadrants that don't contain any points.        
        grouped_param_edge = grouped_param_edge[[len(g) > 1 for g in grouped_param_edge]]

        q1 = param_isocurve[x_positive*y_positive]
        q2 = param_isocurve[x_negative*y_positive]
        q3 = param_isocurve[x_negative*y_negative]
        q4 = param_isocurve[x_positive*y_negative]
        
        grouped_param_isocurve = np.array([q1, q2, q3, q4])
        
        # Remove quadrants that don't contain any points.    
        grouped_param_isocurve = grouped_param_isocurve[[len(g) > 1 for g in
                                                         grouped_param_isocurve]]

        q1 = intersecting_edges[x_positive*y_positive]
        q2 = intersecting_edges[x_negative*y_positive]
        q3 = intersecting_edges[x_negative*y_negative]
        q4 = intersecting_edges[x_positive*y_negative]
        
        grouped_intersecting_edges = np.array([q1, q2, q3, q4])
        
        # Remove quadrants that don't contain any points.    
        grouped_intersecting_edges = grouped_intersecting_edges[[len(g) > 1 for g in
                                                            grouped_intersecting_edges]]
        
    return (grouped_pts, grouped_param_edge, grouped_param_isocurve,
            grouped_intersecting_edges)


# def group_by_quad(pts):
#     """Group a list of points by quadrant, starting with the
#     first and ending with the fourth.
    
#     Args:
#         pts (list): a list of points in 2D Cartesian space.
        
#     Returns:
#         grouped_pts (list): a list of groups of points.
#     """
    
#     pts = np.array(pts)
    
#     # Get the x- and y-coordinates.
#     xs = pts[:,0]
#     ys = pts[:,1]    
    
#     # Find the coordinates that are greater than zero and less than zero.
#     x_positive = xs > 0
#     x_negative = xs < 0

#     y_positive = ys > 0
#     y_negative = ys < 0    
    
#     # Find the points in each quadrant.
#     q1 = pts[x_positive*y_positive]
#     q2 = pts[x_negative*y_positive]
#     q3 = pts[x_negative*y_negative]
#     q4 = pts[x_positive*y_negative]
    
#     grouped_pts = np.array([q1, q2, q3, q4])
    
#     # Remove quadrants that don't contain any points.    
#     grouped_pts = grouped_pts[[len(g) > 1 for g in grouped_pts]]    
    
#     return grouped_pts
