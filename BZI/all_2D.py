"""Methods to for calculating and visualizing lattices and Brillouin zones in 2D."""

from numpy.linalg import norm, inv, det
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
from copy import deepcopy, copy
from scipy.spatial import ConvexHull

from BZI.symmetry import get_minmax_indices, bring_into_cell, make_rptvecs
from BZI.utilities import remove_points, rprint, check_inside, check_contained
plt.style.use('seaborn-colorblind')

def make2D_lattice_basis(lattice_constants, lattice_angle):
    """Create the basis vectors that generate a lattice in 2D.
    
    Args:
        lattice_constants ((2,) list or numpy.ndarray): the lengths of the basis vectors.
        lattice_angle (float): the angle between the lattice vectors in radians.
        
    Returns:
        basis ((2,2) numpy.ndarray): the basis vectors of the lattice as columns of a 
            array.
    
    Example:
        >>> lat_consts = [1, 1]
        >>> lat_angle = np.pi/2
        >>> make2D_lattice_basis(lat_consts, lat_angle)
        np.array([[1, 0], [0, 1]])
    """
    
    v1 = lattice_constants[0]*np.array([1, 0])
    v2 = np.array([lattice_constants[1]*np.cos(lattice_angle),
                   lattice_constants[1]*np.sin(lattice_angle)])
    basis = np.transpose([v1, v2])
    return basis


def get_2Dlattice_type(basis, rtol=1e-5, atol=1e-8):
    """Find the lattice type from the lattice generating vectors in 2D.
    
    Args:
        basis ((2,2) numpy.ndarray): the lattice generating vectors as columns of a
            2D array.
        rtol (float): relative tolerance.
        atol (float): absolute tolerance.

    Returns:
        (str): the lattice type.

    Example:
        >>> lat_basis = numpy.array([[1, 0], [0, 1]])
        >>> get_2Dlattice_type(lat_basis)
        "square"
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
        S ((2,2) numpy.ndarray): a array of integers
        eps (int): finite precision parameter.
        rtol (float): relative tolerance.
        atol (float): absolute tolerance.
        
    Returns:
        B ((2,2) numpy.ndarray): the unimodular transformation matrix.
        H ((2,2) numpy.ndarray): the Hermite normal form of the integer matrix.

    Example:
        >>> int_mat = numpy.array([[2, 5], [4, 7]])
        >>> HermiteNormalForm2D(int_mat)
        ( array( [[1, 0], [5, 6]] ), array( [[3, 5], [-1, -2]] ) )
    """
        
    # Make sure the input is integer.
    if not (np.allclose(np.max(S%1), 0, rtol=rtol, atol=atol) or
            np.allclose(np.min(S%1), 0, rtol=rotl, atol=atol)):
        msg = "Please provide an integer 2x2 array."
        raise ValueError(msg)
    
    H = deepcopy(S)
    B = np.eye(2,2).astype(int)
    
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
    """Sample within a parallelogram with a regular grid.

    Args:
        lat_vecs ((2,2) numpy.ndarray): the vectors defining the area in which 
            to sample. The vectors are the columns of the array.
        grid_vecs ((2,2) numpy.ndarray): the vectors that generate the grid as 
            columns of an array.
        offset ((2,) numpy.ndarray): the offset of the coordinate system in grid coordinates.
        coords (str): a string that determines the coordinate of the returen k-points.
            Options include "Cart" for Cartesian and "lat" for lattice coordinates.
        grid_type (str): if "closed" the grid will include points along both 
            boundaries. If open, only points on one boundary are included.       
        rtol (float): relative tolerance
        atol (float): absolute tolerance

    Returns:
        grid ((N,2) numpy.ndarray): an array of sampling-point coordinates.

    Example:
        >>> lat_vecs = np.array([[1, 0], [0, 1]])
        >>> grid_vecs = np.array([[.5, 0], [0, .5]])
        >>> make_cell_points2D(lat_vecs, grid_vecs)
        array([[0. , 0.], [0., 0.5], [0.5, 0. ], [0.5, 0.5]])
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
        return np.array(grid)
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
        return np.array(grid)
    else:
        msg = "Coordinate options include 'Cart' and 'lat'."
        raise ValueError(msg)


def plot_mesh2D(grid, lattice_basis, offset = np.array([0,0]), ax=None, color="black"):
    """Plot points and the unit cell of a lattice in 2D.

    Args:
        grid ((N,2) numpy.ndarray): a list of points two plot in 2D.
        lattice_basis ((2,2) numpy.ndarray): the generating vectors of the lattice as columns
            of anarray.
        offset ((2,) list or numpy.ndarray): the offset of the unit cell in Cartesian 
            coordinates.
    Returns:
        None

    Example:
        >>> lat_vecs = np.array([[1, 0], [0, 1]])
        >>> grid = array([[0. , 0.], [0., 0.5], [0.5, 0. ], [0.5, 0.5]])
        >>> plot_mesh(grid, lat_vecs)
        None
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

    return None

def get_circle_pts(A, r2, offset=[0.,0.], eps=1e-12):
    """ Calculate all the points within a circle that are
    given by an integer linear combination of the columns of 
    A.
    
    Args:
        A ((2,2) numpy.ndarray): the columns representing basis vectors.
        r2 (float): the squared radius of the circle.
        offset((2,) list or numpy.ndarray): a vector that points to the center
            of the circle in Cartesian coordinates.
        offset ((2,) numpy.ndarray): the center of the circle.

    Returns:
        grid (list): an array of grid coordinates in cartesian
            coordinates.
    
    Example:
        >>> lat_vecs = np.array([[.5, 0], [0, .5]])
        >>> r2 = 0.4
        >>> get_circle_pts(grid_vecs, .4)
        array([[-0.5, 0.], [0., -0.5], [0.,  0. ], [0.,  0.5],[0.5,  0. ]])
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
        mesh_points ((N,2) list or np.ndarray): a list of mesh points.
        r2 (float): the squared radius of the circle
        offset ((2,) list or np.ndarray): the offset of the circle.
    
    Returns:
        None
    
    Example:
        >>> mesh = np.array([[-0.5, 0.], [0., -0.5], [0.,  0. ], [0.,  0.5],[0.5,  0. ]])
        >>> r2 = 0.4
        >>> plot_circle_mesh(mesh, r2)
        None
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the sphere
    u = np.linspace(0, 2 * np.pi, 1000)
    r = np.sqrt(r2)
    x = r*np.cos(u) + offset[0]
    y = r*np.sin(u) + offset[0]
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

    return None

def get_perpendicular_vector2D(vector, atol=1e-8, rtol=1e-5):
    """Find a unit vector perpendicular to the input vector in 2D.
    
    Args:
        vector ((2,) list or numpy.ndarray): a vector in 2D.
        rtol (float): relative tolerance.
        atol (float): absolute tolerance.       
        
    Returns:
        perp_vector (numpy.ndarray): a vector perpendicular to the input
            vector.

    Example:
        >>> get_perpendicular_vector2D([1, 0])
        np.array([0, 1])
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
        pt1 ((2,) list or numpy.ndarray): a point in 2D
        pt2 ((2,) list or numpy.ndarray): a point in 2D
        
    Returns:
        _ ((2,) list): the equation of a line. The first element is a 
            vector normal to the line. The second is the closest distance from the
            origin to the line along the direction of the normal vector.

    Example:
        >>> pt1 = [2, 1]
        >>> pt2 = [3, 1]
        >>> get_line_equation2D(pt1, pt2)
        [array([0., 1.]), 1.0]
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
        point ((2,) numpy.ndarray): a point in Cartesian coordinates.
        line ((2,) list): an array with two elements. The first provides
            a vector normal to the line. The second element is the distance of the
            line from the origin in the direction of the vector normal to the line.
        rtol (float): relative tolerance
        atol (float): absolute tolerance

    Returns:
        (str): a string that indicates where the point is located. Options include
            "inside", "outside", and "on".

    Example:
        >>> line = [np.array([0., 1.]), 1.0]
        >>> pt = [1, 0]
        >>> point_line_location(pt, line)
        "inside"
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


def find_2Dbz(reciprocal_lattice_basis):
    """Find the Brillouin zone of a 2D lattice

    Args:
        reciprocal_lattice_basis ((2,2) numpy.ndarray): the lattice generating vectors as
            columns of a 2x2 array.

    Returns:
        convex_hull (scipy.spatial.ConvexHull): the Brillouin zone

    Example:
        >>> lat_basis = np.array([[1, 0], [0, 1]])
        >>> find_2Dbz(lat_basis)
    """
    
    # Find all the lattice points near the origin within a circle of 
    # radius of two times the longest lattice vector.
    r2 = (2*np.max(norm(reciprocal_lattice_basis, axis=0)))**2
    circle_pts = get_circle_pts(reciprocal_lattice_basis, r2)
    
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
    while not np.isclose(volume, det(reciprocal_lattice_basis)):
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
        mesh_points ((N,2) list or numpy.ndarray): a list of lattice points.
        bz (scipy.spatial.ConvexHull): the Brillouin zone.

    Return:
        None
    
    Example:
        >>> lat_basis = np.array([[1, 0], [0, 1]])
        >>> bz = find_2Dbz(lat_basis)
        >>> mesh = [[0, 0], [0, .5], [.5, 0], [.5, .5]]
        >>> plot_all2D_bz(mesh, bz)
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

    return None
        
        
class FreeElectron2D():
    """This is the popular free electron model. In this model the potential is
    zero everywhere. It is useful for testing.
    
    Args:
        lattice_basis ((2,2) numpy.ndarray): the lattice basis vectors as columns of
            an array.
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
        lattice_basis ((2,2) numpy.ndarray): the lattice basis vectors as columns of
            an array.
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

    Example:
        >>> lattice_basis = make2D_lattice_basis([1,1], np.pi/2)
        >>> args = {"lattice_basis": lattice_basis,
                    "degree": 2,
                    "prefactor":1,
                    "nvalence_electrons": 6,
                    "nsheets": 10}
        >>> free_2D = FreeElectron2D(**args)
    """
    
    def __init__(self, lattice_basis, degree, nvalence_electrons, energy_shift=None,
                 fermi_level=None, band_energy=None, prefactor=None, nsheets=None,
                 convention="ordinary", rtol=None, atol=None):
        
        self.material = "2D free electron model"
        self.lattice_basis = lattice_basis
        
        if convention is None:
            self.convention = "ordinary"
        else:
            self.convention = convention
        
        self.reciprocal_lattice_basis = make_rptvecs(lattice_basis,
                                                     convention=self.convention)
        
        if prefactor is None:
            self.prefactor = 1
        else:
            self.prefactor = prefactor
        
        if degree == 3:
            msg = "In 2D, the dispersion relation cannot be of degree three."
            raise ValueError(msg.format(degree))
        self.degree = degree
        self.nvalence_electrons = nvalence_electrons
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.
        self.fermi_level_ans = self.prefactor*(self.nvalence_electrons*
                                               det(self.reciprocal_lattice_basis)/
                                               (2*np.pi))**(self.degree/2)        
        self.band_energy = band_energy or 0.
        self.band_energy_ans = 2*np.pi*self.prefactor/(self.degree + 2)*(
            self.fermi_level_ans/self.prefactor)**(1 + 2/self.degree)
        self.nfilled_states = self.nvalence_electrons/2.
        
        if nsheets is None:
            self.nsheets = 3
        else:
            self.nsheets = nsheets
        
        if rtol is None:
            self.rtol = rtol
        else:
            self.rtol = rtol
        
        if atol is None:
            self.atol = atol
        else:
            self.atol = atol
            
        
    def eval(self, kpoint, nsheets=None, sigma=False, refine=False):
        """Evaluate the free electron model at a point in k-space.
        """

        if nsheets is None:
            nsheets = self.nsheets
        
        kpoint = np.array(kpoint)
        
        # offset = np.dot(self.reciprocal_lattice_basis, [0.5]*2)
        pts = [np.dot(self.reciprocal_lattice_basis, [i,j]) for i,j in it.product(range(-2,3),
                                                                       repeat=2)]
        # Return the sum of the bands
        if sigma:
            
            values = np.array([np.linalg.norm(kpoint - pt)**self.degree for pt in pts])
            return np.sum(values[values < self.fermi_level])
        
        else:
            return np.sort([np.linalg.norm(kpoint - pt)**self.degree
                            for pt in pts])[:nsheets]
        
    def change_potential(self, prefactor, degree):
        if degree == 3:
            msg = "In 2D, the dispersion relation cannot be of degree three."
            raise ValueError(msg.format(degree))
        self.degree = degree
        self.prefactor = prefactor
        self.fermi_level_ans = self.prefactor*(self.nvalence_electrons*
                                               det(self.reciprocal_lattice_basis)/
                                               2*np.pi)**(self.degree/2)
        self.band_energy_ans = 2*np.pi*self.prefactor*self.degree*(
            (self.fermi_level_ans/self.prefactor)**(1-1/self.degree))

    def eval_dos(self, energy, rtol=1e-5, atol=1e-8):
        """Calculate the exact density of states at a given energy.
        """
        if energy > 0 or np.isclose(0, energy, rtol=rtol, atol=atol):
            
            return (2*np.pi/(self.degree*self.prefactor)*
                    (energy/self.prefactor)**(2/self.degree - 1))*2
        else:
            return 0

    def eval_nos(self, energy, rtol=1e-5, atol=1e-8):
        """Calculate the exact number of states at a given energy.
        """
        if energy > 0 or np.isclose(0, energy, rtol=rtol, atol=atol):            
            return np.pi*(energy/self.prefactor)**(2/self.degree)*2
        else:            
            return 0
        
        
def plot_2Dbands(EPM, sigma=False):
    """Plot the band structure of a 2D empirical pseudopotential.
    
    Args:
        EPM (class): an empirical pseudopotential object.

    Returns:
        None

    Example:
        >>> lattice_basis = make2D_lattice_basis([1,1], np.pi/2)
        >>> args = {"lattice_basis": lattice_basis,
                    "degree": 2,
                    "prefactor":1,
                    "nvalence_electrons": 6,
                    "nsheets": 10}
        >>> free_2D = FreeElectron2D(**args)
        >>> plot_2Dbands(free_2D)
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("equal")
    
    grid_basis = EPM.reciprocal_lattice_basis/100
    offset = np.dot(inv(grid_basis), np.dot(EPM.reciprocal_lattice_basis, [-.5]*2)) # + [.5]*2
    # offset = [0, 0]
    grid = make_cell_points2D(EPM.reciprocal_lattice_basis, grid_basis, offset,
                              grid_type="closed")
    
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

    return None
        
def plot_2Dfermi_curve(EPM, neigvals, ndivs, atol=1e-2, ax=None):
    """Plot the bands of a 2D empirical pseudopotential
    
    Args:
        EPM (class): an empirical pseudopotential object.
        neigvals (int): the number of eigenvalues to plot.
        ndivs (int): the number of divisions for the grid.
        atol (float): the tolerance for the Fermi level.
        ax (matplotlib.axes): an axes object.

    Return:
        None

    Example:
        >>> lattice_basis = make2D_lattice_basis([1,1], np.pi/2)
        >>> args = {"lattice_basis": lattice_basis,
                    "degree": 2,
                    "prefactor":1,
                    "nvalence_electrons": 6,
                    "nsheets": 10}
        >>> free_2D = FreeElectron2D(**args)
        >>> plot_2Dfermi_curve(free_2D, 5, 100, 1e-3)
    """

    if ax is None:
        fig,ax = plt.subplots()

    ax.set_aspect("equal")        
        
    grid_basis = EPM.reciprocal_lattice_basis/ndivs
    offset = np.dot(inv(grid_basis), np.dot(EPM.reciprocal_lattice_basis, [-.5]*2)) + [.5]*2
    grid = make_cell_points2D(EPM.reciprocal_lattice_basis, grid_basis, offset)

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

    offset = np.dot(EPM.reciprocal_lattice_basis, [-.5]*2)
    c1 = EPM.reciprocal_lattice_basis[:,0]
    c2 = EPM.reciprocal_lattice_basis[:,1] 
    O = np.asarray([0.,0.])

    l1 = zip(O + offset, c1 + offset)
    l2 = zip(c2 + offset, c1 + c2 + offset)
    l3 = zip(O + offset, c2 + offset)
    l4 = zip(c1 + offset, c1 + c2 + offset)
    ls = [l1, l2, l3, l4]
    for l in ls:
        ax.plot(*l, c="blue")

    return None
        
        
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
    band_energy = np.sum(energies)*np.linalg.det(EPM.reciprocal_lattice_basis)/(
        np.sum(weights))
    
    return fermi_level, band_energy


def square_tesselation(grid, atol=1e-5, rtol=1e-8):
    """Create a tesselation of squares from a grid.
    
    Args:
        grid (list or numpy.ndarray): a list of points in 2D.
        rtol (float): relative tolerance
        atol (float): absolute tolerance

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
            # Sort the points in the tesselation in counter-clockwise order starting with
            # the lower left point.
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
    """Calculate the values of the parameter where the constant energy curve 
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
        unique_edge_indices (list): a list of edge indices where the isocurve intersects
            the edges of the parallelogram.
        unique_intersecting_edges (list): a list of edges intersected by the isocurve.
        unique_xy (list): the xy-coordinates where the isocurve intersects the boundary of
            the paralellogram.
    """
    
    # Initialize parameters associated with the intersections.
    param_edge = []
    param_isocurve = []
    intersecting_edges = []
    edge_list = [[square_pts[i], square_pts[ (i+1)%4 ]] for i in range(4)]

    edge_indices = []

    # Initialize unique parameters associated with the intersections.
    unique_param_edge = []
    unique_param_isocurve = []
    unique_edge_indices = []    
    unique_intersecting_edges = []
    unique_xy = []
        
    # c0, c1, c2, c3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    c0, c1, c2, c3 = coeffs
    Ec = isovalue

    if (np.isclose(c1, 0, rtol=rtol, atol=atol) and
        np.isclose(c2, 0, rtol=rtol, atol=atol) and
        np.isclose(c3, 0, rtol=rtol, atol=atol)):
        
        return (unique_param_edge, unique_param_isocurve,
                unique_intersecting_edges, unique_edge_indices, unique_xy)

    for i,edge in enumerate(edge_list):
        
        x0, x1, y0, y1 = edge[0][0], edge[1][0], edge[0][1], edge[1][1]
                                    
        # Have to consider c2 == c3 == 0 as a separate case.
        if (np.isclose(c2, 0, rtol=rtol, atol=atol) and
              np.isclose(c3, 0, rtol=rtol, atol=atol)):
            
            # If the x-coordinates are the same.
            if np.isclose(x0, x1, rtol=rtol, atol=atol):

                if np.isclose(x0, (Ec-c0)/c1, rtol=rtol, atol=atol):
                    
                    unique_param_edge = [0, 1]
                    unique_param_isocurve = [y0, y1]
                    unique_edge_indices = [i, i]
                    unique_intersecting_edges = [edge, edge]
                    unique_xy = [[x0, y0], [x0, y1]]
                    return (unique_param_edge, unique_param_isocurve,
                            unique_edeg_indices, unique_intersecting_edges, unique_xy)
                else:
                    continue
                
            # If the y-coordinates are the same.
            elif np.isclose(y0, y1, rtol=rtol, atol=atol):
                te = check_inside((c0 - Ec + c1*x0)/(c1*x0 - c1*x1), rtol=rtol, atol=atol)

                if te is not None:                 
                    param_edge.append(te)
                    param_isocurve.append(y0)
                    edge_indices.append(i)
                    intersecting_edges.append(edge)                    

            # If both the x- and y-coordinates are different.
            else:
                te = check_inside((c0 - Ec + c1*x0)/(c1*x0 - c1*x1), rtol=rtol, atol=atol)
                if te is not None:
                    x,tc = get_param_xy(te, edge)
                    param_edge.append(te)
                    param_isocurve.append(tc)
                    edge_indices.append(i)
                    intersecting_edges.append(edge)                    
                else:
                    continue
        
        # Have to consider c1 == c3 == 0 as a separate case.
        elif (np.isclose(c1, 0, rtol=rtol, atol=atol) and
              np.isclose(c3, 0, rtol=rtol, atol=atol)):
            
            # If the y-coordinates are the same.
            if np.isclose(y0, y1, rtol=rtol, atol=atol):
                
                if np.isclose(y0, (Ec-c0)/c2, rtol=rtol, atol=atol):
                    
                    unique_param_edge = [0, 1]
                    unique_param_isocurve = [x0, x1]
                    unique_edge_indices = [i, i]
                    unique_intersecting_edges = [edge, edge]
                    unique_intersecting_pts = [[x0, y0],[x1, y0]]
                    return (unique_param_edge, unique_param_isocurve,
                            unique_edge_indices, unique_intersecting_edges, intersecting_pts)
                
                else:
                    continue
            
            # If the x-coordinates are the same.
            elif np.isclose(x0, x1, rtol=rtol, atol=atol):
                te = check_inside((c0 - Ec + c2*y0)/(c2*y0 - c2*y1), rtol=rtol, atol=atol)
                
                if te is not None:
                    param_edge.append(te)
                    param_isocurve.append(x0)
                    edge_indices.append(i)
                    intersecting_edges.append(edge)
            
            # If both the x- and y-coordinates are different.
            else:
                te = check_inside((c0 - Ec + c2*y0)/(c2*y0 - c2*y1), rtol=rtol, atol=atol)
                
                if te is not None:
                    tc,y = get_param_xy(te, edge)
                    param_edge.append(te)
                    param_isocurve.append(tc)
                    edge_indices.append(i)
                    intersecting_edges.append(edge)
                
                else:
                    continue
        
        # General case with x0 == x1 and y0 != y1
        elif np.isclose(x0, x1, atol=atol, rtol=rtol):
                        
            te = (c0 - Ec + c1*x0 + c2*y0 + c3*x0*y0)/((c2 + c3*x0)*(y0 - y1))
            te = check_inside(te, rtol=rtol, atol=atol)            
            tc =  x0

            if te is not None:
                param_edge.append(te)
                param_isocurve.append(tc)
                edge_indices.append(i)                
                intersecting_edges.append(edge)
            else:
                continue
        
        # General case with x0 != x1 and y0 == y1
        elif np.isclose(y0, y1, atol=atol, rtol=rtol):
            
            te = (c0 - Ec + c1*x0 + c2*y0 + c3*x0*y0)/((x0 - x1)*(c1 + c3*y0))
            te = check_inside(te, rtol=rtol, atol=atol)
            tc = -((c0 - Ec + c2*y0)/(c1 + c3*y0))
            
            if te is not None:
                print("i2?: ", i)
                param_edge.append(te)
                param_isocurve.append(tc)
                edge_indices.append(i)
                intersecting_edges.append(edge)
            else:
                continue
        
        # General case with x0 != x1 and y0 != y1
        else:
            with np.errstate(invalid='raise'):
                try:                    
                    
                    te1 = -(-(c1*x0) + c1*x1 - c2*y0 - 2*c3*x0*y0 + c3*x1*y0 + c2*y1 +
                            c3*x0*y1 + np.sqrt(-4*c3*(x0 - x1)*(c0 - Ec + c1*x0 + c2*y0 +
                            c3*x0*y0)*(y0 - y1) + (-(c1*x0) + c1*x1 - c2*y0 - 2*c3*x0*y0 +
                            c3*x1*y0 + c2*y1 + c3*x0*y1)**2))/(2.*c3*(x0 - x1)*(y0 - y1))
                except:
                    te1 = np.nan            

                try:
                    tc1 = (-(c1*x0) + c1*x1 - c2*y0 + c3*x1*y0 + c2*y1 - c3*x0*y1 +
                           np.sqrt(-4*c3*(x0 - x1)*(c0 - Ec + c1*x0 + c2*y0 +
                           c3*x0*y0)*(y0 - y1)+ (-(c1*x0) + c1*x1 - c2*y0 - 2*c3*x0*y0 +
                                  c3*x1*y0 + c2*y1 + c3*x0*y1)**2))/(2.*c3*(y0 - y1))
                except:
                    tc1 = np.nan

                try:
                    te2 = (c1*x0 - c1*x1 + c2*y0 + 2*c3*x0*y0 - c3*x1*y0 - c2*y1 -
                           c3*x0*y1 + np.sqrt(-4*c3*(x0 - x1)*(c0 - Ec + c1*x0 + c2*y0 +
                           c3*x0*y0)*(y0 - y1) + (-(c1*x0) + c1*x1 - c2*y0 - 2*c3*x0*y0 +
                           c3*x1*y0 + c2*y1 + c3*x0*y1)**2))/(2.*c3*(x0 - x1)*(y0 - y1))

                except:
                    te2 = np.nan
                                    
                try:
                    tc2 = -(c1*x0 - c1*x1 + c2*y0 - c3*x1*y0 - c2*y1 + c3*x0*y1 +
                            np.sqrt(-4*c3*(x0 - x1)*(c0 - Ec + c1*x0 + c2*y0 +
                                    c3*x0*y0)*(y0 - y1) + (-(c1*x0) + c1*x1 - c2*y0 - 2*c3*x0*y0 +
                                    c3*x1*y0 + c2*y1 + c3*x0*y1)**2))/(2.*c3*(y0 - y1))
                    
                except:
                    tc2 = np.nan

            te1 = check_inside(te1, rtol=rtol, atol=atol)
            te2 = check_inside(te2, rtol=rtol, atol=atol)
            
            if te1 is not None:
                param_edge.append(te1)
                param_isocurve.append(tc1)
                edge_indices.append(i)                
                intersecting_edges.append(edge)
            
            if te2 is not None:
                param_edge.append(te2)
                param_isocurve.append(tc2)
                edge_indices.append(i)                
                intersecting_edges.append(edge)

    # When the intersection occurs at a corner, there may be duplicate intersections.
    # Let's remove duplicate points

    # Find the points of intersections.
    xy = []
    for p,e in zip(param_edge, intersecting_edges):
        xy.append(get_param_xy(p, e))

    # Only keep the unique intersections.
    for i in range(len(xy)):
        if not check_contained([xy[i]], unique_xy, rtol=rtol, atol=atol):
            unique_xy.append(xy[i])
            unique_param_edge.append(param_edge[i])
            unique_param_isocurve.append(param_isocurve[i])
            unique_intersecting_edges.append(intersecting_edges[i])
            unique_edge_indices.append(edge_indices[i])

    return (unique_param_edge, unique_param_isocurve, unique_edge_indices,
            unique_intersecting_edges, unique_xy)


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


def group_by_quad(pts):
    """Group a list of points by quadrant, starting with the
    first and ending with the fourth.
    
    Args:
        pts (list): a list of points in 2D Cartesian space.
        
    Returns:
        grouped_pts (list): a list of groups of points.
    """
    
    pts = np.array(pts)
    
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
    
    return grouped_pts

def get_integration_case(edges):
    """Determine the integration case given the edges intersected by the
    bilinear isocurve.
    
    Args:
        edges (list): a list of two edge indices.
    
    Returns:
        _ (int): the integration case.
    """

    # Determine what case this integral falls under. The cases
    # are ordered by the number of unique integration limits.
    # Case 0 has no unique integration limits on v with the limits being [0, 1]. It
    # is also the case where the isocurve intersects opposite sides.
    # Case 1 has one unique integration limit on v with limits [0 v_upper]. It is also the
    # case where the isocurve intersects adjacent sides.
    # Case 2 has two unique integration limits on v with limits [v_lower, v_upper]. It is
    # also the case where the isocurve intersects the same edge twice.
    
    # Sorting the indices will reduce the number of cases.
    edges = np.sort(edges)

    if all(edges == [0, 2]) or all(edges == [1, 3]):
        return 0
    elif (all(edges == [0, 1]) or all(edges == [1, 2]) or
          all(edges == [2, 3]) or all(edges == [0, 3])):
        return 1
    elif (edges[0] == edges[1]) and (edges[0] < 4):
        return 2
    else:
        msg = "This isn't a valid integration case"
        raise ValueError(msg)


def get_integration_cases(square_pts, coeffs, isovalue,
                          atol=1e-8, rtol=1e-5, eps=1e-3):
    """Find the integration case and subcase for a bilinear in a parallelogram.
    
    Args:
        square_pts (list): A list of coordinates of the corners of the parallelogram.
        coeffs (list): a list of coefficients for the bilinear interpolation.
        isovalue (float): the value of the function on the isocontour. 
        atol (float): the absolute tolerance used when comparing the parameter to 0 and 1.
        rtol (float): the relative tolerance used when comparing the parameter to 0 and 1.
        eps (float): a finite precision parameter that is used in calculating the subcase
            for integration case 0. This value is added and subtracted from the edge
            parameter, the xy-coordinates are calculated, and then the bilinear 
            is evaluated at these points to determine the subcase.        
            
    Returns:
        integration_case_list (list): a list of integration cases.
        integration_subcase_list (list): a list of integration subcases.
    """
    
    (edge_params, isocurve_params, edge_indices,
     edges, intersect_pts) = find_param_intersect(square_pts, coeffs, isovalue)

    (grouped_pts, grouped_edge_params, grouped_isocurve_params,
     grouped_edge_indices, grouped_intersecting_edges) = (
         group_bilinear_intersections(coeffs, intersect_pts, edge_params, isocurve_params,
                      edge_indices, edges))    

    integration_case_list = []
    integration_subcase_list = []
    
    for (intersect_pts, edge_params, isocurve_params,
         edge_indices, edges) in zip(grouped_pts, grouped_edge_params,
                                     grouped_isocurve_params, grouped_edge_indices,
                                     grouped_intersecting_edges):
        
        # Find the integration case for this set of intersections.
        integration_case = get_integration_case(edge_indices)
        integration_case_list.append(integration_case)

        if integration_case == 2:

            # Find the value of the edge parameter between the isocurve intersections.
            middle_param = np.mean(edge_params)

            # Calculate the xy-coordinates of the point between the isocurve intersections
            # with the parallelogram.

            middle_xy = get_param_xy(middle_param, edges[0])

            # Evaluate the bilinear at point between intersections.
            val = eval_bilin(coeffs, middle_xy)

            # If the value is less then the isovalue, the desired area is inside the isocurve.
            if val < isovalue:
                integration_subcase = "inside"
            else:
                integration_subcase = "outside"

        elif integration_case == 1:

            # We sort the edge indices in ascending order and use the edge with the
            # lower index to calculate the value of the edge parameter. The edge and 
            # edge index will be used to evaluate the bilinear. The area around the corner
            # that is separated from the others by the isocurve is considered "inside".

            # Sort the edges where the isocurve intersectios the parallelogram.
            sorted_edge_indices = np.sort(edge_indices)

            print("sorted edge indices: ", sorted_edge_indices)

            # Calculate the value of the edge parameter that lies inside the isocurve.
            if all(sorted_edge_indices == [0,3]):
                corner_param = 0
            elif all(sorted_edge_indices == [0,1]):
                corner_param = 1
            elif all(sorted_edge_indices == [1,2]):
                corner_param = 1
            elif all(sorted_edge_indices == [2,3]):
                corner_param = 1
            else:
                msg = "Invalid edge indices"
                raise ValueError(msg)

            print("corner param: ", corner_param)

            # We used the edge index with the lower value. Here we find the
            # edge that corresponds to that index.
            corner_edge = np.array(edges)[np.argsort(edge_indices)][0]

            print("corner edge: ", corner_edge)
            
            # Calculate the xy-coordinates of the corner inside the isocurve.
            corner_xy = get_param_xy(corner_param, corner_edge)

            print("corner xy: ", corner_xy)
            
            # Evaluate the bilinear at the corner point.
            val = eval_bilin(coeffs, corner_xy)

            print("val: ", val)
            
            # If the value is less then the isovalue, the desired area is inside the isocurve.
            if val < isovalue:
                integration_subcase = "inside"
            else:
                integration_subcase = "outside"
        
        elif integration_case == 0:
            
            # This case is different because the isocurve splits the parallelogram and it
            # isn't very clear what is considered inside or outside the isocurve. If the
            # isocurve splits the parallelogram vertically (horizontally), what is
            # considered "inside" the isocurve will be the area to the left (below) the
            # isocurve.
            
            # Sort the edges where the isocurve intersects the parallelogram to guarantee
            # we select the correct edge.
            sorted_ind = np.argsort(edge_indices)

            sorted_edge_indices = np.array(edge_indices)[sorted_ind]
            sorted_edges = np.array(edges)[sorted_ind]
            sorted_edge_params = np.array(edge_params)[sorted_ind]
            
            iedge = sorted_edges[0]
    
            # Find values of the parametric variable on both sides of the isocurve but
            # not too far from it.
            param_val1 = sorted_edge_params[0] - eps
            param_val2 = sorted_edge_params[0] + eps

            # Find the xy-coordinates of these points on both sides of the isocurve.
            xy_1 = get_param_xy(param_val1, iedge)
            xy_2 = get_param_xy(param_val2, iedge)

            # Evaluate the bilinear on both sides of the isocurve.
            val1 = eval_bilin(coeffs, xy_1)
            val2 = eval_bilin(coeffs, xy_2)

            if (val1 < isovalue) and (val2 > isovalue):
                integration_subcase = "inside"

            elif (val1 > isovalue) and (val2 < isovalue):
                integration_subcase = "outside"

            else:
                msg = "Can't determine the subcase for integration case 0."
                raise ValueError(msg)        

        else:
            msg = "Invalid integration case."
            raise ValueError(msg)

        integration_subcase_list.append(integration_subcase)
    return integration_case_list, integration_subcase_list


def group_bilinear_intersections(coeffs, pts, param_edge, param_isocurve,
                                 edge_indices, intersecting_edges, atol=1e-8,
                                 rtol=1e-5):
    """Group the intersections, parameters, and edges so that each
    is associated with one isocontour.
    
    Args:
        coeffs (list): a list of coefficients for the bilinear interpolation.    
        pts (list): a list of points in 2D Cartesian space.        
        param_edge (list): a list of parametric variables for the edges.
        param_isocurve (list): a list of parametric variables for the curve.
        edge_indices (list): a list of edge indices where the curve intersects the 
            parallelogram.
        intersecting_edges (list): a list of edges given by two points at the ends of the
            edge.
        
    Returns:
        grouped_pts (numpy.array): a list of groups of points.
        grouped_param_edge (numpy.array): a list of grouped edge parameters.
        grouped_param_isocurve (numpy.array): a list of grouped curve parameters.
        grouped_edge_indices (numpy.array): a list of grouped edge indices.
        grouped_intersecting_edges (numpy.array): a list of grouped edges that intersect 
            the isocurve
    """
    
    # Make all inputs numpy arrays.
    pts = np.array(pts)
    param_edge = np.array(param_edge)
    param_isocurve = np.array(param_isocurve)
    edge_indices = np.array(edge_indices)
    intersecting_edges = np.array(intersecting_edges)
    
    # What matters is the point's position relative to the isocontour cross since
    # isocontours never cross over it.
    c0, c1, c2, c3 = coeffs

    if np.isclose(c3, 0, atol=atol, rtol=rtol):
        # Technically x_cross and y_cross go to +/- infinity when c3 = 0, but we don't
        # have to worry too much about it because we're guaranteed to have only one set of
        # intersections in this case. This is just to avoid infinities.
        x_cross = 0
        y_cross = 0
    else:
        x_cross = -c2/c3
        y_cross = -c1/c3
    
    # Get the x- and y-coordinates relative to the cross.
    xs = pts[:,0] - x_cross
    ys = pts[:,1] - y_cross
    
    # Find the coordinates that are greater than zero and less than zero relative
    # to the cross.
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
                np.array(edge_indices), np.array([intersecting_edges]))
    
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

        q1 = edge_indices[x_positive*y_positive]
        q2 = edge_indices[x_negative*y_positive]
        q3 = edge_indices[x_negative*y_negative]
        q4 = edge_indices[x_positive*y_negative]
        
        grouped_edge_indices = np.array([q1, q2, q3, q4])
        
        # Remove quadrants that don't contain any points.    
        grouped_edge_indices = grouped_edge_indices[[len(g) > 1 for g in
                                                         grouped_edge_indices]]
        
        q1 = intersecting_edges[x_positive*y_positive]
        q2 = intersecting_edges[x_negative*y_positive]
        q3 = intersecting_edges[x_negative*y_negative]
        q4 = intersecting_edges[x_positive*y_negative]
        
        grouped_intersecting_edges = np.array([q1, q2, q3, q4])
        
        # Remove quadrants that don't contain any points.    
        grouped_intersecting_edges = grouped_intersecting_edges[[len(g) > 1 for g in
                                                            grouped_intersecting_edges]]
        
    return (grouped_pts, grouped_param_edge, grouped_param_isocurve,
            grouped_edge_indices, grouped_intersecting_edges)


# def calc_bilinear_area(case, subcase, edge_or_corner):
#     """Calculate the between a bilinear iso-curve and a parallelogram.

#     Args:
#         case (int): the integration case. Options include 0, 1, and 2. The case number
#             is the same as the number of unique integration limits.
#         subcase (float): the integration subcase. This determines whether the calculated
#             is the area inside or outside the iso-contour.
#         edge_or corner (int): the edge or corner that the iso-contour surrounds.
#     """

#     # Consider the case where the iso-contour intersects the same edge twice.
#     if case == 2:


# -(2*c1*u*x3 + c3*u**2*x3*y1 + 2*c2*u*y3 + c3*u**2*x1*y3 + 
# ((c1*x3 + c3*u*x3*y1 - (c2 + c3*u*x1)*y3)*
# Sqrt(x3**2*(c1 + c3*u*y1)**2 2*x3*(c1*(c2 - c3*u*x1) + c3*(2*C - 2*c0 - u*(c2 + c3*u*x1)*y1))*y3 + 
# (c2 + c3*u*x1)**2*y3**2))/(c3*(x3*y1 - x1*y3)) + 
# (4*(c1*c2 + (C - c0)*c3)*x3*y3*
# Log(c1*x3 - c2*y3 + c3*u*(x3*y1 - x1*y3) + 
# Sqrt(x3**2*(c1 + c3*u*y1)**2 + 
# 2*x3*(c1*c2 + 2*C*c3 - 2*c0*c3 - c1*c3*u*x1 - c3*u*(c2 + c3*u*x1)*y1)*y3 + 
#      -            (c2 + c3*u*x1)**2*y3**2)))/(c3*(x3*y1 - x1*y3)))/(4.*c3*x3*y3)


#         # Each edge has different integration limits and needs to be treated separately.
#         if edge_or_corner == 0:
            
            
#         elif edge_or_corner == 1:
            
#         elif edge_or_corner == 2:
            
#         elif edge_or_corner == 3:
            
#         else:
#             msg = "Invalid edge or corner"
#             raise ValueError(msg)
        
#     else:
#         msg = "Invalid integration case provided. Options include 0, 1, 2, and 3."
#         rais ValueError(msg)
    
    
    
#     if subcase == "inside":

            
#     elif subcase == "outside":
        
#     else:
#         msg = ("Invalid subcase provided. Options include 'inside' or "
#                "'outside'."
#         raise ValueError(msg)
            
