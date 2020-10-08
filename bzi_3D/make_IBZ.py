"""Methods to calculate the irreducible Brillouin zone (IBZ).

This module is a modified version of code written by Dr. Hess at BYU. See
https://www.physics.byu.edu/department/directory/hess.
"""

import numpy as np
from numpy.linalg import norm, det, inv
import itertools as it
import pandas as pd
from scipy.spatial import ConvexHull
from bzi_3D.utilities import trim_small, check_contained

def get_bragg_planes(lat_vecs):
    """Calculate a subset of Bragg planes.

    Args:
        lat_vecs (numpy.ndarray): the lattice vectors as columns of a 3x3 numpy array.

    Returns:
        bragg_planes (pandas.DataFrame): a dataframe of bragg planes, sorted by distance
            from the origin. Columns are labeled 'Direct' for the lattice points in direct
            coordinates, "Cartesian" for lattice points in Cartesian coordinates,
            "Bragg plane" for the Bragg planes in general form, and "Distance" for
            the minimum distance from the origin to the Bragg plane.
    """

    # Lattice points in direct coordinates.
    dir_lat_pts = np.array([[i,j,k] for i,j,k in it.product(range(-2, 3), repeat=3)])
    # Remove the Bragg plane at the origin.
    mask = list(map(lambda x: not np.allclose(x, [0,0,0]), dir_lat_pts))
    dir_lat_pts = dir_lat_pts[mask]
    # Lattice points in Cartesian coordinates.
    car_lat_pts = np.array([np.dot(lat_vecs, lpt) for lpt in dir_lat_pts])
    bragg_planes = np.array([[i, j, norm(j)/2, np.append(j/norm(j), norm(j)/2)] for i,j in
                             zip(dir_lat_pts, car_lat_pts)], dtype=object)
    indices = np.argsort(bragg_planes[:,2])
    columns = ["Direct", "Cartesian", "Distance", "Bragg plane"]
    bindices = range(len(bragg_planes))
    bragg_planes = pd.DataFrame(bragg_planes[indices], columns=columns,
                                index=bindices)
    return bragg_planes


def three_planes_intersect(planes):
    """This routine finds the point of intersection of three planes.

    Three planes, in general form with a single point in common, can be written as

    a0x + b0y + c0z = d0,   d0 = (a0*a0 + b0*b0 + c0*c0)/2
    a1x + b1y + c1z = d1,   d1 = (a1*a1 + b1*b1 + c1*c1)/2
    a2x + b2y + c2z = d2,   d2 = (a2*a2 + b2*b2 + c2*c2)/2

    where the point in the plane is the same as the vector normal to the plane divided by
    two. These equations can be put in matrix form

    [[a0, b0, c0], [a1, b1, c1], [a2, b2, c2]]*[x, y, z] = [d0, d1, d2]

    M = [[a0, b0, c0], [a1, b1, c1], [a2, b2, c2]]
    r = [x, y, z]
    d = [d0, d1, d2]

    Mr = d  ==>  r = inv(M)d

    As long as M has an inverse, we can find a point common to all three planes.

    Args:
        planes (numpy.ndarray): a 3x4 array whose rows are Bragg planes. The first three
            elements of each row make a unit vector normal to the plane. The last element
            in each row gives the distance from the origin of the plane along the line
            normal to he plane.

    Returns:
         (numpy.ndarray): the point of intersection of the three planes.
    """

    planes = np.array(planes)
    normal_vecs = planes[:,:3]
    d = planes[:,3]
    if np.isclose(det(normal_vecs), 0):
        return None
    else:
        return trim_small(np.dot(inv(normal_vecs), d))


def get_bragg_shells(bragg_planes):
    """Find the starting index of Bragg planes that have the same minimum distance from
    the origin.

    Args:
        bragg_planes (pandas.DataFrame): an array of Bragg planes ordered by
            increasing distance from the origin.

    Returns:
        indices (list): a list of indices that indicate where in the list of
            Bragg planes the minimum distance from the origin changes.
    """

    indices = [0]
    old_dist = bragg_planes["Distance"][0]
    for i in range(len(bragg_planes)):
        new_dist = bragg_planes["Distance"][i]
        if np.isclose(new_dist, old_dist):
            continue
        else:
            old_dist = new_dist
            indices.append(i)
    return indices


def point_plane_location(point, plane):
    """Determine if a point is inside the plane, outside the plane, or lies on the plane.

    Inside is the side of the plane opposite that in which the vector normal to the
    plane points.

    Args:
        point (numpy.ndarray): a point in Cartesian coordinates.
        plane (numpy.ndarray): an array with four elements. The first three provide
            a vector normal to the plane. The fourth element is the distance of the
            plane from the origin in the direction of the vector normal to the plane.

    Returns:
        (str): a string that indicates where the point is located. Options include
            "inside", "outside", and "on".
    """

    n = np.array(plane[:3])
    d = plane[-1]
    loc = np.dot(point, n) - d

    if np.isclose(loc, 0):
        return "on"
    elif loc > 0:
        return "outside"
    else:
        return "inside"


def find_bz(lat_vecs):
    """Find the Brillouin zone.

    Args:
        lat_vecs (numpy.ndarray): the lattice vectors as columns of a 3x3 numpy array.

    Returns:
        BZ (scipy.spatial.ConvexHull): the Brillouin zone for the given lattice.
    """

    eps = 1e-6
    # Get the Bragg planes.
    bragg_planes = get_bragg_planes(lat_vecs)
    # Find the shells of the Bragg planes. A shell is a set of Bragg planes that have
    # the same minimum distance from the origin.
    bragg_shells = get_bragg_shells(bragg_planes)
    shell_index = 0
    BZvolume = 0


    while not np.isclose(BZvolume, det(lat_vecs)):
        shell_index += 1
        stop_index = bragg_shells[shell_index]
        this_shell = bragg_planes["Bragg plane"][:stop_index]
        ipintersects = []
        this_shell = bragg_planes["Bragg plane"][:stop_index]
        ipintersects = []
        for three_planes in it.combinations(np.array(this_shell), 3):
            ipt = three_planes_intersect(three_planes)
            if not ipt is None:
                if not any(map(lambda elem: np.allclose(ipt, elem), ipintersects)):
                    ipintersects.append(ipt)

        inside_intersects = []
        # Make sure the intersections don't cross any of the planes that make up the
        # Brillouin zone.
        for pt in ipintersects:
            crossed = False
            for plane in this_shell:
                if point_plane_location(pt, plane) == "outside":
                    crossed = True
            if not crossed:
                inside_intersects.append(pt)
        try:
            BZ = ConvexHull(inside_intersects)
            BZvolume = BZ.volume
            if np.isclose(BZvolume, det(lat_vecs)):
                return BZ
        except:
            if shell_index > 9:
                return None
            continue

    return BZ


def get_unique_planes(BZ, rtol=1e-5, atol=1e-8):
    """Find the unique planes that form the boundaries of a Brillouin zone.

    Args:
        BZ (scipy.spatial.ConvexHull): a convex hull object

    Returns:
        unique_plane (numpy.ndarray): an array of unique planes in general form.
    """

    unique_planes = []
    for plane in BZ.equations:
        if not check_contained([plane], unique_planes, rtol=rtol, atol=atol):
            unique_planes.append(plane)

    return np.array(unique_planes)


def planar3dTo2d(points, eps=1e-6):
    """From points on a plane, create a new coordinate system whose origin
    is the point at the center of all the other points. Project the points
    onto this 2D coordinate system.

    Args:
        points (numpy.ndarray): a list of points in a plane.
        eps (float): finite precision tolerance for the vector normal to the plane.

    Returns:
        coords (numpy.ndarray): a list of points in the coordinate system of the plane.
    """

    coords = np.zeros((len(points), 2))

    # Find the vector normal to the plane
    uvec = plane3pts(points,eps)[0]

    # Find the point at the center of all the points. This is a point in the plane that
    # acts as the origin in a coordinate system in the plane. The sum function must not
    # be the one from numpy.
    rcenter = sum(points)/len(points)

    # This is the x-axis in the new coordinate system. As long as it points to a point in
    # the plane, there is freedom in choosing the x-axis.
    xunitv =  (points[0] - rcenter)/norm(points[0] - rcenter)

    # The y-axis in the new coordinate system.
    crss = np.cross(xunitv, uvec)
    yunitv = crss/norm(crss)

    for i, vec in enumerate(points):
        # Find a vector that points from the new orign to the point in the plane.
        vc = vec - rcenter
        # Project this vector onto the new coordinate system.
        coords[i,0] = np.dot(vc, xunitv); coords[i,1] = np.dot(vc, yunitv)
    return coords

def orderAngle(facet, eps=1e-6):
    """Get the angle of each vector in the plane of the facet relative to a
    coordinate system in the plane of points. If there are only three points,
    leave the points as they were.

    Args:
        facet (numpy.ndarray): a list of points in the plane of a facet
        eps (float): a finite-tolerance precision parameter

    Returns:
        (list): a list of points.
    """

    # if len(facet) == 3:
    #     return facet
    xy = planar3dTo2d(facet, eps)
    angles = []
    for i, vec in enumerate(facet):
        angle = np.arctan2(xy[i,1], xy[i,0])
        if angle < (0 - eps):
            angle += 2*np.pi
        angles.append(angle)
    return [point for (angle,point) in sorted(zip(angles,facet), key = lambda x: x[0])]

def plane3pts(points, eps=1e-3):
    """From a list of points in a plane, find the vector normal to the plane (pointing
    away from the origin) and the closest distance from the origin to the plane. If the
    plane passes through the origin, no choice is made concerning the direction of the
    vector normal to the plane.

    Args:
        points (numpy.ndarray): a list of points in plane
        eps (float): finite tolerance parameter that determines if two vectors are
            parallel.
    Returns:
        n (numpy.ndarray): a vector normal to the plane of points.
        dist (float): the distance from the origin to the plane.
    """

    # """From the first 3 (noncollinear) points of a list of points,
    # returns a normal and closest distance to origin of the plane
    # The closest distance is d = dot(r0, u). The normal's direction
    # (degenerate vs multiplication by -1)
    # is chosen to be that which points away from the side the origin is on.  If the
    # plane goes through the origin, then all the r's are in the plane, and no choice is made
    # """

    r0 = points[0]; r1 = points[1]; r2 = points[2];
    vec = np.cross(r1-r0, r2-r0)
    nv = norm(vec)

    # Two points are very close, use a 4th point.
    if nv < eps and len(points) > 3:
        if norm(r1-r0) > norm(r2-r0):
            r2 = points[3]
        else:
            r1 = points[3]

        vec = np.cross(r1-r0, r2-r0)
        nv = norm(vec)

    n = vec/nv
    if np.dot(n, r0) < -eps:
        n = -n
    dist = np.dot(n, r0)
    return n, dist
