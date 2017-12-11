"""Quaternion methods for finding rotation, reflection, and improper rotation matrices."""

def get_rotation_matrix(a,b):
    """Find the rotation matrix that takes a to b.
    """
    a = np.array(a)
    b = np.array(b)
    a2 = np.dot(a,a)
    b2 = np.dot(b,b)
    if not np.isclose(a2, b2):
        msg = "The vectors must be the same length."
        raise ValueError(msg.format(a))
        
    v = np.cross(a,b)
    w = np.sqrt(a2*b2) + np.dot(a,b)
    Q = np.hstack([v,w])
    if np.count_nonzero(Q) == 0:
        if (np.allclose(a,b) or np.allclose(-a,b)):
            msg = ("The vectors provided are parallel and the "
                   "rotation axis is illdefined.")            
            raise ValueError(msg.format(a))        
        else:
            msg = "There is something wrong with the provided vectors."
            raise ValueError(msg.format(a))        
    else:
        Q = Q/np.linalg.norm(Q)
        x = Q[0]
        y = Q[1]
        z = Q[2]
        w = Q[3]
        return np.array([[w**2 + x**2 - y**2 - z**2, 2*(x*y - w*z), 2*(x*z + w*y)],
                         [2*(x*y + w*z), w**2 - x**2 + y**2 - z**2, 2*(y*z - w*x)],
                         [2*(x*z - w*y), 2*(y*z + w*x), w**2 - x**2 - y**2 + z**2]])

    
def get_improper_rotation_matrix(a,b):
    """Find the improper rotation that takes a to b.
    """
    
    R = RQ(a, -b)
    return np.dot(-np.eye(3), R)


def get_reflection_matrix(a,b):
    """Find the reflection matrix that takes a to b.
    """
    
    a = np.array(a)
    b = np.array(b)
    a2 = np.dot(a,a)
    b2 = np.dot(b,b)
    if not np.isclose(a2, b2):
        msg = "The vectors must be the same length."
        raise ValueError(msg.format(a))
        
    n = (np.dot(b,b) + np.dot(a,b))*a - (np.dot(a,a) + np.dot(a,b))*b
    Q = n/norm(n)
    if np.count_nonzero(Q) == 0:
        if (np.allclose(a,b) or np.allclose(-a,b)):
            sign = lambda x: x and (1, -1)[x<0]
            return sign(np.dot(a,b))*np.eye(3,3)
        else:
            msg = "There is something wrong with the provided vectors."
            raise ValueError(msg.format(a))        
    else:
        Q = Q/np.linalg.norm(Q)
        x = Q[0]
        y = Q[1]
        z = Q[2]

        return np.array([[-x**2 + y**2 + z**2, -2*x*y, -2*x*z],
                         [-2*x*y, x**2 - y**2 + z**2, -2*y*z],
                         [-2*x*z, -2*y*z, x**2 + y**2 - z**2]])
