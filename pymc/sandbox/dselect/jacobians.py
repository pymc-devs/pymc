"""Determinant of the Jacobian of the transformation from the parameters of
distributions to their mean and variance.

    |d mean(a,b)   d mean(a,b)|
    |-----------   -----------|
    |    da            db     |
    |                         |
    | d var(a,b)   d var(a,b) |
    |-----------   -----------|
    |    da            db     |

"""

def beta(alpha, beta):
    """Return the determinant of the jacobian for the beta function.
    """
    return -1.0*alpha*beta/((beta+alpha)**3*(beta+alpha+1)**2)



