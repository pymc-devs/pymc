import numpy as np
import theano
import theano.tensor as tt


def augment_system(ode_func, n, m):
    """
    Function to create augmented system.

    Take a function which specifies a set of differential equations and return
    a compiled function which allows for computation of gradients of the
    differential equation's solition with repsect to the parameters.

    Uses float64 even if floatX=float32, because the scipy integrator always uses float64.

    Parameters
    ----------
    ode_func : function
        Differential equation.  Returns array-like.
    n : int
        Number of rows of the sensitivity matrix. (n_states)
    m : int
        Number of columns of the sensitivity matrix. (n_states + n_theta)

    Returns
    -------
    system : function
        Augemted system of differential equations.
    """

    # Present state of the system
    t_y = tt.vector("y", dtype='float64')
    t_y.tag.test_value = np.zeros((n,), dtype='float64')
    # Parameter(s).  Should be vector to allow for generaliztion to multiparameter
    # systems of ODEs.  Is m dimensional because it includes all initial conditions as well as ode parameters
    t_p = tt.vector("p", dtype='float64')
    t_p.tag.test_value = np.zeros((m,), dtype='float64')
    # Time.  Allow for non-automonous systems of ODEs to be analyzed
    t_t = tt.scalar("t", dtype='float64')
    t_t.tag.test_value = 2.459

    # Present state of the gradients:
    # Will always be 0 unless the parameter is the inital condition
    # Entry i,j is partial of y[i] wrt to p[j]
    dydp_vec = tt.vector("dydp", dtype='float64')
    dydp_vec.tag.test_value = np.zeros(n * m, dtype='float64')

    dydp = dydp_vec.reshape((n, m))

    # Get symbolic representation of the ODEs by passing tensors for y, t and theta
    yhat = ode_func(t_y, t_t, t_p[n:])
    # Stack the results of the ode_func into a single tensor variable
    if not isinstance(yhat, (list, tuple)):
        yhat = (yhat,)
    t_yhat = tt.stack(yhat, axis=0)

    # Now compute gradients
    J = tt.jacobian(t_yhat, t_y)

    Jdfdy = tt.dot(J, dydp)

    grad_f = tt.jacobian(t_yhat, t_p)

    # This is the time derivative of dydp
    ddt_dydp = (Jdfdy + grad_f).flatten()

    system = theano.function(
        inputs=[t_y, t_t, t_p, dydp_vec],
        outputs=[t_yhat, ddt_dydp],
        on_unused_input="ignore"
    )

    return system
