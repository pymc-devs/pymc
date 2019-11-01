import numpy as np
import theano
import theano.tensor as tt


def augment_system(ode_func, n, m):
    """
    Function to create augmented system.

    Take a function which specifies a set of differential equations and return
    a compiled function which allows for computation of gradients of the
    differential equation's solition with repsect to the parameters.

    Parameters
    ----------
    ode_func : function
        Differential equation.  Returns array-like.
    n : int
        Number of rows of the sensitivity matrix.
    m : int
        Number of columns of the sensitivity matrix.

    Returns
    -------
    system : function
        Augemted system of differential equations.
    """

    # Present state of the system
    t_y = tt.vector("y", dtype=theano.config.floatX)
    t_y.tag.test_value = np.zeros((n,))
    # Parameter(s).  Should be vector to allow for generaliztion to multiparameter
    # systems of ODEs.  Is m dimensional because it includes all ode parameters as well as initical conditions
    t_p = tt.vector("p", dtype=theano.config.floatX)
    t_p.tag.test_value = np.zeros((m,))
    # Time.  Allow for non-automonous systems of ODEs to be analyzed
    t_t = tt.scalar("t", dtype=theano.config.floatX)
    t_t.tag.test_value = 2.459

    # Present state of the gradients:
    # Will always be 0 unless the parameter is the inital condition
    # Entry i,j is partial of y[i] wrt to p[j]
    dydp_vec = tt.vector("dydp", dtype=theano.config.floatX)
    dydp_vec.tag.test_value = np.zeros(n * m)

    dydp = dydp_vec.reshape((n, m))

    # Stack the results of the ode_func
    f_tensor = tt.stack(ode_func(t_y, t_t, t_p))

    # Now compute gradients
    J = tt.jacobian(f_tensor, t_y)

    Jdfdy = tt.dot(J, dydp)

    grad_f = tt.jacobian(f_tensor, t_p)

    # This is the time derivative of dydp
    ddt_dydp = (Jdfdy + grad_f).flatten()

    system = theano.function(
        inputs=[t_y, t_t, t_p, dydp_vec],
        outputs=[f_tensor, ddt_dydp],
        on_unused_input="ignore",
    )

    return system
