#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import aesara
import aesara.tensor as at
import numpy as np


def make_sens_ic(n_states, n_theta, floatX):
    r"""
    The sensitivity matrix will always have consistent form. (n_states, n_states + n_theta)

    If the first n_states entries of the parameters vector in the simulate call
    correspond to initial conditions of the system,
    then the first n_states columns of the sensitivity matrix should form
    an identity matrix.

    If the last n_theta entries of the parameters vector in the simulate call
    correspond to ode paramaters, then the last n_theta columns in
    the sensitivity matrix will be 0.

    Parameters
    ----------
    n_states : int
        Number of state variables in the ODE
    n_theta : int
        Number of ODE parameters
    floatX : str
        dtype to be used for the array

    Returns
    -------
    dydp : array
        1D-array of shape (n_states * (n_states + n_theta),), representing the initial condition of the sensitivities
    """

    # Initialize the sensitivity matrix to be 0 everywhere
    sens_matrix = np.zeros((n_states, n_states + n_theta), dtype=floatX)

    # Slip in the identity matrix in the appropirate place
    sens_matrix[:, :n_states] = np.eye(n_states, dtype=floatX)

    # We need the sensitivity matrix to be a vector (see augmented_function)
    # Ravel and return
    dydp = sens_matrix.ravel()
    return dydp


def augment_system(ode_func, n_states, n_theta):
    """
    Function to create augmented system.

    Take a function which specifies a set of differential equations and return
    a compiled function which allows for computation of gradients of the
    differential equation's solition with repsect to the parameters.

    Uses float64 even if floatX=float32, because the scipy integrator always uses float64.

    Parameters
    ----------
    ode_func: function
        Differential equation.  Returns array-like.
    n_states: int
        Number of rows of the sensitivity matrix. (n_states)
    n_theta: int
        Number of ODE parameters

    Returns
    -------
    system: function
        Augemted system of differential equations.
    """

    # Present state of the system
    t_y = at.vector("y", dtype="float64")
    t_y.tag.test_value = np.ones((n_states,), dtype="float64")
    # Parameter(s).  Should be vector to allow for generaliztion to multiparameter
    # systems of ODEs.  Is m dimensional because it includes all initial conditions as well as ode parameters
    t_p = at.vector("p", dtype="float64")
    t_p.tag.test_value = np.ones((n_states + n_theta,), dtype="float64")
    # Time.  Allow for non-automonous systems of ODEs to be analyzed
    t_t = at.scalar("t", dtype="float64")
    t_t.tag.test_value = 2.459

    # Present state of the gradients:
    # Will always be 0 unless the parameter is the inital condition
    # Entry i,j is partial of y[i] wrt to p[j]
    dydp_vec = at.vector("dydp", dtype="float64")
    dydp_vec.tag.test_value = make_sens_ic(n_states, n_theta, "float64")

    dydp = dydp_vec.reshape((n_states, n_states + n_theta))

    # Get symbolic representation of the ODEs by passing tensors for y, t and theta
    yhat = ode_func(t_y, t_t, t_p[n_states:])
    if isinstance(yhat, at.TensorVariable):
        t_yhat = at.atleast_1d(yhat)
    else:
        # Stack the results of the ode_func into a single tensor variable
        if not isinstance(yhat, (list, tuple)):
            raise TypeError(
                f"Unexpected type, {type(yhat)}, returned by ode_func. TensorVariable, list or tuple is expected."
            )
        t_yhat = at.stack(yhat, axis=0)
    if t_yhat.ndim > 1:
        raise ValueError(
            f"The odefunc returned a {t_yhat.ndim}-dimensional tensor, but 0 or 1 dimensions were expected."
        )

    # Now compute gradients
    J = at.jacobian(t_yhat, t_y)

    Jdfdy = at.dot(J, dydp)

    grad_f = at.jacobian(t_yhat, t_p)

    # This is the time derivative of dydp
    ddt_dydp = (Jdfdy + grad_f).flatten()

    system = aesara.function(
        inputs=[t_y, t_t, t_p, dydp_vec], outputs=[t_yhat, ddt_dydp], on_unused_input="ignore"
    )

    return system
