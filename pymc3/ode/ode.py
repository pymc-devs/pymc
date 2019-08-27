import numpy as np
import scipy
import theano
import theano.tensor as tt
from ..ode.utils import augment_system, ODEGradop


class DifferentialEquation(theano.Op):
    """
    Specify an ordinary differential equation

    .. math::
        \dfrac{dy}{dt} = f(y,t,p) \quad y(t_0) = y_0

    Parameters
    ----------

    func : callable
        Function specifying the differential equation
    t0 : float
        Time corresponding to the initial condition
    times : array
        Array of times at which to evaluate the solution of the differential equation.
    n_states : int
        Dimension of the differential equation.  For scalar differential equations, n_states=1.
        For vector valued differential equations, n_states = number of differential equations in the system.
    n_odeparams : int
        Number of parameters in the differential equation.

    .. code-block:: python

        def odefunc(y, t, p):
            #Logistic differential equation
            return p[0]*y[0]*(1-y[0])
       
        times=np.arange(0.5, 5, 0.5)

        ode_model = DifferentialEquation(func=odefunc, t0=0, times=times, n_states=1, n_odeparams=1)
    """

    __props__ = ("func", "t0", "times", "n_states", "n_odeparams")

    def __init__(self, func, times, n_states, n_odeparams, t0=0):
        if not callable(func):
            raise ValueError("Argument func must be callable.")
        if n_states < 1:
            raise ValueError("Argument n_states must be at least 1.")
        if n_odeparams <= 0:
            raise ValueError("Argument n_odeparams must be positive.")

        # Public
        self.func = func
        self.t0 = t0
        self.times = tuple(times)
        self.n_states = n_states
        self.n_odeparams = n_odeparams

        # Private
        self._n = n_states
        self._m = n_odeparams + n_states

        self._augmented_times = np.insert(times, t0, 0)
        self._augmented_func = augment_system(func, self._n, self._m)
        self._sens_ic = self._make_sens_ic()

        self._cached_y = None
        self._cached_sens = None
        self._cached_parameters = None

        self._grad_op = ODEGradop(self._numpy_vsp)

    def _make_sens_ic(self):
        """The sensitivity matrix will always have consistent form.
        If the first n_odeparams entries of the parameters vector in the simulate call
        correspond to ode paramaters, then the first n_odeparams columns in
        the sensitivity matrix will be 0 

        If the last n_states entries of the paramters vector in the simulate call
        correspond to initial conditions of the system,
        then the last n_states columns of the sensitivity matrix should form
        an identity matrix
        """

        # Initialize the sensitivity matrix to be 0 everywhere
        sens_matrix = np.zeros((self._n, self._m))

        # Slip in the identity matrix in the appropirate place
        sens_matrix[:, -self.n_states :] = np.eye(self.n_states)

        # We need the sensitivity matrix to be a vector (see augmented_function)
        # Ravel and return
        dydp = sens_matrix.ravel()

        return dydp

    def _system(self, Y, t, p):
        """This is the function that will be passed to odeint.
        Solves both ODE and sensitivities
        Args:
            Y (vector): current state and current gradient state
            t (scalar): current time
            p (vector): parameters
        Returns:
            derivatives (vector): derivatives of state and gradient
        """

        dydt, ddt_dydp = self._augmented_func(Y[: self._n], t, p, Y[self._n :])
        derivatives = np.concatenate([dydt, ddt_dydp])
        return derivatives

    def _simulate(self, parameters):
        # Initial condition comprised of state initial conditions and raveled
        # sensitivity matrix
        y0 = np.concatenate([parameters[self.n_odeparams :], self._sens_ic])

        # perform the integration
        sol = scipy.integrate.odeint(
            func=self._system, y0=y0, t=self._augmented_times, args=(parameters,)
        )
        # The solution
        y = sol[1:, : self.n_states]

        # The sensitivities, reshaped to be a sequence of matrices
        sens = sol[1:, self.n_states :].reshape(len(self.times), self._n, self._m)

        return y, sens

    def _cached_simulate(self, parameters):
        if np.array_equal(np.array(parameters), self._cached_parameters):

            return self._cached_y, self._cached_sens

        return self._simulate(np.array(parameters))

    def _state(self, parameters):
        y, sens = self._cached_simulate(np.array(parameters))
        self._cached_y, self._cached_sens, self._cached_parameters = y, sens, parameters
        return y.ravel()

    def _numpy_vsp(self, parameters, g):
        _, sens = self._cached_simulate(np.array(parameters))

        # Each element of sens is an nxm sensitivity matrix
        # There is one sensitivity matrix per time step, making sens a (len(times), n_states, len(parameter))
        # dimensional array.  Reshaping the sens array in this way is like stacking each of the elements of sens on top
        # of one another.
        numpy_sens = sens.reshape((self.n_states * len(self.times), len(parameters)))
        # The dot product here is equivalent to np.einsum('ijk,jk', sens, g)
        # if sens was not reshaped and if g had the same shape as yobs
        return numpy_sens.T.dot(g)

    def make_node(self, odeparams, y0):
        if len(odeparams) != self.n_odeparams:
            raise ValueError(
                "odeparams has too many or too few parameters.  Expected {a} parameter(s) but got {b}".format(
                    a=self.n_odeparams, b=len(odeparams)
                )
            )
        if len(y0) != self.n_states:
            raise ValueError(
                "y0 has too many or too few parameters.  Expected {a} parameter(s) but got {b}".format(
                    a=self.n_states, b=len(y0)
                )
            )

        if np.ndim(odeparams) > 1:
            odeparams = np.ravel(odeparams)
        if np.ndim(y0) > 1:
            y0 = np.ravel(y0)

        odeparams = tt.as_tensor_variable(odeparams)
        y0 = tt.as_tensor_variable(y0)
        parameters = tt.concatenate([odeparams, y0])
        return theano.Apply(self, [parameters], [parameters.type()])

    def perform(self, node, inputs_storage, output_storage):
        parameters = inputs_storage[0]
        out = output_storage[0]
        # get the numerical solution of ODE states
        out[0] = self._state(parameters)

    def grad(self, inputs, output_grads):
        x = inputs[0]
        g = output_grads[0]
        # pass the VSP when asked for gradient
        grad_op_apply = self._grad_op(x, g)

        return [grad_op_apply]
