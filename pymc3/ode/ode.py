import logging
import numpy as np
import scipy
import theano
import theano.tensor as tt
from ..ode.utils import augment_system, ODEGradop

_log = logging.getLogger('pymc3')


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
            return p[0] * y[0] * (1 - y[0])
       
        times = np.arange(0.5, 5, 0.5)
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

        self._augmented_times = np.insert(times, 0, t0)
        self._augmented_func = augment_system(func, self._n, self._m)
        self._sens_ic = self._make_sens_ic()

        self._cached_y = None
        self._cached_sens = None
        self._cached_parameters = None

        self._grad_op = ODEGradop(self._numpy_vsp)

    def _make_sens_ic(self):
        """
        The sensitivity matrix will always have consistent form.
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
        """This is the function that will be passed to odeint. Solves both ODE and sensitivities
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
        parameters = tt.concatenate([odeparams, y0])
        inputs = [parameters]
        _log.debug(f'make_node for inputs {hash(tuple(inputs))}')
        return theano.Apply(self, [parameters], [parameters.type()])
    
    def __call__(self, y0, odeparams, **kwargs):
        # convert inputs to tensors (and check their types)
        y0 = tt.unbroadcast(tt.as_tensor_variable(y0), 0)
        theta = tt.unbroadcast(tt.as_tensor_variable(odeparams), 0)
        inputs = [y0, theta]
        
        states = super(theano.Op, self).__call__(y0=y0, odeparams=theta, **kwargs)
        return states

    def perform(self, node, inputs_storage, output_storage):
        parameters = inputs_storage[0]
        out = output_storage[0]
        # get the numerical solution of ODE states
        out[0] = self._state(parameters)

    def grad(self, inputs, output_grads):
        _log.debug(f'grad w.r.t. inputs {hash(tuple(inputs))}')
        x = inputs[0]
        g = output_grads[0]
        # pass the VSP when asked for gradient
        grad_op_apply = self._grad_op(x, g)

        return [grad_op_apply]


class NewDifferentialEquation(theano.Op):
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
    n_theta : int
        Number of parameters in the differential equation.

    .. code-block:: python

        def odefunc(y, t, p):
            #Logistic differential equation
            return p[0] * y[0] * (1 - y[0])
       
        times = np.arange(0.5, 5, 0.5)

        ode_model = DifferentialEquation(func=odefunc, times=times, n_states=1, n_theta=1, t0=0)
    """
    _itypes = [
        tt.TensorType(theano.config.floatX, (False,)),                  # y0 as 1D flloatX vector
        tt.TensorType(theano.config.floatX, (False,))                   # theta as 1D floatX vector
    ]
    _otypes = [
        tt.TensorType(theano.config.floatX, (False, False)),            # model states as floatX of shape (T, S)
        tt.TensorType(theano.config.floatX, (False, False, False)),     # sensitivities as floatX of shape (T, S, len(y0) + len(theta))
    ]
    __props__ = ("func", "times", "n_states", "n_theta", "t0")

    def __init__(self, func, times, *, n_states, n_theta, t0=0):
        if not callable(func):
            raise ValueError("Argument func must be callable.")
        if n_states < 1:
            raise ValueError("Argument n_states must be at least 1.")
        if n_theta <= 0:
            raise ValueError("Argument n_theta must be positive.")

        # Public
        self.func = func
        self.t0 = t0
        self.times = tuple(times)
        self.n_times = len(times)
        self.n_states = n_states
        self.n_theta = n_theta
        self.n_p = n_states + n_theta

        # Private
        self._augmented_times = np.insert(times, 0, t0)
        self._augmented_func = augment_system(func, self.n_states, self.n_p)
        self._sens_ic = self._make_sens_ic()

        # Cache symbolic sensitivities by the hash of inputs
        self._apply_nodes = {}
        self._output_sensitivities = {}
    
    def _make_sens_ic(self):
        """
        The sensitivity matrix will always have consistent form.
        If the first n_theta entries of the parameters vector in the simulate call
        correspond to ode paramaters, then the first n_theta columns in
        the sensitivity matrix will be 0 

        If the last n_states entries of the paramters vector in the simulate call
        correspond to initial conditions of the system,
        then the last n_states columns of the sensitivity matrix should form
        an identity matrix
        """

        # Initialize the sensitivity matrix to be 0 everywhere
        sens_matrix = np.zeros((self.n_states, self.n_p))

        # Slip in the identity matrix in the appropirate place
        sens_matrix[:, -self.n_states :] = np.eye(self.n_states)

        # We need the sensitivity matrix to be a vector (see augmented_function)
        # Ravel and return
        dydp = sens_matrix.ravel()

        return dydp

    def _system(self, Y, t, p):
        """This is the function that will be passed to odeint. Solves both ODE and sensitivities.
        """
        dydt, ddt_dydp = self._augmented_func(Y[:self.n_states], t, p, Y[self.n_states:])
        derivatives = np.concatenate([dydt, ddt_dydp])
        return derivatives

    def _simulate(self, y0, theta):
        # Initial condition comprised of state initial conditions and raveled sensitivity matrix
        s0 = np.concatenate([y0, self._sens_ic])

        # perform the integration
        sol = scipy.integrate.odeint(
            func=self._system, y0=s0, t=self._augmented_times, args=(np.concatenate([theta, y0]),)
        )
        # The solution
        y = sol[1:, :self.n_states]

        # The sensitivities, reshaped to be a sequence of matrices
        sens = sol[1:, self.n_states:].reshape(self.n_times, self.n_states, self.n_p)

        return y, sens

    def make_node(self, y0, theta):
        inputs = (y0, theta)
        _log.debug(f'make_node for inputs {hash(inputs)}')
        states = self._otypes[0]()
        sens = self._otypes[1]()

        # store symbolic output in dictionary such that it can be accessed in the grad method
        self._output_sensitivities[hash(inputs)] = sens
        return theano.Apply(self, inputs, (states, sens))

    def __call__(self, y0, theta, return_sens=False, **kwargs):
        # convert inputs to tensors (and check their types)
        y0 = tt.unbroadcast(tt.as_tensor_variable(y0), 0)
        theta = tt.unbroadcast(tt.as_tensor_variable(theta), 0)
        inputs = [y0, theta]
        for i, (input, itype) in enumerate(zip(inputs, self._itypes)):
            assert input.type == itype, f'Input {i} of type {input.type} does not have the expected type of {itype}'

        # use default implementation to prepare symbolic outputs (via make_node)
        states, sens = super(theano.Op, self).__call__(y0, theta, **kwargs)

        if theano.config.compute_test_value != 'off':
            # compute test values from input test values
            test_states, test_sens = self._simulate(
                y0=self._get_test_value(y0),
                theta=self._get_test_value(theta)
            )

            # check types of simulation result
            assert test_states.dtype == self._otypes[0].dtype, 'Simulated states have the wrong type'
            assert test_sens.dtype == self._otypes[1].dtype, 'Simulated sensitivities have the wrong type'

            # check shapes of simulation result
            expected_states_shape = (self.n_times, self.n_states)
            expected_sens_shape = (self.n_times, self.n_states, self.n_p)
            assert test_states.shape == expected_states_shape, f'States were simulated with shape {test_states.shape} but expected as {expected_states_shape}'
            assert test_sens.shape == expected_sens_shape, f'Sensitivities were simulated with shape {test_sens.shape} but expected as {expected_sens_shape}'

            # attach results as test values to the outputs
            states.tag.test_value = test_states
            sens.tag.test_value = test_sens
        
        if return_sens:
            return states, sens
        return states

    def perform(self, node, inputs_storage, output_storage):
        y0, theta = inputs_storage[0], inputs_storage[1]
        # simulate states and sensitivities in one forward pass
        output_storage[0][0], output_storage[1][0] = self._simulate(y0, theta)

    def infer_shape(self, node, input_shapes):
        s_y0, s_theta = input_shapes
        output_shapes = [(self.n_times, self.n_states), (self.n_times, self.n_states, self.n_p)]
        return output_shapes

    def grad(self, inputs, output_grads):
        _log.debug(f'grad w.r.t. inputs {hash(tuple(inputs))}')
        
        # fetch symbolic sensitivity output node from cache
        ihash = hash(tuple(inputs))
        if ihash in self._output_sensitivities:
            sens = self._output_sensitivities[ihash]
        else:
            _log.debug('No cached sensitivities found!')
            _, sens = self.__call__(*inputs, return_sens=True)
        ograds = output_grads[0]
        
        # for each parameter, multiply sensitivities with the output gradient and sum the result
        # sens is (n_times, n_states, n_p)
        # ograds is (n_times, n_states)
        grads = [
            tt.sum(sens[:,:,p] * ograds)
            for p in range(self.n_p)
        ]
        
        # return separate gradient tensors for y0 and theta inputs
        result = tt.stack(grads[:self.n_states]), tt.stack(grads[self.n_states:])
        return result
