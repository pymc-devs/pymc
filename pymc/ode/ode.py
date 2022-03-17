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

import logging

import aesara
import aesara.tensor as at
import numpy as np
import scipy

from aesara.graph.basic import Apply
from aesara.graph.op import Op, get_test_value
from aesara.tensor.type import TensorType

from pymc.exceptions import DtypeError, ShapeError
from pymc.ode import utils

_log = logging.getLogger("pymc")
floatX = aesara.config.floatX


class DifferentialEquation(Op):
    r"""
    Specify an ordinary differential equation

    .. math::
        \dfrac{dy}{dt} = f(y,t,p) \quad y(t_0) = y_0

    Parameters
    ----------

    func : callable
        Function specifying the differential equation. Must take arguments y (n_states,), t (scalar), p (n_theta,)
    times : array
        Array of times at which to evaluate the solution of the differential equation.
    n_states : int
        Dimension of the differential equation.  For scalar differential equations, n_states=1.
        For vector valued differential equations, n_states = number of differential equations in the system.
    n_theta : int
        Number of parameters in the differential equation.
    t0 : float
        Time corresponding to the initial condition

    Examples
    --------

    .. code-block:: python

        def odefunc(y, t, p):
            #Logistic differential equation
            return p[0] * y[0] * (1 - y[0])

        times = np.arange(0.5, 5, 0.5)

        ode_model = DifferentialEquation(func=odefunc, times=times, n_states=1, n_theta=1, t0=0)
    """
    _itypes = [
        TensorType(floatX, (False,)),  # y0 as 1D floatX vector
        TensorType(floatX, (False,)),  # theta as 1D floatX vector
    ]
    _otypes = [
        TensorType(floatX, (False, False)),  # model states as floatX of shape (T, S)
        TensorType(
            floatX, (False, False, False)
        ),  # sensitivities as floatX of shape (T, S, len(y0) + len(theta))
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
        self._augmented_times = np.insert(times, 0, t0).astype(floatX)
        self._augmented_func = utils.augment_system(func, self.n_states, self.n_theta)
        self._sens_ic = utils.make_sens_ic(self.n_states, self.n_theta, floatX)

        # Cache symbolic sensitivities by the hash of inputs
        self._apply_nodes = {}
        self._output_sensitivities = {}

    def _system(self, Y, t, p):
        r"""This is the function that will be passed to odeint. Solves both ODE and sensitivities.

        Parameters
        ----------
        Y : array
            augmented state vector (n_states + n_states + n_theta)
        t : float
            current time
        p : array
            parameter vector (y0, theta)
        """
        dydt, ddt_dydp = self._augmented_func(Y[: self.n_states], t, p, Y[self.n_states :])
        derivatives = np.concatenate([dydt, ddt_dydp])
        return derivatives

    def _simulate(self, y0, theta):
        # Initial condition comprised of state initial conditions and raveled sensitivity matrix
        s0 = np.concatenate([y0, self._sens_ic])

        # perform the integration
        sol = scipy.integrate.odeint(
            func=self._system, y0=s0, t=self._augmented_times, args=(np.concatenate([y0, theta]),)
        ).astype(floatX)
        # The solution
        y = sol[1:, : self.n_states]

        # The sensitivities, reshaped to be a sequence of matrices
        sens = sol[1:, self.n_states :].reshape(self.n_times, self.n_states, self.n_p)

        return y, sens

    def make_node(self, y0, theta):
        inputs = (y0, theta)
        _log.debug(f"make_node for inputs {hash(inputs)}")
        states = self._otypes[0]()
        sens = self._otypes[1]()

        # store symbolic output in dictionary such that it can be accessed in the grad method
        self._output_sensitivities[hash(inputs)] = sens
        return Apply(self, inputs, (states, sens))

    def __call__(self, y0, theta, return_sens=False, **kwargs):
        if isinstance(y0, (list, tuple)) and not len(y0) == self.n_states:
            raise ShapeError("Length of y0 is wrong.", actual=(len(y0),), expected=(self.n_states,))
        if isinstance(theta, (list, tuple)) and not len(theta) == self.n_theta:
            raise ShapeError(
                "Length of theta is wrong.", actual=(len(theta),), expected=(self.n_theta,)
            )

        # convert inputs to tensors (and check their types)
        y0 = at.cast(at.unbroadcast(at.as_tensor_variable(y0), 0), floatX)
        theta = at.cast(at.unbroadcast(at.as_tensor_variable(theta), 0), floatX)
        inputs = [y0, theta]
        for i, (input_val, itype) in enumerate(zip(inputs, self._itypes)):
            if not input_val.type.in_same_class(itype):
                raise ValueError(
                    f"Input {i} of type {input_val.type} does not have the expected type of {itype}"
                )

        # use default implementation to prepare symbolic outputs (via make_node)
        states, sens = super().__call__(y0, theta, **kwargs)

        if aesara.config.compute_test_value != "off":
            # compute test values from input test values
            test_states, test_sens = self._simulate(
                y0=get_test_value(y0), theta=get_test_value(theta)
            )

            # check types of simulation result
            if not test_states.dtype == self._otypes[0].dtype:
                raise DtypeError(
                    "Simulated states have the wrong type.",
                    actual=test_states.dtype,
                    expected=self._otypes[0].dtype,
                )
            if not test_sens.dtype == self._otypes[1].dtype:
                raise DtypeError(
                    "Simulated sensitivities have the wrong type.",
                    actual=test_sens.dtype,
                    expected=self._otypes[1].dtype,
                )

            # check shapes of simulation result
            expected_states_shape = (self.n_times, self.n_states)
            expected_sens_shape = (self.n_times, self.n_states, self.n_p)
            if not test_states.shape == expected_states_shape:
                raise ShapeError(
                    "Simulated states have the wrong shape.",
                    test_states.shape,
                    expected_states_shape,
                )
            if not test_sens.shape == expected_sens_shape:
                raise ShapeError(
                    "Simulated sensitivities have the wrong shape.",
                    test_sens.shape,
                    expected_sens_shape,
                )

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

    def infer_shape(self, fgraph, node, input_shapes):
        s_y0, s_theta = input_shapes
        output_shapes = [(self.n_times, self.n_states), (self.n_times, self.n_states, self.n_p)]
        return output_shapes

    def grad(self, inputs, output_grads):
        _log.debug(f"grad w.r.t. inputs {hash(tuple(inputs))}")

        # fetch symbolic sensitivity output node from cache
        ihash = hash(tuple(inputs))
        if ihash in self._output_sensitivities:
            sens = self._output_sensitivities[ihash]
        else:
            _log.debug("No cached sensitivities found!")
            _, sens = self.__call__(*inputs, return_sens=True)
        ograds = output_grads[0]

        # for each parameter, multiply sensitivities with the output gradient and sum the result
        # sens is (n_times, n_states, n_p)
        # ograds is (n_times, n_states)
        grads = [at.sum(sens[:, :, p] * ograds) for p in range(self.n_p)]

        # return separate gradient tensors for y0 and theta inputs
        result = at.stack(grads[: self.n_states]), at.stack(grads[self.n_states :])
        return result
