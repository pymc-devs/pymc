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

import numpy as np
import pytest
import theano

from scipy.integrate import odeint
from scipy.stats import norm

import pymc3 as pm

from pymc3.ode import DifferentialEquation
from pymc3.ode.utils import augment_system


def test_gradients():
    """Tests the computation of the sensitivities from the theano computation graph"""

    # ODE system for which to compute gradients
    def ode_func(y, t, p):
        return np.exp(-t) - p[0] * y[0]

    # Computation of graidients with Theano
    augmented_ode_func = augment_system(ode_func, 1, 1 + 1)

    # This is the new system, ODE + Sensitivities, which will be integrated
    def augmented_system(Y, t, p):
        dydt, ddt_dydp = augmented_ode_func(Y[:1], t, p, Y[1:])
        derivatives = np.concatenate([dydt, ddt_dydp])
        return derivatives

    # Create real sensitivities
    y0 = 0.0
    t = np.arange(0, 12, 0.25).reshape(-1, 1)
    a = 0.472
    p = np.array([y0, a])

    # Derivatives of the analytic solution with respect to y0 and alpha
    # Treat y0 like a parameter and solve analytically.  Then differentiate.
    # I used CAS to get these derivatives
    y0_sensitivity = np.exp(-a * t)
    a_sensitivity = (
        -(np.exp(t * (a - 1)) - 1 + (a - 1) * (y0 * a - y0 - 1) * t) * np.exp(-a * t) / (a - 1) ** 2
    )

    sensitivity = np.c_[y0_sensitivity, a_sensitivity]

    integrated_solutions = odeint(func=augmented_system, y0=[y0, 1, 0], t=t.ravel(), args=(p,))
    simulated_sensitivity = integrated_solutions[:, 1:]

    np.testing.assert_allclose(sensitivity, simulated_sensitivity, rtol=1e-5)


def test_simulate():
    """Tests the integration in DifferentialEquation"""

    # Create an ODe to integrate
    def ode_func(y, t, p):
        return np.exp(-t) - p[0] * y[0]

    # Evaluate exact solution
    y0 = 0
    t = np.arange(0, 12, 0.25).reshape(-1, 1)
    a = 0.472
    y = 1.0 / (a - 1) * (np.exp(-t) - np.exp(-a * t))

    # Instantiate ODE model
    ode_model = DifferentialEquation(func=ode_func, t0=0, times=t, n_states=1, n_theta=1)

    simulated_y, sens = ode_model._simulate([y0], [a])

    assert simulated_y.shape == (len(t), 1)
    assert sens.shape == (len(t), 1, 1 + 1)
    np.testing.assert_allclose(y, simulated_y, rtol=1e-5)


class TestSensitivityInitialCondition:

    t = np.arange(0, 12, 0.25).reshape(-1, 1)

    def test_sens_ic_scalar_1_param(self):
        """Tests the creation of the initial condition for the sensitivities"""
        # Scalar ODE 1 Param
        # Create an ODe to integrate
        def ode_func_1(y, t, p):
            return np.exp(-t) - p[0] * y[0]

        # Instantiate ODE model
        # Instantiate ODE model
        model1 = DifferentialEquation(func=ode_func_1, t0=0, times=self.t, n_states=1, n_theta=1)

        # Sensitivity initial condition for this model should be 1 by 2
        model1_sens_ic = np.array([1, 0])

        np.testing.assert_array_equal(model1_sens_ic, model1._sens_ic)

    def test_sens_ic_scalar_2_param(self):
        # Scalar ODE 2 Param
        def ode_func_2(y, t, p):
            return p[0] * np.exp(-p[0] * t) - p[1] * y[0]

        # Instantiate ODE model
        model2 = DifferentialEquation(func=ode_func_2, t0=0, times=self.t, n_states=1, n_theta=2)

        model2_sens_ic = np.array([1, 0, 0])

        np.testing.assert_array_equal(model2_sens_ic, model2._sens_ic)

    def test_sens_ic_vector_1_param(self):
        # Vector ODE 1 Param
        def ode_func_3(y, t, p):
            ds = -p[0] * y[0] * y[1]
            di = p[0] * y[0] * y[1] - y[1]

            return [ds, di]

        # Instantiate ODE model
        model3 = DifferentialEquation(func=ode_func_3, t0=0, times=self.t, n_states=2, n_theta=1)

        model3_sens_ic = np.array([1, 0, 0, 0, 1, 0])

        np.testing.assert_array_equal(model3_sens_ic, model3._sens_ic)

    def test_sens_ic_vector_2_param(self):
        # Vector ODE 2 Param
        def ode_func_4(y, t, p):
            ds = -p[0] * y[0] * y[1]
            di = p[0] * y[0] * y[1] - p[1] * y[1]

            return [ds, di]

        # Instantiate ODE model
        model4 = DifferentialEquation(func=ode_func_4, t0=0, times=self.t, n_states=2, n_theta=2)

        model4_sens_ic = np.array([1, 0, 0, 0, 0, 1, 0, 0])

        np.testing.assert_array_equal(model4_sens_ic, model4._sens_ic)

    def test_sens_ic_vector_3_params(self):
        # Big System with Many Parameters
        def ode_func_5(y, t, p):
            dx = p[0] * (y[1] - y[0])
            ds = y[0] * (p[1] - y[2]) - y[1]
            dz = y[0] * y[1] - p[2] * y[2]

            return [dx, ds, dz]

        # Instantiate ODE model
        model5 = DifferentialEquation(func=ode_func_5, t0=0, times=self.t, n_states=3, n_theta=3)

        # First three columns are derivatives with respect to ode parameters
        # Last three coluimns are derivatives with repsect to initial condition
        # So identity matrix should appear in last 3 columns
        model5_sens_ic = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

        np.testing.assert_array_equal(np.ravel(model5_sens_ic), model5._sens_ic)


def test_logp_scalar_ode():
    """Test the computation of the log probability for these models"""

    # Differential equation
    def system_1(y, t, p):
        return np.exp(-t) - p[0] * y[0]

    # Parameters and inital condition
    alpha = 0.4
    y0 = 0.0
    times = np.arange(0.5, 8, 0.5)

    yobs = np.array(
        [0.30, 0.56, 0.51, 0.55, 0.47, 0.42, 0.38, 0.30, 0.26, 0.21, 0.22, 0.13, 0.13, 0.09, 0.09]
    )[:, np.newaxis]

    ode_model = DifferentialEquation(func=system_1, t0=0, times=times, n_theta=1, n_states=1)

    integrated_solution, *_ = ode_model._simulate([y0], [alpha])

    assert integrated_solution.shape == yobs.shape

    # compare automatic and manual logp values
    manual_logp = norm.logpdf(x=np.ravel(yobs), loc=np.ravel(integrated_solution), scale=1).sum()
    with pm.Model() as model_1:
        forward = ode_model(theta=[alpha], y0=[y0])
        y = pm.Normal("y", mu=forward, sd=1, observed=yobs)
    pymc3_logp = model_1.logp()

    np.testing.assert_allclose(manual_logp, pymc3_logp)


class TestErrors:
    """Test running model for a scalar ODE with 1 parameter"""

    def system(y, t, p):
        return np.exp(-t) - p[0] * y[0]

    times = np.arange(0, 9)

    ode_model = DifferentialEquation(func=system, t0=0, times=times, n_states=1, n_theta=1)

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_too_many_params(self):
        with pytest.raises(pm.ShapeError):
            self.ode_model(theta=[1, 1], y0=[0])

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_too_many_y0(self):
        with pytest.raises(pm.ShapeError):
            self.ode_model(theta=[1], y0=[0, 0])

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_too_few_params(self):
        with pytest.raises(pm.ShapeError):
            self.ode_model(theta=[], y0=[1])

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_too_few_y0(self):
        with pytest.raises(pm.ShapeError):
            self.ode_model(theta=[1], y0=[])

    def test_func_callable(self):
        with pytest.raises(ValueError):
            DifferentialEquation(func=1, t0=0, times=self.times, n_states=1, n_theta=1)

    def test_number_of_states(self):
        with pytest.raises(ValueError):
            DifferentialEquation(func=self.system, t0=0, times=self.times, n_states=0, n_theta=1)

    def test_number_of_params(self):
        with pytest.raises(ValueError):
            DifferentialEquation(func=self.system, t0=0, times=self.times, n_states=1, n_theta=0)


class TestDiffEqModel:
    def test_op_equality(self):
        """Tests that the equality of mathematically identical Ops evaluates True"""

        # Create ODE to test with
        def ode_func(y, t, p):
            return np.exp(-t) - p[0] * y[0]

        t = np.linspace(0, 2, 12)

        # Instantiate two Ops
        op_1 = DifferentialEquation(func=ode_func, t0=0, times=t, n_states=1, n_theta=1)
        op_2 = DifferentialEquation(func=ode_func, t0=0, times=t, n_states=1, n_theta=1)
        op_other = DifferentialEquation(
            func=ode_func, t0=0, times=np.linspace(0, 2, 16), n_states=1, n_theta=1
        )

        assert op_1 == op_2
        assert op_1 != op_other
        return

    def test_scalar_ode_1_param(self):
        """Test running model for a scalar ODE with 1 parameter"""

        def system(y, t, p):
            return np.exp(-t) - p[0] * y[0]

        times = np.array(
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        )

        yobs = np.array(
            [0.31, 0.57, 0.51, 0.55, 0.47, 0.42, 0.38, 0.3, 0.26, 0.22, 0.22, 0.14, 0.14, 0.09, 0.1]
        )[:, np.newaxis]

        ode_model = DifferentialEquation(func=system, t0=0, times=times, n_states=1, n_theta=1)

        with pm.Model() as model:
            alpha = pm.HalfCauchy("alpha", 1)
            y0 = pm.Lognormal("y0", 0, 1)
            sigma = pm.HalfCauchy("sigma", 1)
            forward = ode_model(theta=[alpha], y0=[y0])
            y = pm.Lognormal("y", mu=pm.math.log(forward), sd=sigma, observed=yobs)
            trace = pm.sample(100, tune=0, chains=1, return_inferencedata=False)

        assert trace["alpha"].size > 0
        assert trace["y0"].size > 0
        assert trace["sigma"].size > 0

    def test_scalar_ode_2_param(self):
        """Test running model for a scalar ODE with 2 parameters"""

        def system(y, t, p):
            return p[0] * np.exp(-p[0] * t) - p[1] * y[0]

        times = np.array(
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        )

        yobs = np.array(
            [0.31, 0.57, 0.51, 0.55, 0.47, 0.42, 0.38, 0.3, 0.26, 0.22, 0.22, 0.14, 0.14, 0.09, 0.1]
        )[:, np.newaxis]

        ode_model = DifferentialEquation(func=system, t0=0, times=times, n_states=1, n_theta=2)

        with pm.Model() as model:
            alpha = pm.HalfCauchy("alpha", 1)
            beta = pm.HalfCauchy("beta", 1)
            y0 = pm.Lognormal("y0", 0, 1)
            sigma = pm.HalfCauchy("sigma", 1)
            forward = ode_model(theta=[alpha, beta], y0=[y0])
            y = pm.Lognormal("y", mu=pm.math.log(forward), sd=sigma, observed=yobs)

            trace = pm.sample(100, tune=0, chains=1, return_inferencedata=False)

        assert trace["alpha"].size > 0
        assert trace["beta"].size > 0
        assert trace["y0"].size > 0
        assert trace["sigma"].size > 0

    def test_vector_ode_1_param(self):
        """Test running model for a vector ODE with 1 parameter"""

        def system(y, t, p):
            ds = -p[0] * y[0] * y[1]
            di = p[0] * y[0] * y[1] - y[1]
            return [ds, di]

        times = np.array([0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2, 8.0])

        yobs = np.array(
            [
                [1.02, 0.02],
                [0.86, 0.12],
                [0.43, 0.37],
                [0.14, 0.42],
                [0.05, 0.43],
                [0.03, 0.14],
                [0.02, 0.08],
                [0.02, 0.04],
                [0.02, 0.01],
                [0.02, 0.01],
                [0.02, 0.01],
            ]
        )

        ode_model = DifferentialEquation(func=system, t0=0, times=times, n_states=2, n_theta=1)

        with pm.Model() as model:
            R = pm.Lognormal("R", 1, 5)
            sigma = pm.HalfCauchy("sigma", 1, shape=2)
            forward = ode_model(theta=[R], y0=[0.99, 0.01])
            y = pm.Lognormal("y", mu=pm.math.log(forward), sd=sigma, observed=yobs)

            trace = pm.sample(100, tune=0, chains=1, return_inferencedata=False)

        assert trace["R"].size > 0
        assert trace["sigma"].size > 0

    def test_vector_ode_2_param(self):
        """Test running model for a vector ODE with 2 parameters"""

        def system(y, t, p):
            ds = -p[0] * y[0] * y[1]
            di = p[0] * y[0] * y[1] - p[1] * y[1]
            return [ds, di]

        times = np.array([0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2, 8.0])

        yobs = np.array(
            [
                [1.02, 0.02],
                [0.86, 0.12],
                [0.43, 0.37],
                [0.14, 0.42],
                [0.05, 0.43],
                [0.03, 0.14],
                [0.02, 0.08],
                [0.02, 0.04],
                [0.02, 0.01],
                [0.02, 0.01],
                [0.02, 0.01],
            ]
        )

        ode_model = DifferentialEquation(func=system, t0=0, times=times, n_states=2, n_theta=2)

        with pm.Model() as model:
            beta = pm.HalfCauchy("beta", 1)
            gamma = pm.HalfCauchy("gamma", 1)
            sigma = pm.HalfCauchy("sigma", 1, shape=2)
            forward = ode_model(theta=[beta, gamma], y0=[0.99, 0.01])
            y = pm.Lognormal("y", mu=pm.math.log(forward), sd=sigma, observed=yobs)

            trace = pm.sample(100, tune=0, chains=1, return_inferencedata=False)

        assert trace["beta"].size > 0
        assert trace["gamma"].size > 0
        assert trace["sigma"].size > 0
