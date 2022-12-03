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
import scipy.integrate as ode

from pymc.ode.utils import augment_system


def test_gradients():
    """Tests the computation of the sensitivities from the PyTensor computation graph"""

    # ODE system for which to compute gradients
    def ode_func(y, t, p):
        return np.exp(-t) - p[0] * y[0]

    # Computation of graidients with PyTensor
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

    integrated_solutions = ode.odeint(func=augmented_system, y0=[y0, 1, 0], t=t.ravel(), args=(p,))
    simulated_sensitivity = integrated_solutions[:, 1:]

    np.testing.assert_allclose(sensitivity, simulated_sensitivity, rtol=1e-5)
