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

try:
    import matplotlib.pyplot as plt
except ImportError:  # mpl is optional
    pass
import numpy as np


def plot_posterior_predictive_glm(trace, eval=None, lm=None, samples=30, **kwargs):
    """Plot posterior predictive of a linear model.
    :Arguments:
        trace: <array>
            Array of posterior samples with columns
        eval: <array>
            Array over which to evaluate lm
        lm: function <default: linear function>
            Function mapping parameters at different points
            to their respective outputs.
            input: point, sample
            output: estimated value
        samples: int <default=30>
            How many posterior samples to draw.
    Additional keyword arguments are passed to pylab.plot().
    """
    if lm is None:
        lm = lambda x, sample: sample["Intercept"] + sample["x"] * x

    if eval is None:
        eval = np.linspace(0, 1, 100)

    # Set default plotting arguments
    if "lw" not in kwargs and "linewidth" not in kwargs:
        kwargs["lw"] = 0.2
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["c"] = "k"

    for rand_loc in np.random.randint(0, len(trace), samples):
        rand_sample = trace[rand_loc]
        plt.plot(eval, lm(eval, rand_sample), **kwargs)
        # Make sure to not plot label multiple times
        kwargs.pop("label", None)

    plt.title("Posterior predictive")
