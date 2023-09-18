#   Copyright 2023 The PyMC Developers
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

import warnings

import numpy as np
import pytensor.tensor as pt

from pytensor.compile import SharedVariable
from pytensor.tensor.variable import TensorConstant
from scipy.cluster.vq import kmeans

# Avoid circular dependency when importing modelcontext
from pymc.distributions.distribution import Distribution
from pymc.model import modelcontext
from pymc.pytensorf import compile_pymc, walk_model

_ = Distribution  # keep both pylint and black happy

JITTER_DEFAULT = 1e-6


def replace_with_values(vars_needed, replacements=None, model=None):
    R"""
    Replace random variable nodes in the graph with values given by the replacements dict.
    Uses untransformed versions of the inputs, performs some basic input validation.

    Parameters
    ----------
    vars_needed: list of TensorVariables
        A list of variable outputs
    replacements: dict with string keys, numeric values
        The variable name and values to be replaced in the model graph.
    model: Model
        A PyMC model object
    """
    model = modelcontext(model)

    inputs, input_names = [], []
    for rv in walk_model(vars_needed):
        if rv in model.named_vars.values() and not isinstance(rv, SharedVariable):
            inputs.append(rv)
            input_names.append(rv.name)

    # Then it's deterministic, no inputs are required, can eval and return
    if len(inputs) == 0:
        return tuple(v.eval() for v in vars_needed)

    fn = compile_pymc(
        inputs,
        vars_needed,
        allow_input_downcast=True,
        accept_inplace=True,
        on_unused_input="ignore",
    )

    # Remove unneeded inputs
    replacements = {name: val for name, val in replacements.items() if name in input_names}
    missing = set(input_names) - set(replacements.keys())

    # Error if more inputs are needed
    if len(missing) > 0:
        missing_str = ", ".join(missing)
        raise ValueError(f"Values for {missing_str} must be included in `replacements`.")

    return fn(**replacements)


def stabilize(K, jitter=JITTER_DEFAULT):
    R"""
    Adds small diagonal to a covariance matrix.

    Often the matrices calculated from covariance functions, `K = cov_func(X)`
    do not appear numerically to be positive semi-definite.  Adding a small
    correction, `jitter`, to the diagonal is usually enough to fix this.

    Parameters
    ----------
    K: array-like
        A square covariance or kernel matrix.
    jitter: float
        A small constant.
    """
    return K + jitter * pt.identity_like(K)


def kmeans_inducing_points(n_inducing, X, **kmeans_kwargs):
    R"""
    Use the K-means algorithm to initialize the locations `X` for the inducing
    points `fu`.

    Parameters
    ----------
    n_inducing: int
        The number of inducing points (or k, the number of clusters)
    X: array-like
        Gaussian process input matrix.
    **kmeans_kwargs:
        Extra keyword arguments that are passed to `scipy.cluster.vq.kmeans`
    """
    # first whiten X
    if isinstance(X, TensorConstant):
        X = X.value
    elif isinstance(X, (np.ndarray, tuple, list)):
        X = np.asarray(X)
    else:
        raise TypeError(
            "To use K-means initialization, "
            "please provide X as a type that "
            "can be cast to np.ndarray, instead "
            "of {}".format(type(X))
        )
    scaling = np.std(X, 0)
    # if std of a column is very small (zero), don't normalize that column
    scaling[scaling <= 1e-6] = 1.0
    Xw = X / scaling

    if "k_or_guess" in kmeans_kwargs:
        warnings.warn("Use `n_inducing` to set the `k_or_guess` parameter instead.")

    Xu, distortion = kmeans(Xw, k_or_guess=n_inducing, **kmeans_kwargs)
    return Xu * scaling


def conditioned_vars(varnames):
    """Decorator for validating attrs that are conditioned on."""

    def gp_wrapper(cls):
        def make_getter(name):
            def getter(self):
                value = getattr(self, name, None)
                if value is None:
                    raise AttributeError(
                        "'{}' not set.  Provide as argument "
                        "to condition, or call 'prior' "
                        "first".format(name.lstrip("_"))
                    )
                else:
                    return value
                return getattr(self, name)

            return getter

        def make_setter(name):
            def setter(self, val):
                setattr(self, name, val)

            return setter

        for name in varnames:
            getter = make_getter("_" + name)
            setter = make_setter("_" + name)
            setattr(cls, name, property(getter, setter))
        return cls

    return gp_wrapper


def plot_gp_dist(
    ax,
    samples: np.ndarray,
    x: np.ndarray,
    plot_samples=True,
    palette="Reds",
    fill_alpha=0.8,
    samples_alpha=0.1,
    fill_kwargs=None,
    samples_kwargs=None,
):
    """A helper function for plotting 1D GP posteriors from trace

    Parameters
    ----------
    ax: axes
        Matplotlib axes.
    samples: numpy.ndarray
        Array of S posterior predictive sample from a GP.
        Expected shape: (S, X)
    x: numpy.ndarray
        Grid of X values corresponding to the samples.
        Expected shape: (X,) or (X, 1), or (1, X)
    plot_samples: bool
        Plot the GP samples along with posterior (defaults True).
    palette: str
        Palette for coloring output (defaults to "Reds").
    fill_alpha: float
        Alpha value for the posterior interval fill (defaults to 0.8).
    samples_alpha: float
        Alpha value for the sample lines (defaults to 0.1).
    fill_kwargs: dict
        Additional arguments for posterior interval fill (fill_between).
    samples_kwargs: dict
        Additional keyword arguments for samples plot.

    Returns
    -------
    ax: Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if fill_kwargs is None:
        fill_kwargs = {}
    if samples_kwargs is None:
        samples_kwargs = {}
    if np.any(np.isnan(samples)):
        warnings.warn(
            "There are `nan` entries in the [samples] arguments. "
            "The plot will not contain a band!",
            UserWarning,
        )

    cmap = plt.get_cmap(palette)
    percs = np.linspace(51, 99, 40)
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    samples = samples.T
    x = x.flatten()
    for i, p in enumerate(percs[::-1]):
        upper = np.percentile(samples, p, axis=1)
        lower = np.percentile(samples, 100 - p, axis=1)
        color_val = colors[i]
        ax.fill_between(x, upper, lower, color=cmap(color_val), alpha=fill_alpha, **fill_kwargs)
    if plot_samples:
        # plot a few samples
        idx = np.random.randint(0, samples.shape[1], 30)
        ax.plot(x, samples[:, idx], color=cmap(0.9), lw=1, alpha=samples_alpha, **samples_kwargs)

    return ax
