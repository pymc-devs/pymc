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

import warnings

import numpy as np
import theano.tensor as tt
import theano.tensor.slinalg  # pylint: disable=unused-import

from scipy.cluster.vq import kmeans

cholesky = tt.slinalg.cholesky
solve_lower = tt.slinalg.Solve(A_structure="lower_triangular")
solve_upper = tt.slinalg.Solve(A_structure="upper_triangular")
solve = tt.slinalg.Solve(A_structure="general")


def infer_shape(X, n_points=None):
    if n_points is None:
        try:
            n_points = int(X.shape[0])
        except TypeError:
            raise TypeError("Cannot infer 'shape', provide as an argument")
    return n_points


def stabilize(K):
    """adds small diagonal to a covariance matrix"""
    return K + 1e-6 * tt.identity_like(K)


def kmeans_inducing_points(n_inducing, X):
    # first whiten X
    if isinstance(X, tt.TensorConstant):
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
    Xu, distortion = kmeans(Xw, n_inducing)
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
