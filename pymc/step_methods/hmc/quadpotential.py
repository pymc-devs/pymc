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

import aesara
import numpy as np
import scipy.linalg

from numpy.random import normal
from scipy.sparse import issparse

from pymc.aesaraf import floatX

__all__ = [
    "quad_potential",
    "QuadPotentialDiag",
    "QuadPotentialFull",
    "QuadPotentialFullInv",
    "QuadPotentialDiagAdapt",
    "QuadPotentialFullAdapt",
    "isquadpotential",
]


def quad_potential(C, is_cov):
    """
    Compute a QuadPotential object from a scaling matrix.

    Parameters
    ----------
    C: arraylike, 0 <= ndim <= 2
        scaling matrix for the potential
        vector treated as diagonal matrix.
    is_cov: Boolean
        whether C is provided as a covariance matrix or hessian

    Returns
    -------
    q: Quadpotential
    """
    if issparse(C):
        if not chol_available:
            raise ImportError("Sparse mass matrices require scikits.sparse")
        elif is_cov:
            return QuadPotentialSparse(C)
        else:
            raise ValueError("Sparse precision matrices are not supported")

    partial_check_positive_definite(C)
    if C.ndim == 1:
        if is_cov:
            return QuadPotentialDiag(C)
        else:
            return QuadPotentialDiag(1.0 / C)
    else:
        if is_cov:
            return QuadPotentialFull(C)
        else:
            return QuadPotentialFullInv(C)


def partial_check_positive_definite(C):
    """Make a simple but partial check for Positive Definiteness."""
    if C.ndim == 1:
        d = C
    else:
        d = np.diag(C)
    (i,) = np.nonzero(np.logical_or(np.isnan(d), d <= 0))

    if len(i):
        raise PositiveDefiniteError("Simple check failed. Diagonal contains negatives", i)


class PositiveDefiniteError(ValueError):
    def __init__(self, msg, idx):
        super().__init__(msg)
        self.idx = idx
        self.msg = msg

    def __str__(self):
        return f"Scaling is not positive definite: {self.msg}. Check indexes {self.idx}."


class QuadPotential:
    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        raise NotImplementedError("Abstract method")

    def energy(self, x, velocity=None):
        raise NotImplementedError("Abstract method")

    def random(self, x):
        raise NotImplementedError("Abstract method")

    def velocity_energy(self, x, v_out):
        raise NotImplementedError("Abstract method")

    def update(self, sample, grad, tune):
        """Inform the potential about a new sample during tuning.

        This can be used by adaptive potentials to change the
        mass matrix.
        """
        pass

    def raise_ok(self, map_info=None):
        """Check if the mass matrix is ok, and raise ValueError if not.

        Parameters
        ----------
        map_info: List of (name, shape, dtype)
            List tuples with variable name, shape, and dtype.

        Raises
        ------
        ValueError if any standard deviations are 0 or infinite

        Returns
        -------
        None
        """
        return None

    def reset(self):
        pass


def isquadpotential(value):
    """Check whether an object might be a QuadPotential object."""
    return isinstance(value, QuadPotential)


class QuadPotentialDiagAdapt(QuadPotential):
    """Adapt a diagonal mass matrix from the sample variances."""

    def __init__(
        self,
        n,
        initial_mean,
        initial_diag=None,
        initial_weight=0,
        adaptation_window=101,
        adaptation_window_multiplier=1,
        dtype=None,
        discard_window=50,
        early_update=False,
        store_mass_matrix_trace=False,
    ):
        """Set up a diagonal mass matrix.

        Parameters
        ----------
        n : int
            The number of parameters.
        initial_mean : np.ndarray
            An initial guess for the posterior mean of each parameter.
        initial_diag : np.ndarray
            An estimate of the posterior variance of each parameter.
        initial_weight : int
            How much weight the initial guess has compared to new samples during tuning.
            Measured in equivalent number of samples.
        adaptation_window : int
            The size of the adaptation window during tuning. It specifies how many samples
            are used to estimate the mass matrix in each section of the adaptation.
        adaptation_window_multiplier : float
            The factor with which we increase the adaptation window after each adaptation
            window.
        dtype : np.dtype
            The dtype used to store the mass matrix
        discard_window : int
            The number of initial samples that are just discarded and not used to estimate
            the mass matrix.
        early_update : bool
            Whether to update the mass matrix live during the first adaptation window.
        store_mass_matrix_trace : bool
            If true, store the mass matrix at each step of the adaptation. Only for debugging
            purposes.
        """
        if initial_diag is not None and initial_diag.ndim != 1:
            raise ValueError("Initial diagonal must be one-dimensional.")
        if initial_mean.ndim != 1:
            raise ValueError("Initial mean must be one-dimensional.")
        if initial_diag is not None and len(initial_diag) != n:
            raise ValueError(f"Wrong shape for initial_diag: expected {n} got {len(initial_diag)}")
        if len(initial_mean) != n:
            raise ValueError(f"Wrong shape for initial_mean: expected {n} got {len(initial_mean)}")

        if dtype is None:
            dtype = aesara.config.floatX

        if initial_diag is None:
            initial_diag = np.ones(n, dtype=dtype)
            initial_weight = 1

        self.dtype = dtype
        self._n = n

        self._discard_window = discard_window
        self._early_update = early_update

        self._initial_mean = initial_mean
        self._initial_diag = initial_diag
        self._initial_weight = initial_weight
        self.adaptation_window = adaptation_window
        self.adaptation_window_multiplier = float(adaptation_window_multiplier)

        self._store_mass_matrix_trace = store_mass_matrix_trace
        self._mass_trace = []

        self.reset()

    def reset(self):
        self._var = np.array(self._initial_diag, dtype=self.dtype, copy=True)
        self._var_aesara = aesara.shared(self._var)
        self._stds = np.sqrt(self._initial_diag)
        self._inv_stds = floatX(1.0) / self._stds
        self._foreground_var = _WeightedVariance(
            self._n, self._initial_mean, self._initial_diag, self._initial_weight, self.dtype
        )
        self._background_var = _WeightedVariance(self._n, dtype=self.dtype)
        self._n_samples = 0

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        return np.multiply(self._var, x, out=out)

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is not None:
            return 0.5 * x.dot(velocity)
        return 0.5 * x.dot(self._var * x)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)

    def random(self):
        """Draw random value from QuadPotential."""
        vals = normal(size=self._n).astype(self.dtype)
        return self._inv_stds * vals

    def _update_from_weightvar(self, weightvar):
        weightvar.current_variance(out=self._var)
        np.sqrt(self._var, out=self._stds)
        np.divide(1, self._stds, out=self._inv_stds)
        self._var_aesara.set_value(self._var)

    def update(self, sample, grad, tune):
        """Inform the potential about a new sample during tuning."""
        if self._store_mass_matrix_trace:
            self._mass_trace.append(self._stds.copy())

        if not tune:
            return

        if self._n_samples > self._discard_window:
            self._foreground_var.add_sample(sample)
            self._background_var.add_sample(sample)

        if self._early_update or self._n_samples > self.adaptation_window:
            self._update_from_weightvar(self._foreground_var)

        if self._n_samples > 0 and self._n_samples % self.adaptation_window == 0:
            self._foreground_var = self._background_var
            self._background_var = _WeightedVariance(self._n, dtype=self.dtype)
            self.adaptation_window = int(self.adaptation_window * self.adaptation_window_multiplier)

        self._n_samples += 1

    def raise_ok(self, map_info):
        """Check if the mass matrix is ok, and raise ValueError if not.

        Parameters
        ----------
        map_info: List of (name, shape, dtype)
            List tuples with variable name, shape, and dtype.

        Raises
        ------
        ValueError if any standard deviations are 0 or infinite

        Returns
        -------
        None
        """
        if np.any(self._stds == 0):
            errmsg = ["Mass matrix contains zeros on the diagonal. "]
            last_idx = 0
            for name, shape, dtype in map_info:
                arr_len = np.prod(shape, dtype=int)
                index = np.where(self._stds[last_idx : last_idx + arr_len] == 0)[0]
                errmsg.append(f"The derivative of RV `{name}`.ravel()[{index}] is zero.")
                last_idx += arr_len

            raise ValueError("\n".join(errmsg))

        if np.any(~np.isfinite(self._stds)):
            errmsg = ["Mass matrix contains non-finite values on the diagonal. "]

            last_idx = 0
            for name, shape, dtype in map_info:
                arr_len = np.prod(shape, dtype=int)
                index = np.where(~np.isfinite(self._stds[last_idx : last_idx + arr_len]))[0]
                errmsg.append(f"The derivative of RV `{name}`.ravel()[{index}] is non-finite.")
                last_idx += arr_len
            raise ValueError("\n".join(errmsg))


class _WeightedVariance:
    """Online algorithm for computing mean of variance."""

    def __init__(
        self, nelem, initial_mean=None, initial_variance=None, initial_weight=0, dtype="d"
    ):
        self._dtype = dtype
        self.n_samples = float(initial_weight)
        if initial_mean is None:
            self.mean = np.zeros(nelem, dtype="d")
        else:
            self.mean = np.array(initial_mean, dtype="d", copy=True)
        if initial_variance is None:
            self.raw_var = np.zeros(nelem, dtype="d")
        else:
            self.raw_var = np.array(initial_variance, dtype="d", copy=True)

        self.raw_var[:] *= self.n_samples

        if self.raw_var.shape != (nelem,):
            raise ValueError("Invalid shape for initial variance.")
        if self.mean.shape != (nelem,):
            raise ValueError("Invalid shape for initial mean.")

    def add_sample(self, x):
        x = np.asarray(x)
        self.n_samples += 1
        old_diff = x - self.mean
        self.mean[:] += old_diff / self.n_samples
        new_diff = x - self.mean
        self.raw_var[:] += old_diff * new_diff

    def current_variance(self, out=None):
        if self.n_samples == 0:
            raise ValueError("Can not compute variance without samples.")
        if out is not None:
            return np.divide(self.raw_var, self.n_samples, out=out)
        else:
            return (self.raw_var / self.n_samples).astype(self._dtype)

    def current_mean(self):
        return self.mean.copy(dtype=self._dtype)


class _ExpWeightedVariance:
    def __init__(self, n_vars, *, init_mean, init_var, alpha):
        self._variance = init_var
        self._mean = init_mean
        self._alpha = alpha

    def add_sample(self, value):
        alpha = self._alpha
        delta = value - self._mean
        self._mean[...] += alpha * delta
        self._variance[...] = (1 - alpha) * (self._variance + alpha * delta**2)

    def current_variance(self, out=None):
        if out is None:
            out = np.empty_like(self._variance)
        np.copyto(out, self._variance)
        return out

    def current_mean(self, out=None):
        if out is None:
            out = np.empty_like(self._mean)
        np.copyto(out, self._mean)
        return out


class QuadPotentialDiagAdaptExp(QuadPotentialDiagAdapt):
    def __init__(self, *args, alpha, use_grads=False, stop_adaptation=None, **kwargs):
        """Set up a diagonal mass matrix.

        Parameters
        ----------
        n : int
            The number of parameters.
        initial_mean : np.ndarray
            An initial guess for the posterior mean of each parameter.
        initial_diag : np.ndarray
            An estimate of the posterior variance of each parameter.
        alpha : float
            Decay rate of the exponetial weighted variance.
        use_grads : bool
            Use gradients, not only samples to estimate the mass matrix.
        stop_adaptation : int
            Stop the mass matrix adaptation after this many samples.
        dtype : np.dtype
            The dtype used to store the mass matrix
        discard_window : int
            The number of initial samples that are just discarded and not used to estimate
            the mass matrix.
        store_mass_matrix_trace : bool
            If true, store the mass matrix at each step of the adaptation. Only for debugging
            purposes.
        """
        if len(args) > 3:
            raise ValueError("Unsupported arguments to QuadPotentialDiagAdaptExp")

        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._use_grads = use_grads

        if stop_adaptation is None:
            stop_adaptation = np.inf
        self._stop_adaptation = stop_adaptation

    def update(self, sample, grad, tune):
        if tune and self._n_samples < self._stop_adaptation:
            if self._n_samples > self._discard_window:
                self._variance_estimator.add_sample(sample)
                if self._use_grads:
                    self._variance_estimator_grad.add_sample(grad)
            elif self._n_samples == self._discard_window:
                self._variance_estimator = _ExpWeightedVariance(
                    self._n,
                    init_mean=sample.copy(),
                    init_var=np.zeros_like(sample),
                    alpha=self._alpha,
                )
                if self._use_grads:
                    self._variance_estimator_grad = _ExpWeightedVariance(
                        self._n,
                        init_mean=grad.copy(),
                        init_var=np.zeros_like(grad),
                        alpha=self._alpha,
                    )

            if self._n_samples > 2 * self._discard_window:
                if self._use_grads:
                    self._update_from_variances(
                        self._variance_estimator, self._variance_estimator_grad
                    )
                else:
                    self._update_from_weightvar(self._variance_estimator)

            self._n_samples += 1

        if self._store_mass_matrix_trace:
            self._mass_trace.append(self._stds.copy())

    def _update_from_variances(self, var_estimator, inv_var_estimator):
        var = var_estimator.current_variance()
        inv_var = inv_var_estimator.current_variance()
        updated = np.sqrt(var / inv_var)
        self._var[:] = updated
        np.sqrt(updated, out=self._stds)
        np.divide(1, self._stds, out=self._inv_stds)


class QuadPotentialDiag(QuadPotential):
    """Quad potential using a diagonal covariance matrix."""

    def __init__(self, v, dtype=None):
        """Use a vector to represent a diagonal matrix for a covariance matrix.

        Parameters
        ----------
        v: vector, 0 <= ndim <= 1
           Diagonal of covariance matrix for the potential vector
        """
        if dtype is None:
            dtype = aesara.config.floatX
        self.dtype = dtype
        v = v.astype(self.dtype)
        s = v**0.5

        self.s = s
        self.inv_s = 1.0 / s
        self.v = v

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        if out is not None:
            np.multiply(x, self.v, out=out)
            return
        return self.v * x

    def random(self):
        """Draw random value from QuadPotential."""
        return floatX(normal(size=self.s.shape)) * self.inv_s

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is not None:
            return 0.5 * np.dot(x, velocity)
        return 0.5 * x.dot(self.v * x)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        np.multiply(x, self.v, out=v_out)
        return 0.5 * np.dot(x, v_out)


class QuadPotentialFullInv(QuadPotential):
    """QuadPotential object for Hamiltonian calculations using inverse of covariance matrix."""

    def __init__(self, A, dtype=None):
        """Compute the lower cholesky decomposition of the potential.

        Parameters
        ----------
        A: matrix, ndim = 2
           Inverse of covariance matrix for the potential vector
        """
        if dtype is None:
            dtype = aesara.config.floatX
        self.dtype = dtype
        self.L = floatX(scipy.linalg.cholesky(A, lower=True))

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        vel = scipy.linalg.cho_solve((self.L, True), x)
        if out is None:
            return vel
        out[:] = vel

    def random(self):
        """Draw random value from QuadPotential."""
        n = floatX(normal(size=self.L.shape[0]))
        return np.dot(self.L, n)

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is None:
            velocity = self.velocity(x)
        return 0.5 * x.dot(velocity)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)


class QuadPotentialFull(QuadPotential):
    """Basic QuadPotential object for Hamiltonian calculations."""

    def __init__(self, cov, dtype=None):
        """Compute the lower cholesky decomposition of the potential.

        Parameters
        ----------
        A: matrix, ndim = 2
            scaling matrix for the potential vector
        """
        if dtype is None:
            dtype = aesara.config.floatX
        self.dtype = dtype
        self._cov = np.array(cov, dtype=self.dtype, copy=True)
        self._chol = scipy.linalg.cholesky(self._cov, lower=True)
        self._n = len(self._cov)

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        return np.dot(self._cov, x, out=out)

    def random(self):
        """Draw random value from QuadPotential."""
        vals = np.random.normal(size=self._n).astype(self.dtype)
        return scipy.linalg.solve_triangular(self._chol.T, vals, overwrite_b=True)

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is None:
            velocity = self.velocity(x)
        return 0.5 * np.dot(x, velocity)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        self.velocity(x, out=v_out)
        return self.energy(x, v_out)

    __call__ = random


class QuadPotentialFullAdapt(QuadPotentialFull):
    """Adapt a dense mass matrix using the sample covariances."""

    def __init__(
        self,
        n,
        initial_mean,
        initial_cov=None,
        initial_weight=0,
        adaptation_window=101,
        adaptation_window_multiplier=2,
        update_window=1,
        dtype=None,
    ):
        warnings.warn("QuadPotentialFullAdapt is an experimental feature")

        if initial_cov is not None and initial_cov.ndim != 2:
            raise ValueError("Initial covariance must be two-dimensional.")
        if initial_mean.ndim != 1:
            raise ValueError("Initial mean must be one-dimensional.")
        if initial_cov is not None and initial_cov.shape != (n, n):
            raise ValueError(f"Wrong shape for initial_cov: expected {n} got {initial_cov.shape}")
        if len(initial_mean) != n:
            raise ValueError(f"Wrong shape for initial_mean: expected {n} got {len(initial_mean)}")

        if dtype is None:
            dtype = aesara.config.floatX

        if initial_cov is None:
            initial_cov = np.eye(n, dtype=dtype)
            initial_weight = 1

        self.dtype = dtype
        self._n = n
        self._initial_mean = initial_mean
        self._initial_cov = initial_cov
        self._initial_weight = initial_weight

        self.adaptation_window = int(adaptation_window)
        self.adaptation_window_multiplier = float(adaptation_window_multiplier)
        self._update_window = int(update_window)

        self.reset()

    def reset(self):
        self._previous_update = 0
        self._cov = np.array(self._initial_cov, dtype=self.dtype, copy=True)
        self._chol = scipy.linalg.cholesky(self._cov, lower=True)
        self._chol_error = None
        self._foreground_cov = _WeightedCovariance(
            self._n, self._initial_mean, self._initial_cov, self._initial_weight, self.dtype
        )
        self._background_cov = _WeightedCovariance(self._n, dtype=self.dtype)
        self._n_samples = 0

    def _update_from_weightvar(self, weightvar):
        weightvar.current_covariance(out=self._cov)
        try:
            self._chol = scipy.linalg.cholesky(self._cov, lower=True)
        except (scipy.linalg.LinAlgError, ValueError) as error:
            self._chol_error = error

    def update(self, sample, grad, tune):
        if not tune:
            return

        # Steps since previous update
        delta = self._n_samples - self._previous_update

        self._foreground_cov.add_sample(sample)
        self._background_cov.add_sample(sample)

        # Update the covariance matrix and recompute the Cholesky factorization
        # every "update_window" steps
        if (delta + 1) % self._update_window == 0:
            self._update_from_weightvar(self._foreground_cov)

        # Reset the background covariance if we are at the end of the adaptation
        # window.
        if delta >= self.adaptation_window:
            self._foreground_cov = self._background_cov
            self._background_cov = _WeightedCovariance(self._n, dtype=self.dtype)

            self._previous_update = self._n_samples
            self.adaptation_window = int(self.adaptation_window * self.adaptation_window_multiplier)

        self._n_samples += 1

    def raise_ok(self, vmap):
        if self._chol_error is not None:
            raise ValueError(str(self._chol_error))


class _WeightedCovariance:
    """Online algorithm for computing mean and covariance

    This implements the `Welford's algorithm
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_ based
    on the implementation in `the Stan math library
    <https://github.com/stan-dev/math>`_.

    """

    def __init__(
        self,
        nelem,
        initial_mean=None,
        initial_covariance=None,
        initial_weight=0,
        dtype="d",
    ):
        self._dtype = dtype
        self.n_samples = float(initial_weight)
        if initial_mean is None:
            self.mean = np.zeros(nelem, dtype="d")
        else:
            self.mean = np.array(initial_mean, dtype="d", copy=True)
        if initial_covariance is None:
            self.raw_cov = np.eye(nelem, dtype="d")
        else:
            self.raw_cov = np.array(initial_covariance, dtype="d", copy=True)

        self.raw_cov[:] *= self.n_samples

        if self.raw_cov.shape != (nelem, nelem):
            raise ValueError("Invalid shape for initial covariance.")
        if self.mean.shape != (nelem,):
            raise ValueError("Invalid shape for initial mean.")

    def add_sample(self, x):
        x = np.asarray(x)
        self.n_samples += 1
        old_diff = x - self.mean
        self.mean[:] += old_diff / self.n_samples
        new_diff = x - self.mean
        self.raw_cov[:] += new_diff[:, None] * old_diff[None, :]

    def current_covariance(self, out=None):
        if self.n_samples == 0:
            raise ValueError("Can not compute covariance without samples.")
        if out is not None:
            return np.divide(self.raw_cov, self.n_samples - 1, out=out)
        else:
            return (self.raw_cov / (self.n_samples - 1)).astype(self._dtype)

    def current_mean(self):
        return np.array(self.mean, dtype=self._dtype)


try:
    import sksparse.cholmod as cholmod

    chol_available = True
except ImportError:
    chol_available = False

if chol_available:
    __all__ += ["QuadPotentialSparse"]

    import aesara.sparse

    class QuadPotentialSparse(QuadPotential):
        def __init__(self, A):
            """Compute a sparse cholesky decomposition of the potential.

            Parameters
            ----------
            A: matrix, ndim = 2
                scaling matrix for the potential vector
            """
            self.A = A
            self.size = A.shape[0]
            self.factor = factor = cholmod.cholesky(A)
            self.d_sqrt = np.sqrt(factor.D())

        def velocity(self, x):
            """Compute the current velocity at a position in parameter space."""
            A = aesara.sparse.as_sparse(self.A)
            return aesara.sparse.dot(A, x)

        def random(self):
            """Draw random value from QuadPotential."""
            n = floatX(normal(size=self.size))
            n /= self.d_sqrt
            n = self.factor.solve_Lt(n)
            n = self.factor.apply_Pt(n)
            return n

        def energy(self, x):
            """Compute kinetic energy at a position in parameter space."""
            return 0.5 * x.T.dot(self.velocity(x))
