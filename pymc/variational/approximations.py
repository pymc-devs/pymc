#   Copyright 2024 The PyMC Developers
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
import pytensor

from arviz import InferenceData
from pytensor import tensor as pt
from pytensor.graph.basic import Variable
from pytensor.graph.replace import graph_replace
from pytensor.tensor.variable import TensorVariable

import pymc as pm

from pymc.blocking import DictToArrayBijection
from pymc.distributions.dist_math import rho2sigma
from pymc.util import makeiter
from pymc.variational import opvi
from pymc.variational.opvi import (
    Approximation,
    Group,
    NotImplementedInference,
    _known_scan_ignored_inputs,
    node_property,
)

__all__ = ["MeanField", "FullRank", "Empirical", "sample_approx"]


@Group.register
class MeanFieldGroup(Group):
    """Mean Field approximation to the posterior.

    Spherical Gaussian family is fitted to minimize KL divergence from posterior.

    It is assumed that latent space variables are uncorrelated that is the main
    drawback of the method.
    """

    __param_spec__ = {"mu": ("d",), "rho": ("d",)}
    short_name = "mean_field"
    alias_names = frozenset(["mf"])

    @node_property
    def mean(self):
        return self.params_dict["mu"]

    @node_property
    def rho(self):
        return self.params_dict["rho"]

    @node_property
    def cov(self):
        var = rho2sigma(self.rho) ** 2
        return pt.diag(var)

    @node_property
    def std(self):
        return rho2sigma(self.rho)

    @pytensor.config.change_flags(compute_test_value="off")
    def __init_group__(self, group):
        super().__init_group__(group)
        if not self._check_user_params():
            self.shared_params = self.create_shared_params(
                self._kwargs.get("start", None), self._kwargs.get("start_sigma", None)
            )
        self._finalize_init()

    def create_shared_params(self, start=None, start_sigma=None):
        # NOTE: `Group._prepare_start` uses `self.model.free_RVs` to identify free variables and
        # `DictToArrayBijection` to turn them into a flat array, while `Approximation.rslice` assumes that the free
        # variables are given by `self.group` and that the mapping between original variables and flat array is given
        # by `self.ordering`. In the cases I looked into these turn out to be the same, but there may be edge cases or
        # future code changes that break this assumption.
        start = self._prepare_start(start)
        rho1 = np.zeros((self.ddim,))

        if start_sigma is not None:
            for name, slice_, *_ in self.ordering.values():
                sigma = start_sigma.get(name)
                if sigma is not None:
                    rho1[slice_] = np.log(np.expm1(np.abs(sigma)))
        rho = rho1

        return {
            "mu": pytensor.shared(pm.floatX(start), "mu"),
            "rho": pytensor.shared(pm.floatX(rho), "rho"),
        }

    @node_property
    def symbolic_random(self):
        initial = self.symbolic_initial
        sigma = self.std
        mu = self.mean
        return sigma * initial + mu

    @node_property
    def symbolic_logq_not_scaled(self):
        z0 = self.symbolic_initial
        std = rho2sigma(self.rho)
        logdet = pt.log(std)
        quaddist = -0.5 * z0**2 - pt.log((2 * np.pi) ** 0.5)
        logq = quaddist - logdet
        return logq.sum(range(1, logq.ndim))


@Group.register
class FullRankGroup(Group):
    """Full Rank approximation to the posterior.

    Multivariate Gaussian family is fitted to minimize KL divergence from posterior.

    In contrast to MeanField approach, correlations between variables are taken
    into account. The main drawback of the method is its computational cost.
    """

    __param_spec__ = {"mu": ("d",), "L_tril": ("int(d * (d + 1) / 2)",)}
    short_name = "full_rank"
    alias_names = frozenset(["fr"])

    @pytensor.config.change_flags(compute_test_value="off")
    def __init_group__(self, group):
        super().__init_group__(group)
        if not self._check_user_params():
            self.shared_params = self.create_shared_params(self._kwargs.get("start", None))
        self._finalize_init()

    def create_shared_params(self, start=None):
        start = self._prepare_start(start)
        n = self.ddim
        L_tril = np.eye(n)[np.tril_indices(n)].astype(pytensor.config.floatX)
        return {"mu": pytensor.shared(start, "mu"), "L_tril": pytensor.shared(L_tril, "L_tril")}

    @node_property
    def L(self):
        L = pt.zeros((self.ddim, self.ddim))
        L = pt.set_subtensor(L[self.tril_indices], self.params_dict["L_tril"])
        Ld = L[..., np.arange(self.ddim), np.arange(self.ddim)]
        L = pt.set_subtensor(Ld, rho2sigma(Ld))
        return L

    @node_property
    def mean(self):
        return self.params_dict["mu"]

    @node_property
    def cov(self):
        L = self.L
        return L.dot(L.T)

    @node_property
    def std(self):
        return pt.sqrt(pt.diag(self.cov))

    @property
    def num_tril_entries(self):
        n = self.ddim
        return int(n * (n + 1) / 2)

    @property
    def tril_indices(self):
        return np.tril_indices(self.ddim)

    @node_property
    def symbolic_logq_not_scaled(self):
        z0 = self.symbolic_initial
        diag = pt.diagonal(self.L, 0, self.L.ndim - 2, self.L.ndim - 1)
        logdet = pt.log(diag)
        quaddist = -0.5 * z0**2 - pt.log((2 * np.pi) ** 0.5)
        logq = quaddist - logdet
        return logq.sum(range(1, logq.ndim))

    @node_property
    def symbolic_random(self):
        initial = self.symbolic_initial
        L = self.L
        mu = self.mean
        return initial.dot(L.T) + mu


@Group.register
class EmpiricalGroup(Group):
    """Builds Approximation instance from a given trace.

    It has the same interface as variational approximation.
    """

    has_logq = False
    __param_spec__ = {"histogram": ("s", "d")}
    short_name = "empirical"

    @pytensor.config.change_flags(compute_test_value="off")
    def __init_group__(self, group):
        super().__init_group__(group)
        self._check_trace()
        if not self._check_user_params(spec_kw={"s": -1}):
            self.shared_params = self.create_shared_params(
                trace=self._kwargs.get("trace", None),
                size=self._kwargs.get("size", None),
                jitter=self._kwargs.get("jitter", 1),
                start=self._kwargs.get("start", None),
            )
        self._finalize_init()

    def create_shared_params(self, trace=None, size=None, jitter=1, start=None):
        if trace is None:
            if size is None:
                raise opvi.ParametrizationError("Need `trace` or `size` to initialize")
            else:
                start = self._prepare_start(start)
                # Initialize particles
                histogram = np.tile(start, (size, 1))
                histogram += pm.floatX(np.random.normal(0, jitter, histogram.shape))
        else:
            histogram = np.empty((len(trace) * len(trace.chains), self.ddim))
            i = 0
            for t in trace.chains:
                for j in range(len(trace)):
                    histogram[i] = DictToArrayBijection.map(trace.point(j, t)).data
                    i += 1
        return {"histogram": pytensor.shared(pm.floatX(histogram), "histogram")}

    def _check_trace(self):
        trace = self._kwargs.get("trace", None)
        if isinstance(trace, InferenceData):
            raise NotImplementedError(
                "The `Empirical` approximation does not yet support `InferenceData` inputs."
                " Pass `pm.sample(return_inferencedata=False)` to get a `MultiTrace` to use with `Empirical`."
                " Please help us to refactor: https://github.com/pymc-devs/pymc/issues/5884"
            )
        elif trace is not None and not all(
            self.model.rvs_to_values[var].name in trace.varnames for var in self.group
        ):
            raise ValueError("trace has not all free RVs in the group")

    def randidx(self, size=None):
        if size is None:
            size = (1,)
        elif isinstance(size, TensorVariable):
            if size.ndim < 1:
                size = size[None]
            elif size.ndim > 1:
                raise ValueError("size ndim should be no more than 1d")
            else:
                pass
        else:
            size = tuple(np.atleast_1d(size))
        return pt.random.integers(
            size=size,
            low=0,
            high=self.histogram.shape[0],
            rng=pytensor.shared(np.random.default_rng()),
        )

    def _new_initial(self, size, deterministic, more_replacements=None):
        pytensor_condition_is_here = isinstance(deterministic, Variable)
        if size is None:
            size = 1
        size = pt.as_tensor(size)
        if pytensor_condition_is_here:
            return pt.switch(
                deterministic,
                pt.repeat(self.mean.reshape((1, -1)), size, -1),
                self.histogram[self.randidx(size)],
            )
        else:
            if deterministic:
                raise NotImplementedInference(
                    "Deterministic sampling from a Histogram is broken in v4"
                )
                return pt.repeat(self.mean.reshape((1, -1)), size, -1)
            else:
                return self.histogram[self.randidx(size)]

    @property
    def symbolic_random(self):
        return self.symbolic_initial

    @property
    def histogram(self):
        return self.params_dict["histogram"]

    @node_property
    def mean(self):
        return self.histogram.mean(0)

    @node_property
    def cov(self):
        x = self.histogram - self.mean
        return x.T.dot(x) / pm.floatX(self.histogram.shape[0])

    @node_property
    def std(self):
        return pt.sqrt(pt.diag(self.cov))

    def __str__(self):
        if isinstance(self.histogram, pytensor.compile.SharedVariable):
            shp = ", ".join(map(str, self.histogram.shape.eval()))
        else:
            shp = "None, " + str(self.ddim)
        return f"{self.__class__.__name__}[{shp}]"


def sample_approx(approx, draws=100, include_transformed=True):
    """Draw samples from variational posterior.

    Parameters
    ----------
    approx: :class:`Approximation`
        Approximation to sample from
    draws: `int`
        Number of random samples.
    include_transformed: `bool`
        If True, transformed variables are also sampled. Default is True.

    Returns
    -------
    trace: class:`pymc.backends.base.MultiTrace`
        Samples drawn from variational posterior.
    """
    return approx.sample(draws=draws, include_transformed=include_transformed)


# single group shortcuts exported to user
class SingleGroupApproximation(Approximation):
    """Base class for Single Group Approximation."""

    _group_class: type | None = None

    def __init__(self, *args, **kwargs):
        groups = [self._group_class(None, *args, **kwargs)]
        super().__init__(groups, model=kwargs.get("model"))

    def __getattr__(self, item):
        return getattr(self.groups[0], item)

    def __dir__(self):
        d = set(super().__dir__())
        d.update(self.groups[0].__dir__())
        return sorted(d)


class MeanField(SingleGroupApproximation):
    __doc__ = """**Single Group Mean Field Approximation**

    """ + str(MeanFieldGroup.__doc__)
    _group_class = MeanFieldGroup


class FullRank(SingleGroupApproximation):
    __doc__ = """**Single Group Full Rank Approximation**

    """ + str(FullRankGroup.__doc__)
    _group_class = FullRankGroup


class Empirical(SingleGroupApproximation):
    __doc__ = """**Single Group Full Rank Approximation**

    """ + str(EmpiricalGroup.__doc__)
    _group_class = EmpiricalGroup

    def __init__(self, trace=None, size=None, **kwargs):
        super().__init__(trace=trace, size=size, **kwargs)

    def evaluate_over_trace(self, node):
        R"""
        Allow to statically evaluate any symbolic expression over the trace.

        Parameters
        ----------
        node: PyTensor Variables (or PyTensor expressions)

        Returns
        -------
        evaluated node(s) over the posterior trace contained in the empirical approximation
        """
        node = self.to_flat_input(node)

        def sample(post, *_):
            return graph_replace(node, {self.input: post}, strict=False)

        nodes, _ = pytensor.scan(
            sample, self.histogram, non_sequences=_known_scan_ignored_inputs(makeiter(node))
        )
        return nodes
