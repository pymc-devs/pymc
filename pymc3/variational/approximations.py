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
import theano

from theano import tensor as tt

import pymc3 as pm

from pymc3.distributions.dist_math import rho2sigma
from pymc3.math import batched_diag
from pymc3.util import update_start_vals
from pymc3.variational import flows, opvi
from pymc3.variational.opvi import Approximation, Group, node_property

__all__ = ["MeanField", "FullRank", "Empirical", "NormalizingFlow", "sample_approx"]


@Group.register
class MeanFieldGroup(Group):
    R"""Mean Field approximation to the posterior where spherical Gaussian family
    is fitted to minimize KL divergence from True posterior. It is assumed
    that latent space variables are uncorrelated that is the main drawback
    of the method
    """
    __param_spec__ = dict(mu=("d",), rho=("d",))
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
        if self.batched:
            return batched_diag(var)
        else:
            return tt.diag(var)

    @node_property
    def std(self):
        return rho2sigma(self.rho)

    @theano.config.change_flags(compute_test_value="off")
    def __init_group__(self, group):
        super().__init_group__(group)
        if not self._check_user_params():
            self.shared_params = self.create_shared_params(self._kwargs.get("start", None))
        self._finalize_init()

    def create_shared_params(self, start=None):
        if start is None:
            start = self.model.test_point
        else:
            start_ = start.copy()
            update_start_vals(start_, self.model.test_point, self.model)
            start = start_
        if self.batched:
            start = start[self.group[0].name][0]
        else:
            start = self.bij.map(start)
        rho = np.zeros((self.ddim,))
        if self.batched:
            start = np.tile(start, (self.bdim, 1))
            rho = np.tile(rho, (self.bdim, 1))
        return {
            "mu": theano.shared(pm.floatX(start), "mu"),
            "rho": theano.shared(pm.floatX(rho), "rho"),
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
        logdet = tt.log(std)
        logq = pm.Normal.dist().logp(z0) - logdet
        return logq.sum(range(1, logq.ndim))


@Group.register
class FullRankGroup(Group):
    """Full Rank approximation to the posterior where Multivariate Gaussian family
    is fitted to minimize KL divergence from True posterior. In contrast to
    MeanField approach correlations between variables are taken in account. The
    main drawback of the method is computational cost.
    """

    __param_spec__ = dict(mu=("d",), L_tril=("int(d * (d + 1) / 2)",))
    short_name = "full_rank"
    alias_names = frozenset(["fr"])

    @theano.config.change_flags(compute_test_value="off")
    def __init_group__(self, group):
        super().__init_group__(group)
        if not self._check_user_params():
            self.shared_params = self.create_shared_params(self._kwargs.get("start", None))
        self._finalize_init()

    def create_shared_params(self, start=None):
        if start is None:
            start = self.model.test_point
        else:
            start_ = start.copy()
            update_start_vals(start_, self.model.test_point, self.model)
            start = start_
        if self.batched:
            start = start[self.group[0].name][0]
        else:
            start = self.bij.map(start)
        n = self.ddim
        L_tril = np.eye(n)[np.tril_indices(n)].astype(theano.config.floatX)
        if self.batched:
            start = np.tile(start, (self.bdim, 1))
            L_tril = np.tile(L_tril, (self.bdim, 1))
        return {"mu": theano.shared(start, "mu"), "L_tril": theano.shared(L_tril, "L_tril")}

    @node_property
    def L(self):
        if self.batched:
            L = tt.zeros((self.ddim, self.ddim, self.bdim))
            L = tt.set_subtensor(L[self.tril_indices], self.params_dict["L_tril"].T)
            L = L.dimshuffle(2, 0, 1)
        else:
            L = tt.zeros((self.ddim, self.ddim))
            L = tt.set_subtensor(L[self.tril_indices], self.params_dict["L_tril"])
        return L

    @node_property
    def mean(self):
        return self.params_dict["mu"]

    @node_property
    def cov(self):
        L = self.L
        if self.batched:
            return tt.batched_dot(L, L.swapaxes(-1, -2))
        else:
            return L.dot(L.T)

    @node_property
    def std(self):
        if self.batched:
            return tt.sqrt(batched_diag(self.cov))
        else:
            return tt.sqrt(tt.diag(self.cov))

    @property
    def num_tril_entries(self):
        n = self.ddim
        return int(n * (n + 1) / 2)

    @property
    def tril_indices(self):
        return np.tril_indices(self.ddim)

    @node_property
    def symbolic_logq_not_scaled(self):
        z = self.symbolic_random
        if self.batched:

            def logq(z_b, mu_b, L_b):
                return pm.MvNormal.dist(mu=mu_b, chol=L_b).logp(z_b)

            # it's gonna be so slow
            # scan is computed over batch and then summed up
            # output shape is (batch, samples)
            return theano.scan(logq, [z.swapaxes(0, 1), self.mean, self.L])[0].sum(0)
        else:
            return pm.MvNormal.dist(mu=self.mean, chol=self.L).logp(z)

    @node_property
    def symbolic_random(self):
        initial = self.symbolic_initial
        L = self.L
        mu = self.mean
        if self.batched:
            # initial: bxsxd
            # L: bxdxd
            initial = initial.swapaxes(0, 1)
            return tt.batched_dot(initial, L.swapaxes(1, 2)).swapaxes(0, 1) + mu
        else:
            return initial.dot(L.T) + mu


@Group.register
class EmpiricalGroup(Group):
    """Builds Approximation instance from a given trace,
    it has the same interface as variational approximation
    """

    supports_batched = False
    has_logq = False
    __param_spec__ = dict(histogram=("s", "d"))
    short_name = "empirical"

    @theano.config.change_flags(compute_test_value="off")
    def __init_group__(self, group):
        super().__init_group__(group)
        self._check_trace()
        if not self._check_user_params(spec_kw=dict(s=-1)):
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
                if start is None:
                    start = self.model.test_point
                else:
                    start_ = self.model.test_point.copy()
                    update_start_vals(start_, start, self.model)
                    start = start_
                start = pm.floatX(self.bij.map(start))
                # Initialize particles
                histogram = np.tile(start, (size, 1))
                histogram += pm.floatX(np.random.normal(0, jitter, histogram.shape))

        else:
            histogram = np.empty((len(trace) * len(trace.chains), self.ddim))
            i = 0
            for t in trace.chains:
                for j in range(len(trace)):
                    histogram[i] = self.bij.map(trace.point(j, t))
                    i += 1
        return dict(histogram=theano.shared(pm.floatX(histogram), "histogram"))

    def _check_trace(self):
        trace = self._kwargs.get("trace", None)
        if trace is not None and not all([var.name in trace.varnames for var in self.group]):
            raise ValueError("trace has not all FreeRV in the group")

    def randidx(self, size=None):
        if size is None:
            size = (1,)
        elif isinstance(size, tt.TensorVariable):
            if size.ndim < 1:
                size = size[None]
            elif size.ndim > 1:
                raise ValueError("size ndim should be no more than 1d")
            else:
                pass
        else:
            size = tuple(np.atleast_1d(size))
        return self._rng.uniform(
            size=size, low=pm.floatX(0), high=pm.floatX(self.histogram.shape[0]) - pm.floatX(1e-16)
        ).astype("int32")

    def _new_initial(self, size, deterministic, more_replacements=None):
        theano_condition_is_here = isinstance(deterministic, tt.Variable)
        if theano_condition_is_here:
            return tt.switch(
                deterministic,
                tt.repeat(self.mean.dimshuffle("x", 0), size if size is not None else 1, -1),
                self.histogram[self.randidx(size)],
            )
        else:
            if deterministic:
                return tt.repeat(self.mean.dimshuffle("x", 0), size if size is not None else 1, -1)
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
        return tt.sqrt(tt.diag(self.cov))

    def __str__(self):
        if isinstance(self.histogram, theano.compile.SharedVariable):
            shp = ", ".join(map(str, self.histogram.shape.eval()))
        else:
            shp = "None, " + str(self.ddim)
        return f"{self.__class__.__name__}[{shp}]"


class NormalizingFlowGroup(Group):
    R"""Normalizing flow is a series of invertible transformations on initial distribution.

    .. math::

        z_K &= f_K \circ \dots \circ f_2 \circ f_1(z_0) \\
        & z_0 \sim \mathcal{N}(0, 1)

    In that case we can compute tractable density for the flow.

    .. math::

        \ln q_K(z_K) = \ln q_0(z_0) - \sum_{k=1}^{K}\ln \left|\frac{\partial f_k}{\partial z_{k-1}}\right|


    Every :math:`f_k` here is a parametric function with defined determinant.
    We can choose every step here. For example the here is a simple flow
    is an affine transform:

    .. math::

        z = loc(scale(z_0)) = \mu + \sigma * z_0

    Here we get mean field approximation if :math:`z_0 \sim \mathcal{N}(0, 1)`

    **Flow Formulas**

    In PyMC3 there is a flexible way to define flows with formulas. We have 5 of them by the moment:

    -   Loc (:code:`loc`): :math:`z' = z + \mu`
    -   Scale (:code:`scale`): :math:`z' = \sigma * z`
    -   Planar (:code:`planar`): :math:`z' = z + u * \tanh(w^T z + b)`
    -   Radial (:code:`radial`): :math:`z' = z + \beta (\alpha + (z-z_r))^{-1}(z-z_r)`
    -   Householder (:code:`hh`): :math:`z' = H z`

    Formula can be written as a string, e.g. `'scale-loc'`, `'scale-hh*4-loc'`, `'panar*10'`.
    Every step is separated with `'-'`, repeated flow is marked with `'*'` producing `'flow*repeats'`.

    References
    ----------
    -   Danilo Jimenez Rezende, Shakir Mohamed, 2015
        Variational Inference with Normalizing Flows
        arXiv:1505.05770

    -   Jakub M. Tomczak, Max Welling, 2016
        Improving Variational Auto-Encoders using Householder Flow
        arXiv:1611.09630
    """
    default_flow = "scale-loc"

    @theano.config.change_flags(compute_test_value="off")
    def __init_group__(self, group):
        super().__init_group__(group)
        # objects to be resolved
        # 1. string formula
        # 2. not changed default value
        # 3. Formula
        formula = self._kwargs.get("flow", self._vfam)
        jitter = self._kwargs.get("jitter", 1)
        if formula is None or isinstance(formula, str):
            # case 1 and 2
            has_params = self._check_user_params(f=formula)
        elif isinstance(formula, flows.Formula):
            # case 3
            has_params = self._check_user_params(f=formula.formula)
        else:
            raise TypeError(
                "Wrong type provided for NormalizingFlow as `flow` argument, "
                "expected Formula or string"
            )
        if not has_params:
            if formula is None:
                formula = self.default_flow
        else:
            formula = "-".join(
                flows.flow_for_params(self.user_params[i]).short_name
                for i in range(len(self.user_params))
            )
        if not isinstance(formula, flows.Formula):
            formula = flows.Formula(formula)
        if self.local:
            bs = -1
        elif self.batched:
            bs = self.bdim
        else:
            bs = None
        self.flow = formula(
            dim=self.ddim,
            z0=self.symbolic_initial,
            jitter=jitter,
            params=self.user_params,
            batch_size=bs,
        )
        self._finalize_init()

    def _check_user_params(self, **kwargs):
        params = self._user_params = self.user_params
        formula = kwargs.pop("f")
        if params is None:
            return False
        if formula is not None:
            raise opvi.ParametrizationError("No formula is allowed if user params are provided")
        if not isinstance(params, dict):
            raise TypeError("params should be a dict")
        if not all(isinstance(k, int) for k in params.keys()):
            raise TypeError("params should be a dict with `int` keys")
        needed = set(range(len(params)))
        givens = set(params.keys())
        if givens != needed:
            raise opvi.ParametrizationError(
                "Passed parameters do not have a needed set of keys, "
                "they should be equal, needed {needed}, got {givens}".format(
                    givens=list(sorted(givens)), needed="[0, 1, ..., %d]" % len(formula.flows)
                )
            )
        for i in needed:
            flow = flows.flow_for_params(params[i])
            flow_keys = set(flow.__param_spec__)
            user_keys = set(params[i].keys())
            if flow_keys != user_keys:
                raise opvi.ParametrizationError(
                    "Passed parameters for flow `{i}` ({cls}) do not have a needed set of keys, "
                    "they should be equal, needed {needed}, got {givens}".format(
                        givens=user_keys, needed=flow_keys, i=i, cls=flow.__name__
                    )
                )
        return True

    @property
    def shared_params(self):
        if self.user_params is not None:
            return None
        params = dict()
        current = self.flow
        i = 0
        params[i] = current.shared_params
        while not current.isroot:
            i += 1
            current = current.parent
            params[i] = current.shared_params
        return params

    @shared_params.setter
    def shared_params(self, value):
        if self.user_params is not None:
            raise AttributeError("Cannot set when having user params")
        current = self.flow
        i = 0
        current.shared_params = value[i]
        while not current.isroot:
            i += 1
            current = current.parent
            current.shared_params = value[i]

    @property
    def params(self):
        return self.flow.all_params

    @node_property
    def symbolic_logq_not_scaled(self):
        z0 = self.symbolic_initial
        q0 = pm.Normal.dist().logp(z0).sum(range(1, z0.ndim))
        return q0 - self.flow.sum_logdets

    @property
    def symbolic_random(self):
        return self.flow.forward

    @node_property
    def bdim(self):
        if not self.local:
            return super().bdim
        else:
            return next(iter(self.user_params[0].values())).shape[0]

    @classmethod
    def get_param_spec_for(cls, flow, **kwargs):
        return flows.Formula(flow).get_param_spec_for(**kwargs)


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
    trace: class:`pymc3.backends.base.MultiTrace`
        Samples drawn from variational posterior.
    """
    return approx.sample(draws=draws, include_transformed=include_transformed)


# single group shortcuts exported to user
class SingleGroupApproximation(Approximation):
    """Base class for Single Group Approximation"""

    _group_class = None

    def __init__(self, *args, **kwargs):
        local_rv = kwargs.get("local_rv")
        groups = [self._group_class(None, *args, **kwargs)]
        if local_rv is not None:
            groups.extend(
                [
                    Group([v], params=p, local=True, model=kwargs.get("model"))
                    for v, p in local_rv.items()
                ]
            )
        super().__init__(groups, model=kwargs.get("model"))

    def __getattr__(self, item):
        return getattr(self.groups[0], item)

    def __dir__(self):
        d = set(super().__dir__())
        d.update(self.groups[0].__dir__())
        return list(sorted(d))


class MeanField(SingleGroupApproximation):
    __doc__ = """**Single Group Mean Field Approximation**

    """ + str(
        MeanFieldGroup.__doc__
    )
    _group_class = MeanFieldGroup


class FullRank(SingleGroupApproximation):
    __doc__ = """**Single Group Full Rank Approximation**

    """ + str(
        FullRankGroup.__doc__
    )
    _group_class = FullRankGroup


class Empirical(SingleGroupApproximation):
    __doc__ = """**Single Group Full Rank Approximation**

    """ + str(
        EmpiricalGroup.__doc__
    )
    _group_class = EmpiricalGroup

    def __init__(self, trace=None, size=None, **kwargs):
        if kwargs.get("local_rv", None) is not None:
            raise opvi.LocalGroupError("Empirical approximation does not support local variables")
        super().__init__(trace=trace, size=size, **kwargs)

    def evaluate_over_trace(self, node):
        R"""
        This allows to statically evaluate any symbolic expression over the trace.

        Parameters
        ----------
        node: Theano Variables (or Theano expressions)

        Returns
        -------
        evaluated node(s) over the posterior trace contained in the empirical approximation
        """
        node = self.to_flat_input(node)

        def sample(post):
            return theano.clone(node, {self.input: post})

        nodes, _ = theano.scan(sample, self.histogram)
        return nodes


class NormalizingFlow(SingleGroupApproximation):
    __doc__ = """**Single Group Normalizing Flow Approximation**

    """ + str(
        NormalizingFlowGroup.__doc__
    )
    _group_class = NormalizingFlowGroup

    def __init__(self, flow=NormalizingFlowGroup.default_flow, *args, **kwargs):
        kwargs["flow"] = flow
        super().__init__(*args, **kwargs)
