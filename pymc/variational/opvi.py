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

R"""
Variational inference is a great approach for doing really complex,
often intractable Bayesian inference in approximate form. Common methods
(e.g. ADVI) lack from complexity so that approximate posterior does not
reveal the true nature of underlying problem. In some applications it can
yield unreliable decisions.

Recently on NIPS 2017 `OPVI  <https://arxiv.org/abs/1610.09033/>`_ framework
was presented. It generalizes variational inference so that the problem is
build with blocks. The first and essential block is Model itself. Second is
Approximation, in some cases :math:`log Q(D)` is not really needed. Necessity
depends on the third and fourth part of that black box, Operator and
Test Function respectively.

Operator is like an approach we use, it constructs loss from given Model,
Approximation and Test Function. The last one is not needed if we minimize
KL Divergence from Q to posterior. As a drawback we need to compute :math:`loq Q(D)`.
Sometimes approximation family is intractable and :math:`loq Q(D)` is not available,
here comes LS(Langevin Stein) Operator with a set of test functions.

Test Function has more unintuitive meaning. It is usually used with LS operator
and represents all we want from our approximate distribution. For any given vector
based function of :math:`z` LS operator yields zero mean function under posterior.
:math:`loq Q(D)` is no more needed. That opens a door to rich approximation
families as neural networks.

References
----------
-   Rajesh Ranganath, Jaan Altosaar, Dustin Tran, David M. Blei
    Operator Variational Inference
    https://arxiv.org/abs/1610.09033 (2016)
"""

import collections
import itertools
import warnings

import aesara
import aesara.tensor as at
import numpy as np

from aesara.graph.basic import Variable

import pymc as pm

from pymc.aesaraf import at_rng, compile_pymc, identity, rvs_to_value_vars
from pymc.backends import NDArray
from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pymc.model import modelcontext
from pymc.util import WithMemoization, locally_cachedmethod
from pymc.variational.updates import adagrad_window
from pymc.vartypes import discrete_types

__all__ = ["ObjectiveFunction", "Operator", "TestFunction", "Group", "Approximation"]


class VariationalInferenceError(Exception):
    """Exception for VI specific cases"""


class NotImplementedInference(VariationalInferenceError, NotImplementedError):
    """Marking non functional parts of code"""


class ExplicitInferenceError(VariationalInferenceError, TypeError):
    """Exception for bad explicit inference"""


class AEVBInferenceError(VariationalInferenceError, TypeError):
    """Exception for bad aevb inference"""


class ParametrizationError(VariationalInferenceError, ValueError):
    """Error raised in case of bad parametrization"""


class GroupError(VariationalInferenceError, TypeError):
    """Error related to VI groups"""


class BatchedGroupError(GroupError):
    """Error with batched variables"""


class LocalGroupError(BatchedGroupError, AEVBInferenceError):
    """Error raised in case of bad local_rv usage"""


def append_name(name):
    def wrap(f):
        if name is None:
            return f

        def inner(*args, **kwargs):
            res = f(*args, **kwargs)
            res.name = name
            return res

        return inner

    return wrap


def node_property(f):
    """A shortcut for wrapping method to accessible tensor"""

    if isinstance(f, str):

        def wrapper(fn):
            ff = append_name(f)(fn)
            f_ = aesara.config.change_flags(compute_test_value="off")(ff)
            return property(locally_cachedmethod(f_))

        return wrapper
    else:
        f_ = aesara.config.change_flags(compute_test_value="off")(f)
        return property(locally_cachedmethod(f_))


@aesara.config.change_flags(compute_test_value="ignore")
def try_to_set_test_value(node_in, node_out, s):
    _s = s
    if s is None:
        s = 1
    s = aesara.compile.view_op(at.as_tensor(s))
    if not isinstance(node_in, (list, tuple)):
        node_in = [node_in]
    if not isinstance(node_out, (list, tuple)):
        node_out = [node_out]
    for i, o in zip(node_in, node_out):
        if hasattr(i.tag, "test_value"):
            if not hasattr(s.tag, "test_value"):
                continue
            else:
                tv = i.tag.test_value[None, ...]
                tv = np.repeat(tv, s.tag.test_value, 0)
                if _s is None:
                    tv = tv[0]
                o.tag.test_value = tv


class ObjectiveUpdates(aesara.OrderedUpdates):
    """OrderedUpdates extension for storing loss"""

    loss = None


def _warn_not_used(smth, where):
    warnings.warn(f"`{smth}` is not used for {where} and ignored")


class ObjectiveFunction:
    """Helper class for construction loss and updates for variational inference

    Parameters
    ----------
    op : :class:`Operator`
        OPVI Functional operator
    tf : :class:`TestFunction`
        OPVI TestFunction
    """

    def __init__(self, op, tf):
        self.op = op
        self.tf = tf

    obj_params = property(lambda self: self.op.approx.params)
    test_params = property(lambda self: self.tf.params)
    approx = property(lambda self: self.op.approx)

    def updates(
        self,
        obj_n_mc=None,
        tf_n_mc=None,
        obj_optimizer=adagrad_window,
        test_optimizer=adagrad_window,
        more_obj_params=None,
        more_tf_params=None,
        more_updates=None,
        more_replacements=None,
        total_grad_norm_constraint=None,
    ):
        """Calculate gradients for objective function, test function and then
        constructs updates for optimization step

        Parameters
        ----------
        obj_n_mc : int
            Number of monte carlo samples used for approximation of objective gradients
        tf_n_mc : int
            Number of monte carlo samples used for approximation of test function gradients
        obj_optimizer : function (loss, params) -> updates
            Optimizer that is used for objective params
        test_optimizer : function (loss, params) -> updates
            Optimizer that is used for test function params
        more_obj_params : list
            Add custom params for objective optimizer
        more_tf_params : list
            Add custom params for test function optimizer
        more_updates : dict
            Add custom updates to resulting updates
        more_replacements : dict
            Apply custom replacements before calculating gradients
        total_grad_norm_constraint : float
            Bounds gradient norm, prevents exploding gradient problem

        Returns
        -------
        :class:`ObjectiveUpdates`
        """
        if more_updates is None:
            more_updates = dict()
        resulting_updates = ObjectiveUpdates()
        if self.test_params:
            self.add_test_updates(
                resulting_updates,
                tf_n_mc=tf_n_mc,
                test_optimizer=test_optimizer,
                more_tf_params=more_tf_params,
                more_replacements=more_replacements,
                total_grad_norm_constraint=total_grad_norm_constraint,
            )
        else:
            if tf_n_mc is not None:
                _warn_not_used("tf_n_mc", self.op)
            if more_tf_params:
                _warn_not_used("more_tf_params", self.op)
        self.add_obj_updates(
            resulting_updates,
            obj_n_mc=obj_n_mc,
            obj_optimizer=obj_optimizer,
            more_obj_params=more_obj_params,
            more_replacements=more_replacements,
            total_grad_norm_constraint=total_grad_norm_constraint,
        )
        resulting_updates.update(more_updates)
        return resulting_updates

    def add_test_updates(
        self,
        updates,
        tf_n_mc=None,
        test_optimizer=adagrad_window,
        more_tf_params=None,
        more_replacements=None,
        total_grad_norm_constraint=None,
    ):
        if more_tf_params is None:
            more_tf_params = []
        if more_replacements is None:
            more_replacements = dict()
        tf_target = self(
            tf_n_mc, more_tf_params=more_tf_params, more_replacements=more_replacements
        )
        grads = pm.updates.get_or_compute_grads(tf_target, self.obj_params + more_tf_params)
        if total_grad_norm_constraint is not None:
            grads = pm.total_norm_constraint(grads, total_grad_norm_constraint)
        updates.update(test_optimizer(grads, self.test_params + more_tf_params))

    def add_obj_updates(
        self,
        updates,
        obj_n_mc=None,
        obj_optimizer=adagrad_window,
        more_obj_params=None,
        more_replacements=None,
        total_grad_norm_constraint=None,
    ):
        if more_obj_params is None:
            more_obj_params = []
        if more_replacements is None:
            more_replacements = dict()
        obj_target = self(
            obj_n_mc, more_obj_params=more_obj_params, more_replacements=more_replacements
        )
        grads = pm.updates.get_or_compute_grads(obj_target, self.obj_params + more_obj_params)
        if total_grad_norm_constraint is not None:
            grads = pm.total_norm_constraint(grads, total_grad_norm_constraint)
        updates.update(obj_optimizer(grads, self.obj_params + more_obj_params))
        if self.op.returns_loss:
            updates.loss = obj_target

    @aesara.config.change_flags(compute_test_value="off")
    def step_function(
        self,
        obj_n_mc=None,
        tf_n_mc=None,
        obj_optimizer=adagrad_window,
        test_optimizer=adagrad_window,
        more_obj_params=None,
        more_tf_params=None,
        more_updates=None,
        more_replacements=None,
        total_grad_norm_constraint=None,
        score=False,
        fn_kwargs=None,
    ):
        R"""Step function that should be called on each optimization step.

        Generally it solves the following problem:

        .. math::

                \mathbf{\lambda^{\*}} = \inf_{\lambda} \sup_{\theta} t(\mathbb{E}_{\lambda}[(O^{p,q}f_{\theta})(z)])

        Parameters
        ----------
        obj_n_mc: `int`
            Number of monte carlo samples used for approximation of objective gradients
        tf_n_mc: `int`
            Number of monte carlo samples used for approximation of test function gradients
        obj_optimizer: function (grads, params) -> updates
            Optimizer that is used for objective params
        test_optimizer: function (grads, params) -> updates
            Optimizer that is used for test function params
        more_obj_params: `list`
            Add custom params for objective optimizer
        more_tf_params: `list`
            Add custom params for test function optimizer
        more_updates: `dict`
            Add custom updates to resulting updates
        total_grad_norm_constraint: `float`
            Bounds gradient norm, prevents exploding gradient problem
        score: `bool`
            calculate loss on each step? Defaults to False for speed
        fn_kwargs: `dict`
            Add kwargs to aesara.function (e.g. `{'profile': True}`)
        more_replacements: `dict`
            Apply custom replacements before calculating gradients

        Returns
        -------
        `aesara.function`
        """
        if fn_kwargs is None:
            fn_kwargs = {}
        if score and not self.op.returns_loss:
            raise NotImplementedError("%s does not have loss" % self.op)
        updates = self.updates(
            obj_n_mc=obj_n_mc,
            tf_n_mc=tf_n_mc,
            obj_optimizer=obj_optimizer,
            test_optimizer=test_optimizer,
            more_obj_params=more_obj_params,
            more_tf_params=more_tf_params,
            more_updates=more_updates,
            more_replacements=more_replacements,
            total_grad_norm_constraint=total_grad_norm_constraint,
        )
        if score:
            step_fn = compile_pymc([], updates.loss, updates=updates, **fn_kwargs)
        else:
            step_fn = compile_pymc([], [], updates=updates, **fn_kwargs)
        return step_fn

    @aesara.config.change_flags(compute_test_value="off")
    def score_function(
        self, sc_n_mc=None, more_replacements=None, fn_kwargs=None
    ):  # pragma: no cover
        R"""Compile scoring function that operates which takes no inputs and returns Loss

        Parameters
        ----------
        sc_n_mc: `int`
            number of scoring MC samples
        more_replacements:
            Apply custom replacements before compiling a function
        fn_kwargs: `dict`
            arbitrary kwargs passed to `aesara.function`

        Returns
        -------
        aesara.function
        """
        if fn_kwargs is None:
            fn_kwargs = {}
        if not self.op.returns_loss:
            raise NotImplementedError("%s does not have loss" % self.op)
        if more_replacements is None:
            more_replacements = {}
        loss = self(sc_n_mc, more_replacements=more_replacements)
        return compile_pymc([], loss, **fn_kwargs)

    @aesara.config.change_flags(compute_test_value="off")
    def __call__(self, nmc, **kwargs):
        if "more_tf_params" in kwargs:
            m = -1.0
        else:
            m = 1.0
        a = self.op.apply(self.tf)
        a = self.approx.set_size_and_deterministic(a, nmc, 0, kwargs.get("more_replacements"))
        return m * self.op.T(a)


class Operator:
    R"""**Base class for Operator**

    Parameters
    ----------
    approx: :class:`Approximation`
        an approximation instance

    Notes
    -----
    For implementing custom operator it is needed to define :func:`Operator.apply` method
    """

    has_test_function = False
    returns_loss = True
    require_logq = True
    objective_class = ObjectiveFunction
    supports_aevb = property(lambda self: not self.approx.any_histograms)
    T = identity

    def __init__(self, approx):
        self.approx = approx
        if not self.supports_aevb and approx.has_local:
            raise AEVBInferenceError(
                "%s does not support AEVB, " "please change inference method" % self
            )
        if self.require_logq and not approx.has_logq:
            raise ExplicitInferenceError(
                "%s requires logq, but %s does not implement it"
                "please change inference method" % (self, approx)
            )

    inputs = property(lambda self: self.approx.inputs)
    logp = property(lambda self: self.approx.logp)
    varlogp = property(lambda self: self.approx.varlogp)
    datalogp = property(lambda self: self.approx.datalogp)
    logq = property(lambda self: self.approx.logq)
    logp_norm = property(lambda self: self.approx.logp_norm)
    varlogp_norm = property(lambda self: self.approx.varlogp_norm)
    datalogp_norm = property(lambda self: self.approx.datalogp_norm)
    logq_norm = property(lambda self: self.approx.logq_norm)
    model = property(lambda self: self.approx.model)

    def apply(self, f):  # pragma: no cover
        R"""Operator itself

        .. math::

            (O^{p,q}f_{\theta})(z)

        Parameters
        ----------
        f: :class:`TestFunction` or None
            function that takes `z = self.input` and returns
            same dimensional output

        Returns
        -------
        TensorVariable
            symbolically applied operator
        """
        raise NotImplementedError

    def __call__(self, f=None):
        if self.has_test_function:
            if f is None:
                raise ParametrizationError("Operator %s requires TestFunction" % self)
            else:
                if not isinstance(f, TestFunction):
                    f = TestFunction.from_function(f)
        else:
            if f is not None:
                warnings.warn("TestFunction for %s is redundant and removed" % self, stacklevel=3)
            else:
                pass
            f = TestFunction()
        f.setup(self.approx)
        return self.objective_class(self, f)

    def __str__(self):  # pragma: no cover
        return "%(op)s[%(ap)s]" % dict(
            op=self.__class__.__name__, ap=self.approx.__class__.__name__
        )


def collect_shared_to_list(params):
    """Helper function for getting a list from
    usable representation of parameters

    Parameters
    ----------
    params: {dict|None}

    Returns
    -------
    List
    """
    if isinstance(params, dict):
        return list(
            t[1]
            for t in sorted(params.items(), key=lambda t: t[0])
            if isinstance(t[1], aesara.compile.SharedVariable)
        )
    elif params is None:
        return []
    else:
        raise TypeError("Unknown type %s for %r, need dict or None")


class TestFunction:
    def __init__(self):
        self._inited = False
        self.shared_params = None

    @property
    def params(self):
        return collect_shared_to_list(self.shared_params)

    def __call__(self, z):
        raise NotImplementedError

    def setup(self, approx):
        pass

    @classmethod
    def from_function(cls, f):
        if not callable(f):
            raise ParametrizationError("Need callable, got %r" % f)
        obj = TestFunction()
        obj.__call__ = f
        return obj


class Group(WithMemoization):
    R"""**Base class for grouping variables in VI**

    Grouped Approximation is used for modelling mutual dependencies
    for a specified group of variables. Base for local and global group.

    Parameters
    ----------
    group: list
        List of PyMC variables or None indicating that group takes all the rest variables
    vfam: str
        String that marks the corresponding variational family for the group.
        Cannot be passed both with `params`
    params: dict
        Dict with variational family parameters, full description can be found below.
        Cannot be passed both with `vfam`
    random_seed: int
        Random seed for underlying random generator
    model :
        PyMC Model
    local: bool
        Indicates whether this group is local. Cannot be passed without `params`.
        Such group should have only one variable
    rowwise: bool
        Indicates whether this group is independently parametrized over first dim.
        Such group should have only one variable
    options: dict
        Special options for the group
    kwargs: Other kwargs for the group

    Notes
    -----
    Group instance/class has some important constants:

    -   **supports_batched**
        Determines whether such variational family can be used for AEVB or rowwise approx.

        AEVB approx is such approx that somehow depends on input data. It can be treated
        as conditional distribution. You can see more about in the corresponding paper
        mentioned in references.

        Rowwise mode is a special case approximation that treats every 'row', of a tensor as
        independent from each other. Some distributions can't do that by
        definition e.g. :class:`Empirical` that consists of particles only.

    -   **has_logq**
        Tells that distribution is defined explicitly

    These constants help providing the correct inference method for given parametrization

    Examples
    --------

    **Basic Initialization**

    :class:`Group` is a factory class. You do not need to call every ApproximationGroup explicitly.
    Passing the correct `vfam` (Variational FAMily) argument you'll tell what
    parametrization is desired for the group. This helps not to overload code with lots of classes.

    .. code:: python

        >>> group = Group([latent1, latent2], vfam='mean_field')

    The other way to select approximation is to provide `params` dictionary that has some
    predefined well shaped parameters. Keys of the dict serve as an identifier for variational family and help
    to autoselect the correct group class. To identify what approximation to use, params dict should
    have the full set of needed parameters. As there are 2 ways to instantiate the :class:`Group`
    passing both `vfam` and `params` is prohibited. Partial parametrization is prohibited by design to
    avoid corner cases and possible problems.

    .. code:: python

        >>> group = Group([latent3], params=dict(mu=my_mu, rho=my_rho))

    Important to note that in case you pass custom params they will not be autocollected by optimizer, you'll
    have to provide them with `more_obj_params` keyword.

    **Supported dict keys:**

    -   `{'mu', 'rho'}`: :class:`MeanFieldGroup`

    -   `{'mu', 'L_tril'}`: :class:`FullRankGroup`

    -   `{'histogram'}`: :class:`EmpiricalGroup`

    -   `{0, 1, 2, 3, ..., k-1}`: :class:`NormalizingFlowGroup` of depth `k`

        NormalizingFlows have other parameters than ordinary groups and should be
        passed as nested dicts with the following keys:

        -   `{'u', 'w', 'b'}`: :class:`PlanarFlow`

        -   `{'a', 'b', 'z_ref'}`: :class:`RadialFlow`

        -   `{'loc'}`: :class:`LocFlow`

        -   `{'rho'}`: :class:`ScaleFlow`

        -   `{'v'}`: :class:`HouseholderFlow`

        Note that all integer keys should be present in the dictionary. An example
        of NormalizingFlow initialization can be found below.

    **Using AEVB**

    Autoencoding variational Bayes is a powerful tool to get conditional :math:`q(\lambda|X)` distribution
    on latent variables. It is well supported by PyMC and all you need is to provide a dictionary
    with well shaped variational parameters, the correct approximation will be autoselected as mentioned
    in section above. However we have some implementation restrictions in AEVB. They require autoencoded
    variable to have first dimension as *batch* dimension and other dimensions should stay fixed.
    With this assumptions it is possible to generalize all variational approximation families as
    batched approximations that have flexible parameters and leading axis.

    Only single variable local group is supported. Params are required.

    >>> # for mean field
    >>> group = Group([latent3], params=dict(mu=my_mu, rho=my_rho), local=True)
    >>> # or for full rank
    >>> group = Group([latent3], params=dict(mu=my_mu, L_tril=my_L_tril), local=True)

    -   An Approximation class is selected automatically based on the keys in dict.

    -   `my_mu` and `my_rho` are usually estimated with neural network or function approximator.

    **Using Row-Wise Group**

    Batch groups have independent row wise approximations, thus using batched
    mean field will give no effect. It is more interesting if you want each row of a matrix
    to be parametrized independently with normalizing flow or full rank gaussian.

    To tell :class:`Group` that group is batched you need set `batched` kwarg as `True`.
    Only single variable group is allowed due to implementation details.

    >>> group = Group([latent3], vfam='fr', rowwise=True) # 'fr' is alias for 'full_rank'

    The resulting approximation for this variable will have the following structure

    .. math::

        latent3_{i, \dots} \sim \mathcal{N}(\mu_i, \Sigma_i) \forall i

    **Note**: Using rowwise and user-parametrized approximation is ok, but
    shape should be checked beforehand, it is impossible to infer it by PyMC

    **Normalizing Flow Group**

    In case you use simple initialization pattern using `vfam` you'll not meet any changes.
    Passing flow formula to `vfam` you'll get correct flow parametrization for group

    .. code:: python

        >>> group = Group([latent3], vfam='scale-hh*5-radial*4-loc')

    **Note**: Consider passing location flow as the last one and scale as the first one for stable inference.

    Rowwise normalizing flow is supported as well

    .. code:: python

        >>> group = Group([latent3], vfam='scale-hh*2-radial-loc', rowwise=True)

    Custom parameters for normalizing flow can be a real trouble for the first time.
    They have quite different format from the rest variational families.


    .. code:: python

        >>> # int is used as key, it also tells the flow position
        ... flow_params = {
        ...     # `rho` parametrizes scale flow, softplus is used to map (-inf; inf) -> (0, inf)
        ...     0: dict(rho=my_scale),
        ...     1: dict(v=my_v1),  # Householder Flow, `v` is parameter name from the original paper
        ...     2: dict(v=my_v2),  # do not miss any number in dict, or else error is raised
        ...     3: dict(a=my_a, b=my_b, z_ref=my_z_ref),  # Radial flow
        ...     4: dict(loc=my_loc)  # Location Flow
        ... }
        ... group = Group([latent3], params=flow_params)
        ... # local=True can be added in case you do AEVB inference
        ... group = Group([latent3], params=flow_params, local=True)

    **Delayed Initialization**

    When you have a lot of latent variables it is impractical to do it all manually.
    To make life much simpler, You can pass `None` instead of list of variables. That case
    you'll not create shared parameters until you pass all collected groups to
    Approximation object that collects all the groups together and checks that every group is
    correctly initialized. For those groups which have group equal to `None` it will collect all
    the rest variables not covered by other groups and perform delayed init.

    .. code:: python

        >>> group_1 = Group([latent1], vfam='fr')  # latent1 has full rank approximation
        >>> group_other = Group(None, vfam='mf')  # other variables have mean field Q
        >>> approx = Approximation([group_1, group_other])

    **Summing Up**

    When you have created all the groups they need to pass all the groups to :class:`Approximation`.
    It does not accept any other parameter rather than `groups`

    .. code:: python

        >>> approx = Approximation(my_groups)

    See Also
    --------
    :class:`Approximation`

    References
    ----------
    -   Kingma, D. P., & Welling, M. (2014).
        `Auto-Encoding Variational Bayes. stat, 1050, 1. <https://arxiv.org/abs/1312.6114>`_
    """
    # needs to be defined in init
    shared_params = None
    symbolic_initial = None
    replacements = None
    input = None

    # defined by approximation
    supports_batched = True
    has_logq = True

    # some important defaults
    initial_dist_name = "normal"
    initial_dist_map = 0.0

    # for handy access using class methods
    __param_spec__ = dict()
    short_name = ""
    alias_names = frozenset()
    __param_registry = dict()
    __name_registry = dict()

    @classmethod
    def register(cls, sbcls):
        assert (
            frozenset(sbcls.__param_spec__) not in cls.__param_registry
        ), "Duplicate __param_spec__"
        cls.__param_registry[frozenset(sbcls.__param_spec__)] = sbcls
        assert sbcls.short_name not in cls.__name_registry, "Duplicate short_name"
        cls.__name_registry[sbcls.short_name] = sbcls
        for alias in sbcls.alias_names:
            assert alias not in cls.__name_registry, "Duplicate alias_name"
            cls.__name_registry[alias] = sbcls
        return sbcls

    @classmethod
    def group_for_params(cls, params):
        if pm.variational.flows.seems_like_flow_params(params):
            return pm.variational.approximations.NormalizingFlowGroup
        if frozenset(params) not in cls.__param_registry:
            raise KeyError(
                "No such group for the following params: {!r}, "
                "only the following are supported\n\n{}".format(params, cls.__param_registry)
            )
        return cls.__param_registry[frozenset(params)]

    @classmethod
    def group_for_short_name(cls, name):
        if pm.variational.flows.seems_like_formula(name):
            return pm.variational.approximations.NormalizingFlowGroup
        if name.lower() not in cls.__name_registry:
            raise KeyError(
                "No such group: {!r}, "
                "only the following are supported\n\n{}".format(name, cls.__name_registry)
            )
        return cls.__name_registry[name.lower()]

    def __new__(cls, group=None, vfam=None, params=None, *args, **kwargs):
        if cls is Group:
            if vfam is not None and params is not None:
                raise TypeError("Cannot call Group with both `vfam` and `params` provided")
            elif vfam is not None:
                return super().__new__(cls.group_for_short_name(vfam))
            elif params is not None:
                return super().__new__(cls.group_for_params(params))
            else:
                raise TypeError("Need to call Group with either `vfam` or `params` provided")
        else:
            return super().__new__(cls)

    def __init__(
        self,
        group,
        vfam=None,
        params=None,
        random_seed=None,
        model=None,
        local=False,
        rowwise=False,
        options=None,
        **kwargs,
    ):
        if local and not self.supports_batched:
            raise LocalGroupError("%s does not support local groups" % self.__class__)
        if local and rowwise:
            raise LocalGroupError("%s does not support local grouping in rowwise mode")
        if isinstance(vfam, str):
            vfam = vfam.lower()
        if options is None:
            options = dict()
        self.options = options
        self._vfam = vfam
        self._local = local
        self._batched = rowwise
        self._rng = at_rng(random_seed)
        model = modelcontext(model)
        self.model = model
        self.group = group
        self.user_params = params
        self._user_params = None
        self.replacements = collections.OrderedDict()
        self.ordering = collections.OrderedDict()
        # save this stuff to use in __init_group__ later
        self._kwargs = kwargs
        if self.group is not None:
            # init can be delayed
            self.__init_group__(self.group)

    def _prepare_start(self, start=None):
        ipfn = make_initial_point_fn(
            model=self.model,
            overrides=start,
            jitter_rvs={},
            return_transformed=True,
        )
        start = ipfn(self.model.rng_seeder.randint(2**30, dtype=np.int64))
        group_vars = {self.model.rvs_to_values[v].name for v in self.group}
        start = {k: v for k, v in start.items() if k in group_vars}
        if self.batched:
            start = start[self.group[0].name][0]
        else:
            start = DictToArrayBijection.map(start).data
        return start

    @classmethod
    def get_param_spec_for(cls, **kwargs):
        res = dict()
        for name, fshape in cls.__param_spec__.items():
            res[name] = tuple(eval(s, kwargs) for s in fshape)
        return res

    def _check_user_params(self, **kwargs):
        R"""*Dev* - checks user params, allocates them if they are correct, returns True.
        If they are not present, returns False

        Parameters
        ----------
        kwargs: special kwargs needed sometimes

        Returns
        -------
        bool indicating whether to allocate new shared params
        """
        user_params = self.user_params
        if user_params is None:
            return False
        if not isinstance(user_params, dict):
            raise TypeError("params should be a dict")
        givens = set(user_params.keys())
        needed = set(self.__param_spec__)
        if givens != needed:
            raise ParametrizationError(
                "Passed parameters do not have a needed set of keys, "
                "they should be equal, got {givens}, needed {needed}".format(
                    givens=givens, needed=needed
                )
            )
        self._user_params = dict()
        spec = self.get_param_spec_for(d=self.ddim, **kwargs.pop("spec_kw", {}))
        for name, param in self.user_params.items():
            shape = spec[name]
            if self.local:
                shape = (-1,) + shape
            elif self.batched:
                shape = (self.bdim,) + shape
            self._user_params[name] = at.as_tensor(param).reshape(shape)
        return True

    def _initial_type(self, name):
        R"""*Dev* - initial type with given name. The correct type depends on `self.batched`

        Parameters
        ----------
        name: str
            name for tensor
        Returns
        -------
        tensor
        """
        if self.batched:
            return at.tensor3(name)
        else:
            return at.matrix(name)

    def _input_type(self, name):
        R"""*Dev* - input type with given name. The correct type depends on `self.batched`

        Parameters
        ----------
        name: str
            name for tensor
        Returns
        -------
        tensor
        """
        if self.batched:
            return at.matrix(name)
        else:
            return at.vector(name)

    @aesara.config.change_flags(compute_test_value="off")
    def __init_group__(self, group):
        if not group:
            raise GroupError("Got empty group")
        if self.local:
            raise NotImplementedInference("Local inferene aka AEVB is not supported in v4")
        if self.batched:
            raise NotImplementedInference("Batched inferene is not supported in v4")
        if self.group is None:
            # delayed init
            self.group = group
        if self.batched and len(group) > 1:
            if self.local:  # better error message
                raise LocalGroupError("Local groups with more than 1 variable are not supported")
            else:
                raise BatchedGroupError(
                    "Batched groups with more than 1 variable are not supported"
                )
        self.symbolic_initial = self._initial_type(
            self.__class__.__name__ + "_symbolic_initial_tensor"
        )
        self.input = self._input_type(self.__class__.__name__ + "_symbolic_input")
        # I do some staff that is not supported by standard __init__
        # so I have to to it by myself

        # 1) we need initial point (transformed space)
        model_initial_point = self.model.compute_initial_point(0)
        # 2) we'll work with a single group, a subset of the model
        # here we need to create a mapping to replace value_vars with slices from the approximation
        start_idx = 0
        for var in self.group:
            if var.type.numpy_dtype.name in discrete_types:
                raise ParametrizationError(f"Discrete variables are not supported by VI: {var}")
            # 3) This is the way to infer shape and dtype of the variable
            value_var = self.model.rvs_to_values[var]
            test_var = model_initial_point[value_var.name]
            if self.batched:
                # Leave a more complicated case for future work
                if var.ndim < 1:
                    if self.local:
                        raise LocalGroupError("Local variable should not be scalar")
                    else:
                        raise BatchedGroupError("Batched variable should not be scalar")
                size = test_var[0].size
                if self.local:
                    shape = (-1,) + test_var.shape[1:]
                else:
                    shape = test_var.shape
            else:
                shape = test_var.shape
                size = test_var.size
            dtype = test_var.dtype
            vr = self.input[..., start_idx : start_idx + size].reshape(shape).astype(dtype)
            vr.name = value_var.name + "_vi_replacement"
            self.replacements[value_var] = vr
            self.ordering[value_var.name] = (
                value_var.name,
                slice(start_idx, start_idx + size),
                shape,
                dtype,
            )
            start_idx += size

    def _finalize_init(self):
        """*Dev* - clean up after init"""
        del self._kwargs

    local = property(lambda self: self._local)
    batched = property(lambda self: self._local or self._batched)

    @property
    def params_dict(self):
        # prefixed are correctly reshaped
        if self._user_params is not None:
            return self._user_params
        else:
            return self.shared_params

    @property
    def params(self):
        # raw user params possibly not reshaped
        if self.user_params is not None:
            return collect_shared_to_list(self.user_params)
        else:
            return collect_shared_to_list(self.shared_params)

    def _new_initial_shape(self, size, dim, more_replacements=None):
        """*Dev* - correctly proceeds sampling with variable batch size

        Parameters
        ----------
        size: scalar
            sample size
        dim: scalar
            latent fixed dim
        more_replacements: dict
            replacements for latent batch shape

        Returns
        -------
        shape vector
        """
        if self.batched:
            bdim = at.as_tensor(self.bdim)
            bdim = aesara.clone_replace(bdim, more_replacements)
            return at.stack([size, bdim, dim])
        else:
            return at.stack([size, dim])

    @node_property
    def bdim(self):
        if not self.local:
            if self.batched:
                return next(iter(self.ordering.values()))[2][0]
            else:
                return 1
        else:
            return next(iter(self.params_dict.values())).shape[0]

    @node_property
    def ndim(self):
        if self.batched:
            return self.ordering.size * self.bdim
        else:
            return self.ddim

    @property
    def ddim(self):
        return sum(s.stop - s.start for _, s, _, _ in self.ordering.values())

    def _new_initial(self, size, deterministic, more_replacements=None):
        """*Dev* - allocates new initial random generator

        Parameters
        ----------
        size: scalar
            sample size
        deterministic: bool or scalar
            whether to sample in deterministic manner
        more_replacements: dict
            more replacements passed to shape

        Notes
        -----
        Suppose you have a AEVB setup that:

            -   input `X` is purely symbolic, and `X.shape[0]` is needed to `initial` second dim
            -   to perform inference, `X` is replaced with data tensor, however, since `X.shape[0]` in `initial`
                remains symbolic and can't be replaced, you get `MissingInputError`
            -   as a solution, here we perform a manual replacement for the second dim in `initial`.

        Returns
        -------
        tensor
        """
        if size is None:
            size = 1
        if not isinstance(deterministic, Variable):
            deterministic = np.int8(deterministic)
        dim, dist_name, dist_map = (self.ddim, self.initial_dist_name, self.initial_dist_map)
        dtype = self.symbolic_initial.dtype
        dim = at.as_tensor(dim)
        size = at.as_tensor(size)
        shape = self._new_initial_shape(size, dim, more_replacements)
        # apply optimizations if possible
        if not isinstance(deterministic, Variable):
            if deterministic:
                return at.ones(shape, dtype) * dist_map
            else:
                return getattr(self._rng, dist_name)(size=shape)
        else:
            sample = getattr(self._rng, dist_name)(size=shape)
            initial = at.switch(deterministic, at.ones(shape, dtype) * dist_map, sample)
            return initial

    @node_property
    def symbolic_random(self):
        """*Dev* - abstract node that takes `self.symbolic_initial` and creates
        approximate posterior that is parametrized with `self.params_dict`.

        Implementation should take in account `self.batched`. If `self.batched` is `True`, then
        `self.symbolic_initial` is 3d tensor, else 2d

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @node_property
    def symbolic_random2d(self):
        """*Dev* - `self.symbolic_random` flattened to matrix"""
        if self.batched:
            return self.symbolic_random.flatten(2)
        else:
            return self.symbolic_random

    @aesara.config.change_flags(compute_test_value="off")
    def set_size_and_deterministic(self, node, s, d, more_replacements=None):
        """*Dev* - after node is sampled via :func:`symbolic_sample_over_posterior` or
        :func:`symbolic_single_sample` new random generator can be allocated and applied to node

        Parameters
        ----------
        node: :class:`Variable`
            Aesara node with symbolically applied VI replacements
        s: scalar
            desired number of samples
        d: bool or int
            whether sampling is done deterministically
        more_replacements: dict
            more replacements to apply

        Returns
        -------
        :class:`Variable` with applied replacements, ready to use
        """
        flat2rand = self.make_size_and_deterministic_replacements(s, d, more_replacements)
        node_out = aesara.clone_replace(node, flat2rand)
        try_to_set_test_value(node, node_out, s)
        return node_out

    def to_flat_input(self, node):
        """*Dev* - replace vars with flattened view stored in `self.inputs`"""
        return aesara.clone_replace(node, self.replacements)

    def symbolic_sample_over_posterior(self, node):
        """*Dev* - performs sampling of node applying independent samples from posterior each time.
        Note that it is done symbolically and this node needs :func:`set_size_and_deterministic` call
        """
        node = self.to_flat_input(node)
        random = self.symbolic_random.astype(self.symbolic_initial.dtype)
        random = at.patternbroadcast(random, self.symbolic_initial.broadcastable)

        def sample(post, node):
            return aesara.clone_replace(node, {self.input: post})

        nodes, _ = aesara.scan(sample, random, non_sequences=[node])
        return nodes

    def symbolic_single_sample(self, node):
        """*Dev* - performs sampling of node applying single sample from posterior.
        Note that it is done symbolically and this node needs
        :func:`set_size_and_deterministic` call with `size=1`
        """
        node = self.to_flat_input(node)
        random = self.symbolic_random.astype(self.symbolic_initial.dtype)
        return aesara.clone_replace(node, {self.input: random[0]})

    def make_size_and_deterministic_replacements(self, s, d, more_replacements=None):
        """*Dev* - creates correct replacements for initial depending on
        sample size and deterministic flag

        Parameters
        ----------
        s: scalar
            sample size
        d: bool or scalar
            whether sampling is done deterministically
        more_replacements: dict
            replacements for shape and initial

        Returns
        -------
        dict with replacements for initial
        """
        initial = self._new_initial(s, d, more_replacements)
        initial = at.patternbroadcast(initial, self.symbolic_initial.broadcastable)
        if more_replacements:
            initial = aesara.clone_replace(initial, more_replacements)
        return {self.symbolic_initial: initial}

    @node_property
    def symbolic_normalizing_constant(self):
        """*Dev* - normalizing constant for `self.logq`, scales it to `minibatch_size` instead of `total_size`"""
        t = self.to_flat_input(at.max([v.tag.scaling for v in self.group]))
        t = self.symbolic_single_sample(t)
        return pm.floatX(t)

    @node_property
    def symbolic_logq_not_scaled(self):
        """*Dev* - symbolically computed logq for `self.symbolic_random`
        computations can be more efficient since all is known beforehand including
        `self.symbolic_random`
        """
        raise NotImplementedError  # shape (s,)

    @node_property
    def symbolic_logq(self):
        """*Dev* - correctly scaled `self.symbolic_logq_not_scaled`"""
        if self.local:
            s = self.group[0].tag.scaling
            s = self.to_flat_input(s)
            s = self.symbolic_single_sample(s)
            return self.symbolic_logq_not_scaled * s
        else:
            return self.symbolic_logq_not_scaled

    @node_property
    def logq(self):
        """*Dev* - Monte Carlo estimate for group `logQ`"""
        return self.symbolic_logq.mean(0)

    @node_property
    def logq_norm(self):
        """*Dev* - Monte Carlo estimate for group `logQ` normalized"""
        return self.logq / self.symbolic_normalizing_constant

    def __str__(self):
        if self.group is None:
            shp = "undefined"
        else:
            shp = str(self.ddim)
            if self.local:
                shp = "None, " + shp
            elif self.batched:
                shp = str(self.bdim) + ", " + shp
        return f"{self.__class__.__name__}[{shp}]"

    @node_property
    def std(self):
        raise NotImplementedError

    @node_property
    def cov(self):
        raise NotImplementedError

    @node_property
    def mean(self):
        raise NotImplementedError


group_for_params = Group.group_for_params
group_for_short_name = Group.group_for_short_name


class Approximation(WithMemoization):
    """**Wrapper for grouped approximations**

    Wraps list of groups, creates an Approximation instance that collects
    sampled variables from all the groups, also collects logQ needed for
    explicit Variational Inference.

    Parameters
    ----------
    groups: list[Group]
        List of :class:`Group` instances. They should have all model variables
    model: Model

    Notes
    -----
    Some shortcuts for single group approximations are available:

        -   :class:`MeanField`
        -   :class:`FullRank`
        -   :class:`NormalizingFlow`
        -   :class:`Empirical`

    Single group accepts `local_rv` keyword with dict mapping PyMC variables
    to their local Group parameters dict

    See Also
    --------
    :class:`Group`
    """

    def __init__(self, groups, model=None):
        self._scale_cost_to_minibatch = aesara.shared(np.int8(1))
        model = modelcontext(model)
        if not model.free_RVs:
            raise TypeError("Model does not have an free RVs")
        self.groups = list()
        seen = set()
        rest = None
        for g in groups:
            if g.group is None:
                if rest is not None:
                    raise GroupError("More than one group is specified for " "the rest variables")
                else:
                    rest = g
            else:
                if set(g.group) & seen:
                    raise GroupError("Found duplicates in groups")
                seen.update(g.group)
                self.groups.append(g)
        if set(model.free_RVs) - seen:
            if rest is None:
                raise GroupError("No approximation is specified for the rest variables")
            else:
                rest.__init_group__(list(set(model.free_RVs) - seen))
                self.groups.append(rest)
        self.model = model

    @property
    def has_logq(self):
        return all(self.collect("has_logq"))

    def collect(self, item, part="total"):
        if part == "total":
            return [getattr(g, item) for g in self.groups]
        elif part == "local":
            return [getattr(g, item) for g in self.groups if g.local]
        elif part == "global":
            return [getattr(g, item) for g in self.groups if not g.local]
        elif part == "batched":
            return [getattr(g, item) for g in self.groups if g.batched]
        else:
            raise ValueError("unknown part %s, expected {'local', 'global', 'total', 'batched'}")

    inputs = property(lambda self: self.collect("input"))
    symbolic_randoms = property(lambda self: self.collect("symbolic_random"))

    @property
    def scale_cost_to_minibatch(self):
        """*Dev* - Property to control scaling cost to minibatch"""
        return bool(self._scale_cost_to_minibatch.get_value())

    @scale_cost_to_minibatch.setter
    def scale_cost_to_minibatch(self, value):
        self._scale_cost_to_minibatch.set_value(np.int8(bool(value)))

    @node_property
    def symbolic_normalizing_constant(self):
        """*Dev* - normalizing constant for `self.logq`, scales it to `minibatch_size` instead of `total_size`.
        Here the effect is controlled by `self.scale_cost_to_minibatch`
        """
        t = at.max(
            self.collect("symbolic_normalizing_constant")
            + [var.tag.scaling for var in self.model.observed_RVs]
        )
        t = at.switch(self._scale_cost_to_minibatch, t, at.constant(1, dtype=t.dtype))
        return pm.floatX(t)

    @node_property
    def symbolic_logq(self):
        """*Dev* - collects `symbolic_logq` for all groups"""
        return at.add(*self.collect("symbolic_logq"))

    @node_property
    def logq(self):
        """*Dev* - collects `logQ` for all groups"""
        return at.add(*self.collect("logq"))

    @node_property
    def logq_norm(self):
        """*Dev* - collects `logQ` for all groups and normalizes it"""
        return self.logq / self.symbolic_normalizing_constant

    @node_property
    def _sized_symbolic_varlogp_and_datalogp(self):
        """*Dev* - computes sampled prior term from model via `aesara.scan`"""
        varlogp_s, datalogp_s = self.symbolic_sample_over_posterior(
            [self.model.varlogpt, self.model.datalogpt]
        )
        return varlogp_s, datalogp_s  # both shape (s,)

    @node_property
    def sized_symbolic_varlogp(self):
        """*Dev* - computes sampled prior term from model via `aesara.scan`"""
        return self._sized_symbolic_varlogp_and_datalogp[0]  # shape (s,)

    @node_property
    def sized_symbolic_datalogp(self):
        """*Dev* - computes sampled data term from model via `aesara.scan`"""
        return self._sized_symbolic_varlogp_and_datalogp[1]  # shape (s,)

    @node_property
    def sized_symbolic_logp(self):
        """*Dev* - computes sampled logP from model via `aesara.scan`"""
        return self.sized_symbolic_varlogp + self.sized_symbolic_datalogp  # shape (s,)

    @node_property
    def logp(self):
        """*Dev* - computes :math:`E_{q}(logP)` from model via `aesara.scan` that can be optimized later"""
        return self.varlogp + self.datalogp

    @node_property
    def varlogp(self):
        """*Dev* - computes :math:`E_{q}(prior term)` from model via `aesara.scan` that can be optimized later"""
        return self.sized_symbolic_varlogp.mean(0)

    @node_property
    def datalogp(self):
        """*Dev* - computes :math:`E_{q}(data term)` from model via `aesara.scan` that can be optimized later"""
        return self.sized_symbolic_datalogp.mean(0)

    @node_property
    def _single_symbolic_varlogp_and_datalogp(self):
        """*Dev* - computes sampled prior term from model via `aesara.scan`"""
        varlogp, datalogp = self.symbolic_single_sample([self.model.varlogpt, self.model.datalogpt])
        return varlogp, datalogp

    @node_property
    def single_symbolic_varlogp(self):
        """*Dev* - for single MC sample estimate of :math:`E_{q}(prior term)` `aesara.scan`
        is not needed and code can be optimized"""
        return self._single_symbolic_varlogp_and_datalogp[0]

    @node_property
    def single_symbolic_datalogp(self):
        """*Dev* - for single MC sample estimate of :math:`E_{q}(data term)` `aesara.scan`
        is not needed and code can be optimized"""
        return self._single_symbolic_varlogp_and_datalogp[1]

    @node_property
    def single_symbolic_logp(self):
        """*Dev* - for single MC sample estimate of :math:`E_{q}(logP)` `aesara.scan`
        is not needed and code can be optimized"""
        return self.single_symbolic_datalogp + self.single_symbolic_varlogp

    @node_property
    def logp_norm(self):
        """*Dev* - normalized :math:`E_{q}(logP)`"""
        return self.logp / self.symbolic_normalizing_constant

    @node_property
    def varlogp_norm(self):
        """*Dev* - normalized :math:`E_{q}(prior term)`"""
        return self.varlogp / self.symbolic_normalizing_constant

    @node_property
    def datalogp_norm(self):
        """*Dev* - normalized :math:`E_{q}(data term)`"""
        return self.datalogp / self.symbolic_normalizing_constant

    @property
    def replacements(self):
        """*Dev* - all replacements from groups to replace PyMC random variables with approximation"""
        return collections.OrderedDict(
            itertools.chain.from_iterable(g.replacements.items() for g in self.groups)
        )

    def make_size_and_deterministic_replacements(self, s, d, more_replacements=None):
        """*Dev* - creates correct replacements for initial depending on
        sample size and deterministic flag

        Parameters
        ----------
        s: scalar
            sample size
        d: bool
            whether sampling is done deterministically
        more_replacements: dict
            replacements for shape and initial

        Returns
        -------
        dict with replacements for initial
        """
        if more_replacements is None:
            more_replacements = {}
        flat2rand = collections.OrderedDict()
        for g in self.groups:
            flat2rand.update(g.make_size_and_deterministic_replacements(s, d, more_replacements))
        flat2rand.update(more_replacements)
        return flat2rand

    @aesara.config.change_flags(compute_test_value="off")
    def set_size_and_deterministic(self, node, s, d, more_replacements=None):
        """*Dev* - after node is sampled via :func:`symbolic_sample_over_posterior` or
        :func:`symbolic_single_sample` new random generator can be allocated and applied to node

        Parameters
        ----------
        node: :class:`Variable`
            Aesara node with symbolically applied VI replacements
        s: scalar
            desired number of samples
        d: bool or int
            whether sampling is done deterministically
        more_replacements: dict
            more replacements to apply

        Returns
        -------
        :class:`Variable` with applied replacements, ready to use
        """
        _node = node
        optimizations = self.get_optimization_replacements(s, d)
        flat2rand = self.make_size_and_deterministic_replacements(s, d, more_replacements)
        node = aesara.clone_replace(node, optimizations)
        node = aesara.clone_replace(node, flat2rand)
        try_to_set_test_value(_node, node, s)
        return node

    def to_flat_input(self, node, more_replacements=None):
        """*Dev* - replace vars with flattened view stored in `self.inputs`"""
        more_replacements = more_replacements or {}
        node = aesara.clone_replace(node, more_replacements)
        return aesara.clone_replace(node, self.replacements)

    def symbolic_sample_over_posterior(self, node, more_replacements=None):
        """*Dev* - performs sampling of node applying independent samples from posterior each time.
        Note that it is done symbolically and this node needs :func:`set_size_and_deterministic` call
        """
        node = self.to_flat_input(node, more_replacements=more_replacements)

        def sample(*post):
            return aesara.clone_replace(node, dict(zip(self.inputs, post)))

        nodes, _ = aesara.scan(sample, self.symbolic_randoms)
        return nodes

    def symbolic_single_sample(self, node, more_replacements=None):
        """*Dev* - performs sampling of node applying single sample from posterior.
        Note that it is done symbolically and this node needs
        :func:`set_size_and_deterministic` call with `size=1`
        """
        node = self.to_flat_input(node, more_replacements=more_replacements)
        post = [v[0] for v in self.symbolic_randoms]
        inp = self.inputs
        return aesara.clone_replace(node, dict(zip(inp, post)))

    def get_optimization_replacements(self, s, d):
        """*Dev* - optimizations for logP. If sample size is static and equal to 1:
        then `aesara.scan` MC estimate is replaced with single sample without call to `aesara.scan`.
        """
        repl = collections.OrderedDict()
        # avoid scan if size is constant and equal to one
        if isinstance(s, int) and (s == 1) or s is None:
            repl[self.varlogp] = self.single_symbolic_varlogp
            repl[self.datalogp] = self.single_symbolic_datalogp
        return repl

    @aesara.config.change_flags(compute_test_value="off")
    def sample_node(self, node, size=None, deterministic=False, more_replacements=None):
        """Samples given node or nodes over shared posterior

        Parameters
        ----------
        node: Aesara Variables (or Aesara expressions)
        size: None or scalar
            number of samples
        more_replacements: `dict`
            add custom replacements to graph, e.g. change input source
        deterministic: bool
            whether to use zeros as initial distribution
            if True - zero initial point will produce constant latent variables

        Returns
        -------
        sampled node(s) with replacements
        """
        node_in = node
        if more_replacements:
            node = aesara.clone_replace(node, more_replacements)
        if not isinstance(node, (list, tuple)):
            node = [node]
        node, _ = rvs_to_value_vars(node, apply_transforms=True)
        if not isinstance(node_in, (list, tuple)):
            node = node[0]
        if size is None:
            node_out = self.symbolic_single_sample(node)
        else:
            node_out = self.symbolic_sample_over_posterior(node)
        node_out = self.set_size_and_deterministic(node_out, size, deterministic)
        try_to_set_test_value(node_in, node_out, size)
        return node_out

    def rslice(self, name):
        """*Dev* - vectorized sampling for named random variable without call to `aesara.scan`.
        This node still needs :func:`set_size_and_deterministic` to be evaluated
        """

        def vars_names(vs):
            return {self.model.rvs_to_values[v].name for v in vs}

        for vars_, random, ordering in zip(
            self.collect("group"), self.symbolic_randoms, self.collect("ordering")
        ):
            if name in vars_names(vars_):
                name_, slc, shape, dtype = ordering[name]
                found = random[..., slc].reshape((random.shape[0],) + shape).astype(dtype)
                found.name = name + "_vi_random_slice"
                break
        else:
            raise KeyError("%r not found" % name)
        return found

    @node_property
    def sample_dict_fn(self):
        s = at.iscalar()
        names = [self.model.rvs_to_values[v].name for v in self.model.free_RVs]
        sampled = [self.rslice(name) for name in names]
        sampled = self.set_size_and_deterministic(sampled, s, 0)
        sample_fn = compile_pymc([s], sampled)

        def inner(draws=100):
            _samples = sample_fn(draws)
            return {v_: s_ for v_, s_ in zip(names, _samples)}

        return inner

    def sample(self, draws=500, return_inferencedata=True, **kwargs):
        """Draw samples from variational posterior.

        Parameters
        ----------
        draws: `int`
            Number of random samples.
        return_inferencedata: `bool`
            Return trace in Arviz format

        Returns
        -------
        trace: :class:`pymc.backends.base.MultiTrace`
            Samples drawn from variational posterior.
        """
        # TODO: add tests for include_transformed case
        kwargs["log_likelihood"] = False

        samples = self.sample_dict_fn(draws)  # type: dict
        points = ({name: records[i] for name, records in samples.items()} for i in range(draws))

        trace = NDArray(
            model=self.model,
            test_point={name: records[0] for name, records in samples.items()},
        )
        try:
            trace.setup(draws=draws, chain=0)
            for point in points:
                trace.record(point)
        finally:
            trace.close()

        trace = pm.sampling.MultiTrace([trace])
        if not return_inferencedata:
            return trace
        else:
            return pm.to_inference_data(trace, model=self.model, **kwargs)

    @property
    def ndim(self):
        return sum(self.collect("ndim"))

    @property
    def ddim(self):
        return sum(self.collect("ddim"))

    @property
    def has_local(self):
        return any(self.collect("local"))

    @property
    def has_global(self):
        return any(not c for c in self.collect("local"))

    @property
    def has_batched(self):
        return any(not c for c in self.collect("batched"))

    @node_property
    def symbolic_random(self):
        return at.concatenate(self.collect("symbolic_random2d"), axis=-1)

    def __str__(self):
        if len(self.groups) < 5:
            return "Approximation{" + " & ".join(map(str, self.groups)) + "}"
        else:
            forprint = self.groups[:2] + ["..."] + self.groups[-2:]
            return "Approximation{" + " & ".join(map(str, forprint)) + "}"

    @property
    def all_histograms(self):
        return all(isinstance(g, pm.approximations.EmpiricalGroup) for g in self.groups)

    @property
    def any_histograms(self):
        return any(isinstance(g, pm.approximations.EmpiricalGroup) for g in self.groups)

    @node_property
    def joint_histogram(self):
        if not self.all_histograms:
            raise VariationalInferenceError("%s does not consist of all Empirical approximations")
        return at.concatenate(self.collect("histogram"), axis=-1)

    @property
    def params(self):
        return sum(self.collect("params"), [])
