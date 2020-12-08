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

from collections import OrderedDict

import theano
import theano.tensor as tt

from pymc3.model import inputvars, modelcontext
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.theanof import make_shared_replacements, tt_rng

__all__ = []

EXPERIMENTAL_WARNING = (
    "Warning: Stochastic Gradient based sampling methods are experimental step methods and not yet"
    " recommended for use in PyMC3!"
)


def _value_error(cond, str):
    """Throws ValueError if cond is False"""
    if not cond:
        raise ValueError(str)


def _check_minibatches(minibatch_tensors, minibatches):
    _value_error(isinstance(minibatch_tensors, list), "minibatch_tensors must be a list.")

    _value_error(hasattr(minibatches, "__iter__"), "minibatches must be an iterator.")


def prior_dlogp(vars, model, flat_view):
    """Returns the gradient of the prior on the parameters as a vector of size D x 1"""
    terms = tt.concatenate([theano.grad(var.logpt, var).flatten() for var in vars], axis=0)
    dlogp = theano.clone(terms, flat_view.replacements, strict=False)

    return dlogp


def elemwise_dlogL(vars, model, flat_view):
    """
    Returns Jacobian of the log likelihood for each training datum wrt vars
    as a matrix of size N x D
    """
    # select one observed random variable
    obs_var = model.observed_RVs[0]
    # tensor of shape (batch_size,)
    logL = obs_var.logp_elemwiset.sum(axis=tuple(range(1, obs_var.logp_elemwiset.ndim)))
    # calculate fisher information
    terms = []
    for var in vars:
        output, _ = theano.scan(
            lambda i, logX=logL, v=var: theano.grad(logX[i], v).flatten(),
            sequences=[tt.arange(logL.shape[0])],
        )
        terms.append(output)
    dlogL = theano.clone(tt.concatenate(terms, axis=1), flat_view.replacements, strict=False)
    return dlogL


class BaseStochasticGradient(ArrayStepShared):
    R"""
    BaseStochasticGradient Object

    For working with BaseStochasticGradient Object
    we need to supply the probabilistic model
    (:code:`model`) with the data supplied to observed
    variables of type `GeneratorOp`

    Parameters
    ----------
    vars: list
        List of variables for sampler
    batch_size`: int
        Batch Size for each step
    total_size: int
        Total size of the training data
    step_size: float
        Step size for the parameter update
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context)
    random_seed: int
        The seed to initialize the Random Stream
    minibatches: iterator
        If the ObservedRV.observed is not a GeneratorOp then this parameter must not be None
    minibatch_tensor: list of tensors
        If the ObservedRV.observed is not a GeneratorOp then this parameter must not be None
        The length of this tensor should be the same as the next(minibatches)

    Notes
    -----
    Defining a BaseStochasticGradient needs
    custom implementation of the following methods:
        - :code: `.mk_training_fn()`
            Returns a theano function which is called for each sampling step
        - :code: `._initialize_values()`
            Returns None it creates class variables which are required for the training fn
    """

    def __init__(
        self,
        vars=None,
        batch_size=None,
        total_size=None,
        step_size=1.0,
        model=None,
        random_seed=None,
        minibatches=None,
        minibatch_tensors=None,
        **kwargs
    ):
        warnings.warn(EXPERIMENTAL_WARNING)

        model = modelcontext(model)

        if vars is None:
            vars = model.vars

        vars = inputvars(vars)

        self.model = model
        self.vars = vars
        self.batch_size = batch_size
        self.total_size = total_size
        _value_error(
            total_size != None or batch_size != None,
            "total_size and batch_size of training data have to be specified",
        )
        self.expected_iter = int(total_size / batch_size)

        # set random stream
        self.random = None
        if random_seed is None:
            self.random = tt_rng()
        else:
            self.random = tt_rng(random_seed)

        self.step_size = step_size

        shared = make_shared_replacements(vars, model)

        self.updates = OrderedDict()
        self.q_size = int(sum(v.dsize for v in self.vars))

        flat_view = model.flatten(vars)
        self.inarray = [flat_view.input]

        self.dlog_prior = prior_dlogp(vars, model, flat_view)
        self.dlogp_elemwise = elemwise_dlogL(vars, model, flat_view)
        self.q_size = int(sum(v.dsize for v in self.vars))

        if minibatch_tensors != None:
            _check_minibatches(minibatch_tensors, minibatches)
            self.minibatches = minibatches

            # Replace input shared variables with tensors
            def is_shared(t):
                return isinstance(t, theano.compile.sharedvalue.SharedVariable)

            tensors = [(t.type() if is_shared(t) else t) for t in minibatch_tensors]
            updates = OrderedDict(
                {t: t_ for t, t_ in zip(minibatch_tensors, tensors) if is_shared(t)}
            )
            self.minibatch_tensors = tensors
            self.inarray += self.minibatch_tensors
            self.updates.update(updates)

        self._initialize_values()
        super().__init__(vars, shared)

    def _initialize_values(self):
        """Initializes the parameters for the stochastic gradient minibatch
        algorithm"""
        raise NotImplementedError

    def mk_training_fn(self):
        raise NotImplementedError

    def training_complete(self):
        """Returns boolean if astep has been called expected iter number of times"""
        return self.expected_iter == self.t

    def astep(self, q0):
        """Perform a single update in the stochastic gradient method.

        Returns new shared values and values sampled
        The size and ordering of q0 and q must be the same
        Parameters
        -------
        q0: list
            List of shared values and values sampled from last estimate

        Returns
        -------
        q
        """
        if hasattr(self, "minibatch_tensors"):
            return q0 + self.training_fn(q0, *next(self.minibatches))
        else:
            return q0 + self.training_fn(q0)
