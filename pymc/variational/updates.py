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

# Taken from the Lasagne project: http://lasagne.readthedocs.io/en/latest/
# License:
# The MIT License (MIT)

# Copyright (c) 2014-2015 Lasagne contributors

# Lasagne uses a shared copyright model: each contributor holds copyright over
# their contributions to Lasagne. The project versioning records all such
# contribution and copyright details.
# By contributing to the Lasagne repository through pull-request, comment,
# or otherwise, the contributor releases their content to the license and
# copyright terms herein.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Functions to generate PyTensor update dictionaries for training.

The update functions implement different methods to control the learning
rate for use with stochastic gradient descent.

Update functions take a loss expression or a list of gradient expressions and
a list of parameters as input and return an ordered dictionary of updates:

.. autosummary::
    :nosignatures:

    sgd
    momentum
    nesterov_momentum
    adagrad
    rmsprop
    adadelta
    adam
    adamax

Two functions can be used to further modify the updates to include momentum:

.. autosummary::
    :nosignatures:

    apply_momentum
    apply_nesterov_momentum

Finally, we provide two helper functions to constrain the norm of tensors:

.. autosummary::
    :nosignatures:

    norm_constraint
    total_norm_constraint

:func:`norm_constraint()` can be used to constrain the norm of parameters
(as an alternative to weight decay), or for a form of gradient clipping.
:func:`total_norm_constraint()` constrain the total norm of a list of tensors.
This is often used when training recurrent neural networks.

Examples
--------
>>> import lasagne
>>> import pytensor
>>> from lasagne.nonlinearities import softmax
>>> from lasagne.layers import InputLayer, DenseLayer, get_output
>>> from lasagne.updates import sgd, apply_momentum
>>> l_in = InputLayer((100, 20))
>>> l1 = DenseLayer(l_in, num_units=3, nonlinearity=softmax)
>>> x = pt.matrix('x')  # shp: num_batch x num_features
>>> y = pt.ivector('y') # shp: num_batch
>>> l_out = get_output(l1, x)
>>> params = lasagne.layers.get_all_params(l1)
>>> loss = pt.mean(pt.nnet.categorical_crossentropy(l_out, y))
>>> updates_sgd = sgd(loss, params, learning_rate=0.0001)
>>> updates = apply_momentum(updates_sgd, params, momentum=0.9)
>>> train_function = pytensor.function([x, y], updates=updates)

Notes
-----
Taken from the Lasagne project: http://lasagne.readthedocs.io/en/latest/

"""
from collections import OrderedDict
from functools import partial, wraps
from typing import Callable

import numpy as np
import pytensor
import pytensor.tensor as pt

import pymc as pm

__all__ = [
    "sgd",
    "apply_momentum",
    "momentum",
    "apply_nesterov_momentum",
    "nesterov_momentum",
    "adagrad",
    "adagrad_window",
    "rmsprop",
    "adadelta",
    "adam",
    "adamax",
    "norm_constraint",
    "total_norm_constraint",
]


def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients

    Parameters
    ----------
    loss_or_grads: symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params: list of shared variables
        The variables to return the gradients for

    Returns
    -------
    list of expressions
        If `loss_or_grads` is a list, it is assumed to be a list of
        gradients and returned as is, unless it does not match the length
        of `params`, in which case a `ValueError` is raised.
        Otherwise, `loss_or_grads` is assumed to be a cost expression and
        the function returns `pytensor.grad(loss_or_grads, params)`.

    Raises
    ------
    ValueError
        If `loss_or_grads` is a list of a different length than `params`, or if
        any element of `params` is not a shared variable (while we could still
        compute its gradient, we can never update it and want to fail early).
    """
    if any(not isinstance(p, pytensor.compile.SharedVariable) for p in params):
        raise ValueError(
            "params must contain shared variables only. If it "
            "contains arbitrary parameter expressions, then "
            "lasagne.utils.collect_shared_vars() may help you."
        )
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError(
                "Got %d gradient expressions for %d parameters" % (len(loss_or_grads), len(params))
            )
        return loss_or_grads
    else:
        grads = pytensor.grad(loss_or_grads, params)
        for grad, param in zip(grads, params):
            grad.name = f'd_loss/d_{param.name}'
        return grads


def _input_to_shared_variable(x, name):
    if isinstance(x, pt.sharedvar.SharedVariable):
        return x
    return pytensor.shared(x, name=name)


def _find_variable_among_args_kwargs(args, kwargs, name):
    """
    Helper function to find a variable among args and kwargs.

    Notes
    -----
    Assumes that the variable being searched for is either a kwarg or the first arg.
    """

    variable = kwargs.pop(name, None)
    if not variable:
        variable = args.pop(0) if len(args) > 0 else None
    return variable


def _partial_initialization_wrapper(optimizer):
    """
    Functional wrapper to allow optimizer to be called without both loss_or_grads and params

    Parameters
    ----------
    optimizer: callable
        Optimizer function to wrap

    Returns
    -------
    optimizer: callable
        Optimizer function that returns itself partially initialized if called without both loss_or_grads and params
    """

    @wraps(optimizer)
    def make_partial_if_necessary(*args, **kwargs):
        args = list(args)
        loss_or_grads = _find_variable_among_args_kwargs(args, kwargs, 'loss_or_grads')
        params = _find_variable_among_args_kwargs(args, kwargs, 'params')

        if loss_or_grads is None and params is None:
            return partial(optimizer, **kwargs)
        elif loss_or_grads is None or params is None:
            raise ValueError("Please provide both `loss_or_grads` and `params` to get updates")

        return optimizer(loss_or_grads=loss_or_grads, params=params, **kwargs)

    return make_partial_if_necessary


def _handle_loss_and_grad_input_wrapper(optimizer):
    """
    Functional wrapper to allow optimizer to take a tuple of (loss, grads) as input, and either discard the loss or
    pass it through.

    Adds a keyword argument to the wrapped optimizer, `discard_loss`, which if True, will discard the loss and only
    return the updates. If False, the optimizer will return both the updates and the loss.

    Parameters
    ----------
    optimizer: callable
        Optimizer function to wrap

    Returns
    -------
    optimizer: callable
        Wrapped optimizer function
    """

    @wraps(optimizer)
    def discard_or_pass_through_loss_optimizer(loss_or_grads, params, discard_loss=True, *args, **kwargs):
        if isinstance(loss_or_grads, tuple):
            loss, grads = loss_or_grads
            updates = optimizer(loss_or_grads=grads, params=params, *args, **kwargs)
        else:
            discard_loss, loss = True, None
            updates = optimizer(loss_or_grads=loss_or_grads, params=params, *args, **kwargs)

        if discard_loss:
            return updates
        else:
            return updates, loss

    return discard_or_pass_through_loss_optimizer


def _sgd(loss_or_grads=None, params=None, *, learning_rate=1e-3):
    """Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:

    * ``param := param - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads: symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params: list of shared variables
        The variables to generate update expressions for
    learning_rate: float or symbolic scalar
        The learning rate controlling the size of update steps

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    Optimizer can be called without both loss_or_grads and params
    in that case partial function is returned

    Examples
    --------
    >>> a = pytensor.shared(1.)
    >>> b = a*2
    >>> updates = sgd(b, [a], learning_rate=.01)
    >>> isinstance(updates, dict)
    True
    >>> optimizer = sgd(learning_rate=.01)
    >>> callable(optimizer)
    True
    >>> updates = optimizer(b, [a])
    >>> isinstance(updates, dict)
    True
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updated_param = param - learning_rate * grad
        updated_param.name = f'{param.name}__updated'
        updates[param] = updated_param

    return updates


sgd = _partial_initialization_wrapper(_handle_loss_and_grad_input_wrapper(_sgd))


def apply_momentum(updates, params=None, momentum=0.9):
    """Returns a modified update dictionary including momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity + updates[param] - param``
    * ``param := param + velocity``

    Parameters
    ----------
    updates: OrderedDict
        A dictionary mapping parameters to update expressions
    params: iterable of shared variables, optional
        The variables to apply momentum to. If omitted, will apply
        momentum to all `updates.keys()`.
    momentum: float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
        A copy of `updates` with momentum updates for all `params`.

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    See Also
    --------
    momentum: Shortcut applying momentum to SGD updates
    """
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape, name='velocity')

        updated_param = momentum * velocity + updates[param]
        updated_param.name = f'{param}__updated'

        updated_velocity = updated_param - param
        updated_velocity.name = 'velocity__updated'

        updates[velocity] = updated_velocity
        updates[param] = updated_param

    return updates


def _momentum(loss_or_grads=None, params=None, learning_rate=1e-3, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + velocity``

    Parameters
    ----------
    loss_or_grads: symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params: list of shared variables
        The variables to generate update expressions for
    learning_rate: float or symbolic scalar
        The learning rate controlling the size of update steps
    momentum: float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    Optimizer can be called without both loss_or_grads and params
    in that case partial function is returned

    See Also
    --------
    apply_momentum: Generic function applying momentum to updates
    nesterov_momentum: Nesterov's variant of SGD with momentum

    Examples
    --------
    >>> a = pytensor.shared(1.)
    >>> b = a*2
    >>> updates = momentum(b, [a], learning_rate=.01)
    >>> isinstance(updates, dict)
    True
    >>> optimizer = momentum(learning_rate=.01)
    >>> callable(optimizer)
    True
    >>> updates = optimizer(b, [a])
    >>> isinstance(updates, dict)
    True
    """
    updates = sgd(loss_or_grads, params, learning_rate)

    return apply_momentum(updates, momentum=momentum)


momentum = _partial_initialization_wrapper(_handle_loss_and_grad_input_wrapper(_momentum))


def apply_nesterov_momentum(updates, params=None, momentum=0.9):
    """Returns a modified update dictionary including Nesterov momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity + updates[param] - param``
    * ``param := param + momentum * velocity + updates[param] - param``

    Parameters
    ----------
    updates: OrderedDict
        A dictionary mapping parameters to update expressions
    params: iterable of shared variables, optional
        The variables to apply momentum to. If omitted, will apply
        momentum to all `updates.keys()`.
    momentum: float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
        A copy of `updates` with momentum updates for all `params`.

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    The classic formulation of Nesterov momentum (or Nesterov accelerated
    gradient) requires the gradient to be evaluated at the predicted next
    position in parameter space. Here, we use the formulation described at
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
    which allows the gradient to be evaluated at the current parameters.

    See Also
    --------
    nesterov_momentum: Shortcut applying Nesterov momentum to SGD updates
    """
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape)

        updated_velocity = momentum * velocity + updates[param] - param
        updated_velocity.name = 'velocity__updated'

        updated_param = momentum * updated_velocity + updates[param]
        updated_param.name = f'{param.name}__updated'

        updates[velocity] = updated_velocity
        updates[param] = updated_param

    return updates


def _nesterov_momentum(loss_or_grads=None, params=None, learning_rate=1e-3, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + momentum * velocity - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads: symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params: list of shared variables
        The variables to generate update expressions for
    learning_rate: float or symbolic scalar
        The learning rate controlling the size of update steps
    momentum: float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    The classic formulation of Nesterov momentum (or Nesterov accelerated
    gradient) requires the gradient to be evaluated at the predicted next
    position in parameter space. Here, we use the formulation described at
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
    which allows the gradient to be evaluated at the current parameters.

    Optimizer can be called without both loss_or_grads and params
    in that case partial function is returned

    See Also
    --------
    apply_nesterov_momentum: Function applying momentum to updates

    Examples
    --------
    >>> a = pytensor.shared(1.)
    >>> b = a*2
    >>> updates = nesterov_momentum(b, [a], learning_rate=.01)
    >>> isinstance(updates, dict)
    True
    >>> optimizer = nesterov_momentum(learning_rate=.01)
    >>> callable(optimizer)
    True
    >>> updates = optimizer(b, [a])
    >>> isinstance(updates, dict)
    True
    """

    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_nesterov_momentum(updates, momentum=momentum)


nesterov_momentum = _partial_initialization_wrapper(_handle_loss_and_grad_input_wrapper(_nesterov_momentum))


def _adagrad(loss_or_grads=None, params=None, learning_rate=1.0, epsilon=1e-6):
    """Adagrad updates

    Scale learning rates by dividing with the square root of accumulated
    squared gradients. See [1]_ for further description.

    Parameters
    ----------
    loss_or_grads: symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params: list of shared variables
        The variables to generate update expressions for
    learning_rate: float or symbolic scalar
        The learning rate controlling the size of update steps
    epsilon: float or symbolic scalar
        Small value added for numerical stability

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    Using step size eta Adagrad calculates the learning rate for feature i at
    time step t as:

    .. math:: \\eta_{t,i} = \\frac{\\eta}
       {\\sqrt{\\sum^t_{t^\\prime} g^2_{t^\\prime,i}+\\epsilon}} g_{t,i}

    as such the learning rate is monotonically decreasing.

    Epsilon is not included in the typical formula, see [2]_.

    Optimizer can be called without both loss_or_grads and params
    in that case partial function is returned

    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
           Adaptive subgradient methods for online learning and stochastic
           optimization. JMLR, 12:2121-2159.

    .. [2] Chris Dyer:
           Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf

    Examples
    --------
    >>> a = pytensor.shared(1.)
    >>> b = a*2
    >>> updates = adagrad(b, [a], learning_rate=.01)
    >>> isinstance(updates, dict)
    True
    >>> optimizer = adagrad(learning_rate=.01)
    >>> callable(optimizer)
    True
    >>> updates = optimizer(b, [a])
    >>> isinstance(updates, dict)
    True
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape,
                               name='gradient_squares')
        accu_new = accu + grad ** 2
        accu_new.name = 'gradient_squares__updated'

        updates[accu] = accu_new

        updated_param = param - (learning_rate * grad / pt.sqrt(accu_new + epsilon))
        updated_param.name = f'{param.name}__updated'

        updates[param] = updated_param

    return updates


adagrad = _partial_initialization_wrapper(_handle_loss_and_grad_input_wrapper(_adagrad))


def _adagrad_window(loss_or_grads=None, params=None, learning_rate=0.001, epsilon=0.1, n_win=10):
    """Returns a function that returns parameter updates.
    Instead of accumulated estimate, uses running window

    Parameters
    ----------
    loss_or_grads: symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params: list of shared variables
        The variables to generate update expressions for
    learning_rate: float
        Learning rate.
    epsilon: float
        Offset to avoid zero-division in the normalizer of adagrad.
    n_win: int
        Number of past steps to calculate scales of parameter gradients.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        i = pytensor.shared(pm.floatX(0), name='window_idx')
        i_int = i.astype("int32")
        value = param.get_value(borrow=True)

        accu = pytensor.shared(np.zeros(value.shape + (n_win,), dtype=value.dtype),
                               name='gradient_squares')

        # Append squared gradient vector to accu_new
        accu_new = pt.set_subtensor(accu[..., i_int], grad ** 2)
        accu_new.name = 'gradient_squares__updated'

        i_new = pt.switch((i + 1) < n_win, i + 1, 0)
        i_new.name = 'window_idx__updated'

        updates[accu] = accu_new
        updates[i] = i_new

        accu_sum = accu_new.sum(axis=-1)

        param_updated = param - (learning_rate * grad / pt.sqrt(accu_sum + epsilon))
        param_updated.name = f'{param.name}__updated'
        updates[param] = param_updated

    return updates


adagrad_window = _partial_initialization_wrapper(_handle_loss_and_grad_input_wrapper(_adagrad_window))


def _rmsprop(loss_or_grads=None, params=None, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """RMSProp updates

    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [1]_ for further description.

    Parameters
    ----------
    loss_or_grads: symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params: list of shared variables
        The variables to generate update expressions for
    learning_rate: float or symbolic scalar
        The learning rate controlling the size of update steps
    rho: float or symbolic scalar
        Gradient moving average decay factor
    epsilon: float or symbolic scalar
        Small value added for numerical stability

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.

    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:

    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}


    Optimizer can be called without both loss_or_grads and params
    in that case partial function is returned

    References
    ----------
    .. [1] Tieleman, at. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)

    Examples
    --------
    >>> a = pytensor.shared(1.)
    >>> b = a*2
    >>> updates = rmsprop(b, [a], learning_rate=.01)
    >>> isinstance(updates, dict)
    True
    >>> optimizer = rmsprop(learning_rate=.01)
    >>> callable(optimizer)
    True
    >>> updates = optimizer(b, [a])
    >>> isinstance(updates, dict)
    True
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    # Using pytensor constant to prevent upcasting of float32
    one = pt.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape,
                               name='gradient_squares')

        accu_new = rho * accu + (one - rho) * grad ** 2
        accu_new.name = 'gradient_squares__updated'

        updates[accu] = accu_new

        param_updated = param - (learning_rate * grad / pt.sqrt(accu_new + epsilon))
        param_updated.name = f'{param.name}__updated'
        updates[param] = param_updated

    return updates


rmsprop = _partial_initialization_wrapper(_handle_loss_and_grad_input_wrapper(_rmsprop))


def _adadelta(loss_or_grads=None, params=None, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    r"""Adadelta updates

    Scale learning rates by the ratio of accumulated gradients to accumulated
    updates, see [1]_ and notes for further description.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Squared gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    rho should be between 0 and 1. A value of rho close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.

    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
    work for multiple datasets (MNIST, speech).

    In the paper, no learning rate is considered (so learning_rate=1.0).
    Probably best to keep it at this value.
    epsilon is important for the very first update (so the numerator does
    not become 0).

    Using the step size eta and a decay factor rho the learning rate is
    calculated as:

    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1} + \\epsilon}}
                             {\sqrt{r_t + \epsilon}}\\\\
       s_t &= \\rho s_{t-1} + (1-\\rho)*(\\eta_t*g)^2

    Optimizer can be called without both loss_or_grads and params
    in that case partial function is returned

    References
    ----------
    .. [1] Zeiler, M. D. (2012):
           ADADELTA: An Adaptive Learning Rate Method.
           arXiv Preprint arXiv:1212.5701.

    Examples
    --------
    >>> a = pytensor.shared(1.)
    >>> b = a*2
    >>> updates = adadelta(b, [a], learning_rate=.01)
    >>> isinstance(updates, dict)
    True
    >>> optimizer = adadelta(learning_rate=.01)
    >>> callable(optimizer)
    True
    >>> updates = optimizer(b, [a])
    >>> isinstance(updates, dict)
    True
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    # Using pytensor constant to prevent upcasting of float32
    one = pt.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        # accu: accumulate gradient magnitudes

        accu = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape,
                               name='gradient_squares')
        # delta_accu: accumulate update magnitudes (recursively!)
        delta_accu = pytensor.shared(
            np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape, name=''
        )

        # update accu (as in rmsprop)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = grad * pt.sqrt(delta_accu + epsilon) / pt.sqrt(accu_new + epsilon)
        updates[param] = param - learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates


adadelta = _partial_initialization_wrapper(_handle_loss_and_grad_input_wrapper(_adadelta))


def _adam(
        loss_or_grads=None, params=None, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
):
    """Adam updates

    Adam updates implemented as in [1]_.

    Parameters
    ----------
    loss_or_grads: symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params: list of shared variables
        The variables to generate update expressions for
    learning_rate: float
        Learning rate
    beta1: float
        Exponential decay rate for the first moment estimates.
    beta2: float
        Exponential decay rate for the second moment estimates.
    epsilon: float
        Constant for numerical stability.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.

    Optimizer can be called without both loss_or_grads and params
    in that case partial function is returned

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.

    Examples
    --------
    >>> a = pytensor.shared(1.)
    >>> b = a*2
    >>> updates = adam(b, [a], learning_rate=.01)
    >>> isinstance(updates, dict)
    True
    >>> optimizer = adam(learning_rate=.01)
    >>> callable(optimizer)
    True
    >>> updates = optimizer(b, [a])
    >>> isinstance(updates, dict)
    True
    """
    all_grads = get_or_compute_grads(loss_or_grads, params)
    t_prev = pytensor.shared(pm.pytensorf.floatX(0.0), name='t')
    updates = OrderedDict()

    # Using pytensor constant to prevent upcasting of float32
    one = pt.constant(1)

    t = t_prev + 1
    t.name = 't__updated'
    a_t = learning_rate * pt.sqrt(one - beta2 ** t) / (one - beta1 ** t)
    a_t.name = 'a'

    for param, g_t in zip(params, all_grads):
        name = param.name
        value = param.get_value(borrow=True)
        m_prev = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape, name=f'{name}_m')
        v_prev = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape, name=f'{name}_v')

        m_t = beta1 * m_prev + (one - beta1) * g_t
        m_t.name = f'{name}_m__updated'
        v_t = beta2 * v_prev + (one - beta2) * g_t ** 2
        v_t.name = f'{name}_v__updated'

        step = a_t * m_t / (pt.sqrt(v_t) + epsilon)
        step.name = f'{name}_step_size'

        updates[m_prev] = m_t
        updates[v_prev] = v_t

        param_updated = param - step
        param_updated.name = f'{name}__updated'
        updates[param] = param_updated

    updates[t_prev] = t
    return updates


adam = _partial_initialization_wrapper(_handle_loss_and_grad_input_wrapper(_adam))


def _adamax(
        loss_or_grads=None, params=None, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8
):
    """Adamax updates

    Adamax updates implemented as in [1]_. This is a variant of the Adam
    algorithm based on the infinity norm.

    Parameters
    ----------
    loss_or_grads: symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params: list of shared variables
        The variables to generate update expressions for
    learning_rate: float
        Learning rate
    beta1: float
        Exponential decay rate for the first moment estimates.
    beta2: float
        Exponential decay rate for the weighted infinity norm estimates.
    epsilon: float
        Constant for numerical stability.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    Optimizer can be called without both loss_or_grads and params
    in that case partial function is returned

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.

    Examples
    --------
    >>> a = pytensor.shared(1.)
    >>> b = a*2
    >>> updates = adamax(b, [a], learning_rate=.01)
    >>> isinstance(updates, dict)
    True
    >>> optimizer = adamax(learning_rate=.01)
    >>> callable(optimizer)
    True
    >>> updates = optimizer(b, [a])
    >>> isinstance(updates, dict)
    True
    """
    all_grads = get_or_compute_grads(loss_or_grads, params)
    t_prev = pytensor.shared(pm.pytensorf.floatX(0.0), name='t')
    updates = OrderedDict()

    # Using pytensor constant to prevent upcasting of float32
    one = pt.constant(1)

    t = t_prev + 1
    t.name = 't__updated'

    a_t = learning_rate / (one - beta1 ** t)
    a_t.name = 'a'

    for param, g_t in zip(params, all_grads):
        name = param.name
        value = param.get_value(borrow=True)
        m_prev = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape, name=f'{name}_m')
        u_prev = pytensor.shared(np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape, name=f'{name}_u')

        m_t = beta1 * m_prev + (one - beta1) * g_t
        m_t.name = f'{name}_m__updated'

        u_t = pt.maximum(beta2 * u_prev, abs(g_t))
        u_t.name = f'{name}_u__updated'

        step = a_t * m_t / (u_t + epsilon)
        step.name = f'{name}_step_size'

        updates[m_prev] = m_t
        updates[u_prev] = u_t

        param_updated = param - step
        param_updated.name = f'{name}__updated'
        updates[param] = param_updated

    updates[t_prev] = t
    return updates


adamax = _partial_initialization_wrapper(_handle_loss_and_grad_input_wrapper(_adamax))


def norm_constraint(tensor_var, max_norm, norm_axes=None, epsilon=1e-7):
    """Max weight norm constraints and gradient clipping

    This takes a TensorVariable and rescales it so that incoming weight
    norms are below a specified constraint value. Vectors violating the
    constraint are rescaled so that they are within the allowed range.

    Parameters
    ----------
    tensor_var: TensorVariable
        PyTensor expression for update, gradient, or other quantity.
    max_norm: scalar
        This value sets the maximum allowed value of any norm in
        `tensor_var`.
    norm_axes: sequence (list or tuple)
        The axes over which to compute the norm.  This overrides the
        default norm axes defined for the number of dimensions
        in `tensor_var`. When this is not specified and `tensor_var` is a
        matrix (2D), this is set to `(0,)`. If `tensor_var` is a 3D, 4D or
        5D tensor, it is set to a tuple listing all axes but axis 0. The
        former default is useful for working with dense layers, the latter
        is useful for 1D, 2D and 3D convolutional layers.
        (Optional)
    epsilon: scalar, optional
        Value used to prevent numerical instability when dividing by
        very small or zero norms.

    Returns
    -------
    TensorVariable
        Input `tensor_var` with rescaling applied to weight vectors
        that violate the specified constraints.

    Examples
    --------
    >>> param = pytensor.shared(
    ...     np.random.randn(100, 200).astype(pytensor.config.floatX))
    >>> update = param + 100
    >>> update = norm_constraint(update, 10)
    >>> func = pytensor.function([], [], updates=[(param, update)])
    >>> # Apply constrained update
    >>> _ = func()
    >>> from lasagne.utils import compute_norms
    >>> norms = compute_norms(param.get_value())
    >>> np.isclose(np.max(norms), 10)
    True

    Notes
    -----
    When `norm_axes` is not specified, the axes over which the norm is
    computed depend on the dimensionality of the input variable. If it is
    2D, it is assumed to come from a dense layer, and the norm is computed
    over axis 0. If it is 3D, 4D or 5D, it is assumed to come from a
    convolutional layer and the norm is computed over all trailing axes
    beyond axis 0. For other uses, you should explicitly specify the axes
    over which to compute the norm using `norm_axes`.
    """
    ndim = tensor_var.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}." "Must specify `norm_axes`".format(ndim)
        )

    dtype = np.dtype(pytensor.config.floatX).type
    norms = pt.sqrt(pt.sum(pt.sqr(tensor_var), axis=sum_over, keepdims=True))
    target_norms = pt.clip(norms, 0, dtype(max_norm))
    constrained_output = tensor_var * (target_norms / (dtype(epsilon) + norms))

    return constrained_output


def total_norm_constraint(tensor_vars, max_norm, epsilon=1e-7, return_norm=False):
    """Rescales a list of tensors based on their combined norm

    If the combined norm of the input tensors exceeds the threshold then all
    tensors are rescaled such that the combined norm is equal to the threshold.

    Scaling the norms of the gradients is often used when training recurrent
    neural networks [1]_.

    Parameters
    ----------
    tensor_vars: List of TensorVariables.
        Tensors to be rescaled.
    max_norm: float
        Threshold value for total norm.
    epsilon: scalar, optional
        Value used to prevent numerical instability when dividing by
        very small or zero norms.
    return_norm: bool
        If true the total norm is also returned.

    Returns
    -------
    tensor_vars_scaled: list of TensorVariables
        The scaled tensor variables.
    norm: PyTensor scalar
        The combined norms of the input variables prior to rescaling,
        only returned if ``return_norms=True``.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> import lasagne
    >>> from lasagne.updates import sgd, total_norm_constraint
    >>> x = pt.matrix()
    >>> y = pt.ivector()
    >>> l_in = InputLayer((5, 10))
    >>> l1 = DenseLayer(l_in, num_units=7, nonlinearity=pt.special.softmax)
    >>> output = lasagne.layers.get_output(l1, x)
    >>> cost = pt.mean(pt.nnet.categorical_crossentropy(output, y))
    >>> all_params = lasagne.layers.get_all_params(l1)
    >>> all_grads = pt.grad(cost, all_params)
    >>> scaled_grads = total_norm_constraint(all_grads, 5)
    >>> updates = sgd(scaled_grads, all_params, learning_rate=0.1)

    Notes
    -----
    The total norm can be used to monitor training.

    References
    ----------
    .. [1] Sutskever, I., Vinyals, O., & Le, Q. V. (2014): Sequence to sequence
       learning with neural networks. In Advances in Neural Information
       Processing Systems (pp. 3104-3112).
    """
    norm = pt.sqrt(sum(pt.sum(tensor ** 2) for tensor in tensor_vars))
    dtype = np.dtype(pytensor.config.floatX).type
    target_norm = pt.clip(norm, 0, dtype(max_norm))
    multiplier = target_norm / (dtype(epsilon) + norm)
    tensor_vars_scaled = [step * multiplier for step in tensor_vars]

    if return_norm:
        return tensor_vars_scaled, norm
    else:
        return tensor_vars_scaled


def _handle_time_updates(updates):
    """
    Create a shared time variable and its update if one does not already exist in the updates dictionary, otherwise
    extract it and delete the entry from the updates dictionary.

    Parameters
    ----------
    updates: dict
        update dictionary created by an optimizer function

    Returns
    -------
    t: pt.shared.SharedVariable
        shared variable representing the current time step
    new_t: pt.shared.SharedVariable
        shared variable representing the next time step

    Notes
    -----
    This function potentially modifies the update dictionary in-place by deleting the entry for the time variable, if
    it exists. This is done to ensure that the time variable is always the last update applied. All schedulers need
    to add this update back in to the update dictionary before returning it.
    """
    old_values = list(updates.keys())
    old_names = [shared_var.name for shared_var in old_values]

    t_idx = old_names.index('t') if 't' in old_names else None
    if t_idx is None:
        t = pytensor.shared(pm.pytensorf.floatX(0.0), name='t')
        new_t = t + 1
        new_t.name = 't__updated'
    else:
        # If t is already present, we will reuse it, but we also need to delete it from the update dict temporarily.
        # We always want it to be the last update applied.
        t = old_values[t_idx]
        new_t = updates[t]
        del updates[t]

    return t, new_t


def exponential_decay_scheduler(optimizer: Callable, decay_steps: int, decay_rate: float, min_lr: float = 1e-6,
                                staircase: bool = False):
    """
    Returns a new optimizer that applies exponential decay to the learning rate.

    Parameters
    ----------
    optimizer: Callable
        Optimizer to apply exponential decay to
    decay_steps: int
        Number of steps between application of a decay.
    decay_rate: float
        Decay factor used to compute new learning rate, with new_lr = max(lr * decay_rate, min_lr). Must be between 0
        and 1.
    min_lr: float
        Minimum learning rate, after which no additional decay is applied. Defaults to 1e-6.
    staircase: bool
        If True, learning rate is decayed in discrete intervals, otherwise decay is applied continuously.
        Defaults to False.

    Returns
    -------
    scheduled_optimizer: Callable
        Optimizer with exponential decay applied to learning rate.
    """
    if not 0 < decay_rate <= 1:
        raise ValueError('decay_rate must be between 0 and 1')

    kwargs = optimizer.keywords
    _initial_lr = pm.floatX(optimizer.keywords['learning_rate'])

    initial_lr = pt.constant(_initial_lr, name='initial_learning_rate')
    shared_lr = _input_to_shared_variable(_initial_lr, 'learning_rate')
    kwargs['learning_rate'] = shared_lr

    @wraps(optimizer)
    def optimizer_with_exponential_decay(loss_or_grads, params, *args, **kwargs):
        updates = optimizer(loss_or_grads, params, *args, **kwargs)
        t, new_t = _handle_time_updates(updates)

        if staircase:
            new_lr = initial_lr * decay_rate ** (t // decay_steps)
        else:
            new_lr = initial_lr * decay_rate ** (t / decay_steps)

        new_lr = pt.maximum(new_lr, min_lr)

        new_lr.name = 'learning_rate__updated'
        updates[shared_lr] = new_lr
        updates[t] = new_t

        return updates

    return optimizer_with_exponential_decay


def reduce_lr_on_plateau_scheduler(optimizer, factor=0.1, patience=10, min_lr=1e-6, cooldown=0):
    kwargs = optimizer.keywords
    _initial_lr = pm.floatX(optimizer.keywords['learning_rate'])
    shared_lr = _input_to_shared_variable(_initial_lr, 'learning_rate')
    kwargs['learning_rate'] = shared_lr

    @wraps(optimizer)
    def optimizer_with_reduce_lr_on_plateau(loss_or_grads, params, *args, **kwargs):
        updates, loss = optimizer(loss_or_grads, params, *args, discard_loss=False, **kwargs)

        cooldown_counter = pytensor.shared(np.zeros((), dtype='int32'), name='cooldown_counter')
        wait = pytensor.shared(np.zeros((), dtype='int32'), name='wait')
        best_loss = pytensor.shared(np.inf, name='best_loss')

        loss_is_inf = pt.isinf(loss)

        in_cooldown = pt.gt(cooldown_counter, 0)
        improving_loss = pt.lt(loss, best_loss)
        patience_exceeded = pt.ge(wait, patience)

        updated_best_loss = pt.switch(loss_is_inf,
                                      best_loss,
                                      pt.switch(improving_loss,
                                                loss,
                                                best_loss))

        updated_best_loss.name = 'best_loss__updated'

        updated_cooldown_counter = pt.switch(loss_is_inf,
                                             cooldown_counter,
                                             pt.switch(in_cooldown,
                                                       cooldown_counter - 1,
                                                       pt.switch(improving_loss,
                                                                 cooldown_counter,
                                                                 pt.switch(patience_exceeded,
                                                                           cooldown,
                                                                           cooldown_counter))))
        updated_cooldown_counter.name = 'cooldown_counter__updated'

        updated_lr = pt.switch(loss_is_inf,
                           shared_lr,
                           pt.switch(in_cooldown,
                                     shared_lr,
                                     pt.switch(improving_loss,
                                               shared_lr,
                                               pt.switch(patience_exceeded,
                                                         pt.maximum(min_lr, shared_lr * factor),
                                                         shared_lr))))

        updated_lr.name = 'learning_rate__updated'

        updated_wait = pt.switch(loss_is_inf,
                              wait,
                              pt.switch(in_cooldown,
                                        0,
                                        pt.switch(improving_loss,
                                                  0,
                                                  pt.switch(patience_exceeded,
                                                            0,
                                                            wait + 1))))
        updated_wait.name = 'wait__updated'

        updates[best_loss] = updated_best_loss
        updates[cooldown_counter] = updated_cooldown_counter
        updates[wait] = updated_wait
        updates[shared_lr] = updated_lr

        return updates

    return optimizer_with_reduce_lr_on_plateau
