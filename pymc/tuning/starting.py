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

"""
Created on Mar 12, 2011

@author: johnsalvatier
"""
import sys

from typing import Optional

import aesara.gradient as tg
import numpy as np

from fastprogress.fastprogress import ProgressBar, progress_bar
from numpy import isfinite, nan_to_num
from scipy.optimize import minimize

import pymc as pm

from pymc.aesaraf import inputvars
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.initial_point import make_initial_point_fn
from pymc.model import modelcontext
from pymc.util import get_default_varnames, get_var_name
from pymc.vartypes import discrete_types, typefilter

__all__ = ["find_MAP"]


def find_MAP(
    start=None,
    vars=None,
    method="L-BFGS-B",
    return_raw=False,
    include_transformed=True,
    progressbar=True,
    maxeval=5000,
    model=None,
    *args,
    seed: Optional[int] = None,
    **kwargs
):
    """Finds the local maximum a posteriori point given a model.

    `find_MAP` should not be used to initialize the NUTS sampler. Simply call
    ``pymc.sample()`` and it will automatically initialize NUTS in a better
    way.

    Parameters
    ----------
    start: `dict` of parameter values (Defaults to `model.initial_point`)
    vars: list
        List of variables to optimize and set to optimum (Defaults to all continuous).
    method: string or callable
        Optimization algorithm (Defaults to 'L-BFGS-B' unless
        discrete variables are specified in `vars`, then
        `Powell` which will perform better).  For instructions on use of a callable,
        refer to SciPy's documentation of `optimize.minimize`.
    return_raw: bool
        Whether to return the full output of scipy.optimize.minimize (Defaults to `False`)
    include_transformed: bool, optional defaults to True
        Flag for reporting automatically transformed variables in addition
        to original variables.
    progressbar: bool, optional defaults to True
        Whether or not to display a progress bar in the command line.
    maxeval: int, optional, defaults to 5000
        The maximum number of times the posterior distribution is evaluated.
    model: Model (optional if in `with` context)
    *args, **kwargs
        Extra args passed to scipy.optimize.minimize

    Notes
    -----
    Older code examples used `find_MAP` to initialize the NUTS sampler,
    but this is not an effective way of choosing starting values for sampling.
    As a result, we have greatly enhanced the initialization of NUTS and
    wrapped it inside ``pymc.sample()`` and you should thus avoid this method.
    """
    model = modelcontext(model)

    if vars is None:
        vars = model.cont_vars
        if not vars:
            raise ValueError("Model has no unobserved continuous variables.")
    else:
        vars = [model.rvs_to_values.get(var, var) for var in vars]

    vars = inputvars(vars)
    disc_vars = list(typefilter(vars, discrete_types))
    allinmodel(vars, model)
    ipfn = make_initial_point_fn(
        model=model,
        jitter_rvs=set(),
        return_transformed=True,
        overrides=start,
    )
    if seed is None:
        seed = model.rng_seeder.randint(2**30, dtype=np.int64)
    start = ipfn(seed)
    model.check_start_vals(start)

    var_names = {var.name for var in vars}
    x0 = DictToArrayBijection.map(
        {var_name: value for var_name, value in start.items() if var_name in var_names}
    )

    # TODO: If the mapping is fixed, we can simply create graphs for the
    # mapping and avoid all this bijection overhead
    compiled_logp_func = DictToArrayBijection.mapf(model.compile_logp(jacobian=False), start)
    logp_func = lambda x: compiled_logp_func(RaveledVars(x, x0.point_map_info))

    rvs = [model.values_to_rvs[value] for value in vars]
    try:
        # This might be needed for calls to `dlogp_func`
        # start_map_info = tuple((v.name, v.shape, v.dtype) for v in vars)
        compiled_dlogp_func = DictToArrayBijection.mapf(
            model.compile_dlogp(rvs, jacobian=False), start
        )
        dlogp_func = lambda x: compiled_dlogp_func(RaveledVars(x, x0.point_map_info))
        compute_gradient = True
    except (AttributeError, NotImplementedError, tg.NullTypeGradError):
        compute_gradient = False

    if disc_vars or not compute_gradient:
        pm._log.warning(
            "Warning: gradient not available."
            + "(E.g. vars contains discrete variables). MAP "
            + "estimates may not be accurate for the default "
            + "parameters. Defaulting to non-gradient minimization "
            + "'Powell'."
        )
        method = "Powell"

    if compute_gradient:
        cost_func = CostFuncWrapper(maxeval, progressbar, logp_func, dlogp_func)
    else:
        cost_func = CostFuncWrapper(maxeval, progressbar, logp_func)

    try:
        opt_result = minimize(
            cost_func, x0.data, method=method, jac=compute_gradient, *args, **kwargs
        )
        mx0 = opt_result["x"]  # r -> opt_result
    except (KeyboardInterrupt, StopIteration) as e:
        mx0, opt_result = cost_func.previous_x, None
        if isinstance(e, StopIteration):
            pm._log.info(e)
    finally:
        last_v = cost_func.n_eval
        if progressbar:
            assert isinstance(cost_func.progress, ProgressBar)
            cost_func.progress.total = last_v
            cost_func.progress.update(last_v)
            print(file=sys.stdout)

    mx0 = RaveledVars(mx0, x0.point_map_info)
    unobserved_vars = get_default_varnames(model.unobserved_value_vars, include_transformed)
    unobserved_vars_values = model.compile_fn(unobserved_vars)(
        DictToArrayBijection.rmap(mx0, start)
    )
    mx = {var.name: value for var, value in zip(unobserved_vars, unobserved_vars_values)}

    if return_raw:
        return mx, opt_result
    else:
        return mx


def allfinite(x):
    return np.all(isfinite(x))


def nan_to_high(x):
    return np.where(isfinite(x), x, 1.0e100)


def allinmodel(vars, model):
    notin = [v for v in vars if v not in model.value_vars]
    if notin:
        notin = list(map(get_var_name, notin))
        raise ValueError("Some variables not in the model: " + str(notin))


class CostFuncWrapper:
    def __init__(self, maxeval=5000, progressbar=True, logp_func=None, dlogp_func=None):
        self.n_eval = 0
        self.maxeval = maxeval
        self.logp_func = logp_func
        if dlogp_func is None:
            self.use_gradient = False
            self.desc = "logp = {:,.5g}"
        else:
            self.dlogp_func = dlogp_func
            self.use_gradient = True
            self.desc = "logp = {:,.5g}, ||grad|| = {:,.5g}"
        self.previous_x = None
        self.progressbar = progressbar
        if progressbar:
            self.progress = progress_bar(range(maxeval), total=maxeval, display=progressbar)
            self.progress.update(0)
        else:
            self.progress = range(maxeval)

    def __call__(self, x):
        neg_value = np.float64(self.logp_func(pm.floatX(x)))
        value = -1.0 * nan_to_high(neg_value)
        if self.use_gradient:
            neg_grad = self.dlogp_func(pm.floatX(x))
            if np.all(np.isfinite(neg_grad)):
                self.previous_x = x
            grad = nan_to_num(-1.0 * neg_grad)
            grad = grad.astype(np.float64)
        else:
            self.previous_x = x
            grad = None

        if self.n_eval % 10 == 0:
            self.update_progress_desc(neg_value, grad)

        if self.n_eval > self.maxeval:
            self.update_progress_desc(neg_value, grad)
            raise StopIteration

        self.n_eval += 1
        if self.progressbar:
            assert isinstance(self.progress, ProgressBar)
            self.progress.update_bar(self.n_eval)

        if self.use_gradient:
            return value, grad
        else:
            return value

    def update_progress_desc(self, neg_value: float, grad: np.float64 = None) -> None:
        if self.progressbar:
            if grad is None:
                self.progress.comment = self.desc.format(neg_value)
            else:
                norm_grad = np.linalg.norm(grad)
                self.progress.comment = self.desc.format(neg_value, norm_grad)
