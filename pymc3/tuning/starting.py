'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy.optimize import minimize
import numpy as np
from numpy import isfinite, nan_to_num
from tqdm import tqdm
import pymc3 as pm
from ..vartypes import discrete_types, typefilter
from ..model import modelcontext, Point
from ..theanof import inputvars
import theano.gradient as tg
from ..blocking import DictToArrayBijection, ArrayOrdering
from ..util import update_start_vals, get_default_varnames

import warnings
from inspect import getargspec

__all__ = ['find_MAP']


def find_MAP(start=None, vars=None, method="L-BFGS-B",
             return_raw=False, include_transformed=True, progressbar=True, maxeval=5000, model=None,
             *args, **kwargs):
    """
    Finds the local maximum a posteriori point given a model.

    Parameters
    ----------
    start : `dict` of parameter values (Defaults to `model.test_point`)
    vars : list
        List of variables to optimize and set to optimum (Defaults to all continuous).
    method : string or callable
        Optimization algorithm (Defaults to 'L-BFGS-B' unless
        discrete variables are specified in `vars`, then
        `Powell` which will perform better).  For instructions on use of a callable,
        refer to SciPy's documentation of `optimize.minimize`.
    return_raw : bool
        Whether to return the full output of scipy.optimize.minimize (Defaults to `False`)
    include_transformed : bool
        Flag for reporting automatically transformed variables in addition
        to original variables (defaults to False).
    progressbar : bool
        Whether or not to display a progress bar in the command line.
    maxeval : int
        The maximum number of times the posterior distribution is evaluated.
    model : Model (optional if in `with` context)
    *args, **kwargs
        Extra args passed to scipy.optimize.minimize

    Notes
    -----
    Older code examples used find_MAP() to initialize the NUTS sampler,
    this turned out to be a rather inefficient method.
    Since then, we have greatly enhanced the initialization of NUTS and
    wrapped it inside pymc3.sample() and you should thus avoid this method.
    """

    warnings.warn('find_MAP should not be used to initialize the NUTS sampler, simply call pymc3.sample() and it will automatically initialize NUTS in a better way.')

    model = modelcontext(model)
    if start is None:
        start = model.test_point
    else:
        update_start_vals(start, model.test_point, model)

    if not set(start.keys()).issubset(model.named_vars.keys()):
        extra_keys = ', '.join(set(start.keys()) - set(model.named_vars.keys()))
        valid_keys = ', '.join(model.named_vars.keys())
        raise KeyError('Some start parameters do not appear in the model!\n'
                       'Valid keys are: {}, but {} was supplied'.format(valid_keys, extra_keys))

    if vars is None:
        vars = model.cont_vars
    vars = inputvars(vars)
    disc_vars = list(typefilter(vars, discrete_types))
    allinmodel(vars, model)

    start = Point(start, model=model)
    bij = DictToArrayBijection(ArrayOrdering(vars), start)
    logp_func = bij.mapf(model.fastlogp_nojac)
    x0 = bij.map(start)

    try:
        dlogp_func = bij.mapf(model.fastdlogp_nojac(vars))
        compute_gradient = True
    except (AttributeError, NotImplementedError, tg.NullTypeGradError):
        compute_gradient = False

    if disc_vars or not compute_gradient:
        pm._log.warning("Warning: gradient not available." +
                        "(E.g. vars contains discrete variables). MAP " +
                        "estimates may not be accurate for the default " +
                        "parameters. Defaulting to non-gradient minimization " +
                        "'Powell'.")
        method = "Powell"

    if "fmin" in kwargs:
        fmin = kwargs.pop("fmin")
        warnings.warn('In future versions, set the optimization algorithm with a string. '
                      'For example, use `method="L-BFGS-B"` instead of '
                      '`fmin=sp.optimize.fmin_l_bfgs_b"`.')

        cost_func = CostFuncWrapper(maxeval, progressbar, logp_func)

        # Check to see if minimization function actually uses the gradient
        if 'fprime' in getargspec(fmin).args:
            def grad_logp(point):
                return nan_to_num(-dlogp_func(point))
            opt_result = fmin(cost_func, bij.map(start), fprime=grad_logp, *args, **kwargs)
        else:
            # Check to see if minimization function uses a starting value
            if 'x0' in getargspec(fmin).args:
                opt_result = fmin(cost_func, bij.map(start), *args, **kwargs)
            else:
                opt_result = fmin(cost_func, *args, **kwargs)

        if isinstance(opt_result, tuple):
            mx0 = opt_result[0]
        else:
            mx0 = opt_result
    else:
        # remove 'if' part, keep just this 'else' block after version change
        if compute_gradient:
            cost_func = CostFuncWrapper(maxeval, progressbar, logp_func, dlogp_func)
        else:
            cost_func = CostFuncWrapper(maxeval, progressbar, logp_func)

        try:
            opt_result = minimize(cost_func, x0, method=method, jac=compute_gradient, *args, **kwargs)
            mx0 = opt_result["x"]  # r -> opt_result
            cost_func.progress.total = cost_func.progress.n + 1
            cost_func.progress.update()
        except (KeyboardInterrupt, StopIteration) as e:
            mx0, opt_result = cost_func.previous_x, None
            cost_func.progress.close()
            if isinstance(e, StopIteration):
                pm._log.info(e)
        finally:
            cost_func.progress.close()

    vars = get_default_varnames(model.unobserved_RVs, include_transformed)
    mx = {var.name: value for var, value in zip(vars, model.fastfn(vars)(bij.rmap(mx0)))}

    if return_raw:
        return mx, opt_result
    else:
        return mx


def allfinite(x):
    return np.all(isfinite(x))


def nan_to_high(x):
    return np.where(isfinite(x), x, 1.0e100)


def allinmodel(vars, model):
    notin = [v for v in vars if v not in model.vars]
    if notin:
        raise ValueError("Some variables not in the model: " + str(notin))


class CostFuncWrapper:
    def __init__(self, maxeval=5000, progressbar=True, logp_func=None, dlogp_func=None):
        self.n_eval = 0
        self.maxeval = maxeval
        self.logp_func = logp_func
        if dlogp_func is None:
            self.use_gradient = False
            self.desc = 'logp = {:,.5g}'
        else:
            self.dlogp_func = dlogp_func
            self.use_gradient = True
            self.desc = 'logp = {:,.5g}, ||grad|| = {:,.5g}'
        self.previous_x = None
        self.progress = tqdm(total=maxeval, disable=not progressbar)
        self.progress.n = 0

    def __call__(self, x):
        neg_value = np.float64(self.logp_func(pm.floatX(x)))
        value = -1.0 * nan_to_high(neg_value)
        if self.use_gradient:
            neg_grad = self.dlogp_func(pm.floatX(x))
            if np.all(np.isfinite(neg_grad)):
                self.previous_x = x
            grad = nan_to_num(-1.0*neg_grad)
            grad = grad.astype(np.float64)
        else:
            self.previous_x = x
            grad = None

        if self.n_eval % 10 == 0:
            self.update_progress_desc(neg_value, grad)

        if self.n_eval > self.maxeval:
            self.update_progress_desc(neg_value, grad)
            self.progress.close()
            raise StopIteration

        self.n_eval += 1
        self.progress.update(1)

        if self.use_gradient:
            return value, grad
        else:
            return value

    def update_progress_desc(self, neg_value, grad=None):
        if grad is None:
            self.progress.set_description(self.desc.format(neg_value))
        else:
            norm_grad = np.linalg.norm(grad)
            self.progress.set_description(self.desc.format(neg_value, norm_grad))
