'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
import numpy as np
from numpy import isfinite, nan_to_num, logical_not
import pymc3 as pm
import time
from tqdm import tqdm

from pymc3.vartypes import discrete_types, typefilter
from pymc3.model import modelcontext, Point
from pymc3.blocking import ArrayOrdering, DictToArrayBijection
from pymc3.theanof import inputvars, floatX
from pymc3.util import update_start_vals
from scipy.optimize import minimize


from inspect import getargspec

__all__ = ['find_MAP']


def find_MAP(start=None, vars=None, method=None, progressbar=True, return_raw=False,
             model=None, maxeval=50000, callback=None, *args, **kwargs):
    """
    Sets state to the local maximum a posteriori point given a model.
    Current default of fmin_Hessian does not deal well with optimizing close
    to sharp edges, especially if they are the minimum.

    Parameters
    ----------
    start : `dict` of parameter values (Defaults to `model.test_point`)
    vars : list
        List of variables to set to MAP point (Defaults to all continuous).
    method : string or callable
        Optimization algorithm (Defaults to `BFGS` unless
        discrete variables are specified in `vars`, then
        `Powell` which will perform better).
    progressbar : bool
        Whether or not to display a progress bar in the command line.
    return_raw : bool
        Whether to return extra values returned by fmin (Defaults to `False`)
    model : Model (optional if in `with` context)
    maxeval : int
        The maximum number of times the posterior distribution is evaluated.
    callback : callable
        Callback function to pass to scipy optimization routine.
    *args, **kwargs
        Extra args passed to fmin.
    """
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

    try:
        model.fastdlogp(vars)
        gradient_avail = True
    except AttributeError:
        gradient_avail = False

    if disc_vars or not gradient_avail:
        pm._log.warning("Warning: gradient not available." +
                        "(E.g. vars contains discrete variables). MAP " +
                        "estimates may not be accurate for the default " +
                        "parameters. Defaulting to non-gradient minimization " +
                        "'Powell'.")
        method = "Powell"

    if method is None:
        if disc_vars:
            method = "Powell"
        else:
            method = "BFGS"

    allinmodel(vars, model)

    start = Point(start, model=model)
    bij = DictToArrayBijection(ArrayOrdering(vars), start)
    logp_func = bij.mapf(model.fastlogp)
    x0 = bij.map(start)


    logp = bij.mapf(model.fastlogp_nojac)
    def logp_o(point):
        return nan_to_high(-logp(point))

    # Check to see if minimization function actually uses the gradient
    if method in ["CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC",
                  "SLSQP", "dogleg", "trust-ncg"]:

        dlogp = bij.mapf(model.fastdlogp_nojac(vars))
        def grad_logp_o(point):
            return nan_to_num(-dlogp(point))

        cost_func = CostFuncWrapper(maxeval, progressbar, logp, dlogp)
        compute_gradient = True
    else:
        cost_func = CostFuncWrapper(maxeval, progressbar, logp_func)
        compute_gradient = False

    try:
        r = minimize(cost_func, x0, method=method, jac=compute_gradient, *args, **kwargs)
        mx0 = r["x"]
    except (KeyboardInterrupt, StopIteration) as e:
        mx0, r = cost_func.previous_x, None
        cost_func.progress.close()
        if isinstance(e, StopIteration):
            pm._log.info(e)
    finally:
        cost_func.progress.close()
    mx = bij.rmap(mx0)

    allfinite_mx0 = allfinite(mx0)
    allfinite_logp = allfinite(model.logp(mx))
    if compute_gradient:
        allfinite_dlogp = allfinite(model.dlogp()(mx))
    else:
        allfinite_dlogp = True

    if (not allfinite_mx0 or
        not allfinite_logp or
        not allfinite_dlogp):

        messages = []
        for var in vars:
            vals = {
                "value": mx[var.name],
                "logp": var.logp(mx)}
            if compute_gradient:
                vals["dlogp"] = var.dlogp()(mx)

            def message(name, values):
                if np.size(values) < 10:
                    return name + " bad: " + str(values)
                else:
                    idx = np.nonzero(logical_not(isfinite(values)))
                    return name + " bad at idx: " + str(idx) + " with values: " + str(values[idx])

            messages += [
                message(var.name + "." + k, v)
                for k, v in vals.items()
                if not allfinite(v)]

        specific_errors = '\n'.join(messages)
        raise ValueError("Optimization error: max, logp or dlogp at " +
                         "max have non-finite values. Some values may be " +
                         "outside of distribution support. max: " +
                         repr(mx) + " logp: " + repr(model.logp(mx)) +
                         " dlogp: " + repr(model.dlogp()(mx)) + "Check that " +
                         "1) you don't have hierarchical parameters, " +
                         "these will lead to points with infinite " +
                         "density. 2) your distribution logp's are " +
                         "properly specified. Specific issues: \n" +
                         specific_errors)

    vars = model.unobserved_RVs
    mx = {var.name: value for var, value in zip(vars, model.fastfn(vars)(mx))}
    if return_raw:
        return mx, r
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


class CostFuncWrapper(object):
    def __init__(self, maxeval=5000, progressbar=True, logp_func=None, dlogp_func=None):
        self.t0 = time.time()
        self.n_eval = 0
        self.maxeval = maxeval
        self.logp_func = logp_func
        if dlogp_func is None:
            self.use_gradient = False
            self.desc = 'lp = {:,.5g}'
        else:
            self.dlogp_func = dlogp_func
            self.use_gradient = True
            self.desc = 'lp = {:,.5g}, ||grad|| = {:,.5g}'
        self.previous_x = None
        self.progress = tqdm(total=maxeval, disable=not progressbar)

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
