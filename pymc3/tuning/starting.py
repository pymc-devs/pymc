'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy import optimize
import numpy as np
from numpy import isfinite, nan_to_num, logical_not
import pymc3 as pm
from ..vartypes import discrete_types, typefilter
from ..model import modelcontext, Point
from ..theanof import inputvars
from ..blocking import DictToArrayBijection, ArrayOrdering

from inspect import getargspec

__all__ = ['find_MAP']


def find_MAP(start=None, vars=None, fmin=None,
             return_raw=False,model=None, *args, **kwargs):
    """
    Sets state to the local maximum a posteriori point given a model.
    Current default of fmin_Hessian does not deal well with optimizing close
    to sharp edges, especially if they are the minimum.

    Parameters
    ----------
    start : `dict` of parameter values (Defaults to `model.test_point`)
    vars : list
        List of variables to set to MAP point (Defaults to all continuous).
    fmin : function
        Optimization algorithm (Defaults to `scipy.optimize.fmin_bfgs` unless
        discrete variables are specified in `vars`, then
        `scipy.optimize.fmin_powell` which will perform better).
    return_raw : Bool
        Whether to return extra value returned by fmin (Defaults to `False`)
    model : Model (optional if in `with` context)
    *args, **kwargs
        Extra args passed to fmin
    """
    model = modelcontext(model)
    if start is None:
        start = model.test_point

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

    if disc_vars or not gradient_avail :
        pm._log.warning("Warning: gradient not available." +
                        "(E.g. vars contains discrete variables). MAP " +
                        "estimates may not be accurate for the default " +
                        "parameters. Defaulting to non-gradient minimization " +
                        "fmin_powell.")
        fmin = optimize.fmin_powell

    if fmin is None:
        if disc_vars:
            fmin = optimize.fmin_powell
        else:
            fmin = optimize.fmin_bfgs

    allinmodel(vars, model)

    start = Point(start, model=model)
    bij = DictToArrayBijection(ArrayOrdering(vars), start)

    logp = bij.mapf(model.fastlogp)
    def logp_o(point):
        return nan_to_high(-logp(point))

    # Check to see if minimization function actually uses the gradient
    if 'fprime' in getargspec(fmin).args:
        dlogp = bij.mapf(model.fastdlogp(vars))
        def grad_logp_o(point):
            return nan_to_num(-dlogp(point))

        r = fmin(logp_o, bij.map(
            start), fprime=grad_logp_o, *args, **kwargs)
        compute_gradient = True
    else:
        compute_gradient = False

        # Check to see if minimization function uses a starting value
        if 'x0' in getargspec(fmin).args:
            r = fmin(logp_o, bij.map(start), *args, **kwargs)
        else:
            r = fmin(logp_o, *args, **kwargs)

    if isinstance(r, tuple):
        mx0 = r[0]
    else:
        mx0 = r

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
    mx = {v.name: mx[v.name].astype(v.dtype) for v in model.vars}

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
