'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy import optimize
import numpy as np
from numpy import isfinite, nan_to_num, logical_not
import pymc3 as pm
import time
from ..vartypes import discrete_types, typefilter
from ..model import modelcontext, Point
from ..theanof import inputvars
from ..blocking import DictToArrayBijection, ArrayOrdering

from inspect import getargspec

__all__ = ['find_MAP']

def find_MAP(start=None, vars=None, fmin=None,
             return_raw=False, model=None, disp=True, callback=None, *args, **kwargs):
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

        if callback is None:
            callback = Monitor(bij, logp_o, grad_logp_o, disp)

        r = fmin(logp_o, bij.map(start), fprime=grad_logp_o, callback=callback, *args, **kwargs)
        compute_gradient = True
    else:
        if callback is None:
            callback = Monitor(bij, logp_o, dlogp=None, disp=disp)

        # Check to see if minimization function uses a starting value
        if 'x0' in getargspec(fmin).args:
            r = fmin(logp_o, bij.map(start), callback=callback, *args, **kwargs)
        else:
            r = fmin(logp_o, callback=callback, *args, **kwargs)
        compute_gradient = False

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



class Monitor(object):
    def __init__(self, bij, logp, dlogp=None, disp=True):
        self.iters = 0
        self.disp = disp
        if disp:
            try:
                from IPython.display import display
                from ipywidgets import HTML, VBox, HBox, FlexBox
                self.prog_table  = HTML(width='100%')
                self.param_table = HTML(width='100%')
                r_col = VBox(children=[self.param_table], padding=3, width='100%')
                l_col = HBox(children=[self.prog_table],  padding=3, width='25%')
                self.hor_align = FlexBox(children = [l_col, r_col], width='100%', orientation='vertical')
                display(self.hor_align)
                self.using_notebook = True
                self.update_interval = 1
            except:
                self.using_notebook = False
                self.update_interval = 2

            self.bij = bij
            self.logp = logp
            self.dlogp = dlogp

            self.t_initial = time.time()
            self.t0 = self.t_initial
            self.paramtable = {}

    def __call__(self, x):
        self.iters += 1
        if not self.disp:
            return
        if self.iters == 1 and not self.using_notebook:
            pm._log.info("iter   runtime   lpost      ||grad||    ")
        if time.time() - self.t0 > self.update_interval:
            self._update_progtable(x)
            self._update_paramtable(x)
            if self.using_notebook:
                self._display_notebook()
            else:
                self._display_terminal()
            self.t0 = time.time()

    def _update_progtable(self, x):
        s = time.time() - self.t_initial
        hours, remainder = divmod(int(s), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.t_elapsed = "{:2d}h{:2d}m{:2d}s".format(hours, minutes, seconds)
        self.logpost = np.float(self.logp(x))
        self.dlogpost = np.linalg.norm(self.dlogp(x))

    def _update_paramtable(self, x):
        names_vals = self.bij.rmap(x)
        for parameter, val in names_vals.items():
            if val.size == 1:
                 valstr = "{:.3f}".format(np.float(val))
            elif val.size < 9:
                 valstr = np.array2string(val, suppress_small=True, separator=", ",
                                          formatter={'float_kind': lambda x: "{:.3f}".format(x)})
            else:
                 valstr_beg = np.array2string(val[:4],  suppress_small=True, separator=", ",
                                          formatter={'float_kind': lambda x: "{:.3f}".format(x)})[:-1]
                 valstr_end = np.array2string(val[-4:], suppress_small=True, separator=", ",
                                          formatter={'float_kind': lambda x: "{:.3f}".format(x)})[1:]
                 valstr = valstr_beg + ", ..., " + valstr_end
            self.paramtable[parameter] = {"size": val.size, "valstr": valstr}

    def _display_terminal(self):
        disp_str = "{:<5d} {:<s}  {:<9.3f}  {:<12.3f}".format(self.iters, self.t_elapsed, self.logpost, self.dlogpost)
        pm._log.info(disp_str)

    def _display_notebook(self):
        ## Progress table
        html = r"""<style type="text/css">
        table { border-collapse:collapse }
        .tg {border-collapse:collapse;border-spacing:0;border:none;}
        .tg td{font-family:Arial, sans-serif;font-size:14px;padding:3px 3px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;}
        .tg th{Impact, Charcoal, sans-serif;font-size:13px;font-weight:bold;padding:3px 3px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal; background-color:#0E688A;color:#ffffff;}
        .tg .tg-vkoh{font-weight:normal;font-family:"Lucida Console", Monaco, monospace !important; background-color:#ffffff;color:#000000}
        .tg .tg-suao{font-weight:bold;font-family:"Lucida Console", Monaco, monospace !important;background-color:#0E688A;color:#ffffff;}
        """
        html += r"""
        </style>
        <table class="tg" style="undefined;">
           <col width="200px" />
           <tr>
             <th class= "tg-vkoh">Time Elapsed: {}</th>
           </tr>
           <tr>
             <th class= "tg-vkoh">Iteration: {}</th>
           </tr>
           <tr>
             <th class= "tg-vkoh">Log Posterior: {:.3f}</th>
           </tr>
        """.format(self.t_elapsed, self.iters, self.logpost)
        if self.dlogp is not None:
           html += r"""
             <tr>
               <th class= "tg-vkoh">||grad||: {:.3f}</th>
             </tr>""".format(self.dlogpost)
        html += "</table>"
        self.prog_table.value = html
        ## Parameter table
        html = r"""<style type="text/css">
          .tg .tg-bgft{font-weight:normal;font-family:"Lucida Console", Monaco, monospace !important;background-color:#0E688A;color:#ffffff;}
          .tg td{font-family:Arial, sans-serif;font-size:12px;padding:3px 3px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#504A4E;color:#333;background-color:#fff;word-wrap: break-word;}
          .tg th{Impact, Charcoal, sans-serif;font-size:13px;font-weight:bold;padding:3px 3px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#504A4E;background-color:#0E688A;color:#ffffff;}
          </style>
          <table class="tg" style="undefined;">
             <col width="130px" />
             <col width="50px" />
             <col width="600px" />
             <tr>
               <th class="tg">Parameter</th>
               <th class="tg">Size</th>
               <th class="tg">Current Value</th>
             </tr>
           """
        for parameter, values in self.paramtable.items():
            html += r"""
              <tr>
                <td class="tg-bgft">{}</td>
                <td class="tg-vkoh">{}</td>
                <td class="tg-vkoh">{}</td>
              </tr>
            """.format(parameter, values["size"], values["valstr"])
        html += "</table>"
        self.param_table.value = html
