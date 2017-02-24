import numpy as np
import pymc3 as pm
import theano
from pymc3 import Point

from .arraystep import BlockedStep
from ..blocking import ArrayOrdering, DictToArrayBijection
from ..model import modelcontext
from ..theanof import inputvars, make_shared_replacements


class ParticleStep(BlockedStep):
    """
    Defined as a controller for a population of walkers/particles which all have individual parameter/logp values.
    This implementation assumes nothing about what is happening in the actual proprietary `astep`
    To use a ParticleStep, write an astep method to except and return a 1d array of variables.
    The higher level `step` method will control where the variables go and what reshapes to perform
    """

    def __init__(self, vars=None, model=None, **kwargs):
        super(ParticleStep, self).__init__()
        model = modelcontext(model)
        if vars is not None and len(vars) != len(model.vars):
            raise NotImplementedError(
                'ParticleStep sampling is not yet implemented along with other sampling techniques!')
        vars = model.vars
        vars = inputvars(vars)
        shared = make_shared_replacements(vars, model)
        self.model = model
        self.vars = vars

        self.dim = sum([i.dsize for i in self.vars])
        self.shared = {str(var): shared for var, shared in shared.items()}
        self.blocked = True
        self.t_func = logp(model.logpt, vars, shared)

    @property
    def ordering(self):
        return ArrayOrdering(self.vars, self.nparticles)

    def astep(self, point_array):
        raise NotImplementedError("This is a ParticleStep template, it doesn't do anything!")

    def step(self, point):
        for var, share in self.shared.items():
            share.container.storage[0] = point[var]

        bij = DictToArrayBijection(self.ordering, point)

        if self.generates_stats:
            apoint, stats = self.astep(bij.map(point))
            return bij.rmap(apoint), stats
        else:
            apoint = self.astep(bij.map(point))
            return bij.rmap(apoint)


def logp(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)
    tensor_type = inarray0.type
    f = theano.function([inarray0], logp0)
    f.trust_input = True
    return f


def transform_start_particles(start, nparticles, model=None):
    """
    Parameters
    ----------
    start : list, dict
        Accepts 3 data types:
        * dict of start positions with variable shape of (nparticles, var.dshape)
        * dict of start positions with variable shape (var.dshape)
        * list of dicts of length nparticles each with start positions of shape (var.dshape)
    nparticles : int
    model : Model (optional if in `with` context)

    Returns
    -------
     transformed_start : a dict each with start positions of shape (nparticles, var.dshape)
    """
    if start is None:
        return {}
    dshapes = {i.name: i.dshape for i in model.vars}
    dshapes.update({i.name: i.transformed.dshape for i in model.deterministics if hasattr(i, 'transformed')})
    if isinstance(start, dict):
        start = {k: v for k,v in start.items() if k in dshapes}
        if all(dshapes[k] == np.asarray(v).shape for k, v in start.items()):
            if nparticles is not None:
                start = {k: np.asarray([v]*nparticles) for k, v in start.items()}  # duplicate
            return Point(**start)
        else:
            extra = tuple() if nparticles is None else (nparticles, )
            if all(extra+dshapes[k] == np.asarray(v).shape for k, v in start.items()):
                return Point(**start)
            else:
                raise TypeError("Start dicts must have a shape of (nparticles, dshape) or (dshape,)")
    elif isinstance(start, list):
        start = [{k: v for k, v in s.items() if k in dshapes} for s in start]
        assert len(start) == nparticles, "If start is a list, it must have a length of nparticles"
        assert all(isinstance(i, dict) for i in start), "Start dicts must have a shape of (dshape,)"
        assert all(s.keys() == start[0].keys() for s in start), "All start positions must have the same variables"
        d = {}
        for varname, varshape in dshapes.items():
            if varname in start[0].keys():
                assert all(varshape == s[varname].shape for s in start), "Start dicts must have a shape of (dshape,)"
                d[varname] = np.asarray([s[varname] for s in start])
        return Point(**d)
    raise TypeError("Start must be a dict or a list of dicts")