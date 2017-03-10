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

    @property
    def nparticles(self):
        if self._nparticles is None:
            raise AttributeError("nparticles has not been set")
        return self._nparticles

    @nparticles.setter
    def nparticles(self, value):
        assert value >= self.min_nparticles, "{} <= minimum required particles for {}".format(value, self)
        self._nparticles = int(value)
        self.ordering = ArrayOrdering(self.vars, self._nparticles)


def logp(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)
    f = theano.function([inarray0], logp0)
    f.trust_input = True
    return f

def transform_start_positions(start, nparticles, njobs, model=None):
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
     transformed_start : a list of dicts each with start positions of shape (nparticles, var.dshape)
    """
    assert nparticles is None or nparticles >= 1
    assert njobs >= 1
    if start is None:
        return [{}]
    model = modelcontext(model)
    dshapes = {i.name: i.dshape for i in model.vars}
    dshapes.update({i.name: i.transformed.dshape for i in model.deterministics if hasattr(i, 'transformed')})

    if isinstance(start, dict):
        start = {k: v for k,v in start.items() if k in dshapes}
        if all(dshapes[k] == np.asarray(v).shape for k, v in start.items()):
            if nparticles is not None:
                start = {k: np.asarray([v]*nparticles) for k, v in start.items()}  # duplicate
            return [Point(**start)]
        else:
            extra_particlesjobs = tuple() if nparticles is None else (nparticles*njobs,)
            extra_particlesjobs_sep = (njobs, ) if nparticles is None else (njobs, nparticles,)
            if all(extra_particlesjobs+dshapes[k] == np.asarray(v).shape for k, v in start.items()):
                return [Point(**{k: v[j:(j+1)*nparticles] for k, v in start.items()}) for j in range(njobs)]
            elif all(extra_particlesjobs_sep+dshapes[k] == np.asarray(v).shape for k, v in start.items()):
                return [Point(**{k: v[j] for k, v in start.items()}) for j in range(njobs)]
            else:
                raise TypeError("Start dicts must have a shape of (nparticles, dshape) or (dshape,)")
    elif isinstance(start, list):
        start = [{k: v for k, v in s.items() if k in dshapes} for s in start]
        assert all(isinstance(i, dict) for i in start)
        assert all(s.keys() == start[0].keys() for s in start), "All start positions must have the same variables"

        d = {}
        for varname, varshape in dshapes.items():
            if varname in start[0].keys():
                d[varname] = np.asarray([s[varname] for s in start])
        return transform_start_positions(d, nparticles, njobs, model)
    raise TypeError("Start must be a dict or a list of dicts")


def get_required_nparticles(*steps):
    try:
        return max(s.min_nparticles for s in steps if hasattr(s, 'min_nparticles'))
    except (ValueError, TypeError):
        return
