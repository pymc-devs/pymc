import pymc3 as pm
import theano
from ...blocking import VarMap, Compose
from ...step_methods.arraystep import ArrayStepShared
from theano.gradient import np

import emcee

LARGENUMBER = 100000


class EnsembleArrayOrdering(object):
    """
    An ordering for an array space
    """

    def __init__(self, vars, nparticles):
        self.vmap = []
        dim = 0

        for var in vars:
            slc = slice(dim, dim + var.dsize)
            self.vmap.append(VarMap(str(var), slc, var.dshape, var.dtype))
            dim += var.dsize

        self.dimensions = (nparticles, dim)

    @property
    def nparticles(self):
        return self.dimensions[0]


class EnsembleDictToArrayBijection(object):
    """
    A mapping between a dict space and an array space
    """

    def __init__(self, ordering, dpoint):
        self.ordering = ordering
        self.dpt = dpoint

        # determine smallest float dtype that will fit all data
        if all([x.dtyp == 'float16' for x in ordering.vmap]):
            self.array_dtype = 'float16'
        elif all([x.dtyp == 'float32' for x in ordering.vmap]):
            self.array_dtype = 'float32'
        else:
            self.array_dtype = 'float64'

    def map(self, dpt):
        """
        Maps value from dict space to array space

        Parameters
        ----------
        dpt : dict
        """
        apt = np.empty(self.ordering.dimensions, dtype=self.array_dtype)
        for var, slc, shp, dty in self.ordering.vmap:
            apt[:, slc] = dpt[var].reshape(self.ordering.nparticles, -1)
        return apt

    def rmap(self, apt):
        """
        Maps value from array space to dict space

        Parameters
        ----------
        apt : array
        """
        dpt = self.dpt.copy()

        for var, slc, shp, dtyp in self.ordering.vmap:
            dpt[var] = np.atleast_2d(apt)[:, slc].reshape(self.ordering.nparticles, *shp).astype(dtyp)

        return dpt

    def mapf(self, f):
        """
         function f : DictSpace -> T to ArraySpace -> T

        Parameters
        ----------

        f : dict -> T

        Returns
        -------
        f : array -> T
        """
        return Compose(f, self.rmap)


class ExternalEnsembleStepShared(ArrayStepShared):
    """Ensemble version of ArrayStepShared for exportation to external step functions"""

    def __init__(self, vars, shared, blocked=True, nparticles=None):
        """
        Parameters
        ----------
        vars : list of sampling variables
        shared : dict of theano variable -> shared variable
        blocked : Boolean (default True)
        """
        self.vars = vars
        self.dimensions = sum(v.dsize for v in vars)
        if nparticles is None:
            nparticles = self.dimensions * 2
        self.nparticles = nparticles
        self.ordering = EnsembleArrayOrdering(vars, nparticles)
        self.shared = {str(var): shared for var, shared in shared.items()}
        self.blocked = blocked

    def step(self, points):
        for var, share in self.shared.items():
            share.set_value(points[var])

        bij = EnsembleDictToArrayBijection(self.ordering, points)

        if self.generates_stats:
            apoints, stats = self.astep(bij.map(points))
            return bij.rmap(apoints), stats
        else:
            apoints = self.astep(bij.map(points))
            return bij.rmap(apoints)


def logpostfn(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)

    f = theano.function([inarray0], logp0)
    f.trust_input = True
    return f


class AffineInvariantEnsemble(ExternalEnsembleStepShared):
    """
    Affine Invariant ensemble sampling step

    Parameters
    ----------
    vars : list
        List of variables for sampler
    tune : bool
        Flag for tuning. Defaults to True.
    tune_interval : int
        The frequency of tuning. Defaults to 100 iterations.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode :  string or `Mode` instance.
        compilation mode passed to Theano functions
    """

    default_blocked = True
    generates_stats = False

    def __init__(self, vars=None, nparticles=None, tune=True, tune_interval=100, model=None, mode=None, **kwargs):

        model = pm.modelcontext(model)

        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)

        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        # Determine type of variables
        self.discrete = np.concatenate([[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in vars])
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        self.mode = mode

        shared = pm.make_shared_replacements(vars, model)
        self.logp = logpostfn(model.logpt, vars, shared)
        self.emcee_kwargs = kwargs.pop('emcee_kwargs', {})
        super(AffineInvariantEnsemble, self).__init__(vars, shared, nparticles=nparticles)
        self.sample_generator = None

    def setup_step(self, point_array):
        # very hacky way to make emcee keep going without `iterations` information
        self.emcee_sampler = emcee.EnsembleSampler(self.nparticles, self.dimensions, self.logp, **self.emcee_kwargs)
        self.sample_generator = self.emcee_sampler.sample(point_array, storechain=False, iterations=100000)

    def astep(self, point_array):
        if self.sample_generator is None:
            self.setup_step(point_array)
        try:
            q, lnprob, state = next(self.sample_generator)
        except StopIteration:
            self.sample_generator = None
            return self.astep(point_array)
        return q.astype(theano.config.floatX)
