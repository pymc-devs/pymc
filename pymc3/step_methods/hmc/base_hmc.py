from ..arraystep import ArrayStepShared
from .trajectory import get_theano_hamiltonian_functions

from pymc3.tuning import guess_scaling
from pymc3.model import modelcontext, Point
from .quadpotential import quad_potential
from pymc3.theanof import inputvars, make_shared_replacements


class BaseHMC(ArrayStepShared):
    default_blocked = True

    def __init__(self, vars=None, scaling=None, step_scale=0.25, is_cov=False,
                 model=None, blocked=True, use_single_leapfrog=False, **theano_kwargs):
        """
        Parameters
        ----------
            vars : list of theano variables
            scaling : array_like, ndim = {1,2}
                Scaling for momentum distribution. 1d arrays interpreted matrix diagonal.
            step_scale : float, default=0.25
                Size of steps to take, automatically scaled down by 1/n**(1/4)
            is_cov : bool, default=False
                Treat scaling as a covariance matrix/vector if True, else treat it as a
                precision matrix/vector
            state
                State object
            model : pymc3 Model instance.  default=Context model
            blocked: Boolean, default True
            use_single_leapfrog: Boolean, will leapfrog steps take a single step at a time.
                default False.
            **theano_kwargs: passed to theano functions
        """
        model = modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = inputvars(vars)

        if scaling is None:
            scaling = model.test_point

        if isinstance(scaling, dict):
            scaling = guess_scaling(Point(scaling, model=model), model=model, vars=vars)

        n = scaling.shape[0]
        self.step_size = step_scale / (n ** 0.25)
        self.potential = quad_potential(scaling, is_cov, as_cov=False)

        shared = make_shared_replacements(vars, model)
        if theano_kwargs is None:
            theano_kwargs = {}

        self.H, self.compute_energy, self.leapfrog, self._vars = get_theano_hamiltonian_functions(
            vars, shared, model.logpt, self.potential, use_single_leapfrog, **theano_kwargs)

        super(BaseHMC, self).__init__(vars, shared, blocked=blocked)
