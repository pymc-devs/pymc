from ..arraystep import ArrayStepShared
from .trajectory import get_theano_hamiltonian_functions

from pymc3.tuning import guess_scaling
from pymc3.model import modelcontext, Point
from .quadpotential import quad_potential
from pymc3.theanof import inputvars, make_shared_replacements


class BaseHMC(ArrayStepShared):
    default_blocked = True

    def __init__(self, vars=None, scaling=None, step_scale=0.25, is_cov=False,
                 model=None, blocked=True, use_single_leapfrog=False,
                 potential=None, integrator="leapfrog", **theano_kwargs):
        """Superclass to implement Hamiltonian/hybrid monte carlo

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
        model : pymc3 Model instance.  default=Context model
        blocked: Boolean, default True
        use_single_leapfrog: Boolean, will leapfrog steps take a single step at a time.
            default False.
        potential : Potential, optional
            An object that represents the Hamiltonian with methods `velocity`,
            `energy`, and `random` methods.
        **theano_kwargs: passed to theano functions
        """
        model = modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = inputvars(vars)

        if scaling is None and potential is None:
            scaling = model.test_point

        if isinstance(scaling, dict):
            scaling = guess_scaling(Point(scaling, model=model), model=model, vars=vars)

        if scaling is not None and potential is not None:
            raise ValueError("Can not specify both potential and scaling.")

        self.step_size = step_scale / (model.ndim ** 0.25)
        if potential is not None:
            self.potential = potential
        else:
            self.potential = quad_potential(scaling, is_cov, as_cov=False)

        shared = make_shared_replacements(vars, model)
        if theano_kwargs is None:
            theano_kwargs = {}

        self.H, self.compute_energy, self.compute_velocity, self.leapfrog, self.dlogp = get_theano_hamiltonian_functions(
            vars, shared, model.logpt, self.potential, use_single_leapfrog, integrator, **theano_kwargs)

        super(BaseHMC, self).__init__(vars, shared, blocked=blocked)
