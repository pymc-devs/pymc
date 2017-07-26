import numpy as np

from pymc3.model import modelcontext, Point
from pymc3.step_methods import arraystep
from pymc3.step_methods.hmc import integration
from pymc3.theanof import inputvars, floatX
from pymc3.tuning import guess_scaling
from .quadpotential import quad_potential, QuadPotentialDiagAdapt


class BaseHMC(arraystep.GradientSharedStep):
    """Superclass to implement Hamiltonian/hybrid monte carlo."""

    default_blocked = True

    def __init__(self, vars=None, scaling=None, step_scale=0.25, is_cov=False,
                 model=None, blocked=True, potential=None,
                 integrator="leapfrog", dtype=None, **theano_kwargs):
        """Set up Hamiltonian samplers with common structures.

        Parameters
        ----------
        vars : list of theano variables
        scaling : array_like, ndim = {1,2}
            Scaling for momentum distribution. 1d arrays interpreted matrix
            diagonal.
        step_scale : float, default=0.25
            Size of steps to take, automatically scaled down by 1/n**(1/4)
        is_cov : bool, default=False
            Treat scaling as a covariance matrix/vector if True, else treat
            it as a precision matrix/vector
        model : pymc3 Model instance
        blocked: bool, default=True
        potential : Potential, optional
            An object that represents the Hamiltonian with methods `velocity`,
            `energy`, and `random` methods.
        **theano_kwargs: passed to theano functions
        """
        model = modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = inputvars(vars)

        super(BaseHMC, self).__init__(vars, blocked=blocked, model=model,
                                      dtype=dtype, **theano_kwargs)

        size = self._logp_dlogp_func.size

        if scaling is None and potential is None:
            mean = floatX(np.zeros(size))
            var = floatX(np.ones(size))
            potential = QuadPotentialDiagAdapt(size, mean, var, 10)

        if isinstance(scaling, dict):
            point = Point(scaling, model=model)
            scaling = guess_scaling(point, model=model, vars=vars)

        if scaling is not None and potential is not None:
            raise ValueError("Can not specify both potential and scaling.")

        self.step_size = step_scale / (size ** 0.25)
        if potential is not None:
            self.potential = potential
        else:
            self.potential = quad_potential(scaling, is_cov)

        self.integrator = integration.CpuLeapfrogIntegrator(
            size, self.potential, self._logp_dlogp_func)
