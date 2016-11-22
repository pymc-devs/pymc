from .quadpotential import quad_potential
from .arraystep import ArrayStepShared, ArrayStep, SamplerHist, Competence
from ..model import modelcontext, Point
from ..vartypes import continuous_types
from numpy import exp, log, array
from numpy.random import uniform
from .hmc import leapfrog, Hamiltonian, bern, energy
from ..tuning import guess_scaling
import theano
from ..theanof import (make_shared_replacements, join_nonshared_inputs, CallableTensor,
                       gradient, inputvars)
import theano.tensor as tt

__all__ = ['NUTS']


class NUTS(ArrayStepShared):
    """
    Automatically tunes step size and adjust number of steps for good performance.

    Implements "Algorithm 6: Efficient No-U-Turn Sampler with Dual Averaging" in:

    Hoffman, Matthew D., & Gelman, Andrew. (2011).
    The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.
    """
    default_blocked = True

    def __init__(self, vars=None, scaling=None, step_scale=0.25, is_cov=False, state=None,
                 Emax=1000,
                 target_accept=0.8,
                 gamma=0.05,
                 k=0.75,
                 t0=10,
                 model=None,
                 profile=False, **kwargs):
        """
        Parameters
        ----------
            vars : list of Theano variables, default continuous vars
            scaling : array_like, ndim = {1,2} or point
                Scaling for momentum distribution. 1d arrays interpreted matrix diagonal.
            step_scale : float, default=.25
                Size of steps to take, automatically scaled down by 1/n**(1/4)
            is_cov : bool, default=False
                Treat C as a covariance matrix/vector if True, else treat it as a precision matrix/vector
            state
                state to start from
            Emax : float, default 1000
                maximum energy
            target_accept : float (0,1) default .8
                target for avg accept probability between final branch and initial position
            gamma : float, default .05
            k : float (.5,1) default .75
                scaling of speed of adaptation
            t0 : int, default 10
                slows inital adapatation
            model : Model
            profile : bool or ProfileStats
                sets the functions to be profiled
        """
        model = modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = inputvars(vars)

        if scaling is None:
            scaling = model.test_point

        if isinstance(scaling, dict):
            scaling = guess_scaling(
                Point(scaling, model=model), model=model, vars=vars)

        n = scaling.shape[0]

        self.step_size = step_scale / n**(1 / 4.)

        self.potential = quad_potential(scaling, is_cov, as_cov=False)

        if state is None:
            state = SamplerHist()
        self.state = state
        self.Emax = Emax

        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.k = k

        self.Hbar = 0
        self.u = log(self.step_size * 10)
        self.m = 1

        shared = make_shared_replacements(vars, model)

        def create_hamiltonian(vars, shared, model):
            dlogp = gradient(model.logpt, vars)
            (logp, dlogp), q = join_nonshared_inputs(
                [model.logpt, dlogp], vars, shared)
            logp = CallableTensor(logp)
            dlogp = CallableTensor(dlogp)

            return Hamiltonian(logp, dlogp, self.potential), q

        def create_energy_func(q):
            p = tt.dvector('p')
            p.tag.test_value = q.tag.test_value
            E0 = energy(self.H, q, p)
            E0_func = theano.function([q, p], E0)
            E0_func.trust_input = True

            return E0_func

        self.H, q = create_hamiltonian(vars, shared, model)
        self.compute_energy = create_energy_func(q)

        self.leapfrog1_dE = leapfrog1_dE(self.H, q, profile=profile)

        super(NUTS, self).__init__(vars, shared, **kwargs)

    def astep(self, q0):
        leapfrog = self.leapfrog1_dE
        Emax = self.Emax
        e = self.step_size

        p0 = self.potential.random()
        E0 = self.compute_energy(q0, p0)

        u = uniform()
        q = qn = qp = q0
        p = pn = pp = p0

        n, s, j = 1, 1, 0

        while s == 1:
            v = bern(.5) * 2 - 1

            if v == -1:
                qn, pn, _, _, q1, n1, s1, a, na = buildtree(
                    leapfrog, qn, pn, u, v, j, e, Emax, E0)
            else:
                _, _, qp, pp, q1, n1, s1, a, na = buildtree(
                    leapfrog, qp, pp, u, v, j, e, Emax, E0)

            if s1 == 1 and bern(min(1, n1 * 1. / n)):
                q = q1

            n = n + n1

            span = qp - qn
            s = s1 * (span.dot(pn) >= 0) * (span.dot(pp) >= 0)
            j = j + 1

        p = -p

        w = 1. / (self.m + self.t0)
        self.Hbar = (1 - w) * self.Hbar + w * \
            (self.target_accept - a * 1. / na)

        self.step_size = exp(self.u - (self.m**self.k / self.gamma) * self.Hbar)
        self.m += 1

        return q

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


def buildtree(leapfrog1_dE, q, p, u, v, j, e, Emax, E0):
    if j == 0:
        q1, p1, dE = leapfrog1_dE(q, p, array(v * e), E0)

        n1 = int(log(u) + dE <= 0)
        s1 = int(log(u) + dE < Emax)
        return q1, p1, q1, p1, q1, n1, s1, min(1, exp(-dE)), 1
    else:
        qn, pn, qp, pp, q1, n1, s1, a1, na1 = buildtree(
            leapfrog1_dE, q, p, u, v, j - 1, e, Emax, E0)
        if s1 == 1:
            if v == -1:
                qn, pn, _, _, q11, n11, s11, a11, na11 = buildtree(
                    leapfrog1_dE, qn, pn, u, v, j - 1, e, Emax, E0)
            else:
                _, _, qp, pp, q11, n11, s11, a11, na11 = buildtree(
                    leapfrog1_dE, qp, pp, u, v, j - 1, e, Emax, E0)

            if bern(n11 * 1. / (max(n1 + n11, 1))):
                q1 = q11

            a1 = a1 + a11
            na1 = na1 + na11

            span = qp - qn
            s1 = s11 * (span.dot(pn) >= 0) * (span.dot(pp) >= 0)
            n1 = n1 + n11
        return qn, pn, qp, pp, q1, n1, s1, a1, na1
    return


def leapfrog1_dE(H, q, profile):
    """Computes a theano function that computes one leapfrog step and the energy difference between the beginning and end of the trajectory.
    Parameters
    ----------
    H : Hamiltonian
    q : theano.tensor
    profile : Boolean

    Returns
    -------
    theano function which returns
    q_new, p_new, dE
    """
    p = tt.dvector('p')
    p.tag.test_value = q.tag.test_value

    e = tt.dscalar('e')
    e.tag.test_value = 1

    q1, p1 = leapfrog(H, q, p, 1, e)
    E = energy(H, q1, p1)

    E0 = tt.dscalar('E0')
    E0.tag.test_value = 1

    dE = E - E0

    f = theano.function([q, p, e, E0], [q1, p1, dE], profile=profile)
    f.trust_input = True
    return f
