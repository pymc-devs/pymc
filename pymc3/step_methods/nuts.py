from .quadpotential import quad_potential
from .arraystep import ArrayStepShared, SamplerHist, Competence
from ..model import modelcontext, Point
from ..vartypes import continuous_types
from .hmc import leapfrog, Hamiltonian, energy, bern
from ..tuning import guess_scaling
import numpy as np
import numpy.random as nr
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
                 max_energy=1000,
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
                Treat C as a covariance matrix/vector if True, else treat it as a
                precision matrix/vector
            state
                state to start from
            max_energy : float, default 1000
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
            scaling = guess_scaling(Point(scaling, model=model), model=model, vars=vars)
        self.step_size = step_scale / scaling.shape[0]**0.25
        self.potential = quad_potential(scaling, is_cov, as_cov=False)
        if state is None:
            state = SamplerHist()
        self.state = state
        self.max_energy = max_energy
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.k = k
        self.h_bar = 0
        self.u = np.log(self.step_size * 10)
        self.m = 1

        shared = make_shared_replacements(vars, model)
        self.leapfrog1_dE = leapfrog1_dE(
            model.logpt, vars, shared, self.potential, profile=profile)

        super(NUTS, self).__init__(vars, shared, **kwargs)

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            return Competence.IDEAL
        return Competence.INCOMPATIBLE

    def astep(self, initial_position):
        log_slice_var = np.log(nr.uniform())
        initial_momentum = self.potential.random()
        position = back_position = forward_position = initial_position
        back_momentum = forward_momentum = initial_momentum
        should_continue = True
        trials = 1
        depth = 0
        while should_continue:
            direction = nr.choice((-1, 1))
            step = np.array(direction * self.step_size)
            new_trials = 0
            metropolis_acceptance = 0
            steps = 0
            for _ in range(2 ** depth):
                if not should_continue:
                    break
                if direction == 1:
                    forward_position, forward_momentum, energy_change = self.leapfrog1_dE(
                        forward_position, forward_momentum, step,
                        initial_position, initial_momentum)
                else:
                    back_position, back_momentum, energy_change = self.leapfrog1_dE(
                        back_position, back_momentum, step, initial_position, initial_momentum)
                new_trials += int(log_slice_var + energy_change <= 0)
                if should_update_position(new_trials, trials):
                    if direction == 1:
                        position = forward_position
                    else:
                        position = back_position

                should_continue = (self._energy_is_bounded(log_slice_var, energy_change) and
                                   no_u_turns(forward_position, forward_momentum,
                                              back_position, back_momentum))
                metropolis_acceptance += min(1., np.exp(-energy_change))
                steps += 1
            trials += new_trials
            depth += 1
        w = 1. / (self.m + self.t0)
        self.h_bar = (1 - w) * self.h_bar + w * (self.target_accept - metropolis_acceptance / steps)
        self.step_size = np.exp(self.u - (self.m ** 0.5 / self.gamma) * self.h_bar)
        self.m += 1
        return position

    def _energy_is_bounded(self, log_slice_var, energy_change):
        return log_slice_var + energy_change < self.max_energy


def no_u_turns(forward_position, forward_momentum, back_position, back_momentum):
    span = forward_position - back_position
    return span.dot(back_momentum) >= 0 and span.dot(forward_momentum) >= 0


def should_update_position(new_trials, trials):
    return bern(float(new_trials) / max(trials, 1.))


def leapfrog1_dE(logp, vars, shared, quad_potential, profile):
    """Computes a theano function that computes one leapfrog step and the energy
    difference between the beginning and end of the trajectory.

    Parameters
    ----------
    logp : TensorVariable
    vars : list of tensor variables
    shared : list of shared variables not to compute leapfrog over
    quad_potential : quadpotential
    profile : Boolean

    Returns
    -------
    theano function which returns
    q_new, p_new, delta_E
    """
    dlogp = gradient(logp, vars)
    (logp, dlogp), q = join_nonshared_inputs([logp, dlogp], vars, shared)
    logp = CallableTensor(logp)
    dlogp = CallableTensor(dlogp)

    hamiltonian = Hamiltonian(logp, dlogp, quad_potential)

    p = tt.dvector('p')
    p.tag.test_value = q.tag.test_value

    q0 = tt.dvector('q0')
    q0.tag.test_value = q.tag.test_value
    p0 = tt.dvector('p0')
    p0.tag.test_value = p.tag.test_value

    e = tt.dscalar('e')
    e.tag.test_value = 1

    q1, p1 = leapfrog(hamiltonian, q, p, 1, e)
    energy_change = energy(hamiltonian, q1, p1) - energy(hamiltonian, q0, p0)

    f = theano.function([q, p, e, q0, p0], [q1, p1, energy_change], profile=profile)
    f.trust_input = True
    return f
