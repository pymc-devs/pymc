from collections import namedtuple

from pymc3.theanof import join_nonshared_inputs, gradient, CallableTensor, floatX

import theano
import theano.tensor as tt
import numpy as np


Hamiltonian = namedtuple("Hamiltonian", "logp, dlogp, pot")


def _theano_hamiltonian(model_vars, shared, logpt, potential):
    """Creates a Hamiltonian with shared inputs.

    Parameters
    ----------
    model_vars : array of variables to be sampled
    shared : theano tensors that are already shared
    logpt : model log probability
    potential : hamiltonian potential

    Returns
    -------
    Hamiltonian : namedtuple with log pdf, gradient of log pdf, and potential functions
    q : Starting position variable.
    """
    dlogp = gradient(logpt, model_vars)
    (logp, dlogp), q = join_nonshared_inputs([logpt, dlogp], model_vars, shared)
    dlogp_func = theano.function(inputs=[q], outputs=dlogp)
    dlogp_func.trust_input = True
    logp = CallableTensor(logp)
    dlogp = CallableTensor(dlogp)
    return Hamiltonian(logp, dlogp, potential), q, dlogp_func


def _theano_energy_function(H, q, **theano_kwargs):
    """Creates a Hamiltonian with shared inputs.

    Parameters
    ----------
    H : Hamiltonian namedtuple
    q : theano variable, starting position
    theano_kwargs : passed to theano.function

    Returns
    -------
    energy_function : theano function that computes the energy at a point (p, q) in phase space
    p : Starting momentum variable.
    """
    p = tt.vector('p')
    p.tag.test_value = q.tag.test_value
    total_energy = H.pot.energy(p) - H.logp(q)
    energy_function = theano.function(inputs=[q, p], outputs=total_energy, **theano_kwargs)
    energy_function.trust_input = True

    return energy_function, p


def _theano_velocity_function(H, p, **theano_kwargs):
    v = H.pot.velocity(p)
    velocity_function = theano.function(inputs=[p], outputs=v, **theano_kwargs)
    velocity_function.trust_input = True
    return velocity_function


def _theano_leapfrog_integrator(H, q, p, **theano_kwargs):
    """Computes a theano function that computes one leapfrog step and the energy at the
    end of the trajectory.

    Parameters
    ----------
    H : Hamiltonian
    q : theano.tensor
    p : theano.tensor
    theano_kwargs : passed to theano.function

    Returns
    -------
    theano function which returns
    q_new, p_new, energy_new
    """
    epsilon = tt.scalar('epsilon')
    epsilon.tag.test_value = 1.

    n_steps = tt.iscalar('n_steps')
    n_steps.tag.test_value = 2

    q_new, p_new = leapfrog(H, q, p, epsilon, n_steps)
    energy_new = energy(H, q_new, p_new)

    f = theano.function([q, p, epsilon, n_steps], [q_new, p_new, energy_new], **theano_kwargs)
    f.trust_input = True
    return f


def get_theano_hamiltonian_functions(model_vars, shared, logpt, potential,
                                     use_single_leapfrog=False,
                                     integrator="leapfrog", **theano_kwargs):
    """Construct theano functions for the Hamiltonian, energy, and leapfrog integrator.

    Parameters
    ----------
    model_vars : array of variables to be sampled
    shared : theano tensors that are already shared
    logpt : model log probability
    potential : Hamiltonian potential
    theano_kwargs : dictionary of keyword arguments to pass to theano functions
    use_single_leapfrog : bool
        if only 1 integration step is done at a time (as in NUTS), this
        provides a ~2x speedup
    integrator : str
        Integration scheme to use. One of "leapfog", "two-stage", or
        "three-stage".

    Returns
    -------
    H : Hamiltonian namedtuple
    energy_function : theano function computing energy at a point in phase space
    leapfrog_integrator : theano function integrating the Hamiltonian from a point in phase space
    theano_variables : dictionary of variables used in the computation graph which may be useful
    """
    H, q, dlogp = _theano_hamiltonian(model_vars, shared, logpt, potential)
    energy_function, p = _theano_energy_function(H, q, **theano_kwargs)
    velocity_function = _theano_velocity_function(H, p, **theano_kwargs)
    if use_single_leapfrog:
        try:
            _theano_integrator = INTEGRATORS_SINGLE[integrator]
        except KeyError:
            raise ValueError("Unknown integrator: %s" % integrator)
        integrator = _theano_integrator(H, q, p, H.dlogp(q), **theano_kwargs)
    else:
        if integrator != "leapfrog":
            raise ValueError("Only leapfrog is supported")
        integrator = _theano_leapfrog_integrator(H, q, p, **theano_kwargs)
    return H, energy_function, velocity_function, integrator, dlogp


def energy(H, q, p):
    """Compute the total energy for the Hamiltonian at a given position/momentum"""
    return H.pot.energy(p) - H.logp(q)


def leapfrog(H, q, p, epsilon, n_steps):
    """Leapfrog integrator.

    Estimates `p(t)` and `q(t)` at time :math:`t = n \cdot e`, by integrating the
    Hamiltonian equations

    .. math::

        \frac{dq_i}{dt} = \frac{\partial H}{\partial p_i}

        \frac{dp_i}{dt} = \frac{\partial H}{\partial q_i}

    with :math:`p(0) = p`, :math:`q(0) = q`

    Parameters
    ----------
    H : Hamiltonian instance.
        Tuple of `logp, dlogp, potential`.
    q : Theano.tensor
        initial position vector
    p : Theano.tensor
        initial momentum vector
    epsilon : float, step size
    n_steps : int, number of iterations

    Returns
    -------
    position : Theano.tensor
        position estimate at time :math:`n \cdot e`.
    momentum : Theano.tensor
        momentum estimate at time :math:`n \cdot e`.
    """
    def full_update(p, q):
        p = p + epsilon * H.dlogp(q)
        q += epsilon * H.pot.velocity(p)
        return p, q
    #  This first line can't be +=, possibly because of theano
    p = p + 0.5 * epsilon * H.dlogp(q)  # half momentum update
    q += epsilon * H.pot.velocity(p)  # full position update
    if tt.gt(n_steps, 1):
        (p_seq, q_seq), _ = theano.scan(full_update, outputs_info=[p, q], n_steps=n_steps - 1)
        p, q = p_seq[-1], q_seq[-1]
    p += 0.5 * epsilon * H.dlogp(q)  # half momentum update
    return q, p


def _theano_single_threestage(H, q, p, q_grad, **theano_kwargs):
    """Perform a single step of a third order symplectic integration scheme.

    References
    ----------
    Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
    Integrators for the Hybrid Monte Carlo Method." SIAM Journal on
    Scientific Computing 36, no. 4 (January 2014): A1556-80.
    doi:10.1137/130932740.

    Mannseth, Janne, Tore Selland Kleppe, and Hans J. Skaug. "On the
    Application of Higher Order Symplectic Integrators in
    Hamiltonian Monte Carlo." arXiv:1608.07048 [Stat],
    August 25, 2016. http://arxiv.org/abs/1608.07048.
    """
    epsilon = tt.scalar('epsilon')
    epsilon.tag.test_value = 1.

    a = 12127897.0 / 102017882
    b = 4271554.0 / 14421423

    # q_{a\epsilon}
    p_ae = p + floatX(a) * epsilon * q_grad
    q_be = q + floatX(b) * epsilon * H.pot.velocity(p_ae)

    # q_{\epsilon / 2}
    p_e2 = p_ae + floatX(0.5 - a) * epsilon * H.dlogp(q_be)

    # p_{(1-b)\epsilon}
    q_1be = q_be + floatX(1 - 2 * b) * epsilon * H.pot.velocity(p_e2)
    p_1ae = p_e2 + floatX(0.5 - a) * epsilon * H.dlogp(q_1be)

    q_e = q_1be + floatX(b) * epsilon * H.pot.velocity(p_1ae)
    grad_e = H.dlogp(q_e)
    p_e = p_1ae + floatX(a) * epsilon * grad_e
    v_e = H.pot.velocity(p_e)

    new_energy = energy(H, q_e, p_e)

    f = theano.function(inputs=[q, p, q_grad, epsilon],
                        outputs=[q_e, p_e, v_e, grad_e, new_energy],
                        **theano_kwargs)
    f.trust_input = True
    return f


def _theano_single_twostage(H, q, p, q_grad, **theano_kwargs):
    """Perform a single step of a second order symplectic integration scheme.

    References
    ----------
    Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
    Integrators for the Hybrid Monte Carlo Method." SIAM Journal on
    Scientific Computing 36, no. 4 (January 2014): A1556-80.
    doi:10.1137/130932740.

    Mannseth, Janne, Tore Selland Kleppe, and Hans J. Skaug. "On the
    Application of Higher Order Symplectic Integrators in
    Hamiltonian Monte Carlo." arXiv:1608.07048 [Stat],
    August 25, 2016. http://arxiv.org/abs/1608.07048.
    """
    epsilon = tt.scalar('epsilon')
    epsilon.tag.test_value = 1.

    a = floatX((3 - np.sqrt(3)) / 6)

    p_ae = p + a * epsilon * q_grad
    q_e2 = q + epsilon / 2 * H.pot.velocity(p_ae)
    p_1ae = p_ae + (1 - 2 * a) * epsilon * H.dlogp(q_e2)
    q_e = q_e2 + epsilon / 2 * H.pot.velocity(p_1ae)
    grad_e = H.dlogp(q_e)
    p_e = p_1ae + a * epsilon * grad_e
    v_e = H.pot.velocity(p_e)

    new_energy = energy(H, q_e, p_e)
    f = theano.function(inputs=[q, p, q_grad, epsilon],
                        outputs=[q_e, p_e, v_e, grad_e, new_energy],
                        **theano_kwargs)
    f.trust_input = True
    return f


def _theano_single_leapfrog(H, q, p, q_grad, **theano_kwargs):
    """Leapfrog integrator for a single step.

    See above for documentation.  This is optimized for the case where only a single step is
    needed, in case of, for example, a recursive algorithm.
    """
    epsilon = tt.scalar('epsilon')
    epsilon.tag.test_value = 1.

    p_new = p + 0.5 * epsilon * q_grad  # half momentum update
    q_new = q + epsilon * H.pot.velocity(p_new)  # full position update
    q_new_grad = H.dlogp(q_new)
    p_new += 0.5 * epsilon * q_new_grad  # half momentum update
    energy_new = energy(H, q_new, p_new)
    v_new = H.pot.velocity(p_new)

    f = theano.function(inputs=[q, p, q_grad, epsilon],
                        outputs=[q_new, p_new, v_new, q_new_grad, energy_new],
                        **theano_kwargs)
    f.trust_input = True
    return f


INTEGRATORS_SINGLE = {
    'leapfrog': _theano_single_leapfrog,
    'two-stage': _theano_single_twostage,
    'three-stage': _theano_single_threestage,
}
