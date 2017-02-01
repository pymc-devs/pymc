from collections import namedtuple

from pymc3.theanof import join_nonshared_inputs, gradient, CallableTensor

import theano
import theano.tensor as tt


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
    epsilon.tag.test_value = 1

    n_steps = tt.iscalar('n_steps')
    n_steps.tag.test_value = 2

    q_new, p_new = leapfrog(H, q, p, epsilon, n_steps)
    energy_new = energy(H, q_new, p_new)

    f = theano.function([q, p, epsilon, n_steps], [q_new, p_new, energy_new], **theano_kwargs)
    f.trust_input = True
    return f


def get_theano_hamiltonian_functions(model_vars, shared, logpt, potential,
                                     use_single_leapfrog=False, **theano_kwargs):
    """Construct theano functions for the Hamiltonian, energy, and leapfrog integrator.

    Parameters
    ----------
    model_vars : array of variables to be sampled
    shared : theano tensors that are already shared
    logpt : model log probability
    potential : Hamiltonian potential
    theano_kwargs : dictionary of keyword arguments to pass to theano functions
    use_single_leapfrog : Boolean, if only 1 integration step is done at a time (as in NUTS),
                          this provides a ~2x speedup

    Returns
    -------
    H : Hamiltonian namedtuple
    energy_function : theano function computing energy at a point in phase space
    leapfrog_integrator : theano function integrating the Hamiltonian from a point in phase space
    theano_variables : dictionary of variables used in the computation graph which may be useful
    """
    H, q, dlogp = _theano_hamiltonian(model_vars, shared, logpt, potential)
    energy_function, p = _theano_energy_function(H, q, **theano_kwargs)
    if use_single_leapfrog:
        leapfrog_integrator = _theano_single_leapfrog(H, q, p, H.dlogp(q), **theano_kwargs)
    else:
        leapfrog_integrator = _theano_leapfrog_integrator(H, q, p, **theano_kwargs)
    return H, energy_function, leapfrog_integrator, dlogp


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

    f = theano.function(inputs=[q, p, q_grad, epsilon],
                        outputs=[q_new, p_new, q_new_grad, energy_new], **theano_kwargs)
    f.trust_input = True
    return f
