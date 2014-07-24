# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'

import pymc as pm
from . import linalg_utils
import copy
import types
import numpy as np
from .gp_submodel import *
import warnings

from pymc import six
from pymc.six import print_
xrange = six.moves.xrange

from .Realization import Realization
from .Mean import Mean
from .Covariance import Covariance
from .GPutils import observe, regularize_array

__all__ = [
    'wrap_metropolis_for_gp_parents',
    'GPEvaluationGibbs',
    'GPParentAdaptiveMetropolis',
    'GPStepMethod',
    'GPEvaluationMetropolis',
    'MeshlessGPMetropolis']


class GPStepMethod(pm.NoStepper):

    @staticmethod
    def competence(stochastic):
        if isinstance(stochastic, GaussianProcess):
            return 1
        else:
            return 0

    def tune(self, verbose=0):
        return False

def wrap_metropolis_for_gp_parents(metro_class):
    """
    Wraps Metropolis step methods so they can handle extended parents of
    Gaussian processes.
    """
    class wrapper(metro_class):
        __doc__ = """A modified version of class %s that handles parents of Gaussian processes.
Docstring of class %s: \n\n%s""" % (metro_class.__name__, metro_class.__name__, metro_class.__doc__)

        def __init__(self, stochastic, *args, **kwds):

            self.metro_class.__init__(self, stochastic, *args, **kwds)

            mb = set(self.markov_blanket)
            for c in list(self.children):
                if isinstance(c, GaussianProcess):
                    self.children |= c.extended_children
                    mb |= c.extended_children
            self.markov_blanket = list(mb)

            # Remove f from the set that will be used to compute
            # logp_plus_loglike.
            self.markov_blanket_no_f = set(
                filter(
                    lambda x: not isinstance(
                        x,
                        GaussianProcess),
                    self.markov_blanket))
            self.fs = filter(
                lambda x: isinstance(
                    x,
                    GaussianProcess),
                self.markov_blanket)
            self.fr_checks = [f.submodel.fr_check for f in self.fs]

        def get_logp_plus_loglike(self):
            return pm.logp_of_set(self.markov_blanket_no_f)
        logp_plus_loglike = property(get_logp_plus_loglike)

        def propose(self):
            self.metro_class.propose(self)
            try:
                # First make sure none of the stochastics handled by
                # metro_method forbid their current values.
                for s in self.stochastics:
                    s.logp
                # Then make sure the covariances are all still full-rank on the
                # observation locations.
                for frc in self.fr_checks:
                    frc.logp
                for f in self.fs:
                    f.rand()
                self.f_proposed = True
            except pm.ZeroProbability:
                self.f_proposed = False

        def reject(self):
            self.metro_class.reject(self)
            if self.f_proposed:
                for f in self.fs:
                    f.revert()

        @staticmethod
        def competence(stochastic, metro_class=metro_class):
            if any([isinstance(child, GaussianProcess)
                    for child in stochastic.extended_children]):
                return metro_class.competence(stochastic) + .01
            else:
                return 0

    wrapper.__name__ = 'GPParent%s' % metro_class.__name__
    wrapper.metro_class = metro_class

    return wrapper


# Wrap all registered Metropolis step methods to use GP parents.
new_sm_dict = {}
filtered_registry = [
    sm for sm in pm.StepMethodRegistry if issubclass(
        sm,
        pm.Metropolis)]
for sm in filtered_registry:
    wrapped_method = wrap_metropolis_for_gp_parents(sm)
    new_sm_dict[wrapped_method.__name__] = wrapped_method
GPParentAdaptiveMetropolis = wrap_metropolis_for_gp_parents(
    pm.AdaptiveMetropolis)
__all__ += new_sm_dict.keys()
locals().update(new_sm_dict)


class MeshlessGPMetropolis(pm.Metropolis):

    def __init__(self, gp):
        pm.Metropolis.__init__(
            self,
            gp,
            proposal_distribution='Prior',
            check_before_accepting=False)

    def propose(self):
        self.stochastic.rand()

    @staticmethod
    def competence(stochastic):
        if isinstance(stochastic, GaussianProcess):
            if len(stochastic.submodel.mesh) == 0:
                return 3
            else:
                return 0
        else:
            return 0


class _GPEvaluationMetropolis(pm.Metropolis):

    """
    Updates a GP evaluation, the 'f_eval' attribute of a GP submodel.
    The stationary distribution of the assymetric proposal is equal
    to the prior distribution, an attempt to minimize jumps to values
    forbidden by the prior.
    """

    def __init__(self, stochastic, proposal_sd=1, **kwds):
        pm.Metropolis.__init__(
            self,
            stochastic,
            proposal_sd=proposal_sd,
            **kwds)

    def propose(self):
        sig = pm.utils.value(self.stochastic.parents['sig'])
        mu = pm.utils.value(self.stochastic.parents['mu'])

        delta = pm.rmv_normal_chol(0 * mu, sig)

        beta = np.minimum(1, self.proposal_sd * self.adaptive_scale_factor)
        bsig = beta * sig
        sb2 = np.sqrt(1 - beta ** 2)
        self.stochastic.value = (
            self.stochastic.value - mu) * sb2 + beta * delta + mu
        xp, x = self.stochastic.value, self.stochastic.last_value
        self._hastings_factor = pm.mv_normal_chol_like(
            x,
            (xp - mu) * sb2 + mu,
            bsig) - pm.mv_normal_chol_like(xp,
                                           (x - mu) * sb2 + mu,
                                           bsig)

        # self.stochastic.value = self.stochastic.value + self.adaptive_scale_factor*self.proposal_sd*delta
        # self._hastings_factor = 0

    def hastings_factor(self):
        return self._hastings_factor

    @staticmethod
    def competence(stochastic):
        if isinstance(stochastic, GPEvaluation):
            return 3
        else:
            return 0

GPEvaluationMetropolis = wrap_metropolis_for_gp_parents(
    _GPEvaluationMetropolis)


class GPEvaluationGibbs(pm.Metropolis):

    """
    Updates a GP evaluation f_eval. Assumes the only children of f_eval
    are as distributed follows:

    eps_p_f ~ Normal(f_eval[ti], 1./V)

    or

    eps_p_f ~ Normal(f_eval, 1./V)

    if ti is None.
    """

    def __init__(self, submod, V, eps_p_f, ti=None, tally=True, verbose=0):

        self.f_eval = submod.f_eval
        self.f = submod.f
        pm.StepMethod.__init__(self, [self.f, self.f_eval], tally=tally)

        self.children_no_data = copy.copy(self.children)
        if isinstance(eps_p_f, pm.Variable):
            self.children_no_data.remove(eps_p_f)
            self.eps_p_f = eps_p_f
        else:
            for epf in eps_p_f:
                self.children_no_data.remove(epf)
            self.eps_p_f = pm.Lambda(
                'eps_p_f',
                lambda e=eps_p_f: np.hstack(
                    e),
                trace=False)

        self.V = pm.Lambda(
            '%s_vect' % V.__name__,
            lambda V=V: np.resize(V,
                                  len(submod.mesh)))
        self.C_eval = submod.C_eval
        self.M_eval = submod.M_eval
        self.S_eval = submod.S_eval

        M_eval_shape = pm.utils.value(self.M_eval).shape
        C_eval_shape = pm.utils.value(self.C_eval).shape
        self.ti = ti or np.arange(M_eval_shape[0])

        # Work arrays
        self.scratch1 = np.asmatrix(np.empty(C_eval_shape, order='F'))
        self.scratch2 = np.asmatrix(np.empty(C_eval_shape, order='F'))
        self.scratch3 = np.empty(M_eval_shape)

        # Initialize hidden attributes
        self.accepted = 0.
        self.rejected = 0.
        self._state = ['rejected', 'accepted', 'proposal_distribution']
        self._tuning_info = []
        self.proposal_distribution = None

        self.verbose = verbose

    def get_logp(self):
        return 0.
    logp = property(get_logp)

    def get_loglike(self):
        return pm.utils.logp_of_set(self.children_no_data)
    loglike = property(get_loglike)

    def get_logp_plus_loglike(self):
        return self.get_loglike()
    logp_plus_loglike = property(get_logp_plus_loglike)

    def reject(self):
        self.rejected += 1
        if self.verbose:
            print_(self._id + ' rejecting')
        # Revert the field evaluation and the rest of the field.
        self.f_eval.revert()
        self.f.revert()

    def tune(self, verbose=0):
        return False

    def propose(self):

        if self.verbose:
            print_(self._id + ' proposing')

        fc = pm.gp.fast_matrix_copy

        eps_p_f = pm.utils.value(self.eps_p_f)
        f = pm.utils.value(self.f_eval)
        for i in xrange(len(self.scratch3)):
            self.scratch3[i] = np.sum(eps_p_f[self.ti[i]] - f[i])

        # Compute Cholesky factor of covariance of eps_p_f, C(x,x) + V
        C_eval_value = pm.utils.value(self.C_eval)
        C_eval_shape = C_eval_value.shape

        # Get the Cholesky factor of C_eval, plus the nugget.
        # I don't think you can use S_eval for speed, unfortunately.
        in_chol = fc(C_eval_value, self.scratch1)

        v_val = pm.utils.value(self.V)
        for i in xrange(pm.utils.value(C_eval_shape)[0]):
            in_chol[i, i] += v_val[i] / np.alen(self.ti[i])

        info = pm.gp.linalg_utils.dpotrf_wrap(in_chol)
        if info > 0:
            raise np.linalg.LinAlgError

        # Compute covariance of f conditional on eps_p_f.
        offdiag = fc(C_eval_value, self.scratch2)
        offdiag = pm.gp.trisolve(
            in_chol,
            offdiag,
            uplo='U',
            transa='T',
            inplace=True)

        C_step = offdiag.T * offdiag
        C_step *= -1
        C_step += C_eval_value

        # Compute mean of f conditional on eps_p_f.
        for i in xrange(len(self.scratch3)):
            self.scratch3[i] = np.mean(eps_p_f[self.ti[i]])
        m_step = pm.utils.value(
            self.M_eval) + np.dot(offdiag.T,
                                  pm.gp.trisolve(in_chol,
                                                 (self.scratch3 -
                                                  self.M_eval.value),
                                                 uplo='U',
                                                 transa='T')).view(np.ndarray).ravel()

        sig_step = C_step
        info = pm.gp.linalg_utils.dpotrf_wrap(C_step.T)
        if info > 0:
            warnings.warn(
                'Full conditional covariance was not positive definite.')
            return

        # Update value of f.
        self.f_eval.value = m_step + np.dot(
            sig_step,
            np.random.normal(
                size=sig_step.shape[
                    1])).view(
                        np.ndarray).ravel(
        )
        # Propose the rest of the field from its conditional prior.
        self.f.rand()
