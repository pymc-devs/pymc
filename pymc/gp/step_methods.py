# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'

import pymc as pm
import linalg_utils
import copy
import types
import numpy as np
from gp_submodel import *

from Realization import Realization
from Mean import Mean
from Covariance import Covariance
from GPutils import observe, regularize_array

__all__ = ['wrap_metropolis_for_gp_parents', 'GPEvaluationGibbs', 'GPParentAdaptiveMetropolis']


def wrap_metropolis_for_gp_parents(metro_class):
    """
    Wraps Metropolis step methods so they can handle extended parents of
    Gaussian processes.
    """
    class wrapper(metro_class):
        def __init__(self, stochastic, *args, **kwds):
            
            self.metro_class.__init__(self, stochastic, *args, **kwds)
            
            # Remove f from the set that will be used to compute logp_plus_loglike.
            self.markov_blanket_no_f = filter(lambda x: not isinstance(x, GaussianProcess), self.markov_blanket)
            self.fs = filter(lambda x: isinstance(x, GaussianProcess), self.markov_blanket)
            self.fr_checks = [f.submodel.fr_check for f in self.fs]

        def get_logp_plus_loglike(self):
            return pm.logp_of_set(self.markov_blanket_no_f)
        logp_plus_loglike = property(get_logp_plus_loglike)
    
        def propose(self):
            self.metro_class.propose(self)
            try:
                # First make sure none of the stochastics handled by metro_method forbid their current values.
                for s in self.stochastics:
                    s.logp
                # Then make sure the covariances are all still full-rank on the observation locations.
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
            if any([isinstance(child, GaussianProcess) for child in stochastic.extended_children]):
                return metro_class.competence(stochastic)+.01
            else:
                return 0
        
    wrapper.__name__ = 'GPParent%s'%metro_class.__name__
    wrapper.metro_class = metro_class
    wrapper.__doc__ = """A modified version of class %s that handles parents of Gaussian processes.
Docstring of class %s: \n\n%s"""%(metro_class.__name__,metro_class.__name__,metro_class.__doc__)
            
    return wrapper


# Wrap all registered Metropolis step methods to use GP parents.
new_sm_dict = {}
filtered_registry = filter(lambda x: issubclass(x, pm.Metropolis), pm.StepMethodRegistry)
for sm in filtered_registry:
    wrapped_method = wrap_metropolis_for_gp_parents(sm)
    new_sm_dict[wrapped_method.__name__] = wrapped_method
GPParentAdaptiveMetropolis = wrap_metropolis_for_gp_parents(pm.AdaptiveMetropolis)
__all__ += new_sm_dict.keys()
locals().update(new_sm_dict)


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
                
        # Remove f from the set that will be used to compute logp_plus_loglike.
        self.markov_blanket_no_f = copy.copy(self.markov_blanket)
        self.markov_blanket_no_f.remove(self.f)
        
        self.V = V
        self.C_eval = submod.C_eval
        self.M_eval = submod.M_eval
        self.S_eval = submod.S_eval
        self.eps_p_f = eps_p_f

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
        self.proposal_distribution=None
        
    def get_logp_plus_loglike(self):
        return pm.logp_of_set(self.markov_blanket_no_f)
    logp_plus_loglike = property(get_logp_plus_loglike)
        
    def reject(self):
        self.rejected += 1
        if self.verbose:
            print self._id + ' rejecting'
        # Revert the field evaluation and the rest of the field.
        self.f_eval.revert()
        self.f.revert()
    
    def tune(self, verbose=0):
        return False
            
    def propose(self):
        if self.verbose:
            print self._id + ' proposing'

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
        for i in xrange(pm.utils.value(C_eval_shape)[0]):
            in_chol[i,i] += pm.utils.value(self.V) / np.alen(self.ti[i])
        info = pm.gp.linalg_utils.dpotrf_wrap(in_chol)
        if info > 0:
            raise np.linalg.LinAlgError

        # Compute covariance of f conditional on eps_p_f.
        offdiag = fc(C_eval_value, self.scratch2)
        offdiag = pm.gp.trisolve(in_chol, offdiag, uplo='U', transa='T', inplace=True)

        C_step = offdiag.T * offdiag
        C_step *= -1
        C_step += C_eval_value

        # Compute mean of f conditional on eps_p_f.
        for i in xrange(len(self.scratch3)):
            self.scratch3[i] = np.mean(eps_p_f[self.ti[i]])
        m_step = pm.utils.value(self.M_eval) + np.dot(offdiag.T, pm.gp.trisolve(in_chol,(self.scratch3 - self.M_eval.value),uplo='U',transa='T')).view(np.ndarray).ravel()

        sig_step = C_step
        info = pm.gp.linalg_utils.dpotrf_wrap(C_step.T)
        if info > 0:
            warnings.warn('Full conditional covariance was not positive definite.')
            return

        # Update value of f.
        self.f_eval.value = m_step+np.dot(sig_step,np.random.normal(size=sig_step.shape[1])).view(np.ndarray).ravel()
        # Propose the rest of the field from its conditional prior.
        self.f.rand()            