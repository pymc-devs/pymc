from numpy.linalg import cholesky

from ..core import *
from .quadpotential import quad_potential

from .arraystep import *
from numpy.random import normal, standard_cauchy, standard_exponential, poisson, random, uniform
from numpy import round, exp, copy, where, log, isfinite
import numpy as np
from ..distributions.discrete import Categorical
import theano.tensor as T
import theano
from metropolis import *

__all__ = ['MetropolisHastings', 'GenericProposal', 'ProposalAdapter']

'''
More general version of Metropolis, which allows to combine different proposal distributions for
different sets of variables and then do a single accept / reject.

Can also handle asymmetric proposals, i.e. it's an implementation of the Metropolis Hastings
algorithm, not just the Metropolis algorithm.

The main purpose of this class is to allow for tailored proposal distributions for specialized
applications, such as efficiently sampling from a mixture model.

@author Kai Londenberg (Kai.Londenberg@gmail.com) 
'''

class GenericProposal(object):
    
    '''
    Asymmetric proposal distribution. 
    Has to be able to calculate move propabilities 
    '''
    def proposal_logp_difference(self):
        ''' Return log-probability difference of move from one point to another
        for the last proposed move
        
        Returns 0.0 by default, which is true for all symmetric proposal distributions
        '''
        return 0.0   
        
    def propose_move(self, point):
        ''' 
        Propose a move: Return a new point proposal
        Destructively writes to the given point, destroying the old values in it
        '''
        raise NotImplementedError("Called abstract method")
    
    def tune_stepsize(self, stepsize_factor=1.0):
        '''
        Tune proposal distribution to increase / decrease the stepsize of the proposals by the given factor (if possible) 
        '''
        pass


class ProposalAdapter(GenericProposal):
    
    def __init__(self, vars, proposal_dist=NormalProposal, scale=None):
        self.vars = vars
        self.ordering = ArrayOrdering(vars)
        if scale is None:
            scale = np.ones(sum(v.dsize for v in vars))
        self.proposal = proposal_dist(scale)
        self.discrete = np.array([v.dtype in discrete_types for v in vars])
        self.stepsize_factor = 1.0

    def propose_move(self, point):
        i = 0
        delta = np.atleast_1d( self.proposal() * self.stepsize_factor)
        for var, slc, shp in self.ordering.vmap:
            dvar = delta[slc] 
            if self.discrete[i]:
                dvar = round(delta[slc], 0).astype(int)               
            point[var] = point[var] + np.reshape(dvar, shp)
            i += 1
        return point

    def tune_stepsize(self, stepsize_factor=1.0):
        self.stepsize_factor = stepsize_factor
        
class CategoricalProposal(GenericProposal):
    
    def __init__(self, vars, model=None):
        model = modelcontext(model)
        self.vars = vars
        varidx = T.iscalar('varidx')
        logpt_elems = []
        self.paramfuncs = []
        self._proposal_logp_difference = 0.0
        
        for i, v in enumerate(vars):
            assert isinstance(v.distribution, Categorical)
            self.paramfuncs.append(model.fastfn(v.distribution.p))
            logpt_elems.append(v.distribution.logp(v))
        
        self.logpfunc = model.fastfn(T.add(*logpt_elems))
        
    def propose_move(self, point):
        move_logp = 0.0
        for i, v in enumerate(self.vars):
            weights = self.paramfuncs[i](point)
            
            oldvalue = point[v.name]
            newvalue = np.random.choice(len(weights),1, p=weights)[0]
            'Move probability: Probability of moving from new state to old state divided by probability of moving from old state to new state'
            move_logp += log(weights[oldvalue]) - log(weights[newvalue])  
            point[v.name] = newvalue
            
        self._proposal_logp_difference = move_logp
        return point

    def proposal_logp_difference(self):
        return self._proposal_logp_difference
    

class MetropolisHastings(object):
    """
    Metropolis-Hastings sampling step

    Parameters
    ----------
    vars : list
        List of variables for sampler
    proposals : array of GenericProposal instances
    scaling : scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune : bool
        Flag for tuning. Defaults to True.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """
    def __init__(self, proposals=None, scaling=1., tune=True, tune_interval=100, model=None):
        model = modelcontext(model)
        self.model = model    
        if (proposals is None):
            cvars = []
            ovars = []
            for v in model.vars:
                if (isinstance(v.distribution, Categorical)):
                    cvars.append(v)
                else:
                    ovars.append(v)
            if (len(cvars)>0):
                proposals = [ CategoricalProposal(cvars), ProposalAdapter(ovars, NormalProposal) ]
            else:
                proposals = [ ProposalAdapter(ovars, NormalProposal) ]
        self.proposals = proposals
        var_set = set()
        for p in proposals:
            p.tune_stepsize(scaling)
            for v in p.vars:
                var_set.add(v)
        vars = list(var_set)
        
        self.scaling = scaling
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0
        self.fastlogp = self.model.fastlogp
        self.oldlogp = None
    
    def step(self, point):
        self._ensure_tuned()
        proposal = point.copy()
        proposal_symmetry = 0.0
        for prop in self.proposals:
            proposal = prop.propose_move(proposal)
            proposal_symmetry += prop.proposal_logp_difference()

        
        # Log Probability of old state
        if (self.oldlogp is None):
            self.oldlogp = self.fastlogp(point)
        
        # Log-Probability of proposed state
        proposal_logp = self.fastlogp(proposal)
        
        # Log probability of move, takes proposal symmetry into account to ensure detailed balance
        move_logp = proposal_symmetry + proposal_logp - self.oldlogp
        
        # Accept / Reject
        if isfinite(move_logp) and log(uniform()) < move_logp:
            # Accept proposed value
            newpoint = proposal
            self.oldlogp = proposal_logp
            self.accepted += 1
        else:
            # Reject proposed value
            newpoint = point#
        
        self.steps_until_tune -= 1

        return newpoint
    
    def _ensure_tuned(self):
        if self.tune and not self.steps_until_tune:
            # Tune scaling parameter
            self.scaling = mh_tune(
                self.scaling, self.accepted / float(self.tune_interval))
            for p in self.proposals:
                p.tune_stepsize(self.scaling)
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

def mh_tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10

    """
    print "Acceptance rate: %f - scale: %f" % (acc_rate, scale)
    # Switch statement
    if acc_rate < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        scale *= 10.0
    elif acc_rate > 0.75:
        # increase by double
        scale *= 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        scale *= 1.1

    return scale