__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

from pymc.NormalApproximation import *
import pymc as pm
import numpy as np

class EM(MAP):
    """
    N = EM(input, sampler, db='ram', eps=.001, diff_order = 5)

    Normal approximation to the posterior of a model via the EM algorithm.

    Useful methods:
    draw:           Draws values for all stochastic variables using normal approximation
    revert_to_max:  Sets all stochastic variables to mean value under normal approximation
    fit:            Finds the normal approximation.

    Useful attributes (after fit() is called):
    mu[p1, p2, ...]:    Returns the posterior mean vector of stochastic variables p1, p2, ...
    C[p1, p2, ...]:     Returns the posterior covariance of stochastic variables p1, p2, ...
    logp:               Returns the log-probability of the model
    logp_at_max:        Returns the maximum log-probability of the model
    len:                The number of free stochastic variables in the model ('k' in AIC and BIC)
    data_len:           The number of datapoints used ('n' in BIC)
    AIC:                Akaike's Information Criterion for the model
    BIC:                Bayesian Information Criterion for the model

    :Arguments:
    input: As for Model
    sampler: Should be a Sampler instance handling a submodel of input. The variables in
      sampler will be integrated out; only the marginal probability of the other variables in input
      will be maximized. The 'expectation' step will be computed using samples obtained from the 
      sampler.
    db: A database backend.
    eps: 'h' for computing numerical derivatives. May be a dictionary keyed by stochastic variable 
      as well as a scalar.
    diff_order: The order of the approximation used to compute derivatives.

    :SeeAlso: Model, NormApprox, Sampler, scipy.optimize
    """
    def __init__(self, input, sampler, db='ram', eps=.001, diff_order = 5, verbose=0, tune_interval=10):

        Q = pm.Container(input)
        new_input = (Q.nodes | sampler.nodes) - sampler.stochastics

        MAP.__init__(self, input=new_input, eps=eps, diff_order=diff_order)

        self.tune_interval = tune_interval
        self.verbose = verbose
        self.sampler = sampler

        # Figure out which stochastics' log-probabilities need to be averaged.
        self.stochastics_to_integrate = set()

        for s in self.stochastics:
            mb = s.markov_blanket
            if any([other_s in mb for other_s in sampler.stochastics]):
                self.stochastics_to_integrate.add(s)


    def fit(self, iterlim=1000, tol=.0001, 
            na_method = 'fmin', na_iterlim=1000, na_tol=.0001, 
            sa_iter = 10000, sa_burn=1000, sa_thin=10):
        """
        N.fit(iterlim=1000, tol=.0001, 
        na_method='fmin', na_iterlim=1000, na_tol=.0001, 
        sa_iter = 10000, sa_burn=1000, sa_thin=10)
        
        Arguments 'iterlim' and 'tol' control the top-level EM iteration.
        Arguments beginning with 'na' are passed to NormApprox.fit() during the M steps.
        Arguments beginning with 'sa' are passed to self.sampler during the E-steps.
        
        The 'E' step consists of running the sampler, which will keep a trace. In the 'M' step, the 
        log-probability of variables in the sampler's Markov blanket are averaged and combined with 
        the log-probabilities of self's other variables to produce a joint log-probability. This 
        quantity is maximized.
        """

        logps = []
        for i in xrange(iterlim):
            
            # E step
            self.sampler.sample(sa_iter, sa_burn, sa_thin, verbose = self.verbose, tune_interval = self.tune_interval)

            # M step
            MAP.fit(self, method = na_method, iterlim=na_iterlim, tol=na_tol, post_fit_computations=False, verbose=self.verbose)

            logps.append(self.logp)
            if abs(logps[i-1] - logps[i])<= tol:
                print 'EM algorithm converged'
                break

        if i == iterlim-1:
            print 'EM algorithm: Maximum number of iterations exceeded.'
        self.post_fit_computations()


    def i_logp(self, index):
        """
        Evaluates the log-probability of the Markov blanket of
        a stochastic owning a particular index. Used for computing derivatives.

        Averages over the sampler's trace for variables in the sampler's 
        Markov blanket.
        """
        all_relevant_stochastics = set()
        p,i = self.stochastic_indices[index]

        logps = []

        # If needed, run an MCMC loop and use those samples.
        if p in self.stochastics_to_integrate:
            for i in xrange(sampler.db.length):
                sampler.remember(i-1)
                try:
                    logps.append(p.logp + np.sum([child.logp for child in self.extended_children[p]]))
                except ZeroProbability:
                    return -Inf

                return mean(logps)

        # Otherwise, just return the log-probability of the Markov blanket.
        else:
            try:
                return p.logp + np.sum([child.logp for child in self.extended_children[p]])
            except pm.ZeroProbability:
                return -np.Inf

    def func(self, p):
        """
        The function that gets passed to the optimizers.
        """
        self._set_stochastics(p)
        logps = []
        for i in xrange(sampler.db.length):
            sampler.remember(i-1)            
            try:
                logps.append(self.logp)
            except pm.ZeroProbability:
                return np.Inf
        return -np.mean(logps)

class SEM(EM, NormApprox):
    """
    Normal approximation via SEM algorithm
    """
    pass