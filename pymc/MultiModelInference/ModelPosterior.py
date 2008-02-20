# March 30 07 AP: This can work with any Model subclass, not just Sampler.

from pymc import *
from numpy import mean, exp, Inf, zeros

__all__ = ['sample_likelihood', 'weight']

def sample_likelihood(model, iter):
    """
    Returns iter samples of 
    
    log p(data|self.stochastics, self) * sum(self.potentials)
        and 
    sum(self.potentials),
    
    where 'sample' means that self.stochastics are drawn from their joint prior and then these
    quantities are evaluated. See documentation.
    
    Exponentiating, averaging and dividing the return values gives an estimate of the model 
    likelihood, p(data|self).
    """
    
    model.generations = find_generations(model)
    
    loglikes = zeros(iter)

    if len (model.potentials) > 0:
        logpots = zeros(iter)
    else:
        logpots = zeros(1, dtype=float)

    try:
        for i in xrange(iter):
            if i % 10000 == 0:
                print 'Sample ', i, ' of ', iter

            model.draw_from_prior()

            for datum in model.data_stochastics | model.potentials:
                loglikes[i] += datum.logp
            if len (model.potentials) > 0:
                for pot in model.potentials:
                    logpots[i] += pot.logp

    except KeyboardInterrupt:
        print 'Sample ', i, ' of ', iter
        raise KeyboardInterrupt
    
    return loglikes, logpots

def weight(models, iter, priors = None):
    """
    weight(models, iter, priors = None)

    models is a list of Models, iter is the number of samples to use, and
    priors is a dictionary of prior weights keyed by model.

    Example:

    M1 = Model(model_1)
    M2 = Model(model_2)
    weight(models = [M1,M2], iter = 100000, priors = {M1: .8, M2: .2})

    Returns a dictionary keyed by model of the model posterior probabilities.
    """
    
    # TODO: Need to attach a standard error to the return values.
    loglikes = {}
    logpots = {}
    i=0
    for model in models:
        print 'Model ', i
        loglikes[model], logpots[model] = sample_likelihood(model, iter)
        i+=1

    # Find max log-likelihood for regularization purposes
    max_loglike = -Inf
    max_logpot = -Inf
    for model in models:
        max_loglike = max((max_loglike,loglikes[model].max()))
        max_logpot = max((max_logpot,logpots[model].max()))

    posteriors = {}
    sumpost = 0
    for model in models:
        
        # Regularize
        loglikes[model] -= max_loglike
        logpots[model] -= max_logpot
        
        # Exponentiate and average
        posteriors[model] = mean(exp(loglikes[model])) / mean(exp(logpots[model]))

        # Multiply in priors
        if priors is not None:
            posteriors[model] *= priors[model]

        # Count up normalizing constant
        sumpost += posteriors[model]

    # Normalize
    for model in models:
        posteriors[model] /= sumpost

    return posteriors
