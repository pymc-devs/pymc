from PyMC2 import Sampler
from numpy import mean, exp

#
# Get posterior probabilities for a list of models
#
def weight(samplers, iter, priors = None):
    """
    weight(samplers, iter, priors = None)

    samplers is a list of Samplers, iter is the number of samples to use, and
    priors is a dictionary of prior weights keyed by model.

    Example:

    M1 = Sampler(model_1)
    M2 = Sampler(model_2)
    weight(samplers = [M1,M2], iter = 100000, priors = {M1: .8, M2: .2})

    Returns a dictionary keyed by model of the model posterior probabilities.

    Need to attach an MCSE value to the return values!
    """
    loglikes = {}
    i=0
    for sampler in samplers:
        print 'Model ', i
        loglikes[sampler] = sampler.sample_model_likelihood(iter)
        i+=1

    # Find max log-likelihood for regularization purposes
    max_loglike = 0
    for sampler in samplers:
        max_loglike = max((max_loglike,loglikes[sampler].max()))

    posteriors = {}
    sumpost = 0
    for sampler in samplers:
        # Regularize
        loglikes[sampler] -= max_loglike
        # Exponentiate and average
        posteriors[sampler] = mean(exp(loglikes[sampler]))
        # Multiply in priors
        if priors is not None:
            posteriors[sampler] *= priors[sampler]
        # Count up normalizing constant
        sumpost += posteriors[sampler]

    # Normalize
    for sampler in samplers:
        posteriors[sampler] /= sumpost

    return posteriors
