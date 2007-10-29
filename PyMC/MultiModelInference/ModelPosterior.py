# March 30 07 AP: This can work with any Model subclass, not just Sampler.

from PyMC import Model
from numpy import mean, exp

# Get posterior probabilities for a list of models
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
    # TODO: Need to attach an MCSE value to the return values!
    loglikes = {}
    i=0
    for model in models:
        print 'Model ', i
        loglikes[model] = model.sample_likelihood(iter)
        i+=1

    # Find max log-likelihood for regularization purposes
    max_loglike = 0
    for model in models:
        max_loglike = max((max_loglike,loglikes[model].max()))

    posteriors = {}
    sumpost = 0
    for model in models:
        # Regularize
        loglikes[model] -= max_loglike
        # Exponentiate and average
        posteriors[model] = mean(exp(loglikes[model]))
        # Multiply in priors
        if priors is not None:
            posteriors[model] *= priors[model]
        # Count up normalizing constant
        sumpost += posteriors[model]

    # Normalize
    for model in models:
        posteriors[model] /= sumpost

    return posteriors
