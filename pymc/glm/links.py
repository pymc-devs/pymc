import theano.tensor

try:
    # Statsmodels is optional
    from statsmodels.genmod.families.links import (identity, logit, inverse_power, log)
except:
    identity, logit, inverse_power, log = [None] * 4

__all__ = ['Identity', 'Logit', 'Inverse', 'Log']

class LinkFunction(object):
    """Base class to define link functions.

    If initialization via statsmodels is desired, define sm.
    """

    def __init__(self, theano_link=None, sm_link=None):
        if theano_link is not None:
            self.theano = theano_link
        if sm_link is not None:
            self.sm = sm_link

class Identity(LinkFunction):
    theano = lambda self, x: x
    sm = identity

class Logit(LinkFunction):
    theano = theano.tensor.nnet.sigmoid
    sm = logit

class Inverse(LinkFunction):
    theano = theano.tensor.inv
    sm = inverse_power

class Log(LinkFunction):
    theano = theano.tensor.log
    sm = log
