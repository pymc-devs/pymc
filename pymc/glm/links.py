import statsmodels.api as sm
import theano.tensor

from abc import ABCMeta

__all__ = ['Identity', 'Logit']

class LinkFunction(object):
    __metaclass__ = ABCMeta

    def __init__(self, theano_link=None, sm_link=None):
    	if theano_link is not None:
            self.theano = theano_link
        if sm_link is not None:
            self.sm = sm_link

class Identity(LinkFunction):
    theano = lambda self, x: x
    sm = sm.families.links.identity

class Logit(LinkFunction):
    theano = theano.tensor.nnet.sigmoid
    sm = sm.families.links.logit
