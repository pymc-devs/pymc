# 30-3-07 AP: Changed Sampler to Model.

"""
This demo compares three different models for the coal mining disasters data.
It doesn't use RJMCMC, it just integrates out the model stochastics by sampling
their values conditional on their parents. Nothing fancy, but it works OK.

The biggest problem is that the variance of the samples of the model likelihoods
can be pretty big, so it takes a lot of samples to get a good estimate of the
model posterior probabilities. This'll probably be worse for more complicated
models. Maybe there's a literature on this problem.
"""

from pymc import Model
from pymc.MultiModelInference import weight
from numpy import log
from pymc.examples import model_1, model_2, model_3

from numpy.testing import *
class test_model_ave(TestCase):
    def test(self):
        
        # Changepoint model
        M1 = Model(model_1)
        
        # Constant rate model
        M2 = Model(model_2)
        
        # Exponentially varying rate model
        M3 = Model(model_3)
        
        # print 'Docstring of model 1:'
        # print model_1.__doc__
        # print 'Docstring of model 2:'
        # print model_2.__doc__
        # print 'Docstring of model 3:'
        # print model_3.__doc__
        
        
        posterior = weight([M1,M2,M3],10000)

        # print 'Log posterior probability of changepoint model: ',log(posterior[M1])
        # print 'Log posterior probability of constant rate model: ',log(posterior[M2])
        # print 'Log posterior probability of linearly varying rate model: ',log(posterior[M3])

        
        assert(abs(log(posterior[M1]))<1e-10)
        assert(abs(log(posterior[M2])+30)<4)
        assert(abs(log(posterior[M3])+60)<30)
        

if __name__=='__main__':
    unittest.main()
