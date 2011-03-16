'''
Created on Mar 15, 2011

@author: jsalvatier
'''
import numpy as np
class NpHistory(object):
    """
    encapsulates the recording of a process chain
    should handle any burn-in, thinning (though this can also be handled at the sampler level) etc.
    """
    def __init__(self, model, max_draws):
        self.max_draws = max_draws
        samples = {}
        for var in model.free_vars: 
            samples[str(var)] = np.empty((int(max_draws),) + var.dshape)
            
        self._samples = samples
        self.nsamples = 0
    
    def record(self, chain_state, step_method):
        """
        records the position of a chain at a certain point in time
        as well as any variables of interest in the step method
        """
        if self.nsamples < self.max_draws:
            for var, sample in self._samples.iteritems():
                sample[self.nsamples,...] = chain_state.values[var]
            self.nsamples += 1
        else :
            raise ValueError('out of space!')
        
    def __getitem__(self, key):
        return self._samples[key][0:self.nsamples,...]