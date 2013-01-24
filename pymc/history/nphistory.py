'''
Created on Mar 15, 2011

@author: jsalvatier
'''
import numpy as np
class NpHistory(object):
    """
    encapsulates the recording of a process chain
    """
    def __init__(self, max_draws = 10000):
        self.max_draws = max_draws
        self.samples = {}
        self.nsamples = 0
    
    def __add__(self, point):
        """
        records the position of a chain at a certain point in time
        """
        if self.nsamples < self.max_draws:
            for var, value in point.iteritems():
                try :
                    s = self.samples[var]
                except: 
                    s = np.empty((self.max_draws,) + value.shape) 
                    self.samples[var] = s 
                    
                s[self.nsamples,...] = value

            self.nsamples += 1
        else :
            raise ValueError('out of space!')
        return self
        
    def __getitem__(self, key):
        return self.samples[key][0:self.nsamples,...]

class MultiHistory(object): 
    def __init__(self, histories): 
        self.histories = histories 

    def __getitem__(self, key): 
        return [h[key] for h in self.histories]
    def combined():
        h = NpHistory()
        for k in histories[0].samples: 
            h.samples[k] = concatenate([s[k] for s in self.histories])
            h.nsamples = h[k].shape[0]
        return h
