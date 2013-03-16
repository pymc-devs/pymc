'''
Created on Mar 15, 2011

@author: jsalvatier
'''
import numpy as np

__all__ = ['NpTrace', 'MultiTrace']

class NpTrace(object):
    """
    encapsulates the recording of a process chain
    """
    def __init__(self, max_draws = 10000):
        self.max_draws = max_draws
        self.samples = StrDict()
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

    def point(self, index):
        return StrDict((k, v[:self.nsamples][index]) for (k,v) in self.samples.iteritems())

class MultiTrace(object): 
    def __init__(self, traces): 
        self.traces = traces 

    def __getitem__(self, key): 
        return [h[key] for h in self.traces]
    def point(self, index): 
        return [h.point(index) for h in self.traces]

    def combined(self):
        h = NpTrace()
        for k in self.traces[0].samples: 
            h.samples[k] = np.concatenate([s[k] for s in self.traces])
            h.nsamples = h.samples[k].shape[0]
        return h


class StrDict(dict): 
    """
    All keys are cast to strings
    """
    def __getitem__(self,key): 
        return dict.__getitem__(self, str(key))

    def __setitem__(self,key,val):
        return dict.__setitem__(self, str(key), val)

    def __delitem__(self, key):
        return dict.__delitem__(self, str(key))
