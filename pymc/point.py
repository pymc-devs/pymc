import numpy as np

def clean_point(d) : 
    return dict([(k,np.array(v)) for (k,v) in d.iteritems()]) 
 
class PointFunc(object): 
    def __init__(self, f):
        self.f = f
    def __call__(self,state):
        return self.f(**state)
