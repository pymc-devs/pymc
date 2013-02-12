import numpy as np

def clean_point(d) : 
    return dict([(k,np.atleast_1d(v)) for (k,v) in d.iteritems()]) 
 
