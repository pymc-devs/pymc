from Mean import Mean
from Covariance import Covariance
from Realization import Realization
from GPutils import *
import cov_funs

# PyMC-specific stuff
try:
    
    from GP import GaussianProcess
    from GPSamplingMethods import *

except ImportError:
    
    class GaussianProcess(object):
        def __init__(self, *args, **kwargs):
            raise ImportError, 'You must install PyMC to use GaussianProcess.'
            
    class GPMetropolis(object):
        def __init(self, *args, **kwargs):
            raise ImportError, 'You must install PyMC to use GPMetropolis.'