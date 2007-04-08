from fcov import *
from cov_utils import *

try:
    from Matern import *
    pycov_functions = ['Matern', 'NormalizedMatern']
except ImportError:
    print 'Warning, Matern covariance functions not available. Install scipy for access to these.'
    pycov_functions = []
    
fcov_functions = ['axi_gauss', 'axi_exp']

for name in fcov_functions:
    locals()[name] = fwrap(locals()[name])
    