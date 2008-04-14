from isotropic_cov_funs import *
from cov_utils import *
from bases import *

extra_parameters = {'gaussian': {'': ''}, 
                'pow_exp': {'pow': 'The exponent in the exponential.'}, 
                'matern': {'diff_degree': 'The degree of differentiability of realizations.'},
                'sphere': {'': ''},
                'quadratic': {'phi': 'The characteristic (scaled) distance of decorrelation.'}}


for name in extra_parameters.iterkeys():
    
    # Wrap the function
    locals()[name].__name__ = name
    locals()[name].extra_parameters = extra_parameters[name]
    locals()[name] = covariance_function_bundle(locals()[name])
