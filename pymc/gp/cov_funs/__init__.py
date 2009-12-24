from isotropic_cov_funs import *
from cov_utils import *
from bases import *
from wrapped_distances import *
import isotropic_cov_funs
from brownian import *



extra_parameters = {'gaussian': {'': ''},
                'pow_exp': {'pow': 'The exponent in the exponential.'},
                'exponential':{'':''},
                'matern': {'diff_degree': 'The degree of differentiability of realizations.'},
                'sphere': {'': ''},
                'quadratic': {'phi': 'The characteristic (scaled) distance of decorrelation.'},
                'exponential': {'': ''}}


for name in extra_parameters.iterkeys():
    locals()[name] = covariance_function_bundle(name, 'isotropic_cov_funs', extra_parameters[name], ampsq_is_diag=True)

