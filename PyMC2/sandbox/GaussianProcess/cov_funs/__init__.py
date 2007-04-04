from fcov import *
from numpy import matrix
try:
    from Matern import *
    pycov_functions = ['Matern', 'NormalizedMatern']
except ImportError:
    print 'Warning, Matern covariance functions not available. Install scipy for access to these.'
    pycov_functions = []
    
fcov_functions = ['axi_gauss', 'axi_exp']


def fwrap(cov_fun):
    
    def wrapper(x,y,*args,**kwargs):
        symm = (x is y)
        kwargs['symm']=symm
        return cov_fun(x,y,*args,**kwargs).view(matrix)
    
    return wrapper
        

for name in fcov_functions:
    locals()[name] = fwrap(locals()[name])
    