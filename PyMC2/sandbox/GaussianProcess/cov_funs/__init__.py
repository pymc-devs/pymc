from fcov import *
from Matern import *

fcov_functions = ['axi_gauss', 'axi_exp']
pycov_functions = ['Matern', 'NormalizedMatern']

def fwrap(cov_fun):
    
    def wrapper(C,x,y,*args,**kwargs):
        symm = (x is y)
        kwargs['symm']=symm
        return cov_fun(C,x,y,*args,**kwargs)
    
    return wrapper
        

for name in fcov_functions:
    locals()[name] = fwrap(locals()[name])
    