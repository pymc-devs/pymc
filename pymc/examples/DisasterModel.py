# DisasterModel.py

"""
A model for the D data with a changepoint

changepoint ~ U(0,111)
e ~ Exp(1.)
l ~ Exp(1.)
D[t] ~ Po(e if t <= s, l otherwise)
"""
__all__ = ['s','e','l','D']

from pymc import *
from numpy import *

D_array =   array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                   3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                   2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                   1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                   0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                   3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

s = DiscreteUniform('s', lower=0, upper=110)   
e = Exponential('e', beta=1)
l = Exponential('l', beta=1)   

@deterministic
def r(s=s, e=e, l=l):
  out = np.empty(111)
  out[:s] = e
  out[s:] = l
  return out
D = Poisson('D', mu=r, value=D_array, isdata=True)