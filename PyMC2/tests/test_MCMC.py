"""
The DisasterSampler example.

.7s for 10k iterations with SamplingMethod,step() commented out,

"""

from PyMC2 import Model
from PyMC2.examples import DisasterModel
M = Model(DisasterModel)
from pylab import *


def test():
    # Sample
    M.sample(50000,0,100)
    
    # It would be nicer to write plot(M.trace(switchpoint)), since switchpoint is local to M.
    plot(M.switchpoint.trace.gettrace())
    title('switchpoint')
    figure()
    plot(M.early_mean.trace.gettrace())
    title('early mean')
    figure()
    title('late mean')
    plot(M.late_mean.trace.gettrace())
    show()

if __name__=='__main__':
    test()
    
