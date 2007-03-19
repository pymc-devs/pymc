"""
The DisasterSampler example.

.7s for 10k iterations with SamplingMethod,step() commented out,

"""
from numpy.testing import *
from pylab import *
PLOT=False

class test_Model(NumpyTestCase):
    def check(self):
        from PyMC2 import Model
        from PyMC2.examples import DisasterModel
        M = Model(DisasterModel)
        M.sample(500,0,10,verbose=False)
        if PLOT:
            # It would be nicer to write plot(M.trace(switchpoint)), since switchpoint is local to M.
            plot(M.s.trace())
            title('switchpoint')
            figure()
            plot(M.e.trace())
            title('early mean')            
            figure()
            title('late mean')
            plot(M.l.trace())
            show()

if __name__ == '__main__':
    NumpyTest().run()
    
