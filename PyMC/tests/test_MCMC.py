"""
The DisasterSampler example.

.7s for 10k iterations with SamplingMethod,step() commented out,

"""

from PyMC import Model
from PyMC.examples import DisasterModel
M = Model(DisasterModel)


# Sample
M.sample(50000,0,100)


# Get and plot traces.
from pylab import *

# It would be nicer to write plot(M.trace(switchpoint)), since switchpoint is local to M.
plot(M.switchpoint.trace.gettrace(thin=100))
title('switchpoint')
figure()
plot(M.early_mean.trace.gettrace(thin=100))
title('early mean')
figure()
title('late mean')
plot(M.late_mean.trace.gettrace(thin=100))
show()
