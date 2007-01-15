"""
The DisasterSampler example.
"""

from proposition5 import Model
import model_1
M = Model(model_1)

# Sample
M.sample(10000,100,10)


# Get and plot traces.
from pylab import *

# It would be nicer to write plot(M.trace(switchpoint)), since switchpoint is local to M.
plot(M.switchpoint.trace())
title('switchpoint')
figure()
plot(M.early_mean.trace())
title('early mean')
figure()
title('late mean')
plot(M.late_mean.trace())
show()
