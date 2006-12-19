"""
The DisasterSampler example.
"""

from proposition5 import *
import model_1
M = Model(model_1)

# Sample
M.sample(5000,1000,10)

"""
# Get and plot traces.
from pylab import *

# It would be nicer to write plot(M.trace(switchpoint)), since switchpoint is local to M.
plot(M.trace(M.switchpoint))
title('switchpoint')
figure()
plot(M.trace(M.early_mean))
title('early mean')
figure()
title('late mean')
plot(M.trace(M.late_mean))
show()
"""
