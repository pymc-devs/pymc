from PyMC2 import Model, JointMetropolis
from PyMC2.examples import model_for_joint

M = Model(model_for_joint)

M.sample(iter=20000,burn=0,thin=100)

from pylab import *
plot(M.A.trace.gettrace()[:,0],M.B.trace.gettrace()[:,0],'b.')
title('First elements')
xlabel('A')
ylabel('B')
figure()
plot(M.A.trace.gettrace()[:,1],M.B.trace.gettrace()[:,1],'b.')
title('Second elements')
xlabel('A')
ylabel('B')
figure()
plot(M.A.trace.gettrace()[:,0])
show()
