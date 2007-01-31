from PyMC import Model, JointMetropolis
from PyMC.examples import model_for_joint

M = Model(model_for_joint)

M.sample(iter=50000,burn=0,thin=100)

from pylab import *
plot(M.A.trace.gettrace(thin=100)[:,0],M.B.trace.gettrace(thin=100)[:,0],'b.')
title('First elements')
xlabel('A')
ylabel('B')
figure()
plot(M.A.trace.gettrace(thin=100)[:,1],M.B.trace.gettrace(thin=100)[:,1],'b.')
title('Second elements')
xlabel('A')
ylabel('B')
figure()
plot(M.A.trace.gettrace(thin=100)[:,0])
show()