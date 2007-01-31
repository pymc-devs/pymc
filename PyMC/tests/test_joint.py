from PyMC import Model, JointMetropolis
from PyMC.examples import model_for_joint

M = Model(model_for_joint)

M.sample(iter=50000,burn=1000,thin=10)

from pylab import *
plot(M.A.trace()[:,0],M.B.trace()[:,0],'b.')
title('First elements')
xlabel('A')
ylabel('B')
figure()
plot(M.A.trace()[:,1],M.B.trace()[:,1],'b.')
title('Second elements')
xlabel('A')
ylabel('B')
figure()
plot(M.A.trace()[:,0])
show()