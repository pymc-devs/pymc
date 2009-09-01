# Import the mean and covariance
from mean import M
from cov import C
from pymc.gp import *
from numpy import *

# Impose observations on the GP
obs_x = array([-.5,.5])
V = array([.002,.002])*0
data = array([3.1, 2.9])
observe(M=M,
        C=C,
        obs_mesh=obs_x,
        obs_V = V,
        obs_vals = data)

# Generate realizations
f_list=[]
for i in range(3):
    f = Realization(M, C)
    f_list.append(f)

x=arange(-1.,1.,.01)

#### - Plot - ####
if __name__ == '__main__':
    from pylab import *

    x=arange(-1.,1.,.01)

    clf()

    plot_envelope(M, C, mesh=x)

    for f in f_list:
        plot(x, f(x))

    xlabel('x')
    ylabel('f(x)')
    title('Three realizations of the observed GP')
    axis('tight')


    # show()
