from pymc.gp.cov_funs import *
from numpy import *
from copy import copy
from pymc.gp import *

N=100

# Generate mean
def quadfun(x, a, b, c):
    return (a * x ** 2 + b * x + c)

M = Mean(eval_fun = quadfun, a = 1., b = .5, c = 2.)

# Generate basis covariance
coef_cov = ones(2*N+1, dtype=float)
for i in xrange(1,len(coef_cov)):
    coef_cov[i] = 1./int(((i+1)/2))**2.5

basis = fourier_basis([N])

C = SeparableBasisCovariance(basis,coef_cov,xmin = [-2.], xmax = [2.])

obs_x = array([-.5,.5])
V = array([.002,.002])
data = array([3.1, 2.9])

observe(M=M,
        C=C,
        obs_mesh=obs_x,
        obs_V = V,
        obs_vals = data)

if __name__ == '__main__':
    from pylab import *

    close('all')
    x=arange(-1.,1.,.01)

    figure()
    # Plot the covariance function
    subplot(1,2,1)


    contourf(x,x,C(x,x).view(ndarray),origin='lower',extent=(-1.,1.,-1.,1.),cmap=cm.bone)

    xlabel('x')
    ylabel('y')
    title('C(x,y)')
    axis('tight')
    colorbar()

    # Plot a slice of the covariance function
    subplot(1,2,2)

    plot(x,C(x,0.).view(ndarray).ravel(),'k-')

    xlabel('x')
    ylabel('C(x,.5)')
    title('A slice of C')

    figure()
    plot_envelope(M,C,x)
    for i in range(3):
        f = Realization(M,C)
        plot(x,f(x))
        title('Three realizations')


    # show()
