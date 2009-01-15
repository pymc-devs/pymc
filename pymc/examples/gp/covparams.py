from pymc.gp import *
from pymc.gp.cov_funs import *
from numpy import *

# Covariance
C = Covariance(eval_fun = matern.euclidean, diff_degree = 1.4, amp = 1., scale = 1.)
# C = Covariance(eval_fun = pow_exp.euclidean, pow=1., amp=1., scale=1.)
# C = Covariance(eval_fun = quadratic.euclidean, phi=1., amp=1., scale=.2)
# C = Covariance(eval_fun = gaussian.euclidean, amp=1., scale=1.)
# C = Covariance(eval_fun = sphere.euclidean, amp=1., scale=.5)

# Mean
def zero_fun(x):
    return 0.*x
M = Mean(zero_fun)

#### - Plot - ####
if __name__ == '__main__':
    from pylab import *

    x=arange(-1.,1.,.01)

    close('all')
    figure()

    # Plot the covariance function
    subplot(2,2,1)

    contourf(x,x,C(x,x).view(ndarray),origin='lower',extent=(-1.,1.,-1.,1.),cmap=cm.bone)

    xlabel('x')
    ylabel('y')
    title('C(x,y)')
    axis('tight')
    colorbar()

    # Plot a slice of the covariance function
    subplot(2,2,2)

    plot(x,C(x,0).view(ndarray).ravel(),'k-')
    axis([-1,1,0,1])

    xlabel('x')
    ylabel('C(x,0)')
    title('A slice of C')

    subplot(2,1,2)

    # plot_envelope(M, C, mesh=x)
    for i in range(3):
        f = Realization(M, C)
        plot(x, f(x))

    xlabel('x')
    ylabel('f(x)')
    title('Three realizations')
    axis([-1,1,-2,2])

    # show()
