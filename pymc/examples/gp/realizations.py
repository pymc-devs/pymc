# Import the mean and covariance
from .mean import M
from .cov import C
from pymc.gp import *

# Generate realizations
f_list = [Realization(M, C) for i in range(3)]

# - Plot - ####
if __name__ == '__main__':
    from pylab import *

    x = arange(-1., 1., .01)
    clf()

    plot_envelope(M, C, x)

    for f in f_list:
        plot(x, f(x))

    xlabel('x')
    ylabel('f(x)')
    title('Three realizations of the GP')
    axis('tight')

    # show()
