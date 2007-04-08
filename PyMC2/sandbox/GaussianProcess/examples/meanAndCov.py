# GaussianProcess/examples/meanAndCov.py

# Import the covariance from cov.py
from cov import *

# Generate mean
def linfun(x, a, b, c):
    return a * x ** 2 + b * x + c

M = Mean(eval_fun = linfun, C = C, a = 1., b = .5, c = 2.)

# Plot
if __name__ == '__main__':
    x=arange(-1.,1.,.1)

    clf()
    plot(x, M(x), 'k-')
    xlabel('x')
    ylabel('M(x)')
    axis('tight')
    show()
