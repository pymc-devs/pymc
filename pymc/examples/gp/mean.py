from pymc.gp import *

# Generate mean
def quadfun(x, a, b, c):
    return (a * x ** 2 + b * x + c)

M = Mean(quadfun, a = 1., b = .5, c = 2.)

#### - Plot - ####
if __name__ == '__main__':
    from pylab import *
    x=arange(-1.,1.,.1)

    clf()
    plot(x, M(x), 'k-')
    xlabel('x')
    ylabel('M(x)')
    axis('tight')
    # show()
