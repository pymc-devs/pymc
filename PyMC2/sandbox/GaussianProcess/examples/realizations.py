# GaussianProcess/examples/realizations.py

# Import the mean and covariance
from mean_and_cov import *

# Generate realizations
f_list=[]
for i in range(3):
    f = Realization(M, C)
    f_list.append(f)

# Plot
if __name__ == '__main__':
    x=arange(-1.,1.,.01)

    clf()

    plot_envelope(M, C, mesh=x)

    for f in f_list:
        plot(x, f(x))
    plot(obs_x,d,'k.',markersize=16)

    xlabel('x')    
    ylabel('f(x)')
    title('Three realizations from the observed GP')
    axis('tight')