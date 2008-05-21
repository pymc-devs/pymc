from pymc.gp import *
from pymc.gp.cov_funs import matern
from numpy import *

C = Covariance(eval_fun = matern.euclidean, diff_degree = 1.4, amp = .4, scale = 1.)
# C = FullRankCovariance(eval_fun = matern.euclidean, diff_degree = 1.4, amp = .4, scale = 1.)
# C = NearlyFullRankCovariance(eval_fun = matern.euclidean, diff_degree = 1.4, amp = .4, scale = 1.)

#### - Plot - ####
if __name__ == '__main__':
    from pylab import *
    
    x=arange(-1.,1.,.01)
    clf()
    
    # Plot the covariance function
    subplot(1,2,1)
    
    imshow(C(x,x),origin='lower',extent=(-1.,1.,-1.,1.))
    
    xlabel('x')
    ylabel('y')
    title('C(x,y)')
    axis('tight')
    colorbar()
    
    # Plot a slice of the covariance function
    subplot(1,2,2)
    
    plot(x,C(x,0),'k-')
    
    xlabel('x')
    ylabel('C(x,0)')
    title('A slice of C')
    
    # show()