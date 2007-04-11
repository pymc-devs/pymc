import PyMC_model
from PyMC2 import *
from pylab import *

x = arange(-1.,1.,.1)
obs_x = array([-.5,.5])


GPSampler = Sampler(PyMC_model)


# Uncomment this for a long run.
# GPSampler.sample(iter=5000,burn=1000,thin=100,verbose=False)

# Uncomment this for a medium run.
GPSampler.sample(iter=500,burn=0,thin=10,verbose=False)

# Uncomment this for a short run.
# GPSampler.sample(iter=50,burn=0,thin=1,verbose=False)

if __name__ == '__main__':

    clf()
    
    # Plot posterior of f
    N_samps = GPSampler.f.trace().shape[0]
    subplot(1,2,1)
    for i in range(0,N_samps):
        plot(x,GPSampler.f.trace()[i,:])
        plot(obs_x,GPSampler.d.value,'k.',markersize=16)
    title('Some samples of f')
    
    subplot(1,2,2)
    plot(GPSampler.f.trace()[:,10])
    title('The trace of f near midpoint')

    # Plot posterior of C and tau
    figure()
    subplot(2,2,1)
    plot(GPSampler.C_diff_degree.trace())
    title("Degree of differentiability of f")
    
    subplot(2,2,2)
    plot(GPSampler.C_amp.trace())
    title("Pointwise prior variance of f")
    
    subplot(2,2,3)
    plot(GPSampler.C_scale.trace())
    title("X-axis scaling of f")
    
    subplot(2,2,4)
    plot(GPSampler.tau.trace())
    title('Observation precision')
    
    
    # Plot posterior of M
    figure()
    subplot(1,3,1)
    plot(GPSampler.M_a.trace())
    title("Quadratic coefficient of M")
    
    subplot(1,3,2)
    plot(GPSampler.M_b.trace())
    title("Linear coefficient of M")    
    
    subplot(1,3,3)
    plot(GPSampler.M_c.trace())
    title("Constant term of M")
    
    show()
    
