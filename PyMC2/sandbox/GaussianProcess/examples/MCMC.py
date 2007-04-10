import PyMC_observed_form
import PyMC_unobserved_form
from PyMC2 import *
from pylab import *

x = arange(-1.,1.,.1)
obs_x = array([-.5,.5])


GPSampler_unobs = Sampler(PyMC_unobserved_form)
GPSampler_obs = Sampler(PyMC_observed_form)

#
# Uncomment one of the following lines, depending on whether you want to
# try the observed or the unobserved form.
#
# sampler_now = GPSampler_unobs
sampler_now = GPSampler_obs

#
# Uncomment this for a long run.
#
sampler_now.sample(iter=5000,burn=1000,thin=100,verbose=False)

#
# Uncomment this for a short run.
#
# sampler_now.sample(iter=50,burn=0,thin=1,verbose=False)

if __name__ == '__main__':

    clf()
    
    # Plot posterior of f
    N_samps = sampler_now.f.trace().shape[0]
    subplot(1,2,1)
    for i in range(0,N_samps):
        plot(x,sampler_now.f.trace()[i,:])
        plot(obs_x,sampler_now.d.value,'k.',markersize=16)
    title('Some samples from f')
    
    subplot(1,2,2)
    plot(sampler_now.f.trace()[:,10])
    title('The trace of f near midpoint')


    # Plot posterior of C and tau
    figure()
    subplot(2,2,1)
    plot(sampler_now.C_diff_degree.trace())
    title("Degree of differentiability of f")

    subplot(2,2,2)
    plot(sampler_now.C_amp.trace())
    title("Pointwise prior variance of f")

    subplot(2,2,3)
    plot(sampler_now.C_scale.trace())
    title("X-axis scaling of f")

    subplot(2,2,4)
    plot(sampler_now.tau.trace())
    title('Observation precision')


    # Plot posterior of M
    figure()
    subplot(1,3,1)
    plot(sampler_now.M_a.trace())
    title("Quadratic coefficient of M")
    
    subplot(1,3,2)
    plot(sampler_now.M_b.trace())
    title("Linear coefficient of M")    
    
    subplot(1,3,3)
    plot(sampler_now.M_c.trace())
    title("Constant term of M")

    show()

