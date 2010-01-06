import PyMCmodel
from pymc import *
from pylab import *
from PyMCmodel import *

x = linspace(-1,1,400)

GPSampler = MCMC(PyMCmodel)
# Uncomment this to use the GPNormal step method instead of the default GPMetropolis
GPSampler.use_step_method(gp.GPEvaluationGibbs, GPSampler.submod, GPSampler.V, GPSampler.d)

GPSampler.isample(iter=5000,burn=1000,thin=100)

# Uncomment this for a medium run.
# GPSampler.isample(iter=500,burn=0,thin=10)

# Uncomment this for a short run.
# GPSampler.isample(iter=50,burn=0,thin=1)

if __name__ == '__main__':

    N_samps = len(GPSampler.submod.f.trace())

    close('all')

    mid_traces = []
    subplot(1,2,1)
    for i in range(0,N_samps):
        f=GPSampler.submod.f.trace()[i](x)
        plot(x,f)
        mid_traces.append(f[len(f)/2])
        plot(fmesh,GPSampler.d.value,'k.',markersize=16)
    axis([x.min(),x.max(),-5.,10.])
    title('Some samples of f')

    subplot(1,2,2)

    plot(mid_traces)
    title('The trace of f at midpoint')

    # Plot posterior of C and tau
    figure()
    subplot(2,2,1)
    try:
        plot(GPSampler.diff_degree.trace())
        title("Degree of differentiability of f")
    except:
        pass

    subplot(2,2,2)
    plot(GPSampler.amp.trace())
    title("Pointwise prior variance of f")

    subplot(2,2,3)
    plot(GPSampler.scale.trace())
    title("X-axis scaling of f")

    # subplot(2,2,4)
    # plot(GPSampler.V.trace())
    # title('Observation precision')


    # Plot posterior of M
    figure()
    subplot(1,3,1)
    plot(GPSampler.a.trace())
    title("Quadratic coefficient of M")

    subplot(1,3,2)
    plot(GPSampler.b.trace())
    title("Linear coefficient of M")

    subplot(1,3,3)
    plot(GPSampler.c.trace())
    title("Constant term of M")

    # show()

