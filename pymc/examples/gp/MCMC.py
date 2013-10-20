from . import PyMCmodel
from pymc import *
from pylab import *
import matplotlib
matplotlib.rcParams['axes.facecolor'] = 'w'

x = linspace(-1, 1, 400)

n_fmesh = 51
fmesh_is_obsmesh = False

GPSampler = MCMC(PyMCmodel.make_model(n_fmesh, fmesh_is_obsmesh))

GPSampler.assign_step_methods()
sm = GPSampler.step_method_dict[GPSampler.sm.f_eval][0]

GPSampler.isample(iter=5000, burn=2500, thin=100)

# Uncomment this for a medium run.
# GPSampler.isample(iter=500,burn=0,thin=10)

# Uncomment this for a short run.
# GPSampler.isample(iter=50,burn=0,thin=1)

if __name__ == '__main__':

    N_samps = len(GPSampler.sm.f.trace())
    fmesh = GPSampler.fmesh
    obs_locs = GPSampler.actual_obs_locs

    close('all')

    mid_traces = []
    figure(figsize=(12, 6))
    subplot(1, 2, 1)
    for i in range(0, N_samps):
        f = GPSampler.sm.f.trace()[i](x)
        plot(x, f, linewidth=2)
        mid_traces.append(f[len(f) / 2])
        plot(obs_locs, GPSampler.d.value, 'k.', markersize=16)
    axis([x.min(), x.max(), -2., 4.])
    xlabel('x')
    ylabel('f(x)')
    title('Some samples of f')

    subplot(1, 2, 2)

    plot(mid_traces, linewidth=4)
    xlabel('iteration')
    ylabel('f(0)')
    title('The trace of f at midpoint')

    # Plot posterior of C and tau
    figure()
    subplot(2, 2, 1)
    plot(GPSampler.nu.trace())
    title("Degree of differentiability of f")

    subplot(2, 2, 2)
    plot(GPSampler.phi.trace())
    title("Pointwise prior variance of f")

    subplot(2, 2, 3)
    plot(GPSampler.theta.trace())
    title("X-axis scaling of f")

    subplot(2, 2, 4)
    plot(GPSampler.V.trace())
    title('Observation variance')

    # Plot posterior of M
    figure()
    subplot(1, 3, 1)
    plot(GPSampler.a.trace())
    title("Quadratic coefficient of M")

    subplot(1, 3, 2)
    plot(GPSampler.b.trace())
    title("Linear coefficient of M")

    subplot(1, 3, 3)
    plot(GPSampler.c.trace())
    title("Constant term of M")

    try:
        figure()
        plot(GPSampler.obs_locs.trace())
        for al in GPSampler.actual_obs_locs:
            plot([0, N_samps], [al, al], 'k-.')
        title('Observation locations')
    except:
        pass

    # show()
