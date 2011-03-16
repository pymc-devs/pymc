import PyMCmodel
from pymc import *
from pylab import *

x = linspace(-1,1,400)

n_fmesh = 21
fmesh_is_obsmesh = False

# IDEA: SubsetMetropolis, proposes some elements of a GP conditional on others.
# Ooh, or GP hit-and-run.
# Or just a Metropolis in covariance directions, with tunable jump size.

GPSampler = MCMC(PyMCmodel.make_model(n_fmesh, fmesh_is_obsmesh))
obs_V = utils.value(GPSampler.V)

if not fmesh_is_obsmesh:
    # offdiag = GPSampler.sm.C.value(GPSampler.fmesh, utils.value(GPSampler.obs_locs))
    # inner = GPSampler.sm.C.value(utils.value(GPSampler.obs_locs), utils.value(GPSampler.obs_locs)) + obs_V*np.eye(offdiag.shape[1])
    # sm_cov = np.asarray(GPSampler.sm.C_eval.value - offdiag*inner.I*offdiag.T)/10000.
    # GPSampler.use_step_method(gp.GPParentAdaptiveMetropolis, GPSampler.sm.f_eval, cov=sm_cov)
    GPSampler.use_step_method(gp.GPEvaluationMetropolis, GPSampler.sm.f_eval, proposal_sd = .01)
else:
    GPSampler.use_step_method(gp.GPEvaluationGibbs, GPSampler.sm, GPSampler.V, GPSampler.d)

GPSampler.assign_step_methods()
sm = GPSampler.step_method_dict[GPSampler.sm.f_eval][0]

GPSampler.isample(iter=50000,burn=0,thin=1000)

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
    subplot(1,2,1)
    for i in range(0,N_samps):
        f=GPSampler.sm.f.trace()[i](x)
        plot(x,f)
        mid_traces.append(f[len(f)/2])
        plot(obs_locs,GPSampler.d.value,'k.',markersize=16)
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
    plot(GPSampler.phi.trace())
    title("Pointwise prior variance of f")
    
    subplot(2,2,3)
    plot(GPSampler.theta.trace())
    title("X-axis scaling of f")
    
    if isinstance(GPSampler.V, pymc.Variable):
        subplot(2,2,4)
        plot(GPSampler.V.trace())
        title('Observation variance')
    
    
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
    
    try:
        figure()
        plot(GPSampler.obs_locs.trace())
        for al in GPSampler.actual_obs_locs:
            plot([0,N_samps],[al,al],'k-.')
        title('Observation locations')
    except:
        pass
    
    # show()
    
