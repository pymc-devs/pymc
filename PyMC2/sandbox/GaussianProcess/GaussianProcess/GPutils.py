__docformat__='reStructuredText'

# TODO: Implement lintrans, allow obs_taus to be a huge matrix or an ndarray in observe().

from Covariance import Covariance
from Mean import Mean
from numpy import *
from numpy.linalg import solve, cholesky, eigh
from numpy.linalg.linalg import LinAlgError
from pylab import fill, plot, clf, axis
from PyMC2 import LikelihoodError

half_log_2pi = .5 * log(2. * pi)

def GP_logp(f,M,C,eff_zero = 1e-15):
    """
    logp = GP_logp(f,M,C,eff_zero = 1e-15)
    
    raises a LikelihoodError if f doesn't seem to
    be in the support (that is, if the deviation
    has a significant component in a direction with
    a very small eigenvalue).
    
    Note: I thought about excluding contributions to
    the log-probability from directions with small
    eigenvalues if the deviation has a small component
    in those directions. 
    
    This is numerically preferable, but unacceptable 
    for comparing p(f|M,C1) and  p(f|M,C2). Reason: 
    if f is in the support of both C1 and C2, but C2 
    has more zero eigenvalues than C1, C2 should be 
    very strongly preferred. However, if you discount 
    contributions from the forbidden eigendirections, 
    you won't see this.
    """
    dev = (f-M).ravel()
    logp = 0.
    
    max_eval = max(C.Eval)
    scaled_evals = C.Eval / max_eval
    eff_inf = 1. / eff_zero
    
    dot = asarray((C.Evec.T * dev)).ravel()
    dot_sq = dot ** 2
    
    if (scaled_evals == 0. and dot_sq > 0.).any():
        raise LikelihoodError
        
    if (dot_sq / scaled_evals > eff_inf).any():
        raise LikelihoodError
    
    logp = -.5 * sum(half_log_2pi * C.logEval + dot_sq / C.Eval) 
            
    return logp

def plot_envelope(M,C,mesh=None):
    """
    plot_envelope(M,C[, mesh])
    
    plots the pointwise mean +/- sd envelope defined by M and C
    along their base mesh.
    
    :Arguments:
        - M: A Gaussian process mean.
        - C: A Gaussian process covariance
        - mesh: The mesh on which to evaluate the mean and cov.
                Base mesh used by default.
    """
    if mesh is None:
        x = concatenate((C.base_mesh, C.base_mesh[::-1]))
        sig = sqrt(diag(C))
        y = concatenate((M-sig, (M+sig)[::-1]))
        clf()
    
        fill(x,y,facecolor='.8',edgecolor='1.')
        plot(C.base_mesh,M,'k-.')

    else:
        x=concatenate((mesh, mesh[::-1]))
        sig = sqrt(diag(C(mesh,mesh)))
        mean = M(mesh)
        y=concatenate((mean-sig, (mean-sig)[::-1]))
        clf()
        fill(x,y,facecolor='.8',edgecolor='1.')
        plot(mesh, mean, 'k-.')
    

def observe(C, obs_mesh, obs_taus = None, lintrans = None, obs_vals = None, M=None):
    """
    observe(C, obs_mesh[, obs_taus, lintrans, obs_vals, M])
    
    Updates C and M to condition f on the value of obs_vals in:
    
    obs_vals ~ N(lintrans * f(obs_mesh), obs_taus)
    f ~ GP(M, C)
    
    :Arguments:
        - C: A Gaussian process covariance.
        - obs_mesh: 
        - obs_taus:
        - lintrans:
        - obs_vals:
        - M:
    
    :SeeAlso:
    GPMean, GPCovariance, GPRealization, GaussianProcess
    """

    base_mesh = C.base_reshape

    cov_fun = C.eval_fun
    cov_params = C.params

    if M is not None:

        mean_params = M.mean_params
        mean_fun = M.eval_fun

    M.conditioned = True
    C.conditioned = True        

    ndim = base_mesh.shape[1]
    if obs_mesh.shape[0] == 0:
        return
    obs_mesh = obs_mesh.reshape(-1,ndim)
    obs_taus = obs_taus.ravel()
    obs_vals = obs_vals.ravel()
    
    have_obs_taus = obs_taus is not None
    combined_mesh = vstack((base_mesh, obs_mesh))
    
    combined_len = combined_mesh.shape[0]
    base_len = base_mesh.shape[0]
    obs_len = obs_mesh.shape[0]
    
    if M is not None:
        M.obs_mesh = obs_mesh
        M.obs_taus = obs_taus
        M.lintrans = lintrans
        M.obs_vals = obs_vals
        M.obs_len = obs_len
        
    C.obs_mesh = obs_mesh
    C.obs_taus = obs_taus
    C.lintrans = lintrans
    C.obs_len = obs_len
                        
    try:
        
        RF = cov_fun(base_mesh, obs_mesh, **cov_params)
        if lintrans is not None:
            RF = RF * lintrans
        
        Q = cov_fun(obs_mesh, obs_mesh, **cov_params)
        mean_under = mean_fun(obs_mesh, **mean_params)
        if have_obs_taus:
            Q += diag(1./obs_taus)
        Q_mean_under = solve(Q,(asmatrix(obs_vals).T - mean_under))            
            
        M.Q_mean_under = Q_mean_under
        C.Q = Q            
        
        tempC = C.view(matrix)
        tempC -= RF * solve(Q, RF.T)
        
        tempM = M.view(ndarray)
        tempM += (RF * Q_mean_under).view(ndarray).reshape(M.shape)
        
        
    except LinAlgError:
           
        combined_cov = cov_fun(combined_mesh,combined_mesh,**cov_params)
        combined_mean = mean_fun(combined_mesh,**mean_params)       
        
        for i in xrange(obs_len):
                
            if have_obs_taus:
                obs_V_now = 1./obs_taus[i]
            else:
                obs_V_now = 0.
            
            RF = combined_cov[:, base_len + i]
            Q = combined_cov[base_len + i, base_len + i] + obs_V_now
                
            combined_cov -= RF* RF.T / Q
            
            if M is not None:
                combined_mean += (RF * (obs_vals[i] - combined_mean[base_len + i])).T / Q
        
        tempM = M.view(ndarray)
        tempC = C.view(matrix)
        for i in xrange(base_len):
            tempM[i] = combined_mean[i]
            for j in xrange(base_len):
                tempC[i,j] = combined_cov[i,j]
    
    C.update_sig_and_e()
        