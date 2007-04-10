__docformat__='reStructuredText'

# TODO: Implement lintrans, allow obs_taus to be a huge matrix or an ndarray in observe().
# TODO: Use regularize_array in observe.
# TODO: You could speed up the one-at-a-time algorithm. Instead of computing the covariance of
# TODO: all called points together and Cholesky factorizing, you could do the called points one
# TODO: at a time, getting the value of each using the previous. Suck it up and do the operation count.

from numpy import *
from numpy.linalg import solve, cholesky, eigh
from numpy.linalg.linalg import LinAlgError
from pylab import fill, plot, clf, axis

try:
    from PyMC2 import LikelihoodError
except ImportError:
    class LikelihoodError(ValueError):
        pass

half_log_2pi = .5 * log(2. * pi)


def enlarge_covariance(base, offdiag, diag):
    old = base.shape[0]
    new = diag.shape[0]
    
    new_mat = asmatrix(zeros((old+new,old+new),dtype=float))
    new_mat[:old,:old] = base
    new_mat[:old,old:] = offdiag.T
    new_mat[old:,:old] = offdiag
    new_mat[old:,old:] = diag

    return new_mat


def regularize_array(A):
    """
    Takes an ndarray as an input.
    - If the array is one-dimensional, it's assumed to be an array of input values.
    - If the array is more than one-dimensional, its last index is assumed to curse
      over spatial dimension.
    
    Either way, the return value is at least two dimensional. A.shape[-1] gives the
    number of spatial dimensions.
    """
    if not isinstance(A,ndarray):
        A = array(A)
    
    if len(A.shape) <= 1:
        return A.reshape(-1,1)
        
    else:
        return asarray(A, dtype=float)

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
    if C.base_mesh is None:
        return 0.
        
    dev = (f-M).ravel()
    logp = 0.
    
    max_eval = max(C.Eval)
    scaled_evals = C.Eval / max_eval
    eff_inf = 1. / eff_zero
    
    dot = asarray((C.Evec.T * dev)).ravel()
    dot_sq = dot ** 2
    if ((scaled_evals == 0. ) * ( dot_sq > 0.)).any():
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
        sig = sqrt(abs(diag(C))).reshape(C.base_mesh.shape)
        y = concatenate((M-sig, (M+sig)[::-1]))
        clf()
        fill(x,y,facecolor='.8',edgecolor='1.')
        plot(C.base_mesh,M,'k-.')

    else:
        x=concatenate((mesh, mesh[::-1]))
        cov = C(mesh,mesh)
        sig = sqrt(abs(diag(cov)))
        mean = M(mesh)
        y=concatenate((mean-sig, (mean+sig)[::-1]))
        clf()
        fill(x,y,facecolor='.8',edgecolor='1.')
        plot(mesh, mean, 'k-.')    


def observe(C,M,obs_mesh, obs_taus = None, lintrans = None, obs_vals = None):
    observe_cov(C, obs_mesh, obs_taus, lintrans)
    observe_mean_from_cov(M,C,obs_vals)
    

def observe_cov(C, obs_mesh, obs_taus = None, lintrans = None):
    """
    observe_cov(C, M, obs_mesh[, obs_taus, lintrans])
    
    Updates C to condition f in:
    
    obs_vals ~ N(lintrans * f(obs_mesh), obs_taus)
    f ~ GP(M, C)
    
    :Arguments:
        - C: A Gaussian process covariance.
        - obs_mesh: 
        - obs_taus:
        - lintrans:
        - M:
    
    :SeeAlso:
    GPMean, GPCovariance, GPRealization, GaussianProcess
    """

    base_mesh = C.base_reshape

    cov_fun = C.eval_fun
    cov_params = C.params

    C.observed = True        

    obs_mesh = regularize_array(obs_mesh)    
    ndim = obs_mesh.shape[-1]
    obs_mesh = obs_mesh.reshape(-1,ndim)
    

    have_obs_taus = obs_taus is not None
    if have_obs_taus:
        obs_taus = obs_taus.ravel()
            
    if base_mesh is not None:
        combined_mesh = vstack((base_mesh, obs_mesh))
        base_len = base_mesh.shape[0]
    else:
        combined_mesh = obs_mesh
        base_len = 0
        
    combined_len = combined_mesh.shape[0]
    obs_len = obs_mesh.shape[0]
    
    Q = cov_fun(obs_mesh, obs_mesh, **cov_params)
    if have_obs_taus:
        Q += diag(1./obs_taus)
            
    if base_mesh is not None:                    
        try:
        
            RF = cov_fun(base_mesh, obs_mesh, **cov_params)

            if lintrans is not None:
                RF = RF * lintrans
            
            tempC = C.view(matrix)
            tempC -= RF * solve(Q, RF.T)        
        
        except LinAlgError:
            raise LinAlgError, 'Unable to invert covariance matrix. Suggest reducing number of observation points or increasing obs_Vs.'
    else:
        RF = None
    
    C.observed = True
    C.obs_mesh = obs_mesh
    C.obs_taus = obs_taus
    C.lintrans = lintrans
    C.obs_len = obs_len
    C.Q = Q
    C.RF = RF

    C.update_sig_and_e()

def observe_mean_from_cov(M,C,obs_vals):
    """
    If you need to observe the covariance and mean separately (as in PyMC),
    use observe() to hit the covariance and then use this to hit the mean.
    
    RF will be passed to C by observe().
    """
    Q = C.Q
    RF = C.RF
    
    obs_vals = obs_vals.ravel()
    
    M.cov_params = C.params
    M.cov_fun = C.eval_fun
    M.C = C
    
    obs_mesh = C.obs_mesh
    obs_taus = C.obs_taus
    lintrans = C.lintrans
    obs_len = C.obs_len
    
    mean_params = M.mean_params
    mean_fun = M.eval_fun
    
    mean_under = mean_fun(obs_mesh, **mean_params)
    Q_mean_under = solve(Q,(asmatrix(obs_vals).T - mean_under))
    
    if M.base_mesh is not None:
        tempM = M.view(ndarray)
        tempM += (RF * Q_mean_under).view(ndarray).reshape(M.shape)

    M.observed = True
    M.obs_mesh = obs_mesh
    M.obs_taus = obs_taus
    M.lintrans = lintrans
    M.obs_vals = obs_vals
    M.obs_len = obs_len
    M.Q_mean_under = Q_mean_under