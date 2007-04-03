# TODO: Implement lintrans, allow obs_taus to be a huge matrix or an ndarray.

from GPCovariance import *
from GPMean import *
from numpy.linalg import solve
from numpy.linalg.linalg import LinAlgError

def condition(C, obs_mesh, obs_taus = None, lintrans = None, obs_vals = None, M=None):
    """
    condition(C, obs_mesh[, obs_taus, lintrans, obs_vals, M])
    
    Updates C and M to condition f on the value of obs_vals in:
    
    obs_vals ~ N(lintrans * f(obs_mesh), obs_taus)
    f ~ GP(M, C)
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
        
        RF = asmatrix(zeros((base_len,obs_len),dtype=float))
        if lintrans is not None:
            RF = RF * lintrans
        Q = asmatrix(zeros((obs_len,obs_len),dtype=float))
        
        cov_fun(RF, base_mesh, obs_mesh, **cov_params)
        cov_fun(Q,obs_mesh, obs_mesh, **cov_params)
        mean_under = mean_fun(obs_mesh, **mean_params)
        if have_obs_taus:
            Q += diag(1./obs_taus)
        Q_mean_under = solve(Q,(asmatrix(obs_vals).T - mean_under))            
            
        M.Q_mean_under = Q_mean_under
        C.Q = Q            

        C -= RF * solve(Q, RF.T)
        M += (RF * Q_mean_under).view(ndarray).reshape(M.shape)
        
        
    except LinAlgError:
        
        combined_cov = asmatrix(zeros((combined_len, combined_len)))    
        cov_fun(combined_cov,combined_mesh,combined_mesh,**cov_params)
        combined_mean=mean_fun(combined_mesh,**mean_params)        
        
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
                
        for i in xrange(base_len):
            M[i] = combined_mean[i]
            for j in xrange(base_len):
                C[i,j] = combined_cov[i,j]
        