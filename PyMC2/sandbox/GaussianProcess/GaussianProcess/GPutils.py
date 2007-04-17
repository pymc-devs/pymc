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
from futils import *

try:
    from PyMC2 import ZeroProbability
except ImportError:
    class ZeroProbability(ValueError):
        pass

half_log_2pi = .5 * log(2. * pi)

# The following linear algebra routines are too dangerous for
# general usage, but good for the relatively controlled
# GP application.

def downdate(chol_LHS, RF, chol_RHS):
    """
    Returns the Cholesky factor of:
    chol_LHS.T * chol_LHS - RF * (chol_RHS.T * chol_RHS).I * RF.T
    
    in a way that's sensitive to the fact that chol_RHS may be low-rank.
    
    XXX Use linpack.dchex, if the manual ever comes.
    """
    return robust_chol(chol_LHS.T * chol_LHS - RF * (chol_RHS.T * chol_RHS).I * RF.T)
    
def enlarge_chol(diag_chol_old, offdiag, diag_new):
    """
    Returns the (robust) Cholesky factor of:
    [diag_old  offdiag.T]
    [offdiat    diag_new].
    
    XXX This will be the hardest one. Either use the 'obvious' 
    formula, enlarge the array and use Bach and Jordan's 
    algorithm, or possibly ask Michael Jordan if he has a 
    better idea.
    """
    diag_old = diag_chol_old.T * diag_chol_old
    N_old = diag_old.shape[0]
    N=N_old + diag_new.shape[0]
    new_mat = asmatrix(zeros((N,N),dtype=float))
    
    new_mat[:N_old,:N_old] = diag_old
    new_mat[:N_old,N_old:] = offdiag.T
    new_mat[N_old:,:N_old] = offdiag
    new_mat[N_old:,N_old:] = diag_new
    
    return robust_chol(new_mat)

def robust_chol(C):
    """
    U=robust_chol(C)
    
    Computes a Cholesky factorization of C. Works for matrices that are
    positive-semidefinite as well as positive-definite, though in these
    cases the Cholesky factorization isn't unique.
    
    U will be upper triangular.
    
    """
    chol = C.copy()
    good_rows, N_good_rows = robust_dpotf2(chol)
    good_rows = good_rows[:N_good_rows]
    return good_rows, chol[good_rows,]
    
def fragile_chol(C):
    """
    U=fragile_chol(C)
    
    Attempts to compute the Cholesky factorization by normal means.
    If C is not positive definite, a LinAlgError will be raised.
    
    U will be upper triangular.
    """
    chol = C.copy()
    info=dpotrf_wrap(chol)
    if info>0:
        raise LinAlgError
    return chol
        
    
def solve_from_chol(U, b, uplo='U'):
    """
    x = solve_from_chol(U, b, uplo)
    
    Solves C x = b, where C = U.T * U if uplo='U' or C=L * L.T if
    uplo='L'. Much more efficient than algorithms not based on the 
    Cholesky factorization.
    
    Raises a LinAlgError if U is singular.
    """

    b_copy = b.copy()
    info = dpotrs_wrap(U, b_copy, uplo)
    if info<0:
        raise LinAlgError
    return b_copy
        
def trisolve(U,b,uplo='U'):
    """
    x = trisolve(U,b, uplo='U')
    
    Solves U x = b, where U is upper triangular if uplo='U'
    or lower triangular if uplo = 'L'.
    
    If a degenerate column is found, an error is raised.
    """
    x = b.copy()
    dtrsm_wrap(U,x,uplo)
    return x
    
def gentle_trisolve(U, b, good_rows, uplo='U',):
    """
    x=gentle_trisolve(U, b)
    
    Kind of solves U x = b, where U is upper triangular.

    If a degenerate column is found (a zero pivot), the corresponding 
    row of b is simply ignored.
    
    This is good for prediction/regression, BAD for log-probability
    computation.
    """
    x = b[good_rows].copy()
    dtrsm_wrap(U[:,good_rows],x,uplo)
    return x

    
    

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
    
    raises a ZeroProbability if f doesn't seem to
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
    # TODO: This solve should be done using trisolve.
    # Check the determinant of C ahead of time,
    # and just puke if it's singular. You don't need
    # to be responsible for changing base measures.
    
    if C.base_mesh is None:
        return 0.
        
    dev = (f-M).ravel()
    logp = 0.

    try:
        devvec = trisolve(C.S, dev)
    except LinAlgError:
        return -Inf
    
    return -.5 * (C.logD + dot(devvec, devvec))


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
    
            
    if base_mesh is not None:
        combined_mesh = vstack((base_mesh, obs_mesh))
        base_len = base_mesh.shape[0]
    else:
        combined_mesh = obs_mesh
        base_len = 0
        
    combined_len = combined_mesh.shape[0]
    obs_len = obs_mesh.shape[0]
    
    have_obs_taus = obs_taus is not None
    if have_obs_taus:
        if isinstance(obs_taus, ndarray):
            obs_taus = obs_taus.ravel()
            if not len(obs_taus) == obs_len:
                obs_taus.resize(obs_len)
        else:
            obs_taus = obs_taus * ones(obs_len, dtype=float)
    
    Q = cov_fun(obs_mesh, obs_mesh, **cov_params)
    Q_chol = robust_chol(Q)

    if have_obs_taus:
        Q += diag(1./obs_taus)
            
    if base_mesh is not None:                    
        try:
        
            RF = cov_fun(base_mesh, obs_mesh, **cov_params)

            if lintrans is not None:
                RF = RF * lintrans
            
            C.S = downdate(C.S,RF,Q_chol)  
        
        except LinAlgError:
            raise LinAlgError, 'Unable to invert covariance matrix. Suggest reducing number of observation points or increasing obs_Vs.'
    else:
        RF = None
    
    C.observed = True
    C.obs_mesh = obs_mesh
    C.obs_taus = obs_taus
    C.lintrans = lintrans
    C.obs_len = obs_len
    C.Q_chol = Q_chol
    C.RF = RF

def observe_mean_from_cov(M,C,obs_vals):
    """
    If you need to observe the covariance and mean separately (as in PyMC),
    use observe() to hit the covariance and then use this to hit the mean.
    
    RF will be passed to C by observe().
    """
    Q_chol = C.Q_chol
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
    
    #TODO: This solve should be computed using the output of Bach and Jordan's method.
    # This matrix has a name, it's the regression matrix or something.
    # Call it that.
    Q_I_dev=gentle_trisolve(Q_chol, (asmatrix(obs_vals).T - mean_under))
    reg_part = gentle_trisolve(Q_chol, Q_I_dev)
    
    if M.base_mesh is not None:
        tempM = M.view(ndarray)
        tempM += (RF * reg_part).view(ndarray).reshape(M.shape)

    M.observed = True
    M.obs_mesh = obs_mesh
    M.obs_taus = obs_taus
    M.lintrans = lintrans
    M.obs_vals = obs_vals
    M.obs_len = obs_len
    M.reg_part = reg_part