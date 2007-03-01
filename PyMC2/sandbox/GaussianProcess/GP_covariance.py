#TODO: Replace all the ndarrays with matrices, to eliminate all the
#TODO: messing around with array dimensions.
#TODO: Make the entire class a subclass of Matrix.

"""
@node
def GP_cov(base_mesh, obs_mesh, lintrans=None, obs_taus = None, actual_GP_fun, **params):

	Valued as a special GP covariance object. On instantiation,
	
	return GP_cov_obj(base_mesh, obs_mesh, lintrans, obs_taus, actual_GP_fun, **params)
	
	this object produces a covariance matrix over base_mesh, according to actual_GP_fun,
	with parameters **params, conditioned on lintrans acting on its evaluations at 
	obs_mesh, with `observation precision' obs_taus.
	
	The object is indexable and callable. When indexed, it returns one of its stored
	elements (static usage). When called with a point couple as arguments (dynamic usage), 
	it returns the covariance of that point couple, again conditioned on the lintrans and obs_mesh
	and all.
"""

from numpy import *
from GP_cov_funs import *
from linalg import cholesky, eigh, inv
#from weave import inline, converters	

def condition(cov_mat, eval_fun, base_mesh, obs_mesh, **params):

	if not obs_mesh:
		return
		
	npts = shape(base_mesh)[0]
	nobs = shape(obs_mesh)[0]
	ndim = shape(base_mesh[1])[0]
	
	Q = zeros((1,1),dtype=float)
	RF = zeros((npts,1),dtype=float)
	
	for i in range(len(obs_mesh)):
		om_now = reshape(obs_mesh[i,:],(1,ndim))
		eval_fun(Q,om_now,om_now,**params)
		eval_fun(RF,base_mesh,om_now,**params)
		cov_mat -= outer(RF,RF)/Q

class GPCovariance(ndarray):
	
	def __new__(subtype, 
				eval_fun,	
				base_mesh,
				obs_mesh, 
				obs_taus, 
				withsigma = False, 
				withtau = False, 
				lintrans = None, 
				**params):
		
		# You may need to reshape these so f2py doesn't puke.
		subtype.base_mesh = base_mesh
		subtype.obs_mesh = obs_mesh
		subtype.obs_taus = obs_taus
		subtype.withsigma = withsigma
		subtype.withtau = withtau
		subtype.lintrans = lintrans
		subtype.params = params
		subtype.eval_fun = eval_fun

		# Call the covariance evaluation function
		length = (base_mesh.shape)[0]
		subtype.data = zeros((length,length), dtype=float)
		eval_fun(subtype.data, base_mesh, base_mesh, symm=True, **params)

		# Condition
		condition(subtype.data, eval_fun, base_mesh, obs_mesh, **params)
		
		if withsigma:
	        # Try Cholesky factorization
	        try:
	            subtype.sigma = cholesky(subtype.data)

	        # If there's a small eigenvalue, diagonalize
	        except linalg.linalg.LinAlgError:
	            subtype.eigval, subtype.eigvec = eigh(subtype.data)
	            subtype.sigma = subtype.eigvec * sqrt(subtype.eigval)
	
		if withtau:
			# Make this faster.
			subtype.tau = inv(subtype.data)
		
		# Return the data
		return subtype.data.view(subtype)
		
	def __array__(self):
		return self.data
		
	def __call__(self, point_1, point_2):
		value = zeros((1,1),dtype=float)
		
		# Evaluate the covariance
		self.eval_fun(value,point_1,point_2,**self.params)
		if not obs_mesh:
			return value[0,0]
		
		# Condition on the observed values
		nobs = shape(obs_mesh)[0]
		ndim = shape(obs_mesh)[1]
		Q = zeros((1,1),dtype=float)
		RF = zeros((2,1),dtype=float)
		
		base_mesh = vstack([point_1,point_2])

		for i in range(len(obs_mesh)):
			om_now = reshape(obs_mesh[i,:],(1,ndim))
			eval_fun(Q,om_now,om_now,**params)
			eval_fun(RF,base_mesh,om_now,**params)
			value -= RF[0]*RF[1]/Q		
		
		return value[0,0]
		
A = reshape(arange(3),(3,1))
B = GPCovariance(	eval_fun = axi_exp, 
					base_mesh = A, 
					obs_mesh = array([[1.5]]), 
					obs_taus = None,
					scale = 2.,
					amp = 1.,
					pow = 1.)