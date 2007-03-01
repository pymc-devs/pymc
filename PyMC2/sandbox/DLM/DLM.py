"""
PyMC implementations of the Dynamic Linear Model of West and Harrison.

Classes: DiscountDLM.
"""

from numpy import *
from PyMC_base import *

def generate_G(omega=None, r=None, k=None, n=None, depth=None, otherblock = None, **kwargs)
	"""
	Reference: 
		Bayesian Forecasting and Dynamic Models by Mike West and Jeff Harrison
		2nd ed, 1997.
	
	
	generate_G(omega=None, r=None, k=None, n=None, depth=None, otherblock = None, 
				**kwargs)

	The arguments are ndarrays:

		omega:			The oscillation frequencies to be incorporated in the model 
						(W&H Ch 8).

		r:				The exponential rate constants to be incorporated in the 
						model, may be complex. Key 'omega' is easier to use for pure 
						oscillatory components (W&H Sec 6.1.1).

		k:				The polynomial bases to be incorporated in the model 
						(W&H Ch 7).

		n:				The maximum polynomial order to be incorporated in the model 
						for each base k (W&H Ch 7).

		auto_coef:		The autoregressive coefficients (W&H Ch 9).

		otherblock:		A matrix which will be included as a block in G.

		other keyword arguments:	In case you want to write your own GenerateG 
									function.


	Returns:

		G:					The system matrix with the desired components

		component_slices:	A dictionary of slice objects keyed like the arguments, 
							which is useful for slicing the components of the DLM. Keys:

							Oscillatory components:			'osc'

							Other exponential components:	'exp'

							Polynomial components:			'poly'

							Autoregressive components:		'auto'

							Other components:				'other'


	If you update generate_G to handle other components that are useful to you, please 
	email them to us!
						
						
	Example:
	
		
						
	See also:
		DiscountDLM and the pdf documentation distributed with module DLM.
	"""
	
	G = zeros(10,dtype='float')
	component_slices = {}
	
	return G, component_slices


class DiscountDLM(SamplingMethod):
	"""
	
	A SamplingMethod for the discounted DLM with unknown observation variance. 
	This class can be used to fit a DLM directly or to embed the DLM in a larger 
	probability model for fitting with MCMC. It implements the sample() method, 
	so it can be treated just like any other SamplingMethod by Sampler. If the 
	observation variance is known a priori, n_0 can be set to a large number.
	
	DiscountDLM is different from most SamplingMethods in that it creates several
	PyMC Parameters, theta, V and Y, which are attributes of itself and not present 
	in the base namespace. However, these Parameters can be used as parents by other 
	Nodes. In addition, F, G, and delta may be existing PyMC Nodes, allowing for 
	full integration of the DLM in larger probability models.

	Reference: 
		_Bayesian Forecasting and Dynamic Models_, 2nd ed 
		Mike West and Jeff Harrison, 1997.
		
	See also the pdf documentation distributed with this module.


	Instantiation:
	
		D = DiscountDLM(Y, F, G, delta, m_0, C_0, n_0, S_0)
	
		m_0 and C_0 must be numpy arrays, and n_0 and S_0 must be scalars. delta 
		may be a scalar or a PyMC Node. Y MUST be a numpy array, Nodes will not be 
		accepted. The reason is Y's parents will be set by DiscountDLM. F and G may 
		be numpy arrays or PyMC Nodes. To simplify creation of G and access to the 
		components of the DLM, consider using the function generate_G.


	Externally-accessible attributes:				
	
		Each of the following is a PyMC Node.
		
			delta:			Discount factor.

			F:				West and Harrison's F, its value is an ndarray.

			G:				West and Harrison's G, its value is an ndarray.

			Y:				The data, t-distributed around hidden_data. Its parents are 
							hidden_data, S and n.

			V:				Observation covariance matrix, or scalar observation variance. 
							Updated when step() is called.

			theta:			West and Harrison's theta, an ndarray. Its parents are S, n, 
							and G. Updated when step() is called.

			hidden_data:	F' * theta, value is string-sliceable.


		The following are one-dimensional numpy arrays:
		
			S:				Estimate of observation covariance matrix V indexed by time.

			n:				Effective number of observations supporting S.


		The following are dictionaries, keyed by 'forward' and 'backward' (implemented using 
		hidden PyMC Logicals to avoid recomputes):

		If 'forward' is keyed, the return value will be conditional on Y[0] ... Y[t] 
		(W&H Sec 4.6 and 16.4.4).

		If 'backward' is keyed, the return value will be conditional on Y[0] ... Y[T] 
		(W&H Sec 4.8 and 16.4.5).

		Regardless, the return value is a numpy ndarray. Its first index corresponds to time, 
		and its subsequent indices slice m and C.

			m:	Mean of theta

			C:	Covariance of theta.

		Examples:

		m['forward'][,3:10]			returns the mean of theta[3:10] for all t=0..T conditional
									on data up to t as a two-dimensional numpy array, with the 
									first index corresponding to t.

		C['backward'][3:10,22]		returns the covariance of theta[3:10] at time t=22 
									conditional on all the data as a two-dimensional numpy array.


	Externally-accessible methods:
	
		step():									Update V and theta (W&H Sec 15.2.3).

		plot_fit(slice,stage,withdata=True):	stage = 'forwardfilter' or 'backwardsmooth', slice 
												is a slice object.

												Plots mean and standard deviation of of 
												hidden_data[slice], with dataset superimposed if 
												desired.
	
	See also:
		generate_G and the pdf documentation distributed with module DLM.
	"""
	
	def __init__(self):
		pass
