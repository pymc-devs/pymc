from base_classes import *

class Joint(SubSampler):

	def __init__(self, model = None, parameters = None, debug = False):
		Sampler.__init__(self,model,parameters,debug)
		
		self._asf = .1
		self.epoch = 0
		self.counter = 0
		self.epoch_length = 1000
		
	def step_singly(self):
		"""
		Include tuning functionality, etc. here
		"""
		for parameter in self.parameters: 
			parameter.metropolis_step()
		self.counter+= 1
		
	def step_jointly(self):
		"""
		Propose according to x_p(t+1) = x(t) + _asf * SVD( cov( x(0..t) ) ) * epsilon,
		where epsilon is a vector of independent standard normal deviates
		and x is the concatenation of all elements in Nodes.
		"""
		self.counter+= 1
		
	def step(self):
		if self.epoch < 2:
			self.step_singly()
		else:
			self.step_jointly()
		
		if self.counter % self.epoch_length == 0:
			self.compute_covariance()
			self.counter = 0
		
	def compute_covariance(self):
		"""
		Compute covariance matrix for x, x being the concatenation of
		every element of parameter in Nodes. Store the singular value decomposition,
		or another matrix square root.
		"""
		self.epoch += 1
				
	def tune(self):
		if self.epoch < 2:
			for parameter in self.parameters: 
				parameter.tune()
		else:
			"""
			Tune the adaptive scaling factor _asf
			"""
			pass
		
    		    
class Slice(SubSampler):
	"""
	Slice sampling MCMC algorithm (see R.M. Neal, Annals of Statistics,
	31, no. 3 (2003), 705-767).
	"""
    
	def __init__(self, model = None, parameters = None, debug = False):
		Sampler.__init__(self,model,parameters,debug)
		    
	def sample(self, debug=False):
		"""Sampling algorithm"""
		pass
		
class JointNonlinear(SubSampler):
	"""
	Learn transformation of parameter space that makes all parameter samples to date look like draws
	from a standard normal. Do one-at-a-time sampling in the transformed space. Need way to do fast
	learning of transformation, evaluation of inverse transformation, computation of Jacobian of inverse
	transformation.
	
	Maybe use differential geometry package PyDX.
	"""
	pass

class ModelAverage(SubSampler):
	"""
	Contains a number of SubSamplers and an Index Node. The Index points to one SubSampler at a time,
	and directs inquiries about its state to that SubSampler's members.
	
	Indices of samples tallied while Index points to a SubSampler are recorded. While the Index points away, 
	the SubSampler is instructed to sample from its prior. When the Index proposes pointing back at that 
	SubSampler, ModelAverage instructs that SubSampler to propose one of its states sampled while Index was
	pointing at it rather than its prior-sampled current state.
	
	There's a Hastings factor associated with this transition. A SubSampler's state while the 
	Index is pointing at it is considered a sample from its posterior given that it is the 'it' model,
	but its state while the index is pointing away from it is considered a sample from its prior. Will
	have to write out the details.
	
	Note that if you want to do joint adaptive proposals you'll have to slice the traces to only include
	samples tallied while the SubSampler was 'on.'
	
	You may want to have ModelAverage spawn a chain for each model and burn them all in without tallying
	the first time it's instructed to sample. Otherwise the first model that gets chosen can reach a very favorable
	state while the others languish in their initial condition.	
	
	Another option would be to run parallel chains, one for each model, but that would require splitting the
	_entire_ model; that is, you couldn't embed a ModelAverage in a larger model. It would have to inherit
	from Sampler not SubSampler.
	"""
	pass

class DynamicLinearModel(SubSampler):
	pass
	
class StochasticDifferentialEquation(SubSampler):
	pass
	
class LinearStochasticDifferentialEquation(DynamicLinearModel):
	pass
	
class DirichletProcessMixture(SubSampler):
	pass