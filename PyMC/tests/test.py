from proposition5 import *

	
		
@parameter(caching=True)
def B(value = 1., haha=3.):

	def logp(value,haha):
		return 3.		
		
	def random(haha):
		return 1.
		
@data
def A(value = 1., lala=B):
	return lala**2		
		
@node
def C(hmhm=A, zmzm=B):
	return hmhm-zmzm