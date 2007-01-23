from C_decorators import *
from numpy import random

def my_fun(First):
	return float(First)
	
def pmy_dist(value, First, Second):
	return float(First ** 2) * value
	
def rmy_dist(First, Second):
	return random.random() * float(First)
	
B=3
A=17

@parameter
def Parameter_A(value=4.,First = A, Second = B):
	"""Parameter_A ~ my_dist(First, Second, )"""

	def logp(value, First, Second, ):
		return float(First ** 2) * value
		#return 0.
    
	def random(First, Second, ):
		return random.random() * float(First)
		#return 0.

@node
def Node_A(Base = Parameter_A, Exponent = 3.):
	"""Node_A = Exp(Base, Exponent, )"""
	return Base ** Exponent
	#return 0.
	
@parameter
def Parameter_B(value=3.,First = Parameter_A, Second = Node_A):
	"""Parameter_B ~ my_dist(First, Second, )"""
	return float(First) / float(Second)
	#return 0.

"""
The following test is 5.5X faster on my laptop than with pure-Python objects,
with all functions above returning 0.

BUT it gets slower on subsequent runs, indicating there's a memory leak.
"""

for i in range(100000):
	Parameter_A.random()
	Q=Node_A.value
	Q=Parameter_B.logp