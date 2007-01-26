from C_decorators import *
from numpy.random import randn
from sys import getrefcount

def my_fun(First):
	return float(First)
	
def pmy_dist(value, First, Second):
	return float(First ** 2) * value
	
def rmy_dist(First, Second):
	return randn()**2 * float(First)
	
B=3
A=17

@parameter
def P1(value=4.,First = A, Second = B):
	"""P1 ~ my_dist(First, Second, )"""

	def logp(value, First, Second):
		return pmy_dist(value, First, Second)
		#return 0.
    
	def random(First, Second):
		return rmy_dist(First, Second)
		#return 0.

@node
def N1(Base = P1, Exponent = B):
	"""Node_A = Exp(Base, Exponent)"""
	return Base ** Exponent
	#return 0.
	
@parameter
def P2(value=3.,First = P1, Second = N1):
	"""Parameter_B ~ my_dist(First, Second)"""
	return float(First) + float(Second)
	#return 0.

"""
The following test is 5.5X faster on my laptop than with pure-Python objects,
with all functions above returning 0.

BUT it gets slower on subsequent runs, indicating there's a memory leak.
"""
for i in range(100000):
	#print getrefcount(P2)
	P1.random()
	N1.value
	P2.logp
	#A=raw_input()
