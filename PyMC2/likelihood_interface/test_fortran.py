
"""
First compile test_fortran.f as module test_fortran...

OK. This is a comparison of:

old_normal (PyMC's current normal_like function, without the 'constrain' part, from old_normal.py)
vs.
decorator_normal (David's modifications, from test_decorator_ETC.py)
vs.
fortran_only_normal (from test_fortran.py):

It seems that f2py automatically ravels arrays when they're passed in, so just getting the
length of an array is all you need to traverse it. 

That means you don't really need to resize or reshape anything before passing in
to fortran. You just need a couple of if statements to check whether tau or mu is a scalar to 
handle broadcasting.

The results are pretty consistent across all test cases... though the decorator version gets
within a factor of two of the fortran-only version for the discontinuous array.

The only thing I'm not thrilled about with f2py is that it seems to handle discontiguous arrays
by making a copy.



TIMINGS ON MY MAC FOR MANY TEST CASES:

--------------

Scalars:
with x = 1., 10k calls with arguments(x,3.,3.):
old_normal:				2.59355 s
decorator_normal:		2.529468 s (Why this is slower than the previous test I have no idea)
fortran_only_normal:	0.226842 s

------------------

A one-dimensional array:
with x=ones(100,dtype='float'):

10k calls with arguments (x,x,x):
old_normal:				29.533302 s
decorator_normal:		2.386878 s
fortran_only_normal:	0.455361 s

10k calls with arguments (x,3.,3.) (broadcasting):
old_normal:				30.855701 s
decorator_normal:		3.609395 s (slower than previous test due to resize)
fortran_only_normal:	0.434159 s

-------------------

A multidimensional array:
with x=ones((10,3,4),dtype='float'):

10k calls with arguments (x,x,x):
old_normal:				85.492895 s
decorator_normal:		3.146185 s
fortran_only_normal:	0.685444 s

10k calls with arguments (x,3.,3.) (broadcasting):
old_normal:				Fails to convert arguments
decorator_normal:		4.726067 s (slower than previous test due to resize)
fortran_only_normal:	0.528392 s

-----------------------

A discontiguous array:
with x = ones(100,100,dtype='float'):

10k calls with arguments (x[::4,::5],x[::4,::5],x[::4,::5]):
old_normal:				181.326994 s
decorator_normal:		3.898648 s
fortran_only_normal:	1.727569 s (relatively slow because f2py makes a contiguous copy)

10k calls with arguments (x[::4,::5],3.,3.) (broadcasting):
old_normal:				193.750519 s
decorator_normal:		9.764598 s (slower than previous test due to resize)
fortran_only_normal:	1.299643 s


SOME REPRESENTATIVE PROFILER RESULTS


Note that these are for different length runs, don't compare between functions

I don't know why old_normal is so slow. The profiler results say the function spends nearly all
its time on line 21 in 'sum' which calls something else called '<generator expression>':
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   101000    3.570    0.000    3.570    0.000 old_normal.py:21(<generator expression>)
     1000    0.640    0.001    4.210    0.004 :0(sum)
     2000    0.170    0.000    0.340    0.000 fromnumeric.py:267(resize)
     1000    0.140    0.000    0.140    0.000 :0(zip)
     1000    0.120    0.000    4.920    0.005 old_normal.py:5(old_normal)
     2000    0.060    0.000    0.060    0.000 :0(reduce)
     1000    0.040    0.000    4.260    0.004 fromnumeric.py:368(sum)
     1000    0.040    0.000    0.050    0.000 shape_base.py:110(atleast_1d)
        1    0.030    0.030    4.950    4.950 test_fortran.py:8(?)    ********* Actual fortran call here
     2000    0.030    0.000    0.030    0.000 numeric.py:126(asarray)
     3000    0.030    0.000    0.030    0.000 :0(isinstance)
     2000    0.020    0.000    0.020    0.000 fromnumeric.py:54(reshape)
     2000    0.020    0.000    0.020    0.000 :0(concatenate)
     2000    0.020    0.000    0.050    0.000 fromnumeric.py:318(ravel)
     4000    0.010    0.000    0.010    0.000 fromnumeric.py:337(shape)
     3000    0.010    0.000    0.010    0.000 :0(len)
        1    0.000    0.000    4.950    4.950 :0(execfile)


The profiler results for decorator_normal show it's spending its time in a lot of places, but most 
in resize:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    20000    1.320    0.000    3.850    0.000 fromnumeric.py:267(resize)
    20000    0.900    0.000    0.900    0.000 :0(reduce)
    10000    0.830    0.000    6.100    0.001 test_decorator_like_interface_ONLY.py:56(wrapper)
    10000    0.820    0.000    0.910    0.000 test_decorator_like_interface_ONLY.py:342(normal_like)
    20000    0.520    0.000    0.930    0.000 fromnumeric.py:318(ravel)
    30000    0.420    0.000    0.420    0.000 numeric.py:126(asarray)
    20000    0.280    0.000    0.280    0.000 fromnumeric.py:54(reshape)
    20000    0.230    0.000    0.230    0.000 :0(concatenate)
        1    0.230    0.230    6.330    6.330 test_fortran.py:8(?)      ********* Actual fortran call here
    20000    0.200    0.000    0.200    0.000 :0(append)
    20000    0.160    0.000    0.160    0.000 :0(ravel)
    20000    0.110    0.000    0.110    0.000 :0(isinstance)
    10000    0.090    0.000    0.090    0.000 test_decorator_like_interface_ONLY.py:114(constrain)
    20000    0.080    0.000    0.080    0.000 :0(len)
    10000    0.080    0.000    0.080    0.000 :0(iterkeys)
    10000    0.060    0.000    0.060    0.000 fromnumeric.py:337(shape)
        1    0.010    0.010    6.340    6.340 :0(execfile)


The profiler results for fortran_only_normal are pretty darn clean, just about all of its
time gets spent in fortran:
        1    4.500    4.500    4.510    4.510 test_fortran.py:8(?)      ********* Actual fortran call here
        1    0.010    0.010    0.010    0.010 :0(range)
        1    0.010    0.010    4.520    4.520 :0(execfile)
		
"""

from old_normal import old_normal
from test_fortran_module import normal as fortran_only_normal
from test_decorator_like_interface_ONLY import normal_like as decorator_normal
from numpy import *
from scipy import weave
from scipy.weave import converters


x=ones((100,100),dtype='float')
for i in range(10000):


	# Uncomment just one and run -t to do the discontiguous test cases
	
	#old_normal(x[::4,::5],x[::4,::5],x[::4,::5])
	#decorator_normal(x[::4,::5],x[::4,::5],x[::4,::5])
	fortran_only_normal(x[::4,::5],x[::4,::5],x[::4,::5])
	
	#old_normal(x[::4,::5],3.,3.)
	#decorator_normal(x[::4,::5],3.,3.)
	#fortran_only_normal(x[::4,::5],3.,3.)
	
	
	# Uncomment just one and run -t to do the other test cases.

	#old_normal(x,x,x)
	#decorator_normal(x,x,x)
	#fortran_only_normal(x,x,x)
	
	#old_normal(x,3.,3.)
	#decorator_normal(x,3.,3.)
	#fortran_only_normal(x,3.,3.)	