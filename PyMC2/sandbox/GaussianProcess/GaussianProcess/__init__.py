# TODO, overall: You've written a reasonable Cholesky factorizer that's robust
# to zero eigenvalues, it's GPutils.robust_chol. For now, purge the covariance object's
# eigenvalues and eigenvectors, and just stick with S. You can get the determinant from
# it.
# 
# You want to establish a good function structure. Make a dummy function for Bach-Jordan
# Cholesky factorization, even though for now it'll just be an alias. In particular, write
# a function that computes R Q.I R.T using the Cholesky factors of Q. This will be applicable
# regardless of the Choleskification method.
# 
# Also, give Realization a full-on Mean and Covariance object. It'll need to throw them out
# and refresh them every time it's called, but that will still save you a huge amount of
# bookkeeping and not cause you any more operations. You will need to write a function that
# updates a Covariance's Cholesky factorization when rows/ columns are added to it. It may 
# be better in Realization to just do this repeatedly, skipping times when it can't be done
# (because if the kernel is singular, the evaluation won't affect the next evaluation)
# rather than doing the Bach-Jordan method over and over. Yeah, do that! It'll work great.
# Less work for everyone. And once your storage is already big, why bother trying to reduce it?
# 
# Actually for both Realization calls and successive observations maybe Bach-Jordanize if the 
# new info is bigger than fifty or so, and otherwise just try to update the covariance for 
# every new row/column.

from Mean import Mean
from Covariance import Covariance
from Realization import Realization
from GPutils import *
import cov_funs

# PyMC-specific stuff
try:
    
    from GP import GaussianProcess
    from GPSamplingMethods import *

except ImportError:
    
    class GaussianProcess(object):
        def __init__(self, *args, **kwargs):
            raise ImportError, 'You must install PyMC to use GaussianProcess.'
            
    class GPMetropolis(object):
        def __init(self, *args, **kwargs):
            raise ImportError, 'You must install PyMC to use GPMetropolis.'