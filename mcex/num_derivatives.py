'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
import numdifftools as nd
import numpy as np 

def approx_cov(model, chain_state):
    """
    returns an approximation of the hessian at the current chain location 
    """
    def grad_logp(x): 
        return np.nan_to_num(-model.evaluate_as_vector( model.project(chain_state, x))[1])
    
    #find the jacobian of the gradient function at the current position
    #this should be the hessian; invert it to find the approximate covariance matrix
    return np.linalg.inv(nd.Jacobian(grad_logp)(model.subspace(chain_state)))