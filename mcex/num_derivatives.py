'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
import numdifftools as nd
import numpy as np 

def local_cov( model, chain_state):
    """
    returns an approximation of the hessian at the current chain location 
    """
    def grad_logp(x):
        model.mapping.update_with_inverse(chain_state.values_considered, x)
        
        return -model.evaluate_as_vector( chain_state)[1]
    
    #find the jacobian of the gradient function at the current position
    #this should be the hessian
    cov = np.linalg.inv(nd.Jacobian(grad_logp)(model.mapping.apply_to_dict(chain_state.values)))
    chain_state.reject()
    return cov