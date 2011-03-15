'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
import numdifftools as nd

def approx_hessian(mapping, model, chain_state):
    """
    returns an approximation of the hessian at the current chain location 
    """
    def grad_logp(x):
        mapping.update_with_inverse(chain_state.values_considered, x)
        
        return -model.eval.evaluate_as_vector(mapping, chain_state)[1]
    
    #find the jacobian of the gradient function at the current position
    #this should be the hessian
    hess = nd.Jacobian(grad_logp)(mapping.apply_to_dict(chain_state.values))
    chain_state.reject()
    return hess