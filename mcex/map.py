'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy.optimize import fmin_bfgs, fmin_ncg
import numpy as np 

def find_MAP( model, chain_state, disp = False):
    """
    moves the chain to the local maximum a posteriori point given a model
    """
    def logp(x):
        model.mapping.update_with_inverse(chain_state.values_considered, x)
        return np.nan_to_num(-(model.evaluate_as_vector(chain_state))[0])
        
    def grad_logp(x):
        model.mapping.update_with_inverse(chain_state.values_considered, x)
        return np.nan_to_num(-(model.evaluate_as_vector( chain_state))[1])

    
    fmin_ncg(logp, model.mapping.apply_to_dict(chain_state.values), grad_logp, disp = disp)
    chain_state.accept() 

    