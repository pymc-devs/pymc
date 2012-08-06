'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
    
def compound_step(steppers): 
    def step(states, chain_state):
        if states is None:
            states = [None ]*len(steppers)
        # this seems somewhat complicated, I wonder if there's a better way for this
        for i, (state, stepper) in enumerate(zip(states, steppers)): 
            state, chain_state = stepper(state, chain_state)
            states[i] = state
            
        return states, chain_state 
    return step

        