'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
    
class compound_step(object): 
    def __init__(self, steppers):
        self.steppers = steppers
    def step(self, states, chain_state):
        if states is None:
            states = [None ]*len(steppers)
        # this seems somewhat complicated, I wonder if there's a better way for this
        for i, (state, stepper) in enumerate(zip(states, self.steppers)): 
            state, chain_state = stepper(state, chain_state)
            states[i] = state
            
        return states, chain_state 

        
