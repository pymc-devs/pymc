'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
    
def compound_step(steppers): 
    def step(chain_state):
        for stepper in steppers: 
            chain_state = stepper(chain_state)
        return chain_state 

        