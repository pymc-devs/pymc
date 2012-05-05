'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
    
def compound_step(steppers): 
    def step(chain_state):
        for stepper in steppers: 
            chain_state = stepper(chain_state)
        return chain_state 


def array_step(stepa, f, mapping):
    def step(chain_state):
        def fn( a):
            return f(mapping.rproject(a, chain_state))
        
        return stepa(fn, mapping.project(chain_state))
        