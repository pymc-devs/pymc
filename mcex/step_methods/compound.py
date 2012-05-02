'''
Created on Mar 7, 2011

@author: johnsalvatier
'''


class CompoundStep(object):
    """
    compound step object, calls each of the step methods in sequence
    """
    def __init__(self, step_methods):
        self.step_methods = step_methods
        
    def step(self, chain_state):
        
        for step_method in self.step_methods: 
            chain_state = step_method.step(chain_state)
        return chain_state 
    
    