'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import numpy as np 
import core


class Sampler(object):
    
    def __init__(self, step_methods):
        self.step_methods = step_methods
        
    def step(self, chain_state):
        
        for step_method in self.step_methods: 
            step_method.step(chain_state)

class MultiStep(object):
        
    def init(self, model):
        self.model = model
        self.var_mapping = core.VariableMapping(model.free_vars)
        self.evaluator = core.ChainEvaluation(model)
