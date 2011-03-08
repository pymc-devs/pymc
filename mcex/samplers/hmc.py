'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import multi 

class HMCStep(multi.MultiStep):
    def __init__(self,model,  step_size_scaling = .25, trajectory_length = 2., covariance = None, find_mode = True):
        if find_mode:
            fm.find_mode(self)

        if covariance is None:
            self.inv_covariance = ah.approx_hess(self)
            self.covariance = np.linalg.inv(self.inv_covariance) 
        else :
            self.covariance = covariance
            self.inv_covariance = np.linalg.inv(covariance)
        
        this.step_size_scaling = step_size_scaling 
        
        if np.size(step_size) > 1:
            self.step_size_max, self.step_size_min = step_size
        else :
            self.step_size_max = self.step_size_min = step_size 
        self.trajectory_length = trajectory_length   
        
    def init(self, model):
        multi.MultiStep(self, model)
        
        self.zero = np.zeros(self.dimensions)
        
        
        step_size = step_size_scaling * self.dimensions**(1/4.)
        
    def step():
        