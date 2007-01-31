"""
author: Chris Fonnesbeck
date: 2006-06-24
encoding: utf-8
"""

from PyMC2 import MetropolisHastings

def standard(data):
    'Standardize a given dataset'
    
    'Determine shape of data'
    shp = shape(data)
    
    'Flatten data'
    _data = ravel(data)
    
    'Calculate mean'
    xbar = average(_data)
    
    'Calculate standard deviation'
    s = sqrt(sum([(x-xbar)**2 for x in _data])/len(_data)-1)
    
    'Standardize'
    sdata = [(x-xbar)/s for x in _data]
    
    'Reshape and return'
    return reshape(sdata,shp)


class CormackJollySeber(MetropolisHastings):

    def __init__(self):
    
        MetropolisHastings.__init__(self)

    def chi(self, phi, p):
        'Calculate chi parameters for CJS model'
    
        'Initialize vector of chi values'
        X = [None]*(len(p))
        X.append(1.0)
        
        'Reverse iteration'
        i = len(X)-1
        while i:
            'Calculate chi'
            X[i-1] = (1 - phi[i-1]) + phi[i-1] * (1 - p[i-1]) * X[i]
            'Decrement index'
            i -= 1
        return X
        
    def calculate_likelihood(self):
        
        like = 0.0

        'Recapture intercepts'
        alpha0 = self.alpha0

        'Recapture covariate parameters'
        alpha1 = self.alpha1
        
        'Recapture covariates'
        xa = self.recapture_covariates
        
        'Survival intercepts'            
        beta0 = self.beta0
        
        'Survival covariate parameters'
        beta1 = self.beta1
                    
        'Survival covariates'
        xb = self.survival_covariates
        
        'Indices to first capture occasion of each individual'
        first_capture = self.first
        
        'Indices to last capture occasion of each individual'
        last_capture = self.last
        
        'Capture histories'
        captures = self.captures
        
        '''
        Ensure the length of intercept parameter vector is equal 
        the number of capture occasions - 1
        '''
        try:
            if len(alpha0)!=self.occasions-1:
                alpha0 = reshape(alpha0,(self.occasions-1,))
        except TypeError:
            alpha0 = [alpha0]*(self.occasions-1)
            
        try:
            if len(beta0)!=self.occasions-1:
                beta0 = reshape(beta0,(self.occasions-1,))
        except TypeError:
            beta0 = [beta0]*(self.occasions-1)
        
        'Calculate recapture probabilities'
        p = params(alpha0,alpha1,xa)
        
        'Calculate survival probabilities'
        phi = params(beta0,beta1,xb)
        
        X = chi(phi,p)
                
        'Calculate likelihood for each capture history'
        for first,last,capture,_phi,_p,_X in zip(first_capture,last_capture,captures,phi,p,X):
        
            'Loop over the relevant sequence of the capture history'        
            for i in  range(first,last):
                'Survival'
                like += log(_phi[i])
                'Recapture'
                like += self.bernoulli_like(capture[i+1],_p[i])
            
            'Never seen again'
            like += log(_X[last])
                    
        ii = random.randint(0,len(captures))
        print p[ii]
        print phi[ii]
        print like
        print captures[ii];print
        return like

