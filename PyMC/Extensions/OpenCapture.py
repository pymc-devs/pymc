#!/usr/bin/env python
# encoding: utf-8
"""
open_capture.py

Created by Chris Fonnesbeck on 2007-07-08.
Copyright (c) 2007 Fish and Wildlife Research Institute (FWC). All rights reserved.
"""

from PyMC import *
from numpy import random, size, prod
import pdb

"""
Sample set of capture histories. Dictionary keys are the histories (tuples)
and the corresponding values are the frequencies.

Simulated with constaint p = phi = 0.7
"""
data = {(0, 0, 0, 0, 1): 1,
 (0, 0, 0, 1, 0): 4,
 (0, 0, 0, 1, 1): 2,
 (0, 0, 1, 0, 0): 5,
 (0, 0, 1, 0, 1): 5,
 (0, 0, 1, 1, 0): 8,
 (0, 0, 1, 1, 1): 7,
 (0, 1, 0, 0, 0): 45,
 (0, 1, 0, 0, 1): 2,
 (0, 1, 0, 1, 0): 9,
 (0, 1, 0, 1, 1): 7,
 (0, 1, 1, 0, 0): 25,
 (0, 1, 1, 0, 1): 8,
 (0, 1, 1, 1, 0): 23,
 (0, 1, 1, 1, 1): 19,
 (1, 0, 0, 0, 0): 279,
 (1, 0, 0, 0, 1): 2,
 (1, 0, 0, 1, 0): 11,
 (1, 0, 0, 1, 1): 9,
 (1, 0, 1, 0, 0): 29,
 (1, 0, 1, 0, 1): 12,
 (1, 0, 1, 1, 0): 18,
 (1, 0, 1, 1, 1): 15,
 (1, 1, 0, 0, 0): 135,
 (1, 1, 0, 0, 1): 5,
 (1, 1, 0, 1, 0): 13,
 (1, 1, 0, 1, 1): 16,
 (1, 1, 1, 0, 0): 79,
 (1, 1, 1, 0, 1): 12,
 (1, 1, 1, 1, 0): 43,
 (1, 1, 1, 1, 1): 45}

def import_data(datafile):
    """
    Import capture histories from a text file. Records shoulld look 
    like this:
    
    100000
    001000
    110000
    100000
    111111
    
    :Arguments:
        datafile : string
            Name of data file to import.
            
    :Note:
        Returns a dictionary of capture history frequencies for use 
        with open capture estimation models.
        
    :SeeAlso: CormackJollySeber, Pradel
    
    """
    
    # Opein specified input file
    ifile = open(datafile)
    
    # Initialize dictionary
    datadict = {}
    
    # Iterate over lines
    for line in ifile:
        if line:
            # Parse string
            hist = tuple([int(i) for i in line.strip()])
            # Add to data dictionary
            try:
                datadict[hist] += 1
            except KeyError:
                datadict[hist] = 1
    
    return datadict

class CormackJollySeber(MetropolisHastings):
    """
    Estimates parameters of Cormack-Jolly-Seber CMR model for open populations.
    
    :Arguments:
        data : dict
            Data dictionary containing capture frequencies. See sample dataset
            for example.
        survival (optional) : string
            Survival constraint. 't' specifies time-specific parameters, 'c'
            specifies constant.
        capture (optional) : string
            Capture constraint. 't' specifies time-specific parameters, 'c'
            specifies constant.
            
    :Public Attributes:
        data : dict
            Dictionary of capture histories
        occasions : int
            Number of capture occasions
            
    :Public Methods:
        chi(phi, p)
            Returns array of chi probabilities (the probabilities
            of never being seen again after last capture)
        model()
            Specification of the log-posterior of the model; generally
            used only by MetropolisHastings sampler.
    """
    
    def __init__(self, data, survival='t', capture='t'):
        
        MetropolisHastings.__init__(self)
        
        # Register capture histories
        self.data = data
        
        # Number of capture occasions
        self.occasions = len(data.keys()[0])
        
        # Number of parameters of each type
        self._param_count = {'p': self.occasions-1, 'phi':self.occasions-1}
        
        # Initialize parameters
        self._init_params(p=capture, phi=survival)
        
        # Survival probabilities
        if survival == 't':
            self.parameter('phi', init_val=array([0.5] * (self.occasions-1)))
        elif survival == 'c':
            self.parameter('phi', init_val=0.5)
        else:
            raise ParameterError, 'Invalid survival probability type'
        
        # Capture probabilities
        if capture == 't':
            self.parameter('p', init_val=array([0.5] * (self.occasions-1)))
        elif capture == 'c':
            self.parameter('p', init_val=0.5)
        else:
            raise ParameterError, 'Invalid capture probability type'
        
        # Determine indices for first and last captures
        self._firstlast = {}
        for key in data:
            foundfirst = False
            for i,capture in enumerate(key):
                if capture:
                    if not foundfirst:
                        self._firstlast[key] = [i, i]
                        foundfirst = True
                    else:
                        self._firstlast[key][1] = i
                        
    def _init_params(self, **params):
        """
        Initializes parameters of CJS model
        
        :Arguments:
            params : dict
                Dictionary of parameter names and associated constraint
        """
        
        # Loop over parameters
        for param in params:
            
            if params[param] == 't':
                # Time-specific
                k = self._param_count[param]
                self.parameter(param, init_val=array([0.5] * k))
            elif params[param] == 'c':
                # Constant
                self.parameter(param, init_val=0.5)
            else:
                raise ParameterError, 'Invalid constraint for %s' % param
    
    def chi(self, phi, p):
        """
        Calculate chi parameters for CJS model
        
        :Arguments:
            phi : array
                Survival probabilities
            p : array
                Capture probabilities
        
        """
        
        'Initialize vector of chi values'
        X = [None]*(len(p))
        X.append(1.0)
        
        'Reverse iteration'
        for i in range(len(p), 0, -1):
            'Calculate chi'
            X[i-1] = (1 - phi[i-1]) + phi[i-1] * (1 - p[i-1]) * X[i]
            
        return X
    
    def model(self):
        """
        Joint log-posterior of model; used for MetropolisHastings sampling.
        """
        
        # Prior constraints on parameters
        self.beta_prior(self.p, 1, 1)
        self.beta_prior(self.phi, 1, 1)
        
        # Make local variables, and resize if necessary
        phi = resize(self.phi, self.occasions-1)
        p = resize(self.p, self.occasions-1)
        
        # Calculate Chi
        X = self.chi(phi, p)
        
        # Initialize set of capture histories by first capture
        capture_set = dict.fromkeys(range(self.occasions - 1))
        for key in capture_set:
            capture_set[key] = [[], []]
        
        # Calculate probabilities for each capture history
        for history, freq in self.data.iteritems():
            
            # Index out local variables
            first, last = self._firstlast[history]
            
            # This should exclude captures on the final occasion
            if first < (self.occasions - 1):
                # Append frequency of capture history
                capture_set[first][0].append(freq)
                prob = prod([phi[i] * (p[i] * history[i+1] or (1. - p[i])) for i in range(first,last)]) * X[last]
                
                # Probability of capture history
                capture_set[first][1].append(prob)
                
        # Multinomial likelihoods
        for x, pi in capture_set.values():
            
            self.multinomial_like(x[:-1], sum(x), pi[:-1])
            

class Pradel(CormackJollySeber):
    """
    Implementation of the Pradel et al. (1996) reverse-time open population model.
    Estimates seniority (gamma) rather than mortality (phi) in CJS.
    
    :Arguments:
        data : dict
            Data dictionary containing capture frequencies. See sample dataset
            for example.
        seniority (optional) : string
            Seniority constraint. 't' specifies time-specific parameters, 'c'
            specifies constant.
        survival (optional) : string
            Survival constraint. 't' specifies time-specific parameters, 'c'
            specifies constant.
        capture (optional) : string
            Capture constraint. 't' specifies time-specific parameters, 'c'
            specifies constant.
            
    :Public Attributes:
        data : dict
            Dictionary of capture histories
        occasions : int
            Number of capture occasions
            
    :Public Methods:
        zeta(phi, p)
            Returns array of zeta probabilities (the probabilities
            of not being present in the population prior to first capture)
        model()
            Specification of the log-posterior of the model; generally
            used only by MetropolisHastings sampler.
    """
    
    def __init__(self, data, seniority='t', capture='t', survival='t'):
        
        CormackJollySeber.__init__(self, data, capture=capture, survival=survival)
        
        # Number of parameters of each type
        self._param_count = {'gamma': self.occasions-1, 'p': self.occasions, 'phi':self.occasions-1}
        
        # Initialize parameters
        self._init_params(gamma=seniority, p=capture, phi=survival)
        
    
    def zeta(self, gamma, p):
        """
        Calculate zeta parameters for Pradel model
        
        :Arguments:
            gamma : array
                Seniority probabilities
            p : array
                Capture probabilities
        
        """
        
        'Initialize vector of zeta values'
        z = [1.0] + [None]*(len(p))
        
        'Iterate'
        for i in range(1, len(z)):
            'Calculate zeta'
            z[i] = (1 - gamma[i-1]) + gamma[i-1] * (1 - p[i-1]) * z[i-1]
        
        return z
    
    def model(self):
        """
        Joint log-posterior of model; used for MetropolisHastings sampling.
        """
        
        # Diffuse beta priors on parameters
        self.beta_prior(self.p, 1, 1)
        self.beta_prior(self.gamma, 1, 1)
        self.beta_prior(self.phi, 1, 1)
        
        # Make local variables, and resize if necessary
        gamma = resize(self.gamma, self.occasions-1)
        p = resize(self.p, self.occasions)
        phi = resize(self.phi, self.occasions-1)
        
        # Calculate Chi
        X = self.chi(phi, p[1:])
        
        # Calculate Chi
        z = self.zeta(gamma, p[:-1])
        
        # Initialize set of CJS capture histories by first capture
        cjs_set = dict.fromkeys(range(self.occasions - 1))
        for key in cjs_set:
            cjs_set[key] = [[], []]
        
        # Initialize set of Pradel capture histories by last capture
        pradel_set = dict.fromkeys(range(1, self.occasions))
        for key in pradel_set:
            pradel_set[key] = [[], []]
        
        'Calculate probabilities for each capture history'
        for history, freq in self.data.iteritems():
            
            # Index out local variables
            first, last = self._firstlast[history]
            
            # CJS CAPTURE PROBABILITIES
            # This should exclude captures on the final occasion
            if first < (self.occasions - 1):
                # Append frequency of capture history
                cjs_set[first][0].append(freq)
                
                # Probability of capture history
                prob = prod([phi[i] * (p[i+1] * history[i+1] or (1. - p[i+1])) for i in range(first,last)]) * X[last]
                
                cjs_set[first][1].append(prob)
            
            # PRADEL CAPTURE PROBABILITIES
            # Ignore capture histories with the first and last capture at t=0
            if last > 0:
                
                # Append frequency of capture history
                pradel_set[last][0].append(freq)
                
                # Probability of capture history (by iterating backwards)
                prob = prod([gamma[i] * (p[i] * history[i] or (1. - p[i])) for i in range(last-1, first-1, -1)]) * z[first]
                
                pradel_set[last][1].append(prob)
        
        # Multinomial likelihoods for CJS
        for x, pi in cjs_set.values():
            
            self.multinomial_like(x[:-1], sum(x), pi[:-1])
        
        # Multinomial likelihoods from Pradel
        for x, pi in pradel_set.values():
            
            self.multinomial_like(x[:-1], sum(x), pi[:-1])
        

def run():
    
    CJS_sampler = CormackJollySeber(data, 't', 't')
    CJS_sampler.sample(10000, burn=5000)

if __name__=='__main__':
    # Run the following block if called from the command line
    
    run()
                
