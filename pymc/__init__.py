"""
Markov Chain methods in Python.

A toolkit of stochastic methods for biometric analysis. Features
a Metropolis-Hastings MCMC sampler and both linear and unscented 
(non-linear) Kalman filters.

Pre-requisite modules: numpy, matplotlib
Required external components: TclTk

"""
__modules__ = [ 'Node',
                'Container',
                'PyMCObjects',                
                'Model',
                'distributions', 
                'InstantiationDecorators',
                'NormalApproximation', 
                'MCMC',
                'StepMethods',
                'convergencediagnostics',
                'CommonDeterministics',
                #'testsuite'
                ]
                
__sepmodules__ = [  'utils', 
                    'testsuite', 
                    'MultiModelInference',
                    'gp']
                
__optmodules__ = ['ScipyDistributions',
                  'parallel',
                  'sandbox',
                  'graph',
                  'Matplot']
#ClosedCapture, OpenCapture   

for mod in __modules__:
    exec "from %s import *" % mod
    
for mod in __sepmodules__:
    exec "import %s" % mod
    
for mod in __optmodules__:
    try:
      exec "import %s" % mod
    except ImportError:
        pass

##try:
##   import parallel
##except ImportError:
##   print 'For parallel-processing dtrmity install IPython1.'

del mod



    
