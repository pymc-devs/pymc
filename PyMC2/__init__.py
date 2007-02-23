"""
Markov Chain methods in Python.

A toolkit of stochastic methods for biometric analysis. Features
a Metropolis-Hastings MCMC sampler and both linear and unscented 
(non-linear) Kalman filters.

Pre-requisite modules: numpy, matplotlib
Required external components: TclTk

"""
__modules__ = [ 'distributions',
                'SamplingMethods',
                'Model',
                'AbstractBase',
                'PurePyMCObjects',
                'MultiModelInference',
                'PyMCObjectDecorators',
                'PyMCObjectContainer',
                'utils']
                
__optmodules__ = []#['MultiModelInference',]
#ClosedCapture, OpenCapture   

try:
    C_modules = ['PyMCObjects']
    for mod in C_modules:
        exec "from %s import *" % mod
except:
    print '\n'+60*'*'
    print 'C objects were not compiled, using pure Python objects as defaults.'
    print 60*'*'+'\n'
    from PurePyMCObjects import PureParameter as Parameter
    from PurePyMCObjects import PureNode as Node
          
for mod in __modules__:
    exec "from %s import *" % mod

for mod in __optmodules__:
    try:
      exec "import %s" % mod
    except ImportError:
        print 'Error importing module ', mod

##try:
##    import parallel
##except ImportError:
##    print 'For parallel-processing functionality install IPython1.'

del mod



    
