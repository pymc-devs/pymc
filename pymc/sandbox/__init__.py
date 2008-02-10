__modules__ = ['AdaptiveMetropolis','bayes','EM','parallel','GibbsStepMethods','test_Gibbs','GaussianSubmodel','DP']

for mod in __modules__:
    try:
        exec('from %s import *' %mod)
    except:
        pass