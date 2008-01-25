__modules__ = ['AdaptiveMetropolis','bayes','EM','parallel','GibbsStepMethods']

for mod in __modules__:
    try:
        exec('from %s import *' %mod)
    except:
        pass