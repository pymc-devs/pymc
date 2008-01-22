__modules__ = ['AdaptiveMetropolis','bayes','EM','parallel']

for mod in __modules__:
    try:
        exec('from %s import *' %mod)
    except:
        pass