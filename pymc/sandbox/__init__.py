__modules__ = ['AdaptiveMetropolis','bayes','EM','parallel']

for mod in __modules__:
    exec('from %s import *' %mod)