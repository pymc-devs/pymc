__modules__ = ['bayes','EM','parallel','NormalSubmodel','NormalModel','DLM','DP']

for mod in __modules__:
    try:
        exec('import %s' %mod)
    except:
        pass
