__modules__ = ['bayes','EM','parallel','GibbsStepMethods','GaussianSubmodel','DP']

for mod in __modules__:
    try:
        exec('import %s' %mod)
    except:
        pass
