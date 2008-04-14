# Changeset
# 19/03/2007 -DH- Commented modules import. They are now imported by testsuite.

__modules__ = [ 'test_MCMCSampler',
                'test_joint',
                'test_model_ave',
                'test_database',
                'test_distributions',
                'test_container',
                'test_instantiation',
                'test_LazyFunction',
                'test_graph',
                'test_norm_approx',
                "test_mean",
                "test_cov",
                "test_realization",
                "test_observation",
                "test_basiscov",
                "test_GP_MCMC"
                #'test_interactive'
                ]

"""         
for mod in __modules__:
    exec "from %s import *" % mod
del mod
"""
