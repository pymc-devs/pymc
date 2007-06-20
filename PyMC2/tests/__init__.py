# Changeset
# 19/03/2007 -DH- Commented modules import. They are now imported by testsuite.

__modules__ = [ 'test_Sampler',
                'test_joint',
                'test_model_ave',
                'test_database',
                'test_distributions',
                'test_container',
                'test_utils',
                'test_normal_approximation',
                #'test_interactive'
                ]

"""         
for mod in __modules__:
    exec "from %s import *" % mod
del mod
"""
