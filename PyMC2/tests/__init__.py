# TODO: Write tests involving containers and nodes
# TODO: Include speed_benchmark in distrib
# FIXME: Getting all the test modules imported is driving me crazy.

# Changeset
# 19/03/2007 -DH- Commented modules import. They are now imported by testsuite.

__modules__ = [ 'test_MCMC',
                'test_joint',
                'test_model_ave',
                'test_database',
                'test_distributions',
                'test_container',
                'test_utils',
                'test_normal_approximation']

"""         
for mod in __modules__:
    exec "from %s import *" % mod
del mod
"""
