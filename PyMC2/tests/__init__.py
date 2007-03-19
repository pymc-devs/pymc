# TODO: Write tests involving containers and nodes
# TODO: Include speed_benchmark in distrib
# FIXME: Getting all the test modules imported is driving me crazy.

__modules__ = [	'test_MCMC',
				'test_joint',
                'test_model_ave',
				'test_database',
				'test_distributions']
         
for mod in __modules__:
    exec "from %s import *" % mod
del mod