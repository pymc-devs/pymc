import pymc as pm

from pymc.examples import DisasterModel
M = pm.MCMC(DisasterModel)
M.sample(10)
M.trace('e')[:]
#array([ 2.28320992,  2.28320992,  2.28320992,  2.28320992,  2.28320992,
#      2.36982455,  2.36982455,  3.1669422 ,  3.1669422 ,  3.14499489])

M.trace('e')
#<pymc.database.ram.Trace object at 0x7fa4877a8b50>

M.sample(5)
M.trace('e', chain=None)[:]
#array([ 2.28320992,  2.28320992,  2.28320992,  2.28320992,  2.28320992,
#        2.36982455,  2.36982455,  3.1669422 ,  3.1669422 ,  3.14499489,
#        3.14499489,  3.14499489,  3.14499489,  2.94672454,  3.10767686])

M = pm.MCMC(DisasterModel, db='pickle', dbname='Disaster.pickle')
M.db
#<pymc.database.pickle.Database object at 0x7fa486623d90>

M.sample(10)
M.db.close()
