from salmon_sampler import *
from pylab import *

# salmon = SalmonSampler('chum')
# salmon = SalmonSampler('sockeye')
salmon = SalmonSampler('pink')


# Really takes 100k
iter=100000
thin=iter/2000
burn=50000
# salmon.sample(iter=iter,burn=burn,thin=thin)
salmon.isample(iter=iter,burn=burn,thin=thin)


close('all')
salmon.plot_SR()
salmon.plot_traces()
