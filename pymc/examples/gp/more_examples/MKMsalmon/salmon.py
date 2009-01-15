from pymc.gp import *
from pymc.gp.cov_funs import matern
from numpy import *
from pylab import *
from csv import *

# Declare salmon class

class salmon(object):
    """
    Reads and organizes data from csv files,
    acts as a container for mean and covariance objects,
    makes plots.
    """
    def __init__(self, name):

        # Read in data

        self.name = name
        f = file(name+'.csv')
        r = reader(f,dialect='excel')

        lines = []
        for line in r:
            lines.append(line)
        f.close()

        data = zeros((len(lines), 2),dtype=float)
        for i in range(len(lines)):
            data[i,:] = array(lines[i])

        self.abundance = data[:,0].ravel()
        self.frye = data[:,1].ravel()

        # Specify priors

        # Function for prior mean
        def line(x, slope):
            return slope * x

        self.M = Mean(line, slope = mean(self.frye / self.abundance))

        self.C = Covariance( matern.euclidean,
                                diff_degree = 1.4,
                                scale = 100. * self.abundance.max(),
                                amp = 200. * self.frye.max())

        observe(self.M,self.C,obs_mesh = 0, obs_vals = 0, obs_V = 0)

        self.xplot = linspace(0,1.25 * self.abundance.max(),100)



    def plot(self):
        """
        Plot posterior from simple nonstochetric regression.
        """
        figure()
        plot_envelope(self.M, self.C, self.xplot)
        for i in range(3):
            f = Realization(self.M, self.C)
            plot(self.xplot,f(self.xplot))

        plot(self.abundance, self.frye, 'k.', markersize=4)
        xlabel('Female abundance')
        ylabel('Frye density')
        title(self.name)
        axis('tight')
