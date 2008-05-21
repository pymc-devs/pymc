from pymc import *
from RandomRealizations import * 
from RandomRealizations.cov_funs import matern
from numpy import *
from pylab import *
from csv import *

class SalmonSampler(MCMC):
    
    def __init__(self, name):
        # Read in data
        
        self.name = name
        f = file(name+'.csv')
        r = reader(f,dialect='excel')

        lines = []
        for line in r:
            lines.append(line)
        f.close()
    
        csvdata = zeros((len(lines), 2),dtype=float)
        for i in range(len(lines)):
            csvdata[i,:] = array(lines[i])

        abundance = csvdata[:,0].ravel()
        frye = csvdata[:,1].ravel()
        lfrye = log(frye)
        labundance = log(abundance)
        
        rx = abundance.max() - abundance.min()
        ry = lfrye.max() - lfrye.min()


        self.abundance = abundance
        self.frye = frye
        self.lfrye = log(frye)
        self.labundance = log(abundance)
        self.plot_x = linspace(self.abundance.min()*.1,self.abundance.max(),100)

        # The mean function's parameters
        beta_0 = Gamma('beta_0', alpha = log(4.5), beta = 1./(10.*(ry/4.)**2))
        beta_1 = Gamma('beta_1', alpha = 1.6 * log(1000.) / ry**2, beta = 1.6 * log(1000.) / ry**2)
                
        # The covariance function's parameters
        invtausq = Gamma('invtausq', alpha = 2., beta = 1./(10.*(ry/4.)**2))
        
        @deterministic
        def amp(invtausq=invtausq):
            """
            Prior amplitude of f.
            Similar to Munch, Mangel and Kottas' \\tau parameter,
            but remember that we're not log-transforming the
            x axis.
            """
            return 1./sqrt(invtausq)
        
        scale = InverseGamma('scale' , alpha=2., beta=1./(6. / rx))
        diff_degree = Uniform('diff_degree', .1, 3)
        
        @deterministic
        def C(diff_degree=diff_degree, amp=amp, scale=scale):
            """
            The Matern covariance function, observed to be zero at the origin.
            """
            C = Covariance(matern.euclidean, diff_degree = diff_degree, amp = amp, scale = scale)
            return C
            
        @deterministic
        def M(beta_0 = beta_0, beta_1 = beta_1):
            """
            The mean function is the Cushing stock-recruitment function
            """
            M = Mean(lambda x: beta_0+ x*beta_1)
            return M
        
        SR = GP(M, C, mesh = labundance, name = self.name + '.SR')
        
        frye_tau = Gamma('frye_tau', alpha = 2., beta = 1./(10.*(ry/4.)**2))
            
        @deterministic
        def frye_V(frye_tau=frye_tau):
            """
            frye_V = 1/(frye_tau)
            """
            return 1./(frye_tau)
        
        
        @data
        @stochastic
        def obs_frye(value=lfrye, SR = SR, mesh=labundance, tau = frye_tau):
            """
            The log of the frye count.
            """
            return normal_like(value, SR(labundance), tau)        
        
        
        MCMC.__init__(self, locals())
        self.use_step_method(GPNormal, SR, labundance, frye_V, obs_frye)
        
        
    def plot_traces(self):
        for object in [self.beta_0, self.beta_1, self.amp, self.scale, self.diff_degree, self.frye_tau]:
            try:
                y=object.trace()
            except:
                print object.__name__
                break
                
            figure()
            plot(y)
            title(object.__name__)
            

    def plot_SR(self):
        f_trace = self.SR.trace()
        figure()
        subplot(2,1,1)
        hold('on')
        plot_GP_envelopes(self.SR, self.plot_x, transx = log, transy=exp)

        for i in range(3):
            plot(self.plot_x, exp(f_trace[i](log(self.plot_x))), label='draw %i'%i)

        plot(self.abundance, self.frye, 'k.', label='data', markersize=8)
        legend(loc=0)
        axis([self.abundance.min()*.1, self.abundance.max(), 0., self.frye.max()*2.])

        midpoint_trace = []
        for i in range(len(self.SR.trace())):
            midpoint_trace.append(self.SR.trace()[i](mean(self.abundance)))
        subplot(2,1,2)
        plot(midpoint_trace)
        title('SR(mean(abundance))')