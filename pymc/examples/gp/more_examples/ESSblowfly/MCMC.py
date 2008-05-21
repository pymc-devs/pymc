from pymc import *
from pylab import *
import BlowflyModel
# import BlowflyModelParametric
from blowflydata import *
from pymc.gp import *

S = Sampler(BlowflyModel)

iter=10000
thin=iter/500
burn=0

# S.sample(iter,burn,thin)
S.interactive_sample(iter,burn,thin)


xplot=arange(0.,max(blowflydata),100.)
tplot = arange(0.,len(blowflydata),10.)

close('all')

def plot_posterior(GP, x, xlab, titlestr):
    figure()
    f_trace = GP.trace()
    subplot(2,1,1)
    hold('on')
    plot_GP_envelopes(GP,x, transy = lambda y: y)
    randindices = randint(0,len(f_trace),3)
    for i in range(3):
        plot(x, f_trace[randindices[i]](x), label='draw %i'%i)
    title(titlestr)
    xlabel(xlab)
    
    midpoint_trace = []
    for i in range(len(GP.trace())):
        midpoint_trace.append(GP.trace()[i](mean(x)))
    subplot(2,1,2)
    plot(midpoint_trace)
    title('f(mean(x))')

plot_posterior(S.B, xplot, r'$x$', r'$B$')
plot_posterior(S.D, xplot, r'$x$', r'$log(D/x)$')

figure()
mu = mean(S.TS.trace())
err = std(S.TS.trace())
plot(mu,'k-')
plot(mu+err,'k-.')
plot(mu-err,'k-.')
tobs = arange(0.,len(blowflydata))
plot(tobs,blowflydata,'k.',markersize=8)
axis([0., S.TS.trace().shape[1], 0., max(blowflydata)])

stochs_to_trace = [S.RickerRate, S.RickerSlope, S.MortalityMean, S.tau, S.mu_psi, S.V_psi, S.meas_V]

for p in stochs_to_trace:
    figure()
    try:
        plot(p.trace())
        title(p.__name__)
    except:
        close()
