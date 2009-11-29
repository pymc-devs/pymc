import numpy as np
import pymc as pm
import pylab as pl

def mixing_fig():
    v1 = 1
    t1 = pm.Normal('t1',0,1./v1,value=-.9)

    c12 = 2
    v2 = .1
    t2 = pm.Normal('t2',t1*c12,1./v2,value=c12*t1.value+np.random.normal()*np.sqrt(v2))

    cov = np.matrix([[v1,c12*v1],[c12*v1,c12**2*v1+v2]])
    theta = np.linspace(0,2.*np.pi,1001)
    x = np.vstack((np.cos(theta),np.sin(theta))).T
    dist = np.array([y*(y*cov.I).T for y in x]).ravel()
    pl.clf()
    pl.plot(x[:,0]/np.sqrt(dist),x[:,1]/np.sqrt(dist),'r-')
    pl.plot([0],[0],'ro',markeredgewidth=.5, markerfacecolor='w',markeredgecolor='r')
    pl.xlabel(r'$\theta_1$')
    pl.ylabel(r'$\theta_2$')

    m1 = pm.Metropolis(t1, proposal_sd=.1)
    m2 = pm.Metropolis(t2, proposal_sd=.1)

    t1s = []
    t2s = []

    for i in xrange(15):
        m1.step()
        t1s.append(t1.value)
        t2s.append(t2.value)
        m2.step()
        t1s.append(t1.value)
        t2s.append(t2.value)

    pl.plot(t1s,t2s,'k-')

    pl.axis([-1.1,1.1,-2.1,2.1])
    pl.axis('off')
    pl.savefig('../figs/poormixing.pdf')

    return x,dist

def metastab_fig():
    index = pm.Bernoulli('index',.2,value=1)
    m = index*2-1
    v = .2
    t1 = pm.Normal('t1',m,1./v)
    t2 = pm.Normal('t2',m,1./v)


    pl.clf()

    mi = pm.BinaryMetropolis(index)
    m1 = pm.Metropolis(t1, proposal_sd=.5)
    m2 = pm.Metropolis(t2, proposal_sd=.5)

    t1s = []
    t2s = []
    ins = []

    theta = np.linspace(0,2.*np.pi,1001)
    x = np.vstack((np.cos(theta),np.sin(theta)))

    r2 = 1.5
    r1 = np.sqrt(r2**2+np.log(.2/(1-.2)))

    pl.plot(x[0]*r1+1,x[1]*r1+1,'r-')
    pl.plot(x[0]*r2-1,x[1]*r2-1,'r-')
    pl.plot([1],[1],'ro',markeredgewidth=.5, markerfacecolor='w',markeredgecolor='r')
    pl.plot([-1],[-1],'ro',markeredgewidth=.5, markerfacecolor='w',markeredgecolor='r')
    pl.xlabel(r'$\theta_1$')
    pl.ylabel(r'$\theta_2$')

    for i in xrange(200):
        mi.step()
        ins.append(index.value)
        m1.step()
        t1s.append(t1.value)
        t2s.append(t2.value)
        m2.step()
        t1s.append(t1.value)
        t2s.append(t2.value)

    pl.plot(t1s,t2s,'k-')
    pl.axis([-3,2,-3,2])
    pl.axis('off')
    pl.savefig('../figs/metastable.pdf')

metastab_fig()
# mixing_fig()
