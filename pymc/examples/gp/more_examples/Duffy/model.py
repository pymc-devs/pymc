# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################

# The idea: There are two mutations involved, one (Nr33) converting Fya to Fyb and one (Nr125) 
# silencing expression. The latter mutation tends to occur only with the former, but the converse 
# is not true.
# 
# The model for an individual chromosomal genotype is that the first mutation is a spatially-
# correlated random field, but the second occurs independently with some probability in Africa
# and another probability outside Africa.
# 
# There are two types of datapoints: 
#     - one testing individuals for phenotype (a-b-), meaning both chromosomes have the silencing
#       mutation.
#     - one testing individuals for expression of Fya on either chromosome.
# It's easy to make a likelihood model for either of these, just a bit complicated.
# The maps we'll eventually want to make will be of (a-b-) frequency, meaining the postprocessing
# function will need to close on model variables. The generic package doesn't currently support this.


import numpy as np
import pymc as pm
import gc
from map_utils import *
from generic_mbg import *
import generic_mbg

__all__ = ['make_model']

def make_gp_submodel(suffix, mesh, africa, with_africa_covariate):
    """
    A small function that creates the mean and covariance object
    of the random field.
    """
    
    # from duffy import cut_matern
    
    # The partial sill.
    amp = pm.Exponential('amp_%s'%suffix, .1, value=1.)
    
    # The range parameter. Units are RADIANS. 
    # 1 radian = the radius of the earth, about 6378.1 km
    # scale = pm.Exponential('scale', 1./.08, value=.08)
    
    scale = pm.Exponential('scale_%s'%suffix, 1, value=.08)
    scale_in_km = scale*6378.1
    
    # The nugget variance. Lower-bounded to preserve mixing.
    V = pm.Exponential('V_%s'%suffix, 1, value=1.)
    @pm.potential
    def V_bound(V=V):
        if V<.1:
            return -np.inf
        else:
            return 0
    
    # Create the covariance & its evaluation at the data locations.
    @pm.deterministic(trace=True,name='C_%s'%suffix)
    def C(amp=amp, scale=scale):
        return pm.gp.FullRankCovariance(pm.gp.exponential.geo_rad, amp=amp, scale=scale)
    
    # Create the mean function    
    if with_africa_covariate:
        beta = pm.Normal('beta_%s'%suffix, 0, .01, value=1)
        
        @pm.deterministic(trace=True, name='M_%s'%suffix)
        def M(mesh=mesh, africa_val = africa_val, beta=beta):
            def f(x, mesh=mesh, africa_val=africa_val, beta=beta):
                if np.all(x == mesh):
                    return africa_val*beta
                else:
                    raise ValueError, 'Value of Africa covariate on input array is unknown.'
            return pm.gp.Mean(pm.gp.zero_fn)
    else:
        @pm.deterministic(trace=True, name='M_%s'%suffix)
        def M():
            return pm.gp.Mean(pm.gp.zero_fn)
    
    # Create the GP submodel    
    sp_sub = pm.gp.GPSubmodel('sp_sub_%s'%suffix,M,C,mesh)

    sp_sub.f_eval.value = sp_sub.f_eval.value - sp_sub.f_eval.value.mean()    
    
    return locals()
        
# =========================
# = Haplotype frequencies =
# =========================
h_freqs = {'a': lambda pb, p0, p1: (1-pb)*(1-p1),
            'b': lambda pb, p0, p1: pb*(1-p0),
            '0': lambda pb, p0, p1: pb*p0,
            '1': lambda pb, p0, p1: (1-pb)*p1}
hfk = ['a','b','0','1']
hfv = [h_freqs[key] for key in hfk]

# ========================
# = Genotype frequencies =
# ========================
g_freqs = {}
for i in xrange(4):
    for j in xrange(i,4):
        if i != j:
            g_freqs[hfk[i]+hfk[j]] = lambda pb, p0, p1, i=i, j=j: 2 * np.asscalar(hfv[i](pb,p0,p1) * hfv[j](pb,p0,p1))
        else:
            g_freqs[hfk[i]*2] = lambda pb, p0, p1, i=i: np.asscalar(hfv[i](pb,p0,p1))**2
            
for i in xrange(1000):
    pb,p0,p1 = np.random.random(size=3)
    np.testing.assert_almost_equal(np.sum([gfi(pb,p0,p1) for gfi in g_freqs.values()]),1.)
    
def make_model(lon,lat,africa,n,datatype,
                genaa,genab,genbb,gen00,gena0,genb0,gena1,genb1,gen01,gen11,
                pheab,phea,pheb,
                phe0,prom0,promab,
                aphea,aphe0,
                bpheb,bphe0,
                cpus=1):
    """
    This function is required by the generic MBG code.
    """
    # Step method granularity    
    grainsize = 5
    
    logp_mesh = np.vstack((lon,lat)).T*np.pi/180.
    
    # Probability of mutation in the promoter region, given that the other thing is a.
    p1 = pm.Uniform('p1', 0, .04, value=.01)
            
    # Spatial submodels
    spatial_b_vars = make_gp_submodel('b',logp_mesh,with_africa_covariate=True)
    spatial_0_vars = make_gp_submodel('0',logp_mesh)
    sp_sub_b = spatial_b_vars['sp_sub']
    sp_sub_0 = spatial_0_vars['sp_sub']
    
    # Loop over data clusters, adding nugget and applying link function.
    eps_p_f0_d = []
    p0_d = []
    eps_p_fb_d = []
    pb_d = []
    V_b = spatial_b_vars['V']
    V_0 = spatial_0_vars['V']            

    for i in xrange(np.ceil(len(n)/float(grainsize))):
        sl = slice(i*grainsize,(i+1)*grainsize,None)
        
        if sl.stop>sl.start:
            this_fb = pm.Lambda('fb_%i'%i, lambda f=sp_sub_b.f_eval, sl=sl: f[fi[sl]], trace=False)
            this_f0 = pm.Lambda('f0_%i'%i, lambda f=sp_sub_0.f_eval, sl=sl: f[fi[sl]], trace=False)

            # Nuggeted field in this cluster
            eps_p_fb_d.append(pm.Normal('eps_p_fb_%i'%i, this_fb, 1./V_b, value=np.random.normal(size=np.shape(this_fb.value)), trace=False))
            eps_p_f0_d.append(pm.Normal('eps_p_f0_%i'%i, this_f0, 1./V_0, value=np.random.normal(size=np.shape(this_fb.value)), trace=False))
        
            # The allele frequency
            pb_d.append(pm.Lambda('pb_%i'%i,lambda lt=eps_p_fb_d[-1]: invlogit(np.atleast_1d(lt)),trace=False))
            p0_d.append(pm.Lambda('p0_%i'%i,lambda lt=eps_p_f0_d[-1]: invlogit(np.atleast_1d(lt)),trace=False))

    # The fields plus the nugget
    @pm.deterministic
    def eps_p_fb(eps_p_fb_d = eps_p_fb_d):
        """Concatenated version of eps_p_fb, for postprocessing & Gibbs sampling purposes"""
        return np.hstack(eps_p_fb_d)

    @pm.deterministic
    def eps_p_f0(eps_p_f0_d = eps_p_f0_d):
        """Concatenated version of eps_p_f0, for postprocessing & Gibbs sampling purposes"""
        return np.hstack(eps_p_f0_d)

    init_OK = True
        
    # The likelihoods.
    data_d = []    
    for i in xrange(len(n)):

        sl_ind = int(i/grainsize)
        sub_ind = i%grainsize
        
        if sl_ind == len(p0_d):
            break
        
        # See duffy/doc/model.tex for explanations of the likelihoods.
        p0 = pm.Lambda('p0_%i_%i'%(sl_ind,sub_ind), lambda p=p0_d[sl_ind], j=sub_ind: p[j], trace=False)
        pb = pm.Lambda('pb_%i_%i'%(sl_ind,sub_ind), lambda p=pb_d[sl_ind], j=sub_ind: p[j], trace=False)
        
        if datatype[i]=='prom':
            cur_obs = [prom0[i], promab[i]]
            # Need to have either b and 0 or a and 1 on both chromosomes
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0, p1=p1: (pb*p0+(1-pb)*p1)**2, trace=False)
            n = np.sum(cur_obs)
            data_d.append(pm.Binomial('data_%i'%i, p=p, n=n, value=prom0[i], observed=True))
            
        elif datatype[i]=='aphe':
            cur_obs = [aphea[i], aphe0[i]]
            n = np.sum(cur_obs)
            # Need to have (a and not 1) on either chromosome, or not (not (a and not 1) on both chromosomes)
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0, p1=p1: 1-(1-(1-pb)*(1-p1))**2, trace=False)
            data_d.append(pm.Binomial('data_%i'%i, p=p, n=n, value=aphea[i], observed=True))
            
        elif datatype[i]=='bphe':
            cur_obs = [bpheb[i], bphe0[i]]
            n = np.sum(cur_obs)
            # Need to have (b and not 0) on either chromosome
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0, p1=p1: 1-(1-pb*(1-p0))**2, trace=False)
            data_d.append(pm.Binomial('data_%i'%i, p=p, n=n, value=aphea[i], observed=True))            
            
        elif datatype[i]=='phe':
            cur_obs = np.array([pheab[i],phea[i],pheb[i],phe0[i]])
            n = np.sum(cur_obs)
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0, p1=p1: np.array([\
                g_freqs['ab'](pb,p0,p1),
                g_freqs['a0'](pb,p0,p1)+g_freqs['a1'](pb,p0,p1)+g_freqs['aa'](pb,p0,p1),
                g_freqs['b0'](pb,p0,p1)+g_freqs['b1'](pb,p0,p1)+g_freqs['bb'](pb,p0,p1),
                g_freqs['00'](pb,p0,p1)+g_freqs['01'](pb,p0,p1)+g_freqs['11'](pb,p0,p1)]), trace=False)
            np.testing.assert_almost_equal(p.value.sum(), 1)
            data_d.append(pm.Multinomial('data_%i'%i, p=p, n=n, value=cur_obs, observed=True))
            
        elif datatype[i]=='gen':
            cur_obs = np.array([genaa[i],genab[i],gena0[i],gena1[i],genbb[i],genb0[i],genb1[i],gen00[i],gen01[i],gen11[i]])
            n = np.sum(cur_obs)
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0, p1=p1, g_freqs=g_freqs: \
                np.array([g_freqs[key](pb,p0,p1) for key in ['aa','ab','a0','a1','bb','b0','b1','00','01','11']]), trace=False)
            np.testing.assert_almost_equal(p.value.sum(), 1)
            data_d.append(pm.Multinomial('data_%i'%i, p=p, n=n, value=cur_obs, observed=True))
            
        if np.any(np.isnan(cur_obs)):
            raise ValueError

    return locals()