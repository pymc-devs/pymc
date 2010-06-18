import numpy as np
import pymc as pm
import gc

def store_africa_val(M, mesh, africa_val):
    M.params['meshes'].append(mesh)
    M.params['africa_vals'].append(africa_val)

def retrieve_africa_val(x, meshes, africa_vals, beta):
    for mesh, africa_val in zip(meshes, africa_vals):
        if np.all(x == mesh):
            return africa_val*beta
    raise ValueError, 'Value of Africa covariate on input array is unknown.'
    


def make_gp_submodel(suffix, mesh, africa_val=None, with_africa_covariate=False):
    
    # The partial sill.
    amp = pm.Exponential('amp_%s'%suffix, .1, value=1.)
    
    # The range parameter. Units are RADIANS.     
    scale = pm.Exponential('scale_%s'%suffix, 1, value=.08)
    
    # 1 radian = the radius of the earth, about 6378.1 km    
    scale_in_km = scale*6378.1
    
    # The nugget variance, lower-bounded to preserve mixing.
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
            M = pm.gp.Mean(retrieve_africa_val, meshes=[], africa_vals=[], beta=beta)
            store_africa_val(M, mesh, africa_val)
            return M
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
                bpheb,bphe0):
    
    logp_mesh = np.vstack((lon,lat)).T*np.pi/180.
    
    # Probability of mutation in the promoter region, given that the other thing is a.
    p1 = pm.Uniform('p1', 0, .04, value=.01)
            
    # Spatial submodels
    spatial_b_vars = make_gp_submodel('b',logp_mesh,africa,with_africa_covariate=True)
    spatial_s_vars = make_gp_submodel('0',logp_mesh)
    sp_sub_b = spatial_b_vars['sp_sub']
    sp_sub_s = spatial_s_vars['sp_sub']
    
    # Loop over data clusters, adding nugget and applying link function.
    tilde_fs_d = []
    p0_d = []
    tilde_fb_d = []
    pb_d = []
    V_b = spatial_b_vars['V']
    V_s = spatial_s_vars['V']            
    data_d = []    

    for i in xrange(len(n)):        
        this_fb =sp_sub_b.f_eval[i]
        this_fs = sp_sub_s.f_eval[i]

        # Nuggeted field in this cluster
        tilde_fb_d.append(pm.Normal('tilde_fb_%i'%i, this_fb, 1./V_b, value=np.random.normal(), trace=False))
        tilde_fs_d.append(pm.Normal('tilde_fs_%i'%i, this_fs, 1./V_s, value=np.random.normal(), trace=False))
            
        # The frequencies.
        p0 = pm.Lambda('pb_%i'%i,lambda lt=tilde_fb_d[-1]: pm.invlogit(lt),trace=False)
        pb = pm.Lambda('p0_%i'%i,lambda lt=tilde_fs_d[-1]: pm.invlogit(lt),trace=False)
        
        # The likelihoods
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
            
    # The fields plus the nugget, in convenient vector form
    @pm.deterministic
    def tilde_fb(tilde_fb_d = tilde_fb_d):
        """Concatenated version of tilde_fb, for postprocessing & Gibbs sampling purposes"""
        return np.hstack(tilde_fb_d)

    @pm.deterministic
    def tilde_fs(tilde_fs_d = tilde_fs_d):
        """Concatenated version of tilde_fs, for postprocessing & Gibbs sampling purposes"""
        return np.hstack(tilde_fs_d)

    return locals()