from dist_math import * 

def Survival(hazard): 
    def like((start, end, died)) : 
        i = arange(hazard.shape[0])[:, None]
        obs = and_(le(start[None,:], i),lt( i, end[None,:]))

        h_e = hazard[end, 0]
        endp = (died * h_e + (1-died) * -exp(h_e))[None, :]

        return endp - sum(exp(hazard)*obs, 0) 
    return like

def ldcount((start,end,died)):  
    nt = np.max(end) + 1
    i = np.arange(nt)[:, None]
    livec = np.sum((start[None,:] <= i) & (i < end[None,:]), 1)
    deaths = end[died ==1]
    deathc = np.sum(deaths[None,:] == i, 1)
    return livec,deathc

def haz_est(l, d):
    haz = np.log((d+1.)/(l+1.))
    m = np.mean(haz)
    return m, haz - m
