'''
Created on Jul 5, 2012

@author: jsalvatier
'''
from numpy import * 

from Queue import Queue

def HessApproxGen(object):
    def __init__(self, n, S0 = 1e-8):
        self.S0 = S0
        self.lps = Queue(n)
        self.xs = Queue(n)
        self.grads = Queue(n)
        
    def update(self, lp, x, dlp):
        self.lps.put(lp)
        self.xs.put(x)
        self.grads.put(dlp)

        d = zip(sorted(self.xs   , self.lps),
                sorted(self.grads, self.lps))

        bfgs = LBFGS(self.S0)
        x0, g0 = d.next()
        while True:
            while True:
                x1,g1 = d.next()
                s = x1 - x0
                y = g1 - g0
                
                sy = dot(s, y)
                if sy > 0:
                    bfgs.update(s,y, sy)
                        
                x0 = x1
                g0 = g1
        return LBFGS
        

def Ipq_dot((p,q), vi):
    qiTvi = dot(q, vi)
    pq_v = p * -qiTvi
    return vi + pq_v

class LBFGS(object):
    def __init__(self, S0):
        self.S0 =S0
        self.C0 = 1./S0
        self.pq =[]
        self.ut = [] 
        
        
    def update(self, s, y, sy):
        
        By = self.Bdot(y)
        Bs = self.Bdot(s)
        
        p = s /sy
        q = sqrt(sy/dot(s, By))*Bs - y
        self.pq.appen((p,q))
        
        sBs =dot(s, Bs)
        t = s/sBs
        u = sqrt(sBs/sy) * y + Bs
        self.ut.append((u,t))
        
    def Sdot(self, x):
        return reduce(Ipq_dot, self.pq, self.S0 * x)
    def STdot(self, x):
        return self.S0 * reduce(Ipq_dot, self.pq, x)
    def Hdot(self, x):
        return self.Sdot(self.STdot(x))
    
    
    def Cdot(self,x ):
        return reduce(Ipq_dot, self.ut, self.C0 * x)
    
    def CTdot(self, x):
        return self.C0 * reduce(Ipq_dot, self.ut, x)
    def Bdot(self, x):
        return self.Cdot(self.CTdot(x))
        
def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)