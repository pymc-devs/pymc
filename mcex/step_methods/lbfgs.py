'''
Created on Jul 5, 2012

@author: jsalvatier
'''
from numpy import * 

from Queue import Queue

class HessApproxGen(object):
    def __init__(self, n, S0 = 1e-8):
        self.S0 = S0
        self.lps = IterQueue(n)
        self.xs = IterQueue(n)
        self.grads = IterQueue(n)
        
    def update(self, x, lp, dlp):
        self.lps.put(lp)
        self.xs.put(x)
        self.grads.put(dlp)

        d = zip(self.lps, self.xs, self.grads)
        d.sort(key = lambda a: a[0])

        lbfgs = LBFGS(self.S0)
        if len(d) > 0:
            _,x0, g0 = d.pop(0)
            for _,x1,g1 in d:
                s = x1 - x0
                y = g1 - g0
                
                sy = dot(s, y)
                if sy > 0:
                    lbfgs.update(s,y, sy)  
                    x0 = x1
                    g0 = g1
                    
        return lbfgs
        

def Iab_dot(vi, (a, b)):
    return vi + a * -dot(b, vi)

class I_abT_Product(object):
    def __init__(self, A0):
        self.A0= A0 
        self.a = []
        self.b = []
        
    def dot(self, v):
        ab = zip(self.a, self.b)
        return reduce(Iab_dot, ab, self.A0 * v)
    
    def Tdot(self, v):
        ba = reversed(zip(self.b, self.a))
        return self.A0 * reduce(Iab_dot, ba, v)
    
    def update(self, a, b):
        self.a.append(a)
        self.b.append(b)


class LBFGS(object):
    def __init__(self, Var0):
        C0 = sqrt(Var0)

        self.C = I_abT_Product(C0)
        self.S = I_abT_Product(1./C0)

        
    def update(self, s, y, sy):
        
        Bs = self.Bdot(s)
        sBs =dot(s.T, Bs)
        ro = sqrt(sy/sBs)
        
        p = s /sy
        q = Bs*ro + y
        
        t = Bs + y/ro
        u = s/sBs
        
        """
                p = s /sy
        q = sqrt(sy/sBs)*Bs - y
        
        t = s/sBs
        u = sqrt(sBs/sy) * y + Bs"""
        
        self.S.update(p, q)
        self.C.update(t, u)
        
    def Hdot(self, x):
        return self.S.dot(self.S.Tdot(x))
    
    def Bdot(self, x):
        return self.C.dot(self.C.Tdot(x))
        
class IterQueue(list): 
    
    def __init__(self, n):
        self.n = n 
        
    def put(self, v):
        self.append(v)
        if len(self) > self.n: 
            self.pop(0)
            