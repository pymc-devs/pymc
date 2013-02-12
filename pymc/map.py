import numpy as np
import collections

__all__ = ['IdxMap', 'DictArrBij', 'DictElemBij']

VarMap = collections.namedtuple('VarMap', 'var, slc, shp')

# TODO Classes and methods need to be fully documented.
class IdxMap(object):
    def __init__(self, vars):
        self.vmap = []
        dim = 0

        for var in vars:       
            slc = slice(dim, dim + var.dsize)
            self.vmap.append( VarMap(str(var), slc, var.dshape)  )
            dim += var.dsize

        self.dimensions = dim
            

class DictArrBij(object):
    def __init__(self, idxmap, dpoint):
        self.idxmap = idxmap
        self.dpt = dpoint

    def map(self, dpt):
        """
        Maps value from dict space to array space
        
        Parameters
        ----------
        dpt : dict 
        """
        apt = np.empty(self.idxmap.dimensions)
        for var, slc, _ in self.idxmap.vmap:
                apt[slc] = np.ravel(dpt[var])
        return apt

    def rmap(self, apt): 
        """
        Maps value from array space to dict space 

        Parameters
        ----------
        apt : array
        """
        dpt = self.dpt.copy()
            
        for var, slc, shp in self.idxmap.vmap:
            dpt[var] = np.reshape(apt[slc], shp)
                
            
        return dpt

    def mapf(self, f):
        """
        Maps function f : DictSpace -> T to ArraySpace -> T
        
        Parameters
        ---------

        f : dict -> T 
        """
        return BijWrapIn(self,f)

class DictElemBij(object):
    def __init__(self, var, idx, dpoint):
        self.var = str(var)
        self.idx = idx
        self.dpt = dpoint

    def map(self, dpt):
        return dpt[self.var][self.idx]

    def rmap(self, apt):
        dpt = self.dpt.copy()
        if self.idx:   
            dvar = dpt[self.var].copy()
            dvar[self.idx] = apt
        else:
            dvar = apt
        dpt[self.var] = dvar
        
        return dpt 
    def mapf(self, f):
        return BijWrapIn(self, f)


class BijWrapIn(object):
    def __init__(self, bij, fn): 
        self.bij = bij
        self.fn = fn

    def __call__(self, d):
        return self.fn(self.bij.rmap(d))
