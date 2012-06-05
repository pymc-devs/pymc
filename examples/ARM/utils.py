'''
Created on May 22, 2012

@author: john
'''
from csv import reader 
import numpy as np 
import theano.tensor as t

def readtable(file, **args):
    
    r = reader(open(file, 'rb'), **args)

    values =[]
    for name in r.next(): 
        values.append([])
        
    for row in r:
        for va, v in zip(values, row):
            try :
                v = float(v)
            except : 
                pass 
             
            va.append(v)
    return map(np.array, values)

def readtabledict(file, **args):
    
    r = reader(open(file, 'rb'), **args)

    values =[]
    names = list( r.next()) 
    values = [[] for i in range(len(names))]
        
    for row in r:
        for va, v in zip(values, row):
            try :
                v = float(v)
            except : 
                pass 
             
            va.append(v)
            
    return dict((name, np.array(vals)) for name, vals in zip(names, values))
    
def demean(x, axis = 0):
    return x - np.mean(x, axis)

def tocorr(c):
    w = np.diag(1/np.diagonal(c)**.5)
    return w.dot(c).dot(w)

def invlogit(x):
    return t.exp(x)/(1 + t.exp(x)) 
    