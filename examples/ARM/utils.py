'''
Created on May 22, 2012

@author: john
'''
from csv import reader 
import numpy as np 

def readcsv(file, **args):
    
    r = reader(open(file, 'rb'), **args)
    
    names =[] 
    values =[]
    for name in r.next(): 
        names.append(name)
        values.append([])
        
    for row in r:
        for va, v in zip(values, row):
            try :
                v = float(v)
            except : 
                pass 
             
            va.append(v)
    return dict((name, np.array(vals)) for name, vals in zip(names, values))        
    
    
    