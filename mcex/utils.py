'''
Created on May 27, 2012

@author: john
'''
def project_function(f, index, value):
    def fn(x):
        v = value.copy()
        v[index] = x
        return f(v)
    