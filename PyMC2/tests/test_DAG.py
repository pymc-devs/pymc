from PyMC2 import *

def mymodel():
    
    @parameter
    def A(value=0):
        return 0.
        
    @node
    def B(mom = 3):
        return 0.
        
    @parameter
    def C(value=0, mom = A, dad = B):
        return 0.
    
    F = []
    
    @parameter
    def x_0(value=0, mod = C):
        return 0.
    F.append(x_0)
    last_x = x_0
    
    for i in range(1,3):          
        @parameter
        def x(value=0, last = last_x, mod = C):
            return 0.
        x.__name__ = 'x\_%i' % i
        last_x = x
        
        F.append(x)
        
        del x
    
    @node
    def q(pop = B):
        return (0)
    F.append(q)
    
    F.append(5)
    
    F = Container(F, name = 'F')
    
    del q
    del x_0
    
    
    @data
    @parameter
    def D(value=0, mom = C, dad = F):
        return 0.
    
    @potential
    def P(mom = F[0], dad = A):
        return 0.
    
    return locals()
    
A = Model(mymodel())

A.DAG(format='pdf', prog='dot', consts = True, legend=True)