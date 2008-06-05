from pymc import *
from numpy.testing import *
from pymc.graph import *

def mymodel():
    
    @stochastic
    def A(value=0):
        return 0.
        
    @deterministic
    def B(mom = 3, dad=A):
        return 0.
        
    @stochastic
    def C(value=0, mom = A, dad = B):
        return 0.
    
    F = []
    
    @stochastic
    def x_0(value=0, mod = C):
        return 0.
    F.append(x_0)
    last_x = x_0
    
    for i in range(1,3):          
        @stochastic
        def x(value=0, last = last_x, mod = C):
            return 0.
        x.__name__ = r'x_%i' % i
        last_x = x
        
        F.append(x)
        
        del x
    
    @deterministic
    def q(pop = A):
        return (0)
    F.append(q)
    
    F.append(5)
    
    F = Container(F)
    
    del q
    del x_0
    
    
    @data
    @stochastic
    def D(value=0, mom = C, dad = F):
        return 0.
    
    @potential
    def P(mom = F[0], dad = A):
        return 0.
    
    return locals()

class test_graph(NumpyTestCase):
    def check_raw(self):
        A = Model(mymodel())
        graph(A, path='../test_results/full.dot', format='raw', prog='dot', consts = True)
        graph(A, path='../test_results/deterministic.dot', format='raw', prog='dot', collapse_deterministics=True, consts = True)
        graph(A, path='../test_results/pot.dot', format='raw', prog='dot', collapse_potentials=True, consts = True)
        graph(A, path='../test_results/deterministic_pot.dot', format='raw', prog='dot', collapse_deterministics=True, collapse_potentials=True, consts = True)
        moral_graph(A, path='../test_results/moral.dot', format='raw', prog='dot')
    def check_pdf(self):
        A = Model(mymodel())    
        graph(A, path='../test_results/full.pdf', format='pdf', prog='dot', consts = True)
        graph(A, path='../test_results/deterministic.pdf', format='pdf', prog='dot', collapse_deterministics=True, consts = True)
        graph(A, path='../test_results/pot.pdf', format='pdf', prog='dot', collapse_potentials=True, consts = True)
        graph(A, path='../test_results/deterministic_pot.pdf', format='pdf', prog='dot', collapse_deterministics=True, collapse_potentials=True, consts = True)
        moral_graph(A, path='../test_results/moral.pdf', format='pdf', prog='dot')

if __name__ == '__main__':
    os.chdir('../test_results')
    NumpyTest().run()
