from PyMC2 import *
from numpy.testing import *


def mymodel():
    
    @parameter
    def A(value=0):
        return 0.
        
    @node
    def B(mom = 3, dad=A):
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
        x.__name__ = r'x_%i' % i
        last_x = x
        
        F.append(x)
        
        del x
    
    @node
    def q(pop = A):
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

class test_graph(NumpyTestCase):
    def check_raw(self):
        A = Model(mymodel())
        A.graph(path='full.dot', format='pdf', prog='dot', consts = True)
        A.graph(path='container.dot', format='pdf', prog='dot', collapse_containers=True, consts = True)
        A.graph(path='node.dot', format='pdf', prog='dot', collapse_nodes=True, consts = True)
        A.graph(path='pot.dot', format='pdf', prog='dot', collapse_potentials=True, consts = True)
        A.graph(path='node_pot.dot', format='pdf', prog='dot', collapse_nodes=True, collapse_potentials=True, consts = True)
        A.graph(path='node_cont.dot', format='pdf', prog='dot', collapse_nodes=True, collapse_containers=True, consts = True)
        A.graph(path='cont_pot.dot', format='pdf', prog='dot', collapse_potentials=True, collapse_containers=True, consts = True)
        A.graph(path='node_cont_pot.dot', format='pdf', prog='dot', collapse_nodes=True, collapse_containers=True, collapse_potentials=True, consts = True)                
    def check_pdf(self):
        A = Model(mymodel())    
        A.graph(path='full.dot', format='pdf', prog='dot', consts = True)
        A.graph(path='container.dot', format='pdf', prog='dot', collapse_containers=True, consts = True)
        A.graph(path='node.dot', format='pdf', prog='dot', collapse_nodes=True, consts = True)
        A.graph(path='pot.dot', format='pdf', prog='dot', collapse_potentials=True, consts = True)
        A.graph(path='node_pot.dot', format='pdf', prog='dot', collapse_nodes=True, collapse_potentials=True, consts = True)
        A.graph(path='node_cont.dot', format='pdf', prog='dot', collapse_nodes=True, collapse_containers=True, consts = True)
        A.graph(path='cont_pot.dot', format='pdf', prog='dot', collapse_potentials=True, collapse_containers=True, consts = True)
        A.graph(path='node_cont_pot.dot', format='pdf', prog='dot', collapse_nodes=True, collapse_containers=True, collapse_potentials=True, consts = True)                        

if __name__ == '__main__':
    os.chdir('../test_results')
    NumpyTest().run()
