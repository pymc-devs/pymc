from mcex import *
from numpy import *

random.seed(1)
seterr(invalid = 'raise')

n = 6
m =2
c0 = array([[100**2., .3],
               [.3, 1.5]])[:m, :m]
p0 = linalg.inv(c0)
C =linalg.cholesky(c0)

data = dot(C, random.normal(size = (m, n))).T
#data = array([[-1.0, 0],[0,0], [0., 1.]])


v0 = 1e-8
hessgen = HessApproxGen( n ,v0)

for x in data:
    
    l = .5 * x.T.dot(p0).dot(x)
    dl = p0.dot(x)
    h = hessgen.update(x, l, dl)

a = array([1.,0])[:m]
#print h.S.dot(a)
#print h.C.dot(a)

print h.Hdot(array([1,0.]))
print h.Hdot(array([0,1.]))