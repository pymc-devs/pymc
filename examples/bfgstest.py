from mcex import *
from numpy import *

random.seed(1)
seterr(invalid = 'raise')

n = 20
m =2
c0 = array([[4., .2],
               [.2, 1.]])[:m, :m]
p0 = linalg.inv(c0)
C =linalg.cholesky(c0)

data = dot(C, random.normal(size = (m, n))).T

v0 = 1e-4
hessgen = HessApproxGen( n ,v0)

for x in data:
    
    l = .5 * x.T.dot(p0).dot(x)
    dl = p0.dot(x)
    h = hessgen.update(x, l, dl)

a = array([1.,0])[:m]
#print h.S.dot(a)
#print h.C.dot(a)
print h.S.dot(array([1,0.]))
print h.S.dot(array([0,1.]))
print C.dot(a)