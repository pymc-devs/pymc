import pymc

B = pymc.Bernoulli('B', .5)

class Class0(object):
   pass

class Class1(object):
   pass

@pymc.deterministic
def K(B=B):
   if B:
       return Class0()
   else:
       return Class1()
           
           
