from numpy import *

# Whhether to thin dataset; definitely thin it if you're running this example on your laptop!
thin = False

l = file('walker_01.dat').read().splitlines()[8:-1]
a = array([fromstring(line,sep='\t') for line in l])
if thin:
    a=a[::5]
ident,x,y,v,u,t=a.T
mesh = vstack((x,y)).T