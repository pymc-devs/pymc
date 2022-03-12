import pymc as pm
import aesara.tensor as at
x = at.constant(5)
size = at.stack([x, x])
pm.Normal.dist(size=size)
x = at.constant([5, 5])
pm.Normal.dist(size=x)