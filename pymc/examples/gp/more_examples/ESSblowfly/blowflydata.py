from csv import reader
from numpy import array
from pylab import *

f = file("Nichadults.txt")
R = reader(f)
l = []
for line in R:
    l.append(float(line[0]))

blowflydata = array(l)