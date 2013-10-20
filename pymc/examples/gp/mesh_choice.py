from . import PyMCmodel
import pymc as pm
from pymc import six
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
matplotlib.rcParams['axes.facecolor'] = 'w'
import time

n_iter = 10000

rej = []
times = []
mesh_sizes = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
for mesh_size in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    six.print_(mesh_size)
    m = PyMCmodel.make_model(mesh_size, False)
    if mesh_size == 0:
        sm = pm.gp.MeshlessGPMetropolis(m['sm'].f)
    else:
        sm = pm.gp.GPEvaluationMetropolis(m['sm'].f_eval, proposal_sd=.01)
    t1 = time.time()
    for i in xrange(n_iter):
        sm.step()
    times.append((time.time() - t1) / float(n_iter))
    rej.append(sm.rejected / float(sm.rejected + sm.accepted))

m = PyMCmodel.make_model(0, True)
sm = pm.gp.GPEvaluationMetropolis(m['sm'].f_eval, proposal_sd=.01)
t1 = time.time()
for i in xrange(n_iter):
    sm.step()
truemesh_time = (time.time() - t1) / float(n_iter)
truemesh_rej = sm.rejected / float(sm.rejected + sm.accepted)


f = pl.figure(1, figsize=(12, 6))
ax1 = f.add_subplot(111)
ax1.plot(rej, 'b-', label='Rejection', linewidth=4)
ax1.plot([3], [truemesh_rej], 'b.', markersize=8)
ax1.set_ylabel('Rejection rate', color='b')
ax1.set_xticks(range(len(mesh_sizes)))
ax1.set_xticklabels(mesh_sizes)
ax1.set_xlabel('Number of points in mesh')


ax2 = ax1.twinx()
ax2.plot(times, 'r-', label='Time', linewidth=4)
ax2.plot([3], [truemesh_time], 'r.', markersize=8)
ax2.set_ylabel('Time per jump (seconds)', color='r')
ax2.set_xticks(range(len(mesh_sizes)))
ax2.set_xticklabels(mesh_sizes)
ax2.set_xlabel('Number of points in mesh')
