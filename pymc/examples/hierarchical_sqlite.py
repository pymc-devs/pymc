import pymc as pm
import pymc.examples.hierarchical as hier

with hier.model:
    trace = pm.sample(3000, hier.step, hier.start, trace='sqlite')
