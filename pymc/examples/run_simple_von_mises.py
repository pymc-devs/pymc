import pymc
import simple_von_mises

model=pymc.MCMC(simple_von_mises)
model.sample(iter=1000, burn=500, thin=2)

print 'mu',model.mu.value
print 'kappa',model.kappa.value
