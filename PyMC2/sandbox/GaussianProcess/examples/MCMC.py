import PyMC_observed_form
import PyMC_unobserved_form
from PyMC2 import *

GPSampler_unobs = Sampler(PyMC_unobserved_form)
GPSampler_obs = Sampler(PyMC_observed_form)

# GPSampler_unobs.sample(iter=500,burn=0,thin=10,verbose=False)
GPSampler_obs.sample(iter=500,burn=0,thin=10,verbose=False)