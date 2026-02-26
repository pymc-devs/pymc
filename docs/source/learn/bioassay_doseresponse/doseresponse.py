
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pymc as pm
import arviz as az

# ---- data ----
dose = np.array([-0.86, -0.3, 0.05, 0.73])
n_animals = np.array([5, 5, 5, 5])
n_deaths = np.array([0, 1, 3, 5])


def build_and_sample():
    with pm.Model() as bioassay_model:
        dose_data = pm.Data('dose_data', dose)
        deaths_data = pm.Data('deaths_data', n_deaths)

        alpha = pm.Normal('alpha', mu=0, sigma=2.5)
        beta  = pm.Normal('beta',  mu=0, sigma=2.5)

        theta = pm.math.invlogit(alpha + beta * dose_data)
        deaths = pm.Binomial('deaths', n=n_animals, p=theta, observed=deaths_data)

        # parallel sampling: cores>1 uses multiprocessing
        trace = pm.sample(cores=1)
    return trace


if __name__ == "__main__":
    tr = build_and_sample()
    print(az.summary(tr))