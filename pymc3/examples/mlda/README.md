This folder contains code that demonstrates the use 
of the Multi-Level Delayed Acceptance MCMC algorithm in PyMC3.

It uses a groundwater flow toy model as a target model for MCMC. 
Thee likelihood computation has been implemented using the FEniCS open-source
library.

## Installation
- Clone the `mlmcmc` branch of the repository by running in bash:
`git clone --single-branch --branch mlmcmc https://github.com/alan-turing-institute/pymc3.git`
- In the directory that contains the new pymc3 folder, run in bash:
`pip install pymc3` or `pip install -e pymc3` to install in developer model
so that any change to the pymc3 code is immediately applied to the 
installed version without a reinstall.

## Dependencies

The example code has been developed and tested with Python 3.6

It depends on the following libraries/tools:
  - Python libraries (easily installed via pip or conda): 
  [numpy](https://pypi.org/project/numpy/), 
  [pandas](https://pypi.org/project/pandas/), 
  [scipy](https://pypi.org/project/scipy/),
  [matplotlib](https://pypi.org/project/matplotlib/)
  - Other (various options to install on the website): [FEniCS](https://fenicsproject.org/)
  
## Folder contents

The examples/mlda folder contains the following files:

 - `example.py`: Contains the demo code pipeline. 

 - `GwFlow.py`: Contains the core implementation of the groundwater model
 in FEniCS and python. 
 
 - `random_process.py`: Contains code to generate a random field and 
 related tasks.
 
 - `model.py`: Code that wraps the model and random process into one class.

 
## The pipeline

In order to run the pipeline, go to directory `pymc3/pymc3/examples/mlda` and run in bash:
`python example.py`.


The pipeline in `example.py` contains the following steps:
 - **Set parameters**: The user can various parameters, the most important of which
 are:
   - `resolutions`: This is a list of different model resolutions. Each
    resolution added to the list will add one level to the multi-level
    inference. Each element is a tuple (x,y) where x, y are the number of 
     points in each dimension. For example, setting `resolutions = 
     [(2,2), (4,4)]` creates a coarse 2x2 model and a fine 4x4 model.
   - `mkl`: The number of unknown parameters in the model (i.e. dimension of
   theta).
   - `ndraws`: The number of MCMC samples to be drawn from the finest posterior.
   - `nburn`: The number of burn-in samples.
   - `nchains`: The number of independent MCMC chains.
   - `nsub`: The subsampling rate for MLDA.
   - `points_list`: The datapoints list.
 - **Generate models and data**: This section instantiates the set of multi-level
 models and the data for inference (from the finest model). It also creates
 the necessary Theano Ops and preforms eigenpairs projection between the fine
 model and the coarse models.
 - **Inference with PyMC3 using MLDA**: This section instantiates the models
 in PyMC3 and draws samples from the posterior. In order to use MLDA, the user
 needs to:
    - Instantiate an MLDA object, passing the list of models and any
 other parameters they want. For example, if you want to set up an MLDA 
 sampler that use the list `coarse_models` of multi-level models and a 
 subsampling rate of 5, you need to type:
 `step_mlda = pm.MLDA(subsampling_rate=5, coarse_models=coarse_models)`. 
    - Sample from the posterior using 
 `trace = pm.sample(step=step_mlda, ...)`.
    - For more information on MLDA, you can have a look at its implementation within
 `pymc3/step_methods/metropolis.py`. Or in a python shell, type `import pymc3 as pm; 
 help(pm.MLDA)` to see the docstring of the class.
    - The code in `example.py` also samples from 
    the same posterior using a standard Metropolis algorithm for 
    comparison purposes and prints out some summary information (e.g. ESS) and plots. 
    The true parameters are printed to allow the user to see if the sampler
    converged to the correct area.
 
