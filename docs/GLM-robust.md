
# This world is far from Normal(ly distributed): Bayesian Robust Regression in PyMC3

Author: Thomas Wiecki

This tutorial first appeard as a post in small series on Bayesian GLMs on my blog:

  1. [The Inference Button: Bayesian GLMs made easy with PyMC3](http://twiecki.github.com/blog/2013/08/12/bayesian-glms-1/)
  2. [This world is far from Normal(ly distributed): Robust Regression in PyMC3](http://twiecki.github.io/blog/2013/08/27/bayesian-glms-2/)
  3. [The Best Of Both Worlds: Hierarchical Linear Regression in PyMC3](http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/)
  
In this blog post I will write about:

 - How a few outliers can largely affect the fit of linear regression models.
 - How replacing the normal likelihood with Student T distribution produces robust regression.
 - How this can easily be done with `PyMC3` and its new `glm` module by passing a `family` object.
 
This is the second part of a series on Bayesian GLMs (click [here for part I about linear regression](http://twiecki.github.io/blog/2013/08/12/bayesian-glms-1/)). In this prior post I described how minimizing the squared distance of the regression line is the same as maximizing the likelihood of a Normal distribution with the mean coming from the regression line. This latter probabilistic expression allows us to easily formulate a Bayesian linear regression model.

This worked splendidly on simulated data. The problem with simulated data though is that it's, well, simulated. In the real world things tend to get more messy and assumptions like normality are easily violated by a few outliers. 

Lets see what happens if we add some outliers to our simulated data from the last post.

Again, import our modules.


    %matplotlib inline
    
    import pymc3 as pm
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    import theano

Create some toy data but also add some outliers.


    size = 100
    true_intercept = 1
    true_slope = 2
    
    x = np.linspace(0, 1, size)
    # y = a + b*x
    true_regression_line = true_intercept + true_slope * x
    # add noise
    y = true_regression_line + np.random.normal(scale=.5, size=size)
    
    # Add outliers
    x_out = np.append(x, [.1, .15, .2])
    y_out = np.append(y, [8, 6, 9])
    
    data = dict(x=x_out, y=y_out)

Plot the data together with the true regression line (the three points in the upper left corner are the outliers we added).


    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, xlabel='x', ylabel='y', title='Generated data and underlying model')
    ax.plot(x_out, y_out, 'x', label='sampled data')
    ax.plot(x, true_regression_line, label='true regression line', lw=2.)
    plt.legend(loc=0);


![png](GLM-robust_files/GLM-robust_6_0.png)


Robust Regression
=================

Lets see what happens if we estimate our Bayesian linear regression model using the `glm()` function as before. This function takes a [`Patsy`](http://patsy.readthedocs.org/en/latest/quickstart.html) string to describe the linear model and adds a Normal likelihood by default. 


    with pm.Model() as model:
        pm.glm.glm('y ~ x', data)
        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        trace = pm.sample(2000, step, progressbar=False)

    /home/wiecki/envs/pymc3/local/lib/python2.7/site-packages/theano/scan_module/scan_perform_ext.py:133: RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility
      from scan_perform.scan_perform import *


To evaluate the fit, I am plotting the posterior predictive regression lines by taking regression parameters from the posterior distribution and plotting a regression line for each (this is all done inside of `plot_posterior_predictive()`).


    plt.subplot(111, xlabel='x', ylabel='y', 
                title='Posterior predictive regression lines')
    plt.plot(x_out, y_out, 'x', label='data')
    pm.glm.plot_posterior_predictive(trace, samples=100, 
                                     label='posterior predictive regression lines')
    plt.plot(x, true_regression_line, 
             label='true regression line', lw=3., c='y')
    
    plt.legend(loc=0);


![png](GLM-robust_files/GLM-robust_10_0.png)


As you can see, the fit is quite skewed and we have a fair amount of uncertainty in our estimate as indicated by the wide range of different posterior predictive regression lines. Why is this? The reason is that the normal distribution does not have a lot of mass in the tails and consequently, an outlier will affect the fit strongly.

A Frequentist would estimate a [Robust Regression](http://en.wikipedia.org/wiki/Robust_regression) and use a non-quadratic distance measure to evaluate the fit.

But what's a Bayesian to do? Since the problem is the light tails of the Normal distribution we can instead assume that our data is not normally distributed but instead distributed according to the [Student T distribution](http://en.wikipedia.org/wiki/Student%27s_t-distribution) which has heavier tails as shown next (I read about this trick in ["The Kruschke"](http://www.indiana.edu/~kruschke/DoingBayesianDataAnalysis/), aka the puppy-book; but I think [Gelman](http://www.stat.columbia.edu/~gelman/book/) was the first to formulate this).

Lets look at those two distributions to get a feel for them.



    normal_dist = pm.Normal.dist(mu=0, sd=1)
    t_dist = pm.T.dist(mu=0, lam=1, nu=1)
    x_eval = np.linspace(-8, 8, 300)
    plt.plot(x_eval, theano.tensor.exp(normal_dist.logp(x_eval)).eval(), label='Normal', lw=2.)
    plt.plot(x_eval, theano.tensor.exp(t_dist.logp(x_eval)).eval(), label='Student T', lw=2.)
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.legend();


![png](GLM-robust_files/GLM-robust_12_0.png)


As you can see, the probability of values far away from the mean (0 in this case) are much more likely under the `T` distribution than under the Normal distribution.

To define the usage of a T distribution in `PyMC3` we can pass a family object -- `T` -- that specifies that our data is Student T-distributed (see `glm.families` for more choices). Note that this is the same syntax as `R` and `statsmodels` use.


    with pm.Model() as model_robust:
        family = pm.glm.families.T()
        pm.glm.glm('y ~ x', data, family=family)
        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        trace_robust = pm.sample(2000, step, progressbar=False)
    
    plt.figure(figsize=(5, 5))
    plt.plot(x_out, y_out, 'x')
    pm.glm.plot_posterior_predictive(trace_robust,
                                     label='posterior predictive regression lines')
    plt.plot(x, true_regression_line, 
             label='true regression line', lw=3., c='y')
    plt.legend();


![png](GLM-robust_files/GLM-robust_14_0.png)


There, much better! The outliers are barely influencing our estimation at all because our likelihood function assumes that outliers are much more probable than under the Normal distribution.

Summary
-------

- `PyMC3`'s `glm()` function allows you to pass in a `family` object that contains information about the likelihood.
 - By changing the likelihood from a Normal distribution to a Student T distribution -- which has more mass in the tails -- we can perform *Robust Regression*.

The next post will be about logistic regression in PyMC3 and what the posterior and oatmeal have in common.

*Extensions*: 

 - The Student-T distribution has, besides the mean and variance, a third parameter called *degrees of freedom* that describes how much mass should be put into the tails. Here it is set to 1 which gives maximum mass to the tails (setting this to infinity results in a Normal distribution!). One could easily place a prior on this rather than fixing it which I leave as an exercise for the reader ;).
 - T distributions can be used as priors as well. I will show this in a future post on hierarchical GLMs.
 - How do we test if our data is normal or violates that assumption in an important way? Check out this [great blog post](http://allendowney.blogspot.com/2013/08/are-my-data-normal.html) by Allen Downey. 


