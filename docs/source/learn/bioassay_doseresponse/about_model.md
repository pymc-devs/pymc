#Bioassay Problem: Dose_Response Model

This is a small, beginner-friendly and ready to run(in windows) example model of a dose-response bioassay model that can be buily using the PyMC library.  

The model is based on the example that can be found in one of the top recommended youtube videos on PyMC, Titled "Getting Started with PyMC" by Chris Fonnesbeck in a Data Umbrella webinar.

As explained in the video, the Bioassay Problem deals with the estimation of the parameters of a drug dose response curve. The program tries to model the relationship between drug dose and probability of death of mice. It assumes that the probability of death is related to the dose by a logistic function.
$$
\theta = \frac{1}{1 + e^{-(\alpha + \beta d)}}
$$
    It uses Bayesian inference to estimate the parameters of this function based on observed data.