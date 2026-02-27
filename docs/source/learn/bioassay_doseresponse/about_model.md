# Bioassay Problem: Dose_Response Model

This is a small, beginner-friendly and ready to run(in windows) example model of a dose-response bioassay model that can be built using the PyMC library. 

## Background 

The model is based on the example presented in the Data Umbrella webinar: **"Getting Started with PyMC"** by Chris Fonnesbeck. It is one of the top recommended videos on introduction to PyMC in youtube. The link for he video is [Youtube](https://www.youtube.com/watch?v=jrU0UBr2z3k&t=1770s)

## Model Specification
As explained in the video, the Bioassay Problem deals with the estimation of the parameters of a drug dose response curve. The program tries to model the relationship between drug dose and probability of death of mice. It assumes that the probability of death is related to the dose by a logistic function.

![Logistic Dose Response Equation](images/eq1.svg)

Where:

- **α** — intercept parameter  
- **β** — slope parameter  
- **d** — administered dose  
- **θ(d)** — probability of death  

It uses Bayesian inference to estimate the parameters of this function based on observed data.