.. title:: PyMC3 Documentation

.. raw:: html

    <div class="ui vertical stripe segment">
        <div class="ui middle aligned stackable grid container">
            <div class="row">

                <div class="eight wide column">
                    <h3 class="ui header">Friendly modelling API</h3>
                    <p>PyMC3 allows you to write down models using an intuitive syntax to describe a data generating
                        process.</p>
                    <h3 class="ui header">Cutting edge algorithms and model building blocks</h3>
                    <p>Fit your model using gradient-based MCMC algorithms like NUTS, using ADVI for fast approximate
                        inference &mdash; including minibatch-ADVI for scaling to large datasets &mdash; or using
                        Gaussian processes to build Bayesian nonparametric models.</p>
                </div>
                <div class="eight wide right floated column">

.. code-block:: python

    import pymc3 as pm

    X, y = linear_training_data()
    with pm.Model() as linear_model:
        weights = pm.Normal("weights", mu=0, sigma=1)
        noise = pm.Gamma("noise", alpha=2, beta=1)
        y_observed = pm.Normal(
            "y_observed",
            mu=X @ weights,
            sigma=noise,
            observed=y,
        )

        prior = pm.sample_prior_predictive()
        posterior = pm.sample()
        posterior_pred = pm.sample_posterior_predictive(posterior)

.. raw:: html

                </div>
            </div>
        </div>

        <div class="ui container">

        <h2 class="ui dividing header">Installation</h2>

            <div class="row">
                <div class="ui text container">
                    <h3 class="ui header">Via conda-forge:</h3>

.. code-block:: bash

    conda install -c conda-forge pymc3


.. raw:: html

                    <h3 class="ui header">Latest (unstable):</h3>

.. code-block:: bash

    pip install git+https://github.com/pymc-devs/pymc3

.. raw:: html

                </div>
            </div>
        </div>

        <div class="ui vertical segment">
            <h2 class="ui dividing header">In-Depth Guides</h2>
            <div class="ui four stackable cards">

                <a class="ui link card" href="/Probability_Distributions.html">
                    <div class="content">
                        <div class="header">Probability Distributions</div>
                        <div class="description">PyMC3 includes a comprehensive set of pre-defined statistical distributions that can be used as model building blocks.
                        </div>
                    </div>
                </a>

                <a class="ui link card" href="/Gaussian_Processes.html">
                    <div class="content">
                        <div class="header">Gaussian Processes</div>
                        <div class="description">Sometimes an unknown parameter or variable in a model is not a scalar value or a fixed-length vector, but a function. A Gaussian process (GP) can be used as a prior probability distribution whose support is over the space of continuous functions. PyMC3 provides rich support for defining and using GPs.
                        </div>
                    </div>
                </a>

                <a class="ui link card" href="/pymc-examples/examples/variational_inference/variational_api_quickstart.html">
                    <div class="content">
                        <div class="header">Variational Inference</div>
                        <div class="description">Variational inference saves computational cost by turning a problem of integration into one of optimization. PyMC3's variational API supports a number of cutting edge algorithms, as well as minibatch for scaling to large datasets.
                        </div>
                    </div>
                </a>

                <a class="ui link card" href="/PyMC3_and_Aesara.html">
                    <div class="content">
                        <div class="header">PyMC3 and Aesara</div>
                        <div class="description">Aesara is the library PyMC3 uses to construct probability distributions and then access the gradient in order to implement cutting edge inference algorithms. More advanced models may be built by understanding this layer.
                        </div>
                    </div>
                </a>

            </div>
        </div>

        <div class="ui vertical segment">
            <h2 class="ui dividing header">License</h2>
            <p>PyMC3 is licensed <a href="https://github.com/pymc-devs/pymc3/blob/master/LICENSE">under the Apache License, V2.</a></p>
        </div>

        <div class="ui vertical segment">
            <h2 class="ui dividing header">Citing PyMC3</h2>
            <p>Please choose from the following:</p>
            <ul>
                <li><a href="https://doi.org/10.7717/peerj-cs.55" rel="nofollow"><img alt="DOIpaper" src="https://camo.githubusercontent.com/6a7e1c555ea828c2f9253f7cff0868debe9fdc711694424b913bf95f6d2da9dd/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f444f492d31302e37373137253246706565726a2d2d63732e35352d626c7565" data-canonical-src="https://img.shields.io/badge/DOI-10.7717%2Fpeerj--cs.55-blue" style="max-width:100%;"></a> <em>Probabilistic programming in Python using PyMC3</em>, Salvatier J., Wiecki T.V., Fonnesbeck C. (2016)</li>
                <li><a href="https://doi.org/10.5281/zenodo.4603970" rel="nofollow"><img alt="DOIzenodo" src="https://camo.githubusercontent.com/2b33f91dc16d9e7ad37aede0119d197776fbef41b8080c7e38a3629df3d9c201/68747470733a2f2f7a656e6f646f2e6f72672f62616467652f444f492f31302e353238312f7a656e6f646f2e343630333937302e737667" data-canonical-src="https://zenodo.org/badge/DOI/10.5281/zenodo.4603970.svg" style="max-width:100%;"></a> A DOI for all versions.</li>
                <li>DOIs for specific versions are shown on Zenodo and under <a href="https://github.com/pymc-devs/pymc3/releases">Releases</a></li>
            </ul>
            <p>See <a href="https://scholar.google.de/scholar?oi=bibs&hl=en&authuser=1&cites=6936955228135731011">Google Scholar</a> for a continuously updated list of papers citing PyMC3.</p>
        </div>

        <div class="ui bottom attached segment">
            <h2 class="ui dividing header">Support and sponsors</h2>
            <p>PyMC3 is a non-profit project under NumFOCUS umbrella.
            If you value PyMC and want to support its development, consider
            <a href="https://numfocus.org/donate-to-pymc3">donating to the project</a> or
            read our <a href="https://docs.pymc.io/about.html#support">support PyMC3 page</a>.
            </p>

            <div class="ui equal width grid">
                <div class="column">
                    <a href="https://numfocus.org/">
                        <img class="ui image" height="120" src="https://www.numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png"/>
                    </a>
                </div>
                <div class="column">
                    <a href="https://quantopian.com">
                        <img class="ui image" height="120" src="https://raw.githubusercontent.com/pymc-devs/pymc3/master/docs/quantopianlogo.jpg"/>
                    </a>
                </div>
                <div class="column">
                    <a href="https://odsc.com/">
                        <img class="ui image" height="120" src="https://raw.githubusercontent.com/pymc-devs/pymc3/master/docs/odsc_logo.png"/>
                    </a>
                </div>
            </div>
        </div>
    </div>
