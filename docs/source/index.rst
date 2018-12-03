.. title:: PyMC3 Documentation

.. raw:: html

    <div class="ui vertical stripe segment">
        <div class="ui middle aligned stackable grid container">
            <div class="row">

                <div class="eight wide column">
                    <h3 class="ui header">Friendly modelling API</h3>
                    <p>PyMC3 allows you to write down models using an intuitive syntax to describe a data generating
                        process.</p>
                    <h3 class="ui header">Cutting edge algorithms</h3>
                    <p>Fit your model using gradient-based MCMC algorithms like NUTS, using ADVI for fast approximate
                        inference &mdash; including minibatch-ADVI for scaling to large datasets &mdash; or using
                        Gaussian processes to fit a regression model.</p>
                </div>
                <div class="eight wide right floated column">
                    <pre><code class="python">
    X, y = linear_training_data()

    with pm.Model() as linear_model:
        weights = pm.Normal('weights', mu=0, sd=1)
        noise = pm.Gamma('noise', alpha=2, beta=1)
        y_observed = pm.Normal('y_observed',
                    mu=X.dot(weights),
                    sd=noise,
                    observed=y)

        prior = pm.sample_prior_predictive()
        posterior = pm.sample()
        posterior_pred = pm.sample_posterior_predictive(posterior)</code></pre>
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

                    <h3 class="ui header">Via pypi:</h3>

.. code-block:: bash

    pip install pymc3

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

                <a class="ui link card" href="/prob_dists.html">
                    <div class="content">
                        <div class="header">Probability Distributions</div>
                        <div class="description">PyMC3 includes a comprehensive set of pre-defined statistical distributions that can be used as model building blocks.
                        </div>
                    </div>
                </a>

                <a class="ui link card" href="/gp.html">
                    <div class="content">
                        <div class="header">Gaussian Processes</div>
                        <div class="description">Sometimes an unknown parameter or variable in a model is not a scalar value or a fixed-length vector, but a function. A Gaussian process (GP) can be used as a prior probability distribution whose support is over the space of continuous functions. PyMC3 provides rich support for defining and using GPs.
                        </div>
                    </div>
                </a>

                <a class="ui link card" href="/notebooks/variational_api_quickstart.html">
                    <div class="content">
                        <div class="header">Variational Inference</div>
                        <div class="description">Variational inference saves computational cost by turning a problem of integration into one of optimization. PyMC3's variational API supports a number of cutting edge algorithms, as well as minibatch for scaling to large datasets.
                        </div>
                    </div>
                </a>

                <a class="ui link card" href="/theano.html">
                    <div class="content">
                        <div class="header">Theano</div>
                        <div class="description">Theano is the deep-learning library PyMC3 uses to construct probability distributions and then access the gradient in order to implement cutting edge inference algorithms. More advanced models may be built by understanding this layer.
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
            <p>Salvatier J., Wiecki T.V., Fonnesbeck C. (2016) Probabilistic programming in Python using PyMC3. PeerJ
                Computer Science 2:e55 <a href="https://doi.org/10.7717/peerj-cs.55">DOI: 10.7717/peerj-cs.55</a>.</p>
            <p>See <a href="https://scholar.google.de/scholar?oi=bibs&hl=en&authuser=1&cites=6936955228135731011">Google Scholar</a> for a continuously updated list of papers citing PyMC3.</p>
        </div>

        <div class="ui bottom attached segment">
            <h2 class="ui dividing header">Support and sponsors</h2>
            <p>PyMC3 is a non-profit project under NumFOCUS umbrella. If you want to support PyMC3 financially, you <a href="https://www.flipcause.com/widget/widget_home/MTE4OTc=">can donate here</a>.</p>

            <div class="ui equal width grid">
                <div class="column">
                    <a href="https://numfocus.org/">
                        <img class="ui image" src="https://www.numfocus.org/wp-content/uploads/2017/03/1457562110.png"/>
                    </a>
                </div>
                <div class="column">
                    <a href="https://quantopian.com">
                        <img class="ui image" src="https://raw.githubusercontent.com/pymc-devs/pymc3/master/docs/quantopianlogo.jpg"/>
                    </a>
                </div>
                <div class="column">
                    <a href="https://odsc.com/">
                        <img class="ui image" src="https://raw.githubusercontent.com/pymc-devs/pymc3/master/docs/odsc_logo.png"/>
                    </a>
                </div>
            </div>
        </div>
    </div>
