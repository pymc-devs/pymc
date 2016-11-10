from ..distributions import StudentT, Exponential, Uniform, HalfCauchy
from .. import Model
from ..variational import advi, sample_vp
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class BEST(object):
    """BEST Model, based on Kruschke (2013).

    Parameters
    ----------
    data : pandas DataFrame
        A pandas dataframe which has the following data:
        - Each row is one replicate measurement.
        - There is a column that records the treatment name.
        - There is a column that records the measured value for that replicate.

    sample_col : str
        The name of the column containing sample names.

    output_col : str
        The name of the column containing values to estimate.

    baseline_name : str
        The name of the "control" or "baseline".

    Output
    ------
    model : PyMC3 model
        Returns the BEST model containing
    """
    def __init__(self, data, sample_col, output_col, baseline_name):
        super(BEST, self).__init__()
        self.data = data
        self.sample_col = sample_col
        self.output_col = output_col
        self.baseline_name = baseline_name
        self.trace = None

        self._convert_to_indices()

    def _convert_to_indices(self):
        sample_names = dict()
        for i, name in enumerate(
                list(np.unique(self.data[self.sample_col].values))):
            print('Sample name {0} has the index {1}'.format(name, i))
            sample_names[name] = i
        self.data['indices'] = self.data[self.sample_col].apply(
            lambda x: sample_names[x])

    def fit(self, n_steps=500000):
        """
        Creates a Bayesian Estimation model for replicate measurements of
        treatment(s) vs. control.

        Parameters
        ----------
        n_steps : int
            The number of steps to run ADVI.
        """

        sample_names = set(self.data[self.sample_col].values)

        with Model() as model:
            # Hyperpriors
            upper = Exponential('upper', lam=0.05)
            nu = Exponential('nu_minus_one', 1/29.) + 1

            # "fold", which is the estimated fold change.
            fold = Uniform('fold', lower=1E-10, upper=upper,
                           shape=len(sample_names))

            # Assume that data have heteroskedastic (i.e. variable) error but
            # are drawn from the same HalfCauchy distribution.
            sigma = HalfCauchy('sigma', beta=1, shape=len(sample_names))

            # Model prediction
            mu = fold[self.data['indices']]
            sig = sigma[self.data['indices']]

            # Data likelihood
            like = StudentT('like', nu=nu, mu=mu, sd=sig**-2,
                            observed=self.data[self.output_col])

            params = advi(n=n_steps)
            trace = sample_vp(params, draws=2000)

        self.trace = trace
        self.params = params
        self.model = model

    def plot_posterior(self, rotate_xticks=False):
        """
        Plots a swarm plot of the data overlaid on top of the 95% HPD and IQR
        of the posterior distribution.
        """

        # Make summary plot #
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # 1. Get the lower error and upper errorbars for 95% HPD and IQR.
        lower, lower_q, upper_q, upper = np.percentile(self.trace['fold'],
                                                       [2.5, 25, 75, 97.5],
                                                       axis=0)
        summary_stats = pd.DataFrame()
        summary_stats['mean'] = self.trace['fold'].mean(axis=0)
        err_low = summary_stats['mean'] - lower
        err_high = upper - summary_stats['mean']
        iqr_low = summary_stats['mean'] - lower_q
        iqr_high = upper_q - summary_stats['mean']

        # 2. Plot the swarmplot and errorbars.
        summary_stats['mean'].plot(ls='', ax=ax,
                                   yerr=[err_low, err_high])
        summary_stats['mean'].plot(ls='', ax=ax,
                                   yerr=[iqr_low, iqr_high],
                                   elinewidth=4, color='red')
        sns.swarmplot(data=self.data, x=self.sample_col, y=self.output_col,
                      ax=ax, alpha=0.5)

        if rotate_xticks:
            print('rotating xticks')
            plt.xticks(rotation='vertical')
        plt.ylabel(self.output_col)

        return fig

    def plot_elbo(self):
        """
        Plots the ELBO values to help check for convergence.
        """
        fig = plt.figure()
        plt.plot(-np.log10(-self.params.elbo_vals))

        return fig
