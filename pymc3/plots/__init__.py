try:
    import arviz as az
except ImportError:  # arviz is optional, throw exception when used

    class _ImportWarner:
        def __init__(self, attr):
            self.attr = attr

        def __call__(self, *args, **kwargs):
            raise ImportError(
                f"ArviZ is not installed. In order to use `{self.attr}`:\npip install arviz"
            )

    class _ArviZ:
        def __getattr__(self, attr):
            return _ImportWarner(attr)

    az = _ArviZ()


autocorrplot = az.plot_autocorr
compareplot = az.plot_compare
forestplot = az.plot_forest
kdeplot = az.plot_kde
plot_posterior = az.plot_posterior
traceplot = az.plot_trace
energyplot = az.plot_energy
densityplot = az.plot_density
pairplot = az.plot_pair

from .posteriorplot import plot_posterior_predictive_glm
