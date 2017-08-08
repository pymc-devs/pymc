from scipy.cluster.vq import kmeans
import numpy as np

def kmeans_inducing_points(n_inducing, X):
    # first whiten X
    if isinstance(X, tt.TensorConstant):
        X = X.value
    elif isinstance(X, (np.ndarray, tuple, list)):
        X = np.asarray(X)
    else:
        raise ValueError(("To use K-means initialization, "
                          "please provide X as a type that "
                          "can be cast to np.ndarray, instead "
                          "of {}".format(type(X))))
    scaling = np.std(X, 0)
    # if std of a column is very small (zero), don't normalize that column
    scaling[scaling <= 1e-6] = 1.0
    Xw = X / scaling
    Xu, distortion = kmeans(Xw, n_inducing)
    return Xu * scaling


def conditioned_vars(varnames):
    """ Decorator for validating attrs that are conditioned on. """
    def gp_wrapper(cls):
        def make_getter(name):
            def getter(self):
                value = getattr(self, name, None)
                if value is None:
                    raise AttributeError(("'{}' not set.  Provide as argument "
                                          "to condition, or call 'prior' "
                                          "first".format(name.lstrip("_"))))
                else:
                    return value
                return getattr(self, name)
            return getter

        def make_setter(name):
            def setter(self, val):
                setattr(self, name, val)
            return setter

        for name in varnames:
            getter = make_getter('_' + name)
            setter = make_setter('_' + name)
            setattr(cls, name, property(getter, setter))
        return cls
    return gp_wrapper


def plot_gp_dist(ax, samples, x, plot_samples=True, palette="Reds"):
    """ A helper function for plotting 1D GP posteriors from trace """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(palette)
    percs = np.linspace(51, 99, 40)
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    samples = samples.T
    x = x.flatten()
    for i, p in enumerate(percs[::-1]):
        upper = np.percentile(samples, p, axis=1)
        lower = np.percentile(samples, 100-p, axis=1)
        color_val = colors[i]
        ax.fill_between(x, upper, lower, color=cmap(color_val), alpha=0.8)
    if plot_samples:
        # plot a few samples
        idx = np.random.randint(0, samples.shape[1], 30)
        ax.plot(x, samples[:,idx], color=cmap(0.9), lw=1, alpha=0.1)



