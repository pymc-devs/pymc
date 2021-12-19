import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.random import RandomState
from scipy.interpolate import griddata
from scipy.signal import savgol_filter


def predict(idata, rng, X_new=None, size=None):
    """
    Generate samples from the BART-posterior

    Parameters
    ----------
    idata: InferenceData
        InferenceData containing a collection of BART_trees in sample_stats group
    rng: NumPy random generator
    X_new : array-like
        A new covariate matrix. Use it to obtain out-of-sample predictions
    size: int or tuple
        Number of samples.
    """
    bart_trees = idata.sample_stats.bart_trees
    stacked_trees = bart_trees.stack(trees=["chain", "draw"])
    if size is None:
        size = ()
    elif isinstance(size, int):
        size = [size]

    flatten_size = 1
    for s in size:
        flatten_size *= s

    idx = rng.randint(len(stacked_trees.trees), size=flatten_size)

    if X_new is None:
        pred = np.zeros((flatten_size, stacked_trees[0, 0].item().num_observations))
        for ind, p in enumerate(pred):
            for tree in stacked_trees.isel(trees=idx[ind]).values:
                p += tree.predict_output()
    else:
        pred = np.zeros((flatten_size, X_new.shape[0]))
        for ind, p in enumerate(pred):
            for tree in stacked_trees.isel(trees=idx[ind]).values:
                p += np.array([tree.predict_out_of_sample(x) for x in X_new])
    return pred.reshape((*size, -1))


def plot_dependence(
    idata,
    X=None,
    Y=None,
    kind="pdp",
    xs_interval="linear",
    xs_values=None,
    var_idx=None,
    var_discrete=None,
    samples=50,
    instances=10,
    random_seed=None,
    sharey=True,
    rug=True,
    smooth=True,
    indices=None,
    grid="long",
    color="C0",
    color_mean="C0",
    alpha=0.1,
    figsize=None,
    smooth_kwargs=None,
    ax=None,
):
    """
    Partial dependence or individual conditional expectation plot

    Parameters
    ----------
    idata: InferenceData
        InferenceData containing a collection of BART_trees in sample_stats group
    X : array-like
        The covariate matrix.
    Y : array-like
        The response vector.
    kind : str
        Whether to plor a partial dependence plot ("pdp") or an individual conditional expectation
        plot ("ice"). Defaults to pdp.
    xs_interval : str
        Method used to compute the values X used to evaluate the predicted function. "linear",
        evenly spaced values in the range of X. "quantiles", the evaluation is done at the specified
        quantiles of X. "insample", the evaluation is done at the values of X.
        For discrete variables these options are ommited.
    xs_values : int or list
        Values of X used to evaluate the predicted function. If ``xs_interval="linear"`` number of
        points in the evenly spaced grid. If ``xs_interval="quantiles"``quantile or sequence of
        quantiles to compute, which must be between 0 and 1 inclusive.
        Ignored when ``xs_interval="insample"``.
    var_idx : list
        List of the indices of the covariate for which to compute the pdp or ice.
    var_discrete : list
        List of the indices of the covariate treated as discrete.
    samples : int
        Number of posterior samples used in the predictions. Defaults to 50
    instances : int
        Number of instances of X to plot. Only relevant if ice ``kind="ice"`` plots.
    random_seed : int
        random_seed used to sample from the posterior. Defaults to None.
    sharey : bool
        Controls sharing of properties among y-axes. Defaults to True.
    rug : bool
        Whether to include a rugplot. Defaults to True.
    smooth=True,
        If True the result will be smoothed by first computing a linear interpolation of the data
        over a regular grid and then applying the Savitzky-Golay filter to the interpolated data.
        Defaults to True.
    grid : str or tuple
        How to arrange the subplots. Defaults to "long", one subplot below the other.
        Other options are "wide", one subplot next to eachother or a tuple indicating the number of
        rows and columns.
    color : matplotlib valid color
        Color used to plot the pdp or ice. Defaults to "C0"
    color_mean : matplotlib valid color
        Color used to plot the mean pdp or ice. Defaults to "C0",
    alpha : float
        Transparency level, should in the interval [0, 1].
    figsize : tuple
        Figure size. If None it will be defined automatically.
    smooth_kwargs : dict
        Additional keywords modifying the Savitzky-Golay filter.
        See scipy.signal.savgol_filter() for details.
    ax : axes
        Matplotlib axes.

    Returns
    -------
    axes: matplotlib axes
    """
    if kind not in ["pdp", "ice"]:
        raise ValueError(f"kind={kind} is not suported. Available option are 'pdp' or 'ice'")

    if xs_interval not in ["insample", "linear", "quantiles"]:
        raise ValueError(
            f"""{xs_interval} is not suported.
                          Available option are 'insample', 'linear' or 'quantiles'"""
        )

    rng = RandomState(seed=random_seed)

    if isinstance(X, pd.DataFrame):
        X_names = list(X.columns)
        X = X.values
    else:
        X_names = []

    if isinstance(Y, pd.DataFrame):
        Y_label = f"Predicted {Y.name}"
    else:
        Y_label = "Predicted Y"

    num_observations = X.shape[0]
    num_covariates = X.shape[1]

    indices = list(range(num_covariates))

    if var_idx is None:
        var_idx = indices
    if var_discrete is None:
        var_discrete = []

    if X_names:
        X_labels = [X_names[idx] for idx in var_idx]
    else:
        X_labels = [f"X_{idx}" for idx in var_idx]

    if xs_interval == "linear" and xs_values is None:
        xs_values = 10

    if xs_interval == "quantiles" and xs_values is None:
        xs_values = [0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]

    if kind == "ice":
        instances = np.random.choice(range(X.shape[0]), replace=False, size=instances)

    new_Y = []
    new_X_target = []
    y_mins = []

    new_X = np.zeros_like(X)
    idx_s = list(range(X.shape[0]))
    for i in var_idx:
        indices_mi = indices[:]
        indices_mi.pop(i)
        y_pred = []
        if kind == "pdp":
            if i in var_discrete:
                new_X_i = np.unique(X[:, i])
            else:
                if xs_interval == "linear":
                    new_X_i = np.linspace(np.nanmin(X[:, i]), np.nanmax(X[:, i]), xs_values)
                elif xs_interval == "quantiles":
                    new_X_i = np.quantile(X[:, i], q=xs_values)
                elif xs_interval == "insample":
                    new_X_i = X[:, i]

            for x_i in new_X_i:
                new_X[:, indices_mi] = X[:, indices_mi]
                new_X[:, i] = x_i
                y_pred.append(np.mean(predict(idata, rng, X_new=new_X, size=samples), 1))
            new_X_target.append(new_X_i)
        else:
            for instance in instances:
                new_X = X[idx_s]
                new_X[:, indices_mi] = X[:, indices_mi][instance]
                y_pred.append(np.mean(predict(idata, rng, X_new=new_X, size=samples), 0))
            new_X_target.append(new_X[:, i])
        y_mins.append(np.min(y_pred))
        new_Y.append(np.array(y_pred).T)

    if ax is None:
        if grid == "long":
            fig, axes = plt.subplots(len(var_idx), sharey=sharey, figsize=figsize)
        elif grid == "wide":
            fig, axes = plt.subplots(1, len(var_idx), sharey=sharey, figsize=figsize)
        elif isinstance(grid, tuple):
            fig, axes = plt.subplots(grid[0], grid[1], sharey=sharey, figsize=figsize)
        axes = np.ravel(axes)
    else:
        axes = [ax]
        fig = ax.get_figure()

    for i, ax in enumerate(axes):
        if i >= len(var_idx):
            ax.set_axis_off()
            fig.delaxes(ax)
        else:
            var = var_idx[i]
            if var in var_discrete:
                if kind == "pdp":
                    y_means = new_Y[i].mean(0)
                    hdi = az.hdi(new_Y[i])
                    ax.errorbar(
                        new_X_target[i],
                        y_means,
                        (y_means - hdi[:, 0], hdi[:, 1] - y_means),
                        fmt=".",
                        color=color,
                    )
                else:
                    ax.plot(new_X_target[i], new_Y[i], ".", color=color, alpha=alpha)
                    ax.plot(new_X_target[i], new_Y[i].mean(1), "o", color=color_mean)
                ax.set_xticks(new_X_target[i])
            elif smooth:
                if smooth_kwargs is None:
                    smooth_kwargs = {}
                smooth_kwargs.setdefault("window_length", 55)
                smooth_kwargs.setdefault("polyorder", 2)
                x_data = np.linspace(np.nanmin(new_X_target[i]), np.nanmax(new_X_target[i]), 200)
                x_data[0] = (x_data[0] + x_data[1]) / 2
                if kind == "pdp":
                    interp = griddata(new_X_target[i], new_Y[i].mean(0), x_data)
                else:
                    interp = griddata(new_X_target[i], new_Y[i], x_data)

                y_data = savgol_filter(interp, axis=0, **smooth_kwargs)

                if kind == "pdp":
                    az.plot_hdi(
                        new_X_target[i], new_Y[i], color=color, fill_kwargs={"alpha": alpha}, ax=ax
                    )
                    ax.plot(x_data, y_data, color=color_mean)
                else:
                    ax.plot(x_data, y_data.mean(1), color=color_mean)
                    ax.plot(x_data, y_data, color=color, alpha=alpha)

            else:
                idx = np.argsort(new_X_target[i])
                if kind == "pdp":
                    az.plot_hdi(
                        new_X_target[i],
                        new_Y[i],
                        smooth=smooth,
                        fill_kwargs={"alpha": alpha},
                        ax=ax,
                    )
                    ax.plot(new_X_target[i][idx], new_Y[i][idx].mean(0), color=color)
                else:
                    ax.plot(new_X_target[i][idx], new_Y[i][idx], color=color, alpha=alpha)
                    ax.plot(new_X_target[i][idx], new_Y[i][idx].mean(1), color=color_mean)

            if rug:
                lb = np.min(y_mins)
                ax.plot(X[:, var], np.full_like(X[:, var], lb), "k|")

            ax.set_xlabel(X_labels[i])

    fig.text(-0.05, 0.5, Y_label, va="center", rotation="vertical", fontsize=15)
    return axes
