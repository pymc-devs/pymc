import numpy as np
from scipy.stats import kde, mode

from pymc3.stats import hpd
from .utils import fast_kde


def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))


def histplot_op(ax, data, alpha=.35):
    """Add a histogram for each column of the data to the provided axes."""
    hs = []
    for column in data.T:
        hs.append(ax.hist(column, bins=_histplot_bins(column), alpha=alpha, align='left'))
    ax.set_xlim(np.min(data) - 0.5, np.max(data) + 0.5)
    return hs


def kdeplot_op(ax, data, prior=None, prior_alpha=1, prior_style='--'):
    """Geet a list of density and likelihood plots, if a prior is provided."""
    ls = []
    pls = []
    errored = []
    for i, d in enumerate(data.T):
        try:
            density, l, u = fast_kde(d)
            x = np.linspace(l, u, len(density))
            if prior is not None:
                p = prior.logp(x).eval()
                pls.append(ax.plot(x, np.exp(p), alpha=prior_alpha, ls=prior_style))

            ls.append(ax.plot(x, density))
        except ValueError:
            errored.append(str(i))

    if errored:
        ax.text(.27, .47, 'WARNING: KDE plot failed for: ' + ','.join(errored),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
                style='italic')

    return ls, pls


def kde2plot_op(ax, x, y, grid=200, **kwargs):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    extent = kwargs.pop('extent', [])
    if len(extent) != 4:
        extent = [xmin, xmax, ymin, ymax]

    grid = grid * 1j
    X, Y = np.mgrid[xmin:xmax:grid, ymin:ymax:grid]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = kde.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    ax.imshow(np.rot90(Z), extent=extent, **kwargs)


def plot_posterior_op(trace_values, ax, kde_plot, point_estimate, round_to,
                      alpha_level, ref_val, rope, **kwargs):
    """Artist to draw posterior."""
    def format_as_percent(x, round_to=0):
        return '{0:.{1:d}f}%'.format(100 * x, round_to)

    def display_ref_val(ref_val):
        less_than_ref_probability = (trace_values < ref_val).mean()
        greater_than_ref_probability = (trace_values >= ref_val).mean()
        ref_in_posterior = "{} <{:g}< {}".format(
            format_as_percent(less_than_ref_probability, 1),
            ref_val,
            format_as_percent(greater_than_ref_probability, 1))
        ax.axvline(ref_val, ymin=0.02, ymax=.75, color='g',
                   linewidth=4, alpha=0.65)
        ax.text(trace_values.mean(), plot_height * 0.6, ref_in_posterior,
                size=14, horizontalalignment='center')

    def display_rope(rope):
        ax.plot(rope, (plot_height * 0.02, plot_height * 0.02),
                linewidth=20, color='r', alpha=0.75)
        text_props = dict(size=16, horizontalalignment='center', color='r')
        ax.text(rope[0], plot_height * 0.14, rope[0], **text_props)
        ax.text(rope[1], plot_height * 0.14, rope[1], **text_props)

    def display_point_estimate():
        if not point_estimate:
            return
        if point_estimate not in ('mode', 'mean', 'median'):
            raise ValueError(
                "Point Estimate should be in ('mode','mean','median', None)")
        if point_estimate == 'mean':
            point_value = trace_values.mean()
            point_text = '{}={}'.format(
                point_estimate, point_value.round(round_to))
        elif point_estimate == 'mode':
            point_value = mode(trace_values.round(round_to))[0][0]
            point_text = '{}={}'.format(
                point_estimate, point_value.round(round_to))
        elif point_estimate == 'median':
            point_value = np.median(trace_values)
            point_text = '{}={}'.format(
                point_estimate, point_value.round(round_to))

        ax.text(point_value, plot_height * 0.8, point_text,
                size=16, horizontalalignment='center')

    def display_hpd():
        hpd_intervals = hpd(trace_values, alpha=alpha_level)
        ax.plot(hpd_intervals, (plot_height * 0.02,
                                plot_height * 0.02), linewidth=4, color='k')
        text_props = dict(size=16, horizontalalignment='center')
        ax.text(hpd_intervals[0], plot_height * 0.07,
                hpd_intervals[0].round(round_to), **text_props)
        ax.text(hpd_intervals[1], plot_height * 0.07,
                hpd_intervals[1].round(round_to), **text_props)
        ax.text((hpd_intervals[0] + hpd_intervals[1]) / 2, plot_height * 0.2,
                format_as_percent(1 - alpha_level) + ' HPD', **text_props)

    def format_axes():
        ax.yaxis.set_ticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', direction='out', width=1, length=3,
                       color='0.5')
        ax.spines['bottom'].set_color('0.5')

    def set_key_if_doesnt_exist(d, key, value):
        if key not in d:
            d[key] = value

    if kde_plot:
        density, l, u = fast_kde(trace_values)
        x = np.linspace(l, u, len(density))
        ax.plot(x, density, **kwargs)
    else:
        set_key_if_doesnt_exist(kwargs, 'bins', 30)
        set_key_if_doesnt_exist(kwargs, 'edgecolor', 'w')
        set_key_if_doesnt_exist(kwargs, 'align', 'right')
        ax.hist(trace_values, **kwargs)

    plot_height = ax.get_ylim()[1]

    format_axes()
    display_hpd()
    display_point_estimate()
    if ref_val is not None:
        display_ref_val(ref_val)
    if rope is not None:
        display_rope(rope)
