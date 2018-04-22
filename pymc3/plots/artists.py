import numpy as np
from scipy.stats import mode
from collections import OrderedDict

from pymc3.stats import hpd
from .kdeplot import fast_kde, kdeplot


def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))


def histplot_op(ax, data, alpha=.35):
    """Add a histogram for each column of the data to the provided axes."""
    hs = []
    for column in data.T:
        hs.append(ax.hist(column, bins=_histplot_bins(
                  column), alpha=alpha, align='left'))
    ax.set_xlim(np.min(data) - 0.5, np.max(data) + 0.5)
    return hs


def kdeplot_op(ax, data, bw, prior=None, prior_alpha=1, prior_style='--'):
    """Get a list of density and likelihood plots, if a prior is provided."""
    ls = []
    pls = []
    errored = []
    for i, d in enumerate(data.T):
        try:
            density, l, u = fast_kde(d, bw)
            x = np.linspace(l, u, len(density))
            if prior is not None:
                p = prior.logp(x).eval()
                pls.append(ax.plot(x, np.exp(p),
                                   alpha=prior_alpha, ls=prior_style))

            ls.append(ax.plot(x, density))
        except ValueError:
            errored.append(str(i))

    if errored:
        ax.text(.27, .47, 'WARNING: KDE plot failed for: ' + ','.join(errored),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
                style='italic')

    return ls, pls


def plot_posterior_op(trace_values, ax, bw, kde_plot, point_estimate, round_to,
                      alpha_level, ref_val, rope, text_size=16, **kwargs):
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
                size=text_size, horizontalalignment='center')

    def display_rope(rope):
        ax.plot(rope, (plot_height * 0.02, plot_height * 0.02),
                linewidth=20, color='r', alpha=0.75)
        text_props = dict(size=text_size, horizontalalignment='center', color='r')
        ax.text(rope[0], plot_height * 0.14, rope[0], **text_props)
        ax.text(rope[1], plot_height * 0.14, rope[1], **text_props)

    def display_point_estimate():
        if not point_estimate:
            return
        if point_estimate not in ('mode', 'mean', 'median'):
            raise ValueError(
                "Point Estimate should be in ('mode','mean','median')")
        if point_estimate == 'mean':
            point_value = trace_values.mean()
        elif point_estimate == 'mode':
            if isinstance(trace_values[0], float):
                density, l, u = fast_kde(trace_values, bw)
                x = np.linspace(l, u, len(density))
                point_value = x[np.argmax(density)]
            else:
                point_value = mode(trace_values.round(round_to))[0][0]
        elif point_estimate == 'median':
            point_value = np.median(trace_values)
        point_text = '{point_estimate}={point_value:.{round_to}f}'.format(point_estimate=point_estimate,
                                                                          point_value=point_value, round_to=round_to)

        ax.text(point_value, plot_height * 0.8, point_text,
                size=text_size, horizontalalignment='center')

    def display_hpd():
        hpd_intervals = hpd(trace_values, alpha=alpha_level)
        ax.plot(hpd_intervals, (plot_height * 0.02,
                                plot_height * 0.02), linewidth=4, color='k')
        ax.text(hpd_intervals[0], plot_height * 0.07,
                hpd_intervals[0].round(round_to),
                size=text_size, horizontalalignment='right')
        ax.text(hpd_intervals[1], plot_height * 0.07,
                hpd_intervals[1].round(round_to),
                size=text_size, horizontalalignment='left')
        ax.text((hpd_intervals[0] + hpd_intervals[1]) / 2, plot_height * 0.2,
                format_as_percent(1 - alpha_level) + ' HPD',
                size=text_size, horizontalalignment='center')

    def format_axes():
        ax.yaxis.set_ticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', direction='out', width=1, length=3,
                       color='0.5', labelsize=text_size)
        ax.spines['bottom'].set_color('0.5')

    def set_key_if_doesnt_exist(d, key, value):
        if key not in d:
            d[key] = value

    if kde_plot and isinstance(trace_values[0], float):
        kdeplot(trace_values, alpha=kwargs.pop('alpha', 0.35), bw=bw, ax=ax, **kwargs)

    else:
        set_key_if_doesnt_exist(kwargs, 'bins', 30)
        set_key_if_doesnt_exist(kwargs, 'edgecolor', 'w')
        set_key_if_doesnt_exist(kwargs, 'align', 'right')
        set_key_if_doesnt_exist(kwargs, 'color', '#87ceeb')
        ax.hist(trace_values, **kwargs)

    plot_height = ax.get_ylim()[1]

    format_axes()
    display_hpd()
    display_point_estimate()
    if ref_val is not None:
        display_ref_val(ref_val)
    if rope is not None:
        display_rope(rope)

def scale_text(figsize, text_size):
        """Scale text to figsize."""

        if text_size is None and figsize is not None:
            if figsize[0] <= 11:
                return 12
            else:
                return figsize[0]
        else:
            return text_size

def get_trace_dict(tr, varnames):
        traces = OrderedDict()
        for v in varnames:
            vals = tr.get_values(v, combine=True, squeeze=True)
            if vals.ndim > 1:
                vals_flat = vals.reshape(vals.shape[0], -1).T
                for i, vi in enumerate(vals_flat):
                    traces['_'.join([v, str(i)])] = vi
            else:
                traces[v] = vals
        return traces
        