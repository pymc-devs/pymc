from .checks import *
from .models import *
import pymc as pm
import numpy as np

# Test if multiprocessing is available
import multiprocessing
try:
    multiprocessing.Pool(2)
    test_parallel = False
except:
    test_parallel = False


def check_trace(model, trace, n, step, start):
    # try using a trace object a few times

    for i in range(2):
        trace = sample(
            n, step, start, trace, progressbar=False, model=model)

        for (var, val) in start.items():

            assert np.shape(trace[var]) == (n * (i + 1),) + np.shape(val)

def test_trace():
    model, start, step, _ = simple_init()

    for h in [pm.NpTrace]:
        for n in [20, 1000]:
            for vars in [model.vars, model.vars + [model.vars[0] ** 2], model.unobserved_RVs]:
                trace = h(vars)

                yield check_trace, model, trace, n, step, start


def test_multitrace():
    if not test_parallel:
        return
    model, start, step, _ = simple_init()
    trace = None
    for n in [20, 1000]:

        yield check_multi_trace, model, trace, n, step, start


def check_multi_trace(model, trace, n, step, start):

    for i in range(2):
        trace = psample(
            n, step, start, trace, model=model)

        for (var, val) in start.items():
            print([len(tr.samples[var].vals) for tr in trace.traces])
            for t in trace[var]:
                assert np.shape(t) == (n * (i + 1),) + np.shape(val)

        ctrace = trace.combined()
        for (var, val) in start.items():

            assert np.shape(
                ctrace[var]) == (len(trace.traces) * n * (i + 1),) + np.shape(val)


def test_get_point():

    p, model = simple_2model()
    p2 = p.copy()
    p2['x'] *= 2.

    x = pm.NpTrace(model.vars)
    x.record(p)
    x.record(p2)
    assert x.point(1) == x[1]

def test_slice():

    model, start, step, moments = simple_init()

    iterations = 100
    burn = 10

    with model:
        tr = sample(iterations, start=start, step=step, progressbar=False)

    burned = tr[burn:]

    # Slicing returns a trace
    assert type(burned) is pm.trace.NpTrace

    # Indexing returns an array
    assert type(tr[tr.varnames[0]]) is np.ndarray

    # Burned trace is of the correct length
    assert np.all([burned[v].shape == (iterations-burn, start[v].size) for v in burned.varnames])

    # Original trace did not change
    assert np.all([tr[v].shape == (iterations, start[v].size) for v in tr.varnames])

    # Now take more burn-in from the burned trace
    burned_again = burned[burn:]
    assert np.all([burned_again[v].shape == (iterations-2*burn, start[v].size) for v in burned_again.varnames])
    assert np.all([burned[v].shape == (iterations-burn, start[v].size) for v in burned.varnames])

def test_multi_slice():

    model, start, step, moments = simple_init()

    iterations = 100
    burn = 10

    with model:
        tr = psample(iterations, start=start, step=step, threads=2)

    burned = tr[burn:]

    # Slicing returns a MultiTrace
    assert type(burned) is pm.trace.MultiTrace

    # Indexing returns a list of arrays
    assert type(tr[tr.varnames[0]]) is list
    assert type(tr[tr.varnames[0]][0]) is np.ndarray

    # # Burned trace is of the correct length
    assert np.all([burned[v][0].shape == (iterations-burn, start[v].size) for v in burned.varnames])

    # Original trace did not change
    assert np.all([tr[v][0].shape == (iterations, start[v].size) for v in tr.varnames])


## For all the summary tests, the number of dimensions refer to the
## original variable dimensions, not the MCMC trace dimensions.


def test_summary_0d_variable_model():
    mu = -2.1
    tau = 1.3
    with Model() as model:
        x = Normal('x', mu, tau, testval=.1)
        step = Metropolis(model.vars, np.diag([1.]))
        trace = sample(100, step=step)
    pm.summary(trace)


def test_summary_1d_variable_model():
    mu = -2.1
    tau = 1.3
    with Model() as model:
        x = Normal('x', mu, tau, shape=2, testval=[.1, .1])
        step = Metropolis(model.vars, np.diag([1.]))
        trace = sample(100, step=step)
    pm.summary(trace)


def test_summary_2d_variable_model():
    mu = -2.1
    tau = 1.3
    with Model() as model:
        x = Normal('x', mu, tau, shape=(2, 2),
                   testval=np.tile(.1, (2, 2)))
        step = Metropolis(model.vars, np.diag([1.]))
        trace = sample(100, step=step)
    pm.summary(trace)


def test_summary_format_values():
    roundto = 2
    summ = pm.trace._Summary(roundto)
    d = {'nodec': 1, 'onedec': 1.0, 'twodec': 1.00, 'threedec': 1.000}
    summ._format_values(d)
    for val in d.values():
        assert val == '1.00'


def test_stat_summary_format_hpd_values():
    roundto = 2
    summ = pm.trace._StatSummary(roundto, None, 0.05)
    d = {'nodec': 1, 'hpd': [1, 1]}
    summ._format_values(d)
    for key, val in d.items():
        if key == 'hpd':
            assert val == '[1.00, 1.00]'
        else:
            assert val == '1.00'


def test_calculate_stats_0d_variable():
    sample = np.arange(10)
    result = list(pm.trace._calculate_stats(sample, 5, 0.05))
    assert result[0] == ()
    assert len(result) == 2


def test_calculate_stats_variable_1d_variable():
    sample = np.arange(10).reshape(5, 2)
    result= list(pm.trace._calculate_stats(sample, 5, 0.05))
    assert result[0] == ()
    assert len(result) == 3

def test_calculate_pquantiles_0d_variable():
    sample = np.arange(10)[:, None]
    qlist = (0.25, 25, 50, 75, 0.98)
    result = list(pm.trace._calculate_posterior_quantiles(sample, qlist))
    assert result[0] == ()
    assert len(result) == 2


def test_stats_value_line():
    roundto = 1
    summ = pm.trace._StatSummary(roundto, None, 0.05)
    values = [{'mean': 0, 'sd': 1, 'mce': 2, 'hpd': [4, 4]},
              {'mean': 5, 'sd': 6, 'mce': 7, 'hpd': [8, 8]},]

    expected = ['0.0              1.0              2.0              [4.0, 4.0]',
                '5.0              6.0              7.0              [8.0, 8.0]']
    result = list(summ._create_value_output(values))
    assert result == expected


def test_post_quantile_value_line():
    roundto = 1
    summ = pm.trace._PosteriorQuantileSummary(roundto, 0.05)
    values = [{'lo': 0, 'q25': 1, 'q50': 2, 'q75': 4, 'hi': 5},
              {'lo': 6, 'q25': 7, 'q50': 8, 'q75': 9, 'hi': 10},]

    expected = ['0.0            1.0            2.0            4.0            5.0',
                '6.0            7.0            8.0            9.0            10.0']
    result = list(summ._create_value_output(values))
    assert result == expected


def test_stats_output_lines_0d_variable():
    roundto = 1
    x = np.arange(5)

    summ = pm.trace._StatSummary(roundto, 5, 0.05)

    expected = ['  Mean             SD               MC Error         95% HPD interval',
                '  -------------------------------------------------------------------',
                '  ',
                '  2.0              1.4              0.6              [0.0, 4.0]',]

    result = list(summ._get_lines(x))
    assert result == expected


def test_stats_output_lines_1d_variable():
    roundto = 1
    x = np.arange(10).reshape(5, 2)

    summ = pm.trace._StatSummary(roundto, 5, 0.05)

    expected = ['  Mean             SD               MC Error         95% HPD interval',
                '  -------------------------------------------------------------------',
                '  ',
                '  4.0              2.8              1.3              [0.0, 8.0]',
                '  5.0              2.8              1.3              [1.0, 9.0]',]
    result = list(summ._get_lines(x))
    assert result == expected


def test_stats_output_lines_2d_variable():
    roundto = 1
    x = np.arange(20).reshape(5, 2, 2)

    summ = pm.trace._StatSummary(roundto, 5, 0.05)

    expected = ['  Mean             SD               MC Error         95% HPD interval',
                '  -------------------------------------------------------------------',
                '  ..............................[0, :]...............................',
                '  8.0              5.7              2.5              [0.0, 16.0]',
                '  9.0              5.7              2.5              [1.0, 17.0]',
                '  ..............................[1, :]...............................',
                '  10.0             5.7              2.5              [2.0, 18.0]',
                '  11.0             5.7              2.5              [3.0, 19.0]',]
    result = list(summ._get_lines(x))
    assert result == expected


def test_posterior_quantiles_output_lines_0d_variable():
    roundto = 1
    x = np.arange(5)

    summ = pm.trace._PosteriorQuantileSummary(roundto, 0.05)

    expected = ['  Posterior quantiles:',
                '  2.5            25             50             75             97.5',
                '  |--------------|==============|==============|--------------|',
                '  ',
                '  0.0            1.0            2.0            3.0            4.0',]

    result = list(summ._get_lines(x))
    assert result == expected


def test_posterior_quantiles_output_lines_1d_variable():
    roundto = 1
    x = np.arange(10).reshape(5, 2)

    summ = pm.trace._PosteriorQuantileSummary(roundto, 0.05)

    expected = ['  Posterior quantiles:',
                '  2.5            25             50             75             97.5',
                '  |--------------|==============|==============|--------------|',
                '  ',
                '  0.0            2.0            4.0            6.0            8.0',
                '  1.0            3.0            5.0            7.0            9.0']

    result = list(summ._get_lines(x))
    assert result == expected


def test_posterior_quantiles_output_lines_2d_variable():
    roundto = 1
    x = np.arange(20).reshape(5, 2, 2)

    summ = pm.trace._PosteriorQuantileSummary(roundto, 0.05)

    expected = ['  Posterior quantiles:',
                '  2.5            25             50             75             97.5',
                '  |--------------|==============|==============|--------------|',
                '  .............................[0, :].............................',
                '  0.0            4.0            8.0            12.0           16.0',
                '  1.0            5.0            9.0            13.0           17.0',
                '  .............................[1, :].............................',
                '  2.0            6.0            10.0           14.0           18.0',
                '  3.0            7.0            11.0           15.0           19.0',]

    result = list(summ._get_lines(x))
    assert result == expected


def test_groupby_leading_idxs_0d_variable():
    result = {k: list(v) for k, v in pm.trace._groupby_leading_idxs(())}
    assert list(result.keys()) == [()]
    assert result[()] == [()]


def test_groupby_leading_idxs_1d_variable():
    result = {k: list(v) for k, v in pm.trace._groupby_leading_idxs((2,))}
    assert list(result.keys()) == [()]
    assert result[()] == [(0,), (1,)]


def test_groupby_leading_idxs_2d_variable():
    result = {k: list(v) for k, v in pm.trace._groupby_leading_idxs((2, 3))}

    expected_keys = [(0,), (1,)]
    keys = list(result.keys())
    assert len(keys) == len(expected_keys)
    for key in keys:
        assert result[key] == [key + (0,), key + (1,), key + (2,)]


def test_groupby_leading_idxs_3d_variable():
    result = {k: list(v) for k, v in pm.trace._groupby_leading_idxs((2, 3, 2))}

    expected_keys = [(0, 0), (0, 1), (0, 2),
                     (1, 0), (1, 1), (1, 2)]
    keys = list(result.keys())
    assert len(keys) == len(expected_keys)
    for key in keys:
        assert result[key] == [key + (0,), key + (1,)]
