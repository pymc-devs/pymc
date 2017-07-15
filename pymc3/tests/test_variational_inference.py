import pytest
import pickle
import operator
import numpy as np
from theano import theano, tensor as tt
from theano.configparser import change_flags
import pymc3 as pm
from pymc3 import Model, Normal
from pymc3.variational import (
    ADVI, FullRankADVI, SVGD, NFVI,
    Empirical, ASVGD,
    MeanField, FullRank,
    fit, flows
)

from pymc3.variational.operators import KL
from pymc3.tests import models


pytestmark = pytest.mark.usefixtures(
    'strict_float32',
    'seeded_test'
)


@pytest.mark.parametrize(
    'diff',
    [
        'relative',
        'absolute'
    ]
)
@pytest.mark.parametrize(
    'ord',
    [1, 2, np.inf]
)
def test_callbacks_convergence(diff, ord):
    cb = pm.variational.callbacks.CheckParametersConvergence(every=1, diff=diff, ord=ord)

    class _approx:
        params = (theano.shared(np.asarray([1, 2, 3])), )

    approx = _approx()

    with pytest.raises(StopIteration):
        cb(approx, None, 1)
        cb(approx, None, 10)


def test_tracker_callback():
    import time
    tracker = pm.callbacks.Tracker(
        ints=lambda *t: t[-1],
        ints2=lambda ap, h, j: j,
        time=time.time,
    )
    for i in range(10):
        tracker(None, None, i)
    assert 'time' in tracker.hist
    assert 'ints' in tracker.hist
    assert 'ints2' in tracker.hist
    assert (len(tracker['ints'])
            == len(tracker['ints2'])
            == len(tracker['time'])
            == 10)
    assert tracker['ints'] == tracker['ints2'] == list(range(10))
    tracker = pm.callbacks.Tracker(
        bad=lambda t: t  # bad signature
    )
    with pytest.raises(TypeError):
        tracker(None, None, 1)


def test_elbo():
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])

    post_mu = np.array([1.88], dtype=theano.config.floatX)
    post_sd = np.array([1], dtype=theano.config.floatX)
    # Create a model for test
    with Model() as model:
        mu = Normal('mu', mu=mu0, sd=sigma)
        Normal('y', mu=mu, sd=1, observed=y_obs)

    # Create variational gradient tensor
    mean_field = MeanField(model=model)
    with change_flags(compute_test_value='off'):
        elbo = -KL(mean_field)()(10000)

    mean_field.shared_params['mu'].set_value(post_mu)
    mean_field.shared_params['rho'].set_value(np.log(np.exp(post_sd) - 1))

    f = theano.function([], elbo)
    elbo_mc = f()

    # Exact value
    elbo_true = (-0.5 * (
        3 + 3 * post_mu ** 2 - 2 * (y_obs[0] + y_obs[1] + mu0) * post_mu +
        y_obs[0] ** 2 + y_obs[1] ** 2 + mu0 ** 2 + 3 * np.log(2 * np.pi)) +
                 0.5 * (np.log(2 * np.pi) + 1))
    np.testing.assert_allclose(elbo_mc, elbo_true, rtol=0, atol=1e-1)


@pytest.fixture(
    'module',
    params=[
        dict(mini=True, scale=False),
        dict(mini=False, scale=True),
    ],
    ids=['mini-noscale', 'full-scale']
)
def minibatch_and_scaling(request):
    return request.param


@pytest.fixture('module')
def using_minibatch(minibatch_and_scaling):
    return minibatch_and_scaling['mini']


@pytest.fixture('module')
def scale_cost_to_minibatch(minibatch_and_scaling):
    return minibatch_and_scaling['scale']


@pytest.fixture('module')
def simple_model_data(using_minibatch):
    n = 1000
    sd0 = 2.
    mu0 = 4.
    sd = 3.
    mu = -5.

    data = sd * np.random.randn(n) + mu
    d = n / sd ** 2 + 1 / sd0 ** 2
    mu_post = (n * np.mean(data) / sd ** 2 + mu0 / sd0 ** 2) / d
    if using_minibatch:
        data = pm.Minibatch(data)
    return dict(
        n=n,
        data=data,
        mu_post=mu_post,
        d=d,
        mu0=mu0,
        sd0=sd0,
        sd=sd,
    )


@pytest.fixture(scope='module')
def simple_model(simple_model_data):
    with Model() as model:
        mu_ = Normal(
            'mu', mu=simple_model_data['mu0'],
            sd=simple_model_data['sd0'], testval=0)
        Normal('x', mu=mu_, sd=simple_model_data['sd'],
               observed=simple_model_data['data'],
               total_size=simple_model_data['n'])
    return model


@pytest.fixture('module', params=[
        dict(cls=NFVI, init=dict(flow='scale-loc')),
        dict(cls=ADVI, init=dict()),
        dict(cls=FullRankADVI, init=dict()),
        dict(cls=SVGD, init=dict(n_particles=500, jitter=1)),
        dict(cls=ASVGD, init=dict(temperature=1.)),
    ], ids=[
        'NFVI=scale-loc',
        'ADVI',
        'FullRankADVI',
        'SVGD',
        'ASVGD'
    ])
def inference_spec(request):
    cls = request.param['cls']
    init = request.param['init']

    def init_(**kw):
        k = init.copy()
        k.update(kw)
        return cls(**k)
    init_.cls = cls
    return init_


@pytest.fixture('function')
def inference(inference_spec, simple_model, scale_cost_to_minibatch):
    with simple_model:
        return inference_spec(scale_cost_to_minibatch=scale_cost_to_minibatch)


@pytest.fixture('function')
def fit_kwargs(inference, using_minibatch):
    _select = {
        (ADVI, 'full'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.02, n_win=50),
            n=5000
        ),
        (ADVI, 'mini'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50),
            n=12000
        ),
        (NFVI, 'full'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50),
            n=12000
        ),
        (NFVI, 'mini'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50),
            n=12000
        ),
        (FullRankADVI, 'full'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.007, n_win=50),
            n=6000
        ),
        (FullRankADVI, 'mini'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.007, n_win=50),
            n=12000
        ),
        (SVGD, 'full'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.07, n_win=7),
            n=300
        ),
        (SVGD, 'mini'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.07, n_win=7),
            n=300
        ),
        (ASVGD, 'full'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.07, n_win=10),
            n=500, obj_n_mc=300
        ),
        (ASVGD, 'mini'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.07, n_win=10),
            n=500, obj_n_mc=300
        )
    }
    if using_minibatch:
        key = 'mini'
    else:
        key = 'full'
    return _select[(type(inference), key)]


@pytest.mark.run('first')
def test_fit_oo(inference,
                fit_kwargs,
                simple_model_data):
    trace = inference.fit(**fit_kwargs).sample(10000)
    mu_post = simple_model_data['mu_post']
    d = simple_model_data['d']
    np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.05)
    np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.1)


def test_profile(inference):
    inference.run_profiling(n=100).summary()


@pytest.fixture('module')
def another_simple_model():
    _model = models.simple_model()[1]
    with _model:
        pm.Potential('pot', tt.ones((10, 10)))
    return _model


@pytest.fixture(params=[
    dict(name='advi', kw=dict(start={})),
    dict(name='fullrank_advi', kw=dict(start={})),
    dict(name='svgd', kw=dict(start={}))],
    ids=lambda d: d['name']
)
def fit_method_with_object(request, another_simple_model):
    _select = dict(
        advi=ADVI,
        fullrank_advi=FullRankADVI,
        svgd=SVGD
    )
    with another_simple_model:
        return _select[request.param['name']](
            **request.param['kw'])


@pytest.mark.parametrize(
    ['method', 'kwargs', 'error'],
    [
        ('undefined', dict(), KeyError),
        (1, dict(), TypeError),
        ('advi', dict(total_grad_norm_constraint=10), None),
        ('advi->fullrank_advi', dict(frac=.1), None),
        ('advi->fullrank_advi', dict(frac=1), ValueError),
        ('fullrank_advi', dict(), None),
        ('svgd', dict(total_grad_norm_constraint=10), None),
        ('svgd', dict(start={}), None),
        ('asvgd', dict(start={}, total_grad_norm_constraint=10), None),
        ('nfvi', dict(start={}), None),
        ('nfvi=scale-loc', dict(start={}), None),
        ('nfvi=bad-formula', dict(start={}), ValueError),
    ],
)
def test_fit_fn_text(method, kwargs, error, another_simple_model):
    with another_simple_model:
        if error is not None:
            with pytest.raises(error):
                fit(10, method=method, **kwargs)
        else:
            fit(10, method=method, **kwargs)


def test_fit_fn_oo(fit_method_with_object, another_simple_model):
    with another_simple_model:
        fit(10, method=fit_method_with_object)


def test_error_on_local_rv_for_svgd(another_simple_model):
    with another_simple_model:
        with pytest.raises(ValueError) as e:
            fit(10, method='svgd', local_rv={
                another_simple_model.free_RVs[0]: (0, 1)})
        assert e.match('does not support AEVB')


@pytest.fixture('module')
def aevb_model():
    with pm.Model() as model:
        x = pm.Normal('x')
        pm.Normal('y', x)
    x = model.x
    y = model.y
    mu = theano.shared(x.init_value) * 2
    rho = theano.shared(np.zeros_like(x.init_value))
    return {
        'model': model,
        'y': y,
        'x': x,
        'replace': (mu, rho)
    }


def test_aevb(inference_spec, aevb_model):
    # add to inference that supports aevb
    cls = inference_spec.cls
    if not cls.OP.SUPPORT_AEVB:
        raise pytest.skip('%s does not support aevb' % cls)
    x = aevb_model['x']
    y = aevb_model['y']
    model = aevb_model['model']
    replace = aevb_model['replace']
    with model:
        inference = inference_spec(local_rv={x: replace})
        approx = inference.fit(3, obj_n_mc=2)
        approx.sample(10)
        approx.apply_replacements(
            y,
            more_replacements={x: np.asarray([1, 1], dtype=x.dtype)}
        ).eval()


@pytest.fixture('module')
def binomial_model():
    n_samples = 100
    xs = np.random.binomial(n=1, p=0.2, size=n_samples)
    with pm.Model() as model:
        p = pm.Beta('p', alpha=1, beta=1)
        pm.Binomial('xs', n=1, p=p, observed=xs)
    return model


@pytest.fixture('module')
def binomial_model_inference(binomial_model, inference_spec):
    with binomial_model:
        return inference_spec()


@pytest.mark.run(after='test_replacements')
def test_n_mc(binomial_model_inference):
    binomial_model_inference.fit(10, obj_n_mc=2)


@pytest.mark.run(after='test_sample_replacements')
def test_replacements(binomial_model_inference):
    d = tt.bscalar()
    d.tag.test_value = 1
    approx = binomial_model_inference.approx
    p = approx.model.p
    p_t = p ** 3
    p_s = approx.apply_replacements(p_t)
    if theano.config.compute_test_value != 'off':
        assert p_s.tag.test_value.shape == p_t.tag.test_value.shape
    sampled = [p_s.eval() for _ in range(100)]
    assert any(map(
        operator.ne,
        sampled[1:], sampled[:-1])
    )  # stochastic

    p_d = approx.apply_replacements(p_t, deterministic=True)
    sampled = [p_d.eval() for _ in range(100)]
    assert all(map(
        operator.eq,
        sampled[1:], sampled[:-1])
    )  # deterministic

    p_r = approx.apply_replacements(p_t, deterministic=d)
    sampled = [p_r.eval({d: 1}) for _ in range(100)]
    assert all(map(
        operator.eq,
        sampled[1:], sampled[:-1])
    )  # deterministic
    sampled = [p_r.eval({d: 0}) for _ in range(100)]
    assert any(map(
        operator.ne,
        sampled[1:], sampled[:-1])
    )  # stochastic


def test_sample_replacements(binomial_model_inference):
    i = tt.iscalar()
    i.tag.test_value = 1
    approx = binomial_model_inference.approx
    p = approx.model.p
    p_t = p ** 3
    p_s = approx.sample_node(p_t, size=100)
    if theano.config.compute_test_value != 'off':
        assert p_s.tag.test_value.shape == (100, ) + p_t.tag.test_value.shape
    sampled = p_s.eval()
    assert any(map(
        operator.ne,
        sampled[1:], sampled[:-1])
    )  # stochastic
    assert sampled.shape[0] == 100

    p_d = approx.sample_node(p_t, size=i)
    sampled = p_d.eval({i: 100})
    assert any(map(
        operator.ne,
        sampled[1:], sampled[:-1])
    )  # deterministic
    assert sampled.shape[0] == 100
    sampled = p_d.eval({i: 101})
    assert sampled.shape[0] == 101


def test_pickling(binomial_model_inference):
    inference = pickle.loads(pickle.dumps(binomial_model_inference))
    inference.fit(20)


def test_empirical_from_trace(another_simple_model):
    with another_simple_model:
        step = pm.Metropolis()
        trace = pm.sample(100, step=step)
        emp = Empirical(trace)
        assert emp.histogram.shape[0].eval() == 100
        trace = pm.sample(100, step=step, njobs=4)
        emp = Empirical(trace)
        assert emp.histogram.shape[0].eval() == 400


def test_multiple_replacements(inference_spec):
    _, model, _ = models.exponential_beta(n=2)
    x = model.x
    y = model.y
    xy = x*y
    xpy = x+y
    with model:
        ap = inference_spec().approx
        xy_, xpy_ = ap.apply_replacements([xy, xpy])
        xy_s, xpy_s = ap.sample_node([xy, xpy])
        xy_.eval()
        xpy_.eval()
        xy_s.eval()
        xpy_s.eval()


def test_from_mean_field(another_simple_model):
    with another_simple_model:
        advi = ADVI()
        full_rank = FullRankADVI.from_mean_field(advi.approx)
        full_rank.fit(20)


def test_from_advi(another_simple_model):
    with another_simple_model:
        advi = ADVI()
        full_rank = FullRankADVI.from_advi(advi)
        full_rank.fit(20)


def test_from_full_rank(another_simple_model):
    with another_simple_model:
        fr = FullRank()
        full_rank = FullRankADVI.from_full_rank(fr)
        full_rank.fit(20)


def test_from_empirical(another_simple_model):
    with another_simple_model:
        emp = Empirical.from_noise(1000)
        svgd = SVGD.from_empirical(emp)
        svgd.fit(20)


def test_aevb_empirical():
    _, model, _ = models.exponential_beta(n=2)
    x = model.x
    mu = theano.shared(x.init_value)
    rho = theano.shared(np.zeros_like(x.init_value))
    with model:
        inference = ADVI(local_rv={x: (mu, rho)})
        approx = inference.approx
        trace0 = approx.sample(10000)
        approx = Empirical(trace0, local_rv={x: (mu, rho)})
        trace1 = approx.sample(10000)
    np.testing.assert_allclose(trace0['y'].mean(0), trace1['y'].mean(0), atol=0.02)
    np.testing.assert_allclose(trace0['y'].var(0), trace1['y'].var(0), atol=0.02)
    np.testing.assert_allclose(trace0['x'].mean(0), trace1['x'].mean(0), atol=0.02)
    np.testing.assert_allclose(trace0['x'].var(0), trace1['x'].var(0), atol=0.02)


@pytest.fixture(
    params=[
        dict(cls=flows.PlanarFlow, init=dict(jitter=.1)),
        dict(cls=flows.RadialFlow, init=dict(jitter=.1)),
        dict(cls=flows.ScaleFlow, init=dict(jitter=.1)),
        dict(cls=flows.LocFlow, init=dict(jitter=.1)),
        dict(cls=flows.HouseholderFlow, init=dict(jitter=.1)),
    ],
    ids=lambda d: d['cls'].__name__
)
def flow_spec(request):
    cls = request.param['cls']
    init = request.param['init']

    def init_(**kw):
        k = init.copy()
        k.update(kw)
        return cls(**k)
    init_.cls = cls
    return init_


def test_flow_init_loop(flow_spec):
    flow = pm.tt_rng().normal(size=(10, 2))
    for i in range(10):
        flow = flow_spec(z0=flow, dim=2)
    flow.forward.eval()


def test_flow_forward_apply(flow_spec):
    z0 = pm.tt_rng().normal(size=(10, 20))
    flow = flow_spec(dim=20, z0=z0)
    with change_flags(compute_test_value='off'):
        dist = flow.forward
        shape_dist = dist.shape.eval()
    assert tuple(shape_dist) == (10, 20)


def test_flow_det(flow_spec):
    z0 = tt.arange(0, 20).astype('float32')
    flow = flow_spec(dim=20, z0=z0.dimshuffle('x', 0))
    with change_flags(compute_test_value='off'):
        z1 = flow.forward.flatten()
        J = tt.jacobian(z1, z0)
        logJdet = tt.log(tt.abs_(tt.nlinalg.det(J)))
        det = flow.logdet[0]
    np.testing.assert_allclose(logJdet.eval(), det.eval(), atol=0.0001)


def test_flow_det_shape(flow_spec):
    with change_flags(compute_test_value='off'):
        z0 = pm.tt_rng().normal(size=(10, 20))
        flow = flow_spec(dim=20, z0=z0)
        det = flow.logdet
        det_dist = det.shape.eval()
    assert tuple(det_dist) == (10,)


def test_flows_collect_chain():
    initial = tt.ones((3, 2))
    flow1 = flows.PlanarFlow(dim=2, z0=initial)
    flow2 = flows.PlanarFlow(dim=2, z0=flow1)
    assert len(flow2.params) == 3
    assert len(flow2.all_params) == 6
    np.testing.assert_allclose(flow1.logdet.eval() + flow2.logdet.eval(), flow2.sum_logdets.eval())


@pytest.mark.parametrize(
    'formula,length,order',
    [
        ('planar', 1, [flows.PlanarFlow]),
        ('planar*2', 2, [flows.PlanarFlow] * 2),
        ('planar-planar', 2, [flows.PlanarFlow] * 2),
        ('planar-planar*2', 3, [flows.PlanarFlow] * 3),
        ('hh-planar*2', 3, [flows.HouseholderFlow]+[flows.PlanarFlow] * 2)
    ]
)
def test_flow_formula(formula, length, order):
    spec = flows.Formula(formula)
    flows_list = spec.flows
    assert len(flows_list) == length
    if order is not None:
        assert flows_list == order
    spec(dim=2, jitter=1)(tt.ones((3, 2))).eval()  # should work


def test_hh_flow():
    cov = pm.floatX([[2, -1], [-1, 3]])
    with pm.Model():
        pm.MvNormal('mvN', mu=pm.floatX([0, 1]), cov=cov, shape=2)
        nf = NFVI('scale-hh*2-loc')
        nf.fit(25000, obj_optimizer=pm.adam(learning_rate=0.001))
        trace = nf.approx.sample(10000)
        cov2 = pm.trace_cov(trace)
    np.testing.assert_allclose(cov, cov2, rtol=0.07)
