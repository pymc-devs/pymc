from pymc3 import *

from pymc3.examples import disaster_model as dm

def test_gelman_rubin(n=1000):

    with dm.model:
        # Run sampler
        step1 = Slice([dm.early_mean, dm.late_mean])
        step2 = Metropolis([dm.switchpoint])
        start = {'early_mean': 2., 'late_mean': 3., 'switchpoint': 50}
        ptrace = sample(n, [step1, step2], start, njobs=2,
                        random_seed=[1, 3])

    rhat = gelman_rubin(ptrace)

    assert np.all([r < 1.5 for r in rhat.values()])


def test_geweke(n=3000):

    with dm.model:
        # Run sampler
        step1 = Slice([dm.early_mean, dm.late_mean])
        step2 = Metropolis([dm.switchpoint])
        trace = sample(n, [step1, step2], progressbar=False,
                       random_seed=1)

    z_switch = geweke(trace['switchpoint'], last=.5, intervals=20)

    # Ensure `intervals` argument is honored
    assert len(z_switch) == 20

    # Ensure `last` argument is honored
    assert z_switch[-1, 0] < (n / 2)

    # These should all be z-scores
    print(max(abs(z_switch[:, 1])))
    assert max(abs(z_switch[:, 1])) < 1


def test_effective_n(k=3, n=1000):
    """Unit test for effective sample size"""
    
    model = Model()
    with model:
        x = Normal('x', 0, 1., shape=5)

        # start sampling at the MAP
        start = find_MAP()

        step = NUTS(scaling=start)
    
        ptrace = sample(n, step, start, njobs=k,
                        random_seed=42)
        
    n_eff = effective_n(ptrace)['x']
    
    assert np.isclose(n_eff, k*n, 2).all()