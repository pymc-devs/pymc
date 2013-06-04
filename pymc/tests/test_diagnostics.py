from pymc import *

from pymc.examples import disaster_model as dm

def test_gelman_rubin(n=1000):

    with dm.model:
        # Run sampler
        step1 = Slice([dm.early_mean, dm.late_mean])
        step2 = Metropolis([dm.switchpoint])
        start = {'early_mean': 2., 'late_mean': 3., 'switchpoint': 50}
        ptrace = psample(n, [step1, step2], start, threads=2,
            random_seeds=[1,3])

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
    print max(abs(z_switch[:, 1]))
    assert max(abs(z_switch[:, 1])) < 1

