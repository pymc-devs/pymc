from pymc import *

from pymc.examples import disaster_model as dm

def test_gelman_rubin(n=1000):

    with dm.model:
        # Run sampler
        ptrace = psample(n, [dm.step1, dm.step2], dm.start, threads=2)

    rhat = gelman_rubin(ptrace)

    assert np.all([r<1.5 for r in rhat.values()])

def test_geweke(n=1000):

    with dm.model:
        # Run sampler
        trace = sample(n, [dm.step1, dm.step2], dm.start, progressbar=False)

    z_switch = geweke(trace, last=.5, intervals=20)['switchpoint']

    # Ensure `intervals` argument is honored
    assert len(z_switch) == 20

    # Ensure `last` argument is honored
    assert z_switch[-1,0] < (n/2)

    # These should all be z-scores
    assert max(abs(z_switch[:,1])) < 1
