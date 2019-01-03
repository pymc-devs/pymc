import pymc3.parallel_sampling as ps
import pymc3 as pm


def test_abort():
    with pm.Model() as model:
        a = pm.Normal('a', shape=1)
        pm.HalfNormal('b')
        step1 = pm.NUTS([a])
        step2 = pm.Metropolis([model.b_log__])

    step = pm.CompoundStep([step1, step2])

    proc = ps.ProcessAdapter(10, 10, step, chain=3, seed=1,
                             start={'a': 1., 'b_log__': 2.})
    proc.start()
    proc.write_next()
    proc.abort()
    proc.join()


def test_explicit_sample():
    with pm.Model() as model:
        a = pm.Normal('a', shape=1)
        pm.HalfNormal('b')
        step1 = pm.NUTS([a])
        step2 = pm.Metropolis([model.b_log__])

    step = pm.CompoundStep([step1, step2])

    proc = ps.ProcessAdapter(10, 10, step, chain=3, seed=1,
                             start={'a': 1., 'b_log__': 2.})
    proc.start()
    while True:
        proc.write_next()
        out = ps.ProcessAdapter.recv_draw([proc])
        view = proc.shared_point_view
        for name in view:
            view[name].copy()
        if out[1]:
            break
    proc.join()


def test_iterator():
    with pm.Model() as model:
        a = pm.Normal('a', shape=1)
        pm.HalfNormal('b')
        step1 = pm.NUTS([a])
        step2 = pm.Metropolis([model.b_log__])

    step = pm.CompoundStep([step1, step2])

    start = {'a': 1., 'b_log__': 2.}
    sampler = ps.ParallelSampler(10, 10, 3, 2, [2, 3, 4], [start] * 3,
                                 step, 0, False)
    with sampler:
        for draw in sampler:
            pass
