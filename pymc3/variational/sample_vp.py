import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
import pymc3 as pm
from pymc3.backends.base import MultiTrace
from .advi import ADVIFit, gen_random_state, get_transformed

__all__ = ['sample_vp']

def _make_sampling_updates(vs, us, ws, rng):
    updates = {}
    for v, u, w in zip(vs, us, ws):
        # n = rng.normal(size=u.tag.test_value.shape)
        n = rng.normal(size=u.shape)
        updates.update({v: (n * w + u).reshape(v.tag.test_value.shape)})

    return updates


def sample_vp(
        vparams, draws=1000, model=None, local_RVs=None, random_seed=None,
        hide_transformed=True, nfs=None):
    """Draw samples from variational posterior.

    Parameters
    ----------
    vparams : dict or pymc3.variational.ADVIFit
        Estimated variational parameters of the model.
    draws : int
        Number of random samples.
    model : pymc3.Model
        Probabilistic model. 

    random_seed : int or None
        Seed of random number generator.  None to use current seed.
    hide_transformed : bool
        If False, transformed variables are also sampled. Default is True.

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        Samples drawn from the variational posterior.
    """
    local_RVs = list() if local_RVs is None else local_RVs
    nfs = list() if nfs is None else nfs

    model = pm.modelcontext(model)

    if isinstance(vparams, ADVIFit):
        vparams = {
            'means': vparams.means,
            'stds': vparams.stds
        }

    if len(local_RVs) == 0:
        global_RVs = list(model.free_RVs)
    else:
        rvs = [get_transformed(v, model) for v in local_RVs]
        global_RVs = list(set(model.free_RVs) - set(rvs))

    # Random number generator for sampling from variational posterior
    if random_seed is None:
        rng = MRG_RandomStreams(gen_random_state())
    else:
        rng = MRG_RandomStreams(seed=random_seed)

    # Make dict for sampling global RVs
    updates = _make_sampling_updates(
        vs=global_RVs, 
        us=[vparams['means'][str(v)].ravel() for v in global_RVs], 
        ws=[vparams['stds'][str(v)].ravel() for v in global_RVs], 
        rng=rng
    )

    # Make dict for sampling local RVs
    if 0 < len(local_RVs):
        updates_local = _make_sampling_updates(
            vs=[get_transformed(v, model) for v, _ in local_RVs.items()], 
            us=[uw[0].ravel() for _, (uw, _) in local_RVs.items()], 
            ws=[uw[1].ravel() for _, (uw, _) in local_RVs.items()], 
            rng=rng
        )
        updates.update(updates_local)

    # Replace some nodes of the graph with variational distributions
    vars = model.free_RVs
    samples = theano.clone(vars, updates)
    f = theano.function([], samples)

    # Random variables which will be sampled
    vars_sampled = [v for v in model.unobserved_RVs if not str(v).endswith('_')] \
        if hide_transformed else \
                   [v for v in model.unobserved_RVs]

    varnames = [str(var) for var in model.unobserved_RVs]
    trace = pm.sampling.NDArray(model=model, vars=vars_sampled)
    trace.setup(draws=draws, chain=0)

    # Draw samples
    for i in range(draws):
        # 'point' is like {'var1': np.array(0.1), 'var2': np.array(0.2), ...}
        point = {varname: value for varname, value in zip(varnames, f())}
        trace.record(point)

    return MultiTrace([trace])
