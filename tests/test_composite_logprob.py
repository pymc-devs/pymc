import aesara
import aesara.tensor as at
import numpy as np
import scipy.stats as st

from aeppl import joint_logprob
from aeppl.opt import construct_ir_fgraph
from aeppl.truncation import CensoredRV


def test_scalar_clipped_mixture():
    x = at.clip(at.random.normal(loc=1), 0.5, 1.5)
    x.name = "x"
    y = at.random.beta(1, 2, name="y")

    comps = at.stack(x, y)
    comps.name = "comps"
    idxs = at.random.bernoulli(0.4, name="idxs")
    mix = comps[idxs]
    mix.name = "mix"

    mix_vv = mix.clone()
    mix_vv.name = "mix_val"
    idxs_vv = idxs.clone()
    idxs_vv.name = "idxs_val"

    logp = joint_logprob({idxs: idxs_vv, mix: mix_vv})

    logp_fn = aesara.function([idxs_vv, mix_vv], logp)
    assert logp_fn(0, 0.4) == -np.inf
    assert np.isclose(logp_fn(0, 0.5), st.norm.logcdf(0.5, 1) + np.log(0.6))
    assert np.isclose(logp_fn(0, 1.3), st.norm.logpdf(1.3, 1) + np.log(0.6))
    assert np.isclose(logp_fn(1, 0.4), st.beta.logpdf(0.4, 1, 2) + np.log(0.4))


def test_nested_scalar_mixtures():
    x = at.random.normal(loc=-50, name="x")
    y = at.random.normal(loc=50, name="y")
    comps1 = at.stack(x, y)
    comps1.name = "comps1"
    idxs1 = at.random.bernoulli(0.5, name="idxs1")
    mix1 = comps1[idxs1]
    mix1.name = "mix1"

    w = at.random.normal(loc=-100, name="w")
    z = at.random.normal(loc=100, name="z")
    comps2 = at.stack(w, z)
    comps2.name = "comps2"
    idxs2 = at.random.bernoulli(0.5, name="idxs2")
    mix2 = comps2[idxs2]
    mix2.name = "mix2"

    comps12 = at.stack(mix1, mix2)
    comps12.name = "comps12"
    idxs12 = at.random.bernoulli(0.5, name="idxs12")
    mix12 = comps12[idxs12]
    mix12.name = "mix12"

    idxs1_vv = idxs1.clone()
    idxs2_vv = idxs2.clone()
    idxs12_vv = idxs12.clone()
    mix12_vv = mix12.clone()

    logp = joint_logprob(
        {idxs1: idxs1_vv, idxs2: idxs2_vv, idxs12: idxs12_vv, mix12: mix12_vv}
    )
    logp_fn = aesara.function([idxs1_vv, idxs2_vv, idxs12_vv, mix12_vv], logp)

    expected_mu_logpdf = st.norm.logpdf(0) + np.log(0.5) * 3
    assert np.isclose(logp_fn(0, 0, 0, -50), expected_mu_logpdf)
    assert np.isclose(logp_fn(0, 1, 0, -50), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 0, 0, 50), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 1, 0, 50), expected_mu_logpdf)
    assert np.isclose(logp_fn(0, 0, 1, -100), expected_mu_logpdf)
    assert np.isclose(logp_fn(0, 1, 1, 100), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 0, 1, -100), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 1, 1, 100), expected_mu_logpdf)

    assert np.isclose(logp_fn(0, 0, 0, 50), st.norm.logpdf(100) + np.log(0.5) * 3)
    assert np.isclose(logp_fn(0, 0, 1, 50), st.norm.logpdf(150) + np.log(0.5) * 3)


def test_unvalued_ir_reversion():
    """Make sure that un-valued IR rewrites are reverted."""
    x_rv = at.random.normal()
    y_rv = at.clip(x_rv, 0, 1)
    z_rv = at.random.normal(y_rv, 1, name="z")
    z_vv = z_rv.clone()

    # Only the `z_rv` is "valued", so `y_rv` doesn't need to be converted into
    # measurable IR.
    rv_values = {z_rv: z_vv}

    z_fgraph, _, memo = construct_ir_fgraph(rv_values)

    assert memo[y_rv] in z_fgraph.preserve_rv_mappings.measurable_conversions

    measurable_y_rv = z_fgraph.preserve_rv_mappings.measurable_conversions[memo[y_rv]]
    assert isinstance(measurable_y_rv.owner.op, CensoredRV)

    # `construct_ir_fgraph` should've reverted the un-valued measurable IR
    # change
    assert measurable_y_rv not in z_fgraph
