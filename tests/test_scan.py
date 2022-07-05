import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara import Mode
from aesara.raise_op import assert_op
from aesara.scan.utils import ScanArgs

from aeppl.joint_logprob import factorized_joint_logprob, joint_logprob
from aeppl.logprob import logprob
from aeppl.scan import construct_scan, convert_outer_out_to_in, get_random_outer_outputs
from tests.utils import assert_no_rvs


def create_inner_out_logp(value_map):
    """Create a log-likelihood inner-output.

    This is intended to be use with `get_random_outer_outputs`.

    """
    res = []
    for old_inner_out_var, new_inner_in_var in value_map.items():
        logp = logprob(old_inner_out_var, new_inner_in_var)
        if new_inner_in_var.name:
            logp.name = "logp({})".format(new_inner_in_var.name)
        res.append(logp)

    return res


def test_convert_outer_out_to_in_sit_sot():
    """Test a single replacement with `convert_outer_out_to_in`.

    This should be a single SIT-SOT replacement.
    """

    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))
    rng_tt = aesara.shared(rng_state, name="rng", borrow=True)
    rng_tt.tag.is_rng = True
    rng_tt.default_update = rng_tt

    #
    # We create a `Scan` representing a time-series model with normally
    # distributed responses that are dependent on lagged values of both the
    # response `RandomVariable` and a lagged "deterministic" that also depends
    # on the lagged response values.
    #
    def input_step_fn(mu_tm1, y_tm1, rng):
        mu_tm1.name = "mu_tm1"
        y_tm1.name = "y_tm1"
        mu = mu_tm1 + y_tm1 + 1
        mu.name = "mu_t"
        return mu, at.random.normal(mu, 1.0, rng=rng, name="Y_t")

    (mu_tt, Y_rv), _ = aesara.scan(
        fn=input_step_fn,
        outputs_info=[
            {
                "initial": at.as_tensor_variable(0.0, dtype=aesara.config.floatX),
                "taps": [-1],
            },
            {
                "initial": at.as_tensor_variable(0.0, dtype=aesara.config.floatX),
                "taps": [-1],
            },
        ],
        non_sequences=[rng_tt],
        n_steps=10,
    )

    mu_tt.name = "mu_tt"
    mu_tt.owner.inputs[0].name = "mu_all"
    Y_rv.name = "Y_rv"
    Y_all = Y_rv.owner.inputs[0]
    Y_all.name = "Y_all"

    input_scan_args = ScanArgs.from_node(Y_rv.owner.inputs[0].owner)

    # TODO FIXME: Everything below needs to be replaced with explicit asserts
    # on the values in `input_scan_args`

    #
    # Sample from the model and create another `Scan` that computes the
    # log-likelihood of the model at the sampled point.
    #
    Y_obs = at.as_tensor_variable(Y_rv.eval())
    Y_obs.name = "Y_obs"

    def output_step_fn(y_t, y_tm1, mu_tm1):
        mu_tm1.name = "mu_tm1"
        y_tm1.name = "y_tm1"
        mu = mu_tm1 + y_tm1 + 1
        mu.name = "mu_t"
        logp = logprob(at.random.normal(mu, 1.0), y_t)
        logp.name = "logp"
        return mu, logp

    (mu_tt, Y_logp), _ = aesara.scan(
        fn=output_step_fn,
        sequences=[{"input": Y_obs, "taps": [0, -1]}],
        outputs_info=[
            {
                "initial": at.as_tensor_variable(0.0, dtype=aesara.config.floatX),
                "taps": [-1],
            },
            {},
        ],
    )

    Y_logp.name = "Y_logp"
    mu_tt.name = "mu_tt"

    #
    # Get the model output variable that corresponds to the response
    # `RandomVariable`
    #
    oo_idx, oo_var, io_var = get_random_outer_outputs(input_scan_args)[0]

    #
    # Convert the original model `Scan` into another `Scan` that's equivalent
    # to the log-likelihood `Scan` given above.
    # In other words, automatically construct the log-likelihood `Scan` based
    # on the model `Scan`.
    #
    value_map = {Y_all: Y_obs}
    test_scan_args = convert_outer_out_to_in(
        input_scan_args,
        [oo_var],
        value_map,
        inner_out_fn=create_inner_out_logp,
    )

    scan_out, updates = construct_scan(test_scan_args)

    #
    # Evaluate the manually and automatically constructed log-likelihoods and
    # compare.
    #
    res = scan_out[oo_idx].eval()
    exp_res = Y_logp.eval()

    assert np.array_equal(res, exp_res)


def test_convert_outer_out_to_in_mit_sot():
    """Test a single replacement with `convert_outer_out_to_in`.

    This should be a single MIT-SOT replacement.
    """

    rng_state = np.random.default_rng(1234)
    rng_tt = aesara.shared(rng_state, name="rng", borrow=True)
    rng_tt.tag.is_rng = True
    rng_tt.default_update = rng_tt

    #
    # This is a very simple model with only one output, but multiple
    # taps/lags.
    #
    def input_step_fn(y_tm1, y_tm2, rng):
        y_tm1.name = "y_tm1"
        y_tm2.name = "y_tm2"
        return at.random.normal(y_tm1 + y_tm2, 1.0, rng=rng, name="Y_t")

    Y_rv, _ = aesara.scan(
        fn=input_step_fn,
        outputs_info=[
            {"initial": at.as_tensor_variable(np.r_[-1.0, 0.0]), "taps": [-1, -2]},
        ],
        non_sequences=[rng_tt],
        n_steps=10,
    )

    Y_rv.name = "Y_rv"
    Y_all = Y_rv.owner.inputs[0]
    Y_all.name = "Y_all"

    Y_obs = at.as_tensor_variable(Y_rv.eval())
    Y_obs.name = "Y_obs"

    input_scan_args = ScanArgs.from_node(Y_rv.owner.inputs[0].owner)

    # TODO FIXME: Everything below needs to be replaced with explicit asserts
    # on the values in `input_scan_args`

    #
    # The corresponding log-likelihood
    #
    def output_step_fn(y_t, y_tm1, y_tm2):
        y_t.name = "y_t"
        y_tm1.name = "y_tm1"
        y_tm2.name = "y_tm2"
        logp = logprob(at.random.normal(y_tm1 + y_tm2, 1.0), y_t)
        logp.name = "logp(y_t)"
        return logp

    Y_logp, _ = aesara.scan(
        fn=output_step_fn,
        sequences=[{"input": Y_obs, "taps": [0, -1, -2]}],
        outputs_info=[{}],
    )

    #
    # Get the model output variable that corresponds to the response
    # `RandomVariable`
    #
    oo_idx, oo_var, io_var = get_random_outer_outputs(input_scan_args)[0]

    #
    # Convert the original model `Scan` into another `Scan` that's equivalent
    # to the log-likelihood `Scan` given above.
    # In other words, automatically construct the log-likelihood `Scan` based
    # on the model `Scan`.

    value_map = {Y_all: Y_obs}
    test_scan_args = convert_outer_out_to_in(
        input_scan_args,
        [oo_var],
        value_map,
        inner_out_fn=create_inner_out_logp,
    )

    scan_out, updates = construct_scan(test_scan_args)

    #
    # Evaluate the manually and automatically constructed log-likelihoods and
    # compare.
    #
    res = scan_out[oo_idx].eval()
    exp_res = Y_logp.eval()

    assert np.array_equal(res, exp_res)


@pytest.mark.parametrize(
    "require_inner_rewrites",
    [
        False,
        True,
    ],
)
def test_scan_joint_logprob(require_inner_rewrites):
    srng = at.random.RandomStream()

    N_tt = at.iscalar("N")
    N_val = 10
    N_tt.tag.test_value = N_val

    M_tt = at.iscalar("M")
    M_val = 2
    M_tt.tag.test_value = M_val

    mus_tt = at.matrix("mus_t")

    mus_val = np.stack([np.arange(0.0, 10), np.arange(0.0, -10, -1)], axis=-1).astype(
        aesara.config.floatX
    )
    mus_tt.tag.test_value = mus_val

    sigmas_tt = at.ones((N_tt,))
    Gamma_rv = srng.dirichlet(at.ones((M_tt, M_tt)), name="Gamma")

    Gamma_vv = Gamma_rv.clone()
    Gamma_vv.name = "Gamma_vv"

    Gamma_val = np.array([[0.5, 0.5], [0.5, 0.5]])
    Gamma_rv.tag.test_value = Gamma_val

    def scan_fn(mus_t, sigma_t, Gamma_t):
        S_t = srng.categorical(Gamma_t[0], name="S_t")

        if require_inner_rewrites:
            Y_t = srng.normal(mus_t, sigma_t, name="Y_t")[S_t]
        else:
            Y_t = srng.normal(mus_t[S_t], sigma_t, name="Y_t")

        return Y_t, S_t

    (Y_rv, S_rv), _ = aesara.scan(
        fn=scan_fn,
        sequences=[mus_tt, sigmas_tt],
        non_sequences=[Gamma_rv],
        outputs_info=[{}, {}],
        strict=True,
        name="scan_rv",
    )
    Y_rv.name = "Y"
    S_rv.name = "S"

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    s_vv = S_rv.clone()
    s_vv.name = "s"

    y_logp = joint_logprob({Y_rv: y_vv, S_rv: s_vv, Gamma_rv: Gamma_vv})

    y_val = np.arange(10)
    s_val = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 1])

    test_point = {
        y_vv: y_val,
        s_vv: s_val,
        M_tt: M_val,
        N_tt: N_val,
        mus_tt: mus_val,
        Gamma_vv: Gamma_val,
    }

    y_logp_fn = aesara.function(list(test_point.keys()), y_logp)

    assert_no_rvs(y_logp_fn.maker.fgraph.outputs[0])

    # Construct the joint log-probability by hand so we can compare it with
    # `y_logp`
    def scan_fn(mus_t, sigma_t, Y_t_val, S_t_val, Gamma_t):
        S_t = at.random.categorical(Gamma_t[0], name="S_t")
        Y_t = at.random.normal(mus_t[S_t_val], sigma_t, name="Y_t")
        Y_t_logp, S_t_logp = logprob(Y_t, Y_t_val), logprob(S_t, S_t_val)
        Y_t_logp.name = "log(Y_t=y_t)"
        S_t_logp.name = "log(S_t=s_t)"
        return Y_t_logp, S_t_logp

    (Y_rv_logp, S_rv_logp), _ = aesara.scan(
        fn=scan_fn,
        sequences=[mus_tt, sigmas_tt, y_vv, s_vv],
        non_sequences=[Gamma_vv],
        outputs_info=[{}, {}],
        strict=True,
        name="scan_rv",
    )
    Y_rv_logp.name = "logp(Y=y)"
    S_rv_logp.name = "logp(S=s)"

    Gamma_logp = logprob(Gamma_rv, Gamma_vv)

    y_logp_ref = Y_rv_logp.sum() + S_rv_logp.sum() + Gamma_logp.sum()

    assert_no_rvs(y_logp_ref)

    y_logp_val = y_logp.eval(test_point)

    y_logp_ref_val = y_logp_ref.eval(test_point)

    assert np.allclose(y_logp_val, y_logp_ref_val)


@aesara.config.change_flags(compute_test_value="raise")
@pytest.mark.xfail(reason="see #148")
def test_initial_values():
    srng = at.random.RandomStream(seed=2320)

    p_S_0 = np.array([0.9, 0.1])
    S_0_rv = srng.categorical(p_S_0, name="S_0")
    S_0_rv.tag.test_value = 0

    Gamma_at = at.matrix("Gamma")
    Gamma_at.tag.test_value = np.array([[0, 1], [1, 0]])

    s_0_vv = S_0_rv.clone()
    s_0_vv.name = "s_0"

    def step_fn(S_tm1, Gamma):
        S_t = srng.categorical(Gamma[S_tm1], name="S_t")
        return S_t

    S_1T_rv, _ = aesara.scan(
        fn=step_fn,
        outputs_info=[{"initial": S_0_rv, "taps": [-1]}],
        non_sequences=[Gamma_at],
        strict=True,
        n_steps=10,
        name="S_0T",
    )

    S_1T_rv.name = "S_1T"
    s_1T_vv = S_1T_rv.clone()
    s_1T_vv.name = "s_1T"

    logp_parts = factorized_joint_logprob({S_1T_rv: s_1T_vv, S_0_rv: s_0_vv})

    s_0_val = 0
    s_1T_val = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    Gamma_val = np.array([[0.1, 0.9], [0.9, 0.1]])

    exp_res = np.log(p_S_0[s_0_val])
    s_prev = s_0_val
    for s in s_1T_val:
        exp_res += np.log(Gamma_val[s_prev, s])
        s_prev = s

    S_0T_logp = sum(v.sum() for v in logp_parts.values())
    S_0T_logp_fn = aesara.function([s_0_vv, s_1T_vv, Gamma_at], S_0T_logp)
    res = S_0T_logp_fn(s_0_val, s_1T_val, Gamma_val)

    assert res == pytest.approx(exp_res)


@pytest.mark.parametrize("remove_asserts", (True, False))
def test_mode_is_kept(remove_asserts):
    mode = Mode().including("local_remove_all_assert") if remove_asserts else None
    x, _ = aesara.scan(
        fn=lambda x: at.random.normal(assert_op(x, x > 0)),
        outputs_info=[at.ones(())],
        n_steps=10,
        mode=mode,
    )
    x.name = "x"
    x_vv = x.clone()
    x_logp = aesara.function([x_vv], joint_logprob({x: x_vv}))

    x_test_val = np.full((10,), -1)
    if remove_asserts:
        assert x_logp(x=x_test_val)
    else:
        with pytest.raises(AssertionError):
            x_logp(x=x_test_val)
