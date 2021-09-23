# pylint: skip-file
import os
import re
import sys
import warnings

xla_flags = os.getenv("XLA_FLAGS", "").lstrip("--")
xla_flags = re.sub(r"xla_force_host_platform_device_count=.+\s", "", xla_flags).split()
os.environ["XLA_FLAGS"] = " ".join([f"--xla_force_host_platform_device_count={100}"])

import aesara.tensor as at
import arviz as az
import jax
import numpy as np
import pandas as pd

from aesara.compile import SharedVariable
from aesara.graph.basic import Apply, Constant, clone, graph_inputs
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt import MergeOptimizer
from aesara.link.jax.dispatch import jax_funcify
from aesara.tensor.type import TensorType

from pymc3 import modelcontext
from pymc3.aesaraf import compile_rv_inplace

warnings.warn("This module is experimental.")


class NumPyroNUTS(Op):
    def __init__(
        self,
        inputs,
        outputs,
        target_accept=0.8,
        draws=1000,
        tune=1000,
        chains=4,
        seed=None,
        progress_bar=True,
    ):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.progress_bar = progress_bar
        self.seed = seed

        self.inputs, self.outputs = clone(inputs, outputs, copy_inputs=False)
        self.inputs_type = tuple(input.type for input in inputs)
        self.outputs_type = tuple(output.type for output in outputs)
        self.nin = len(inputs)
        self.nout = len(outputs)
        self.nshared = len([v for v in inputs if isinstance(v, SharedVariable)])
        self.samples_bcast = [self.chains == 1, self.draws == 1]

        self.fgraph = FunctionGraph(self.inputs, self.outputs, clone=False)
        MergeOptimizer().optimize(self.fgraph)

        super().__init__()

    def make_node(self, *inputs):

        # The samples for each variable
        outputs = [
            TensorType(v.dtype, self.samples_bcast + list(v.broadcastable))() for v in inputs
        ]

        # The leapfrog statistics
        outputs += [TensorType("int64", self.samples_bcast)()]

        all_inputs = list(inputs)
        if self.nshared > 0:
            all_inputs += self.inputs[-self.nshared :]

        return Apply(self, all_inputs, outputs)

    def do_constant_folding(self, *args):
        return False

    def perform(self, node, inputs, outputs):
        raise NotImplementedError()


@jax_funcify.register(NumPyroNUTS)
def jax_funcify_NumPyroNUTS(op, node, **kwargs):
    from numpyro.infer import MCMC, NUTS

    draws = op.draws
    tune = op.tune
    chains = op.chains
    target_accept = op.target_accept
    progress_bar = op.progress_bar
    seed = op.seed

    # Compile the "inner" log-likelihood function.  This will have extra shared
    # variable inputs as the last arguments
    logp_fn = jax_funcify(op.fgraph, **kwargs)

    if isinstance(logp_fn, (list, tuple)):
        # This handles the new JAX backend, which always returns a tuple
        logp_fn = logp_fn[0]

    def _sample(*inputs):

        if op.nshared > 0:
            current_state = inputs[: -op.nshared]
            shared_inputs = tuple(op.fgraph.inputs[-op.nshared :])
        else:
            current_state = inputs
            shared_inputs = ()

        def log_fn_wrap(x):
            res = logp_fn(
                *(
                    x
                    # We manually obtain the shared values and added them
                    # as arguments to our compiled "inner" function
                    + tuple(
                        v.get_value(borrow=True, return_internal_type=True) for v in shared_inputs
                    )
                )
            )

            if isinstance(res, (list, tuple)):
                # This handles the new JAX backend, which always returns a tuple
                res = res[0]

            return -res

        nuts_kernel = NUTS(
            potential_fn=log_fn_wrap,
            target_accept_prob=target_accept,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
        )

        pmap_numpyro = MCMC(
            nuts_kernel,
            num_warmup=tune,
            num_samples=draws,
            num_chains=chains,
            postprocess_fn=None,
            chain_method="parallel",
            progress_bar=progress_bar,
        )

        pmap_numpyro.run(seed, init_params=current_state, extra_fields=("num_steps",))
        samples = pmap_numpyro.get_samples(group_by_chain=True)
        leapfrogs_taken = pmap_numpyro.get_extra_fields(group_by_chain=True)["num_steps"]
        return tuple(samples) + (leapfrogs_taken,)

    return _sample


def sample_numpyro_nuts(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.8,
    random_seed=10,
    model=None,
    progress_bar=True,
    keep_untransformed=False,
):
    model = modelcontext(model)

    seed = jax.random.PRNGKey(random_seed)

    rv_names = [rv.name for rv in model.value_vars]
    init_state = [model.initial_point[rv_name] for rv_name in rv_names]
    init_state_batched = jax.tree_map(lambda x: np.repeat(x[None, ...], chains, axis=0), init_state)
    init_state_batched_at = [at.as_tensor(v) for v in init_state_batched]

    nuts_inputs = sorted(
        (v for v in graph_inputs([model.logpt]) if not isinstance(v, Constant)),
        key=lambda x: isinstance(x, SharedVariable),
    )
    map_seed = jax.random.split(seed, chains)
    numpyro_samples = NumPyroNUTS(
        nuts_inputs,
        [model.logpt],
        target_accept=target_accept,
        draws=draws,
        tune=tune,
        chains=chains,
        seed=map_seed,
        progress_bar=progress_bar,
    )(*init_state_batched_at)

    # Un-transform the transformed variables in JAX
    sample_outputs = []
    for i, (value_var, rv_samples) in enumerate(zip(model.value_vars, numpyro_samples[:-1])):
        rv = model.values_to_rvs[value_var]
        transform = getattr(value_var.tag, "transform", None)
        if transform is not None:
            untrans_value_var = transform.backward(rv, rv_samples)
            untrans_value_var.name = rv.name
            sample_outputs.append(untrans_value_var)

            if keep_untransformed:
                rv_samples.name = value_var.name
                sample_outputs.append(rv_samples)
        else:
            rv_samples.name = rv.name
            sample_outputs.append(rv_samples)

    print("Compiling...", file=sys.stdout)

    tic1 = pd.Timestamp.now()
    _sample = compile_rv_inplace(
        [],
        sample_outputs + [numpyro_samples[-1]],
        allow_input_downcast=True,
        on_unused_input="ignore",
        accept_inplace=True,
        mode="JAX",
    )
    tic2 = pd.Timestamp.now()

    print("Compilation time = ", tic2 - tic1, file=sys.stdout)

    print("Sampling...", file=sys.stdout)

    *mcmc_samples, leapfrogs_taken = _sample()
    tic3 = pd.Timestamp.now()

    print("Sampling time = ", tic3 - tic2, file=sys.stdout)

    posterior = {k.name: v for k, v in zip(sample_outputs, mcmc_samples)}

    az_trace = az.from_dict(posterior=posterior)

    return az_trace
