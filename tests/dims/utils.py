#   Copyright 2025 - present The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from pytensor import graph_replace
from pytensor.graph import rewrite_graph
from pytensor.printing import debugprint
from pytensor.xtensor import as_xtensor

from pymc import Model
from pymc.testing import equal_computations_up_to_root


def assert_equivalent_random_graph(model: Model, reference_model: Model) -> bool:
    """Check if the random graph of a model with xtensor variables is equivalent."""
    lowered_model = rewrite_graph(
        [var.values for var in model.basic_RVs + model.deterministics + model.potentials],
        include=(
            "lower_xtensor",
            "inline_ofg_expansion_xtensor",
            "canonicalize",
            "local_remove_all_assert",
        ),
    )
    reference_lowered_model = rewrite_graph(
        reference_model.basic_RVs + reference_model.deterministics + reference_model.potentials,
        include=(
            "inline_ofg_expansion",
            "canonicalize",
            "local_remove_all_assert",
        ),
    )
    assert equal_computations_up_to_root(
        lowered_model,
        reference_lowered_model,
        ignore_rng_values=True,
    ), debugprint(lowered_model + reference_lowered_model, print_type=True)


def assert_equivalent_logp_graph(model: Model, reference_model: Model) -> bool:
    """Check if the logp graph of a model with xtensor variables is equivalent."""
    # Replace xtensor value variables by tensor value variables
    replacements = {
        var: as_xtensor(var.values.clone(name=var.name), dims=var.dims) for var in model.value_vars
    }
    model_logp = graph_replace(model.logp(), replacements)
    lowered_model_logp = rewrite_graph(
        [model_logp],
        include=("lower_xtensor", "canonicalize", "local_remove_all_assert"),
    )
    reference_lowered_model_logp = rewrite_graph(
        [reference_model.logp()],
        include=("canonicalize", "local_remove_all_assert"),
    )
    assert equal_computations_up_to_root(
        lowered_model_logp,
        reference_lowered_model_logp,
        ignore_rng_values=False,
    ), debugprint(
        lowered_model_logp + reference_lowered_model_logp,
        print_type=True,
    )
