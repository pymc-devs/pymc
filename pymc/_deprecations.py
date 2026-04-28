#   Copyright 2024 - present The PyMC Developers
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

import importlib
import warnings

# Symbols hand-relocated to a specific submodule during the namespace cleanup.
_RELOCATED: dict[str, str] = {
    # blocking
    "DictToArrayBijection": "pymc.blocking",
    # exceptions
    "BlockModelAccessError": "pymc.exceptions",
    "DtypeError": "pymc.exceptions",
    "ImputationWarning": "pymc.exceptions",
    "IncorrectArgumentsError": "pymc.exceptions",
    "NotConstantValueError": "pymc.exceptions",
    "SamplingError": "pymc.exceptions",
    "ShapeError": "pymc.exceptions",
    "ShapeWarning": "pymc.exceptions",
    "TraceDirectoryError": "pymc.exceptions",
    "TruncationError": "pymc.exceptions",
    "UndefinedMomentException": "pymc.exceptions",
    # math
    "expand_packed_triangular": "pymc.math",
    "invlogit": "pymc.math",
    "invprobit": "pymc.math",
    "logaddexp": "pymc.math",
    "logit": "pymc.math",
    "logsumexp": "pymc.math",
    "probit": "pymc.math",
    # printing
    "str_for_data_var": "pymc.printing",
    "str_for_dist": "pymc.printing",
    "str_for_model": "pymc.printing",
    "str_for_potential_or_deterministic": "pymc.printing",
    # pytensorf
    "CallableTensor": "pymc.pytensorf",
    "cont_inputs": "pymc.pytensorf",
    "convert_data": "pymc.pytensorf",
    "convert_observed_data": "pymc.pytensorf",
    "floatX": "pymc.pytensorf",
    "gradient": "pymc.pytensorf",
    "hessian": "pymc.pytensorf",
    "hessian_diag": "pymc.pytensorf",
    "inputvars": "pymc.pytensorf",
    "intX": "pymc.pytensorf",
    "jacobian": "pymc.pytensorf",
    "join_nonshared_inputs": "pymc.pytensorf",
    "make_shared_replacements": "pymc.pytensorf",
    # stats
    "compute_log_prior": "pymc.stats",
    # step_methods
    "BlockedStep": "pymc.step_methods",
    "CauchyProposal": "pymc.step_methods",
    "CompoundStep": "pymc.step_methods",
    "LaplaceProposal": "pymc.step_methods",
    "MultivariateNormalProposal": "pymc.step_methods",
    "NormalProposal": "pymc.step_methods",
    "PoissonProposal": "pymc.step_methods",
    "STEP_METHODS": "pymc.step_methods",
    "UniformProposal": "pymc.step_methods",
    # tuning
    "guess_scaling": "pymc.tuning",
    "trace_cov": "pymc.tuning",
    # util
    "drop_warning_stat": "pymc.util",
    # variational
    "Approximation": "pymc.variational",
    "ImplicitGradient": "pymc.variational",
    "Inference": "pymc.variational",
    "KLqp": "pymc.variational",
    "Stein": "pymc.variational",
    # vartypes
    "bool_types": "pymc.vartypes",
    "complex_types": "pymc.vartypes",
    "continuous_types": "pymc.vartypes",
    "discrete_types": "pymc.vartypes",
    "float_types": "pymc.vartypes",
    "int_types": "pymc.vartypes",
    "isgenerator": "pymc.vartypes",
    "typefilter": "pymc.vartypes",
}

# Bulk-relocated namespaces. A miss in :data:`_RELOCATED` falls back to looking
# the name up in each of these submodules, which covers the dozens of
# ``arviz_plots`` / ``arviz_stats`` functions that used to leak into the root.
_FALLBACK_SUBMODULES: tuple[str, ...] = ("pymc.plots", "pymc.stats")


def _warn(name: str, new_path: str) -> None:
    warnings.warn(
        f"`pymc.{name}` was moved out of the root namespace and will be removed in "
        f"the first PyMC release of 2027. Use `{new_path}` instead.",
        DeprecationWarning,
        stacklevel=4,
    )


def resolve(name: str):
    """Resolve a removed root-namespace symbol or raise ``AttributeError``."""
    target_module = _RELOCATED.get(name)
    if target_module is not None:
        module = importlib.import_module(target_module)
        _warn(name, f"{target_module}.{name}")
        return getattr(module, name)

    for submodule_path in _FALLBACK_SUBMODULES:
        module = importlib.import_module(submodule_path)
        if hasattr(module, name):
            _warn(name, f"{submodule_path}.{name}")
            return getattr(module, name)

    raise AttributeError(f"module 'pymc' has no attribute {name!r}")
