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

import subprocess
import sys
import types

import pymc

# Submodules whose entire public namespace is re-exported at the pymc root via `from pymc.<submodule> import *`.
_REEXPORTED_SUBMODULES = (
    "backends",
    "data",
    "distributions",
    "logprob",
    "model.core",
    "sampling",
    "smc",
    "step_methods",
    "tuning",
    "variational",
)

# Names imported individually into the root namespace from elsewhere. New entries here should be deliberate — adding
# cruft to the root requires updating this list.
_EXPLICIT_ROOT_NAMES = frozenset(
    {
        "compile",  # from pymc.pytensorf
        "compute_log_likelihood",  # from pymc.stats
        "do",  # from pymc.model.transform.conditioning
        "find_constrained_prior",  # from pymc.func_utils
        "model_to_graphviz",  # from pymc.model_graph
        "model_to_mermaid",  # from pymc.model_graph
        "model_to_networkx",  # from pymc.model_graph
        "observe",  # from pymc.model.transform.conditioning
    }
)


def _resolve(dotted):
    obj = pymc
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def test_reexported_submodules_define_all():
    """
    Each whitelisted submodule must declare its public surface explicitly via __all__.

    Without this guard, removing __all__ from a submodule would  silently start re-exporting every non-underscore name
    to the pymc root.
    """
    missing = [sub for sub in _REEXPORTED_SUBMODULES if not hasattr(_resolve(sub), "__all__")]
    assert not missing, f"Submodules missing __all__: {missing}"


def test_root_module_not_polluted():
    actual = {
        name
        for name in dir(pymc)
        if not name.startswith("_") and not isinstance(getattr(pymc, name), types.ModuleType)
    }

    expected = set(_EXPLICIT_ROOT_NAMES)
    for sub in _REEXPORTED_SUBMODULES:
        expected |= set(_resolve(sub).__all__)

    unexpected = actual - expected
    missing = expected - actual
    hint = (
        "If a name is intentional, add it to _EXPLICIT_ROOT_NAMES or to the appropriate submodule's __all__. Otherwise,"
        " remove the import from pymc/__init__.py."
    )
    assert not unexpected, f"Unexpected names at pymc root: {sorted(unexpected)}. {hint}"
    assert not missing, f"Missing names at pymc root: {sorted(missing)}. {hint}"


def test_eager_import_heavy_dependencies():
    """`import pymc` must not eagerly load heavy optional dependencies.

    These modules are deferred to first use to keep ``import pymc`` fast.
    Note: ``scipy.sparse`` is intentionally omitted because ``xarray`` pulls it
    in transitively, which is outside of pymc's control.
    """
    expensive_modules = (
        "arviz_base",
        "arviz_plots",
        "arviz_stats",
        "pytensor.sparse",
        "scipy.cluster",
        "scipy.interpolate",
        "scipy.linalg",
        "scipy.optimize",
        "scipy.spatial",
        "scipy.special",
        "scipy.stats",
        "zarr",
    )

    code = (
        "import sys\n"
        "import pymc  # noqa: F401\n"
        f"loaded = [m for m in {expensive_modules!r} if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = [m for m in result.stdout.strip().split(",") if m]
    assert not loaded, (
        f"`import pymc` eagerly loaded heavy modules: {loaded}. "
        "These should be deferred to first use."
    )
