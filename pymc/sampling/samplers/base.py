#   Copyright 2026 - present The PyMC Developers
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
import importlib.util

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from pymc.initial_point import StartDict
from pymc.model.core import Model
from pymc.util import RandomState
from pymc.vartypes import discrete_types


class Sampler(ABC):
    """Base for everything ``pm.sample(sampler=...)`` accepts.

    The constructor is algorithm configuration only: nothing model-bound,
    nothing compiled. The model and the run configuration arrive at sample
    time, through :meth:`sample_from_init`.

    ``sample_from_init`` takes exactly the run-level arguments of
    ``pm.sample`` (minus ``sampler``/``backend``; enforced by a test), so the
    funnel is sugar over the sampler rather than a second code path.
    Sampler-specific arguments (algorithm parameters, initialization strategy,
    adaptation options, ...) belong to the subclass constructor and must not
    be routed through ``pm.sample``.
    """

    @abstractmethod
    def sample_from_init(
        self,
        *,
        model: Model | None = None,
        draws: int = 1000,
        tune: int | None = 1000,
        chains: int | None = None,
        cores: int | None = None,
        initvals: StartDict | Sequence[StartDict | None] | None = None,
        random_seed: RandomState = None,
        progressbar: bool = True,
        quiet: bool = False,
        discard_tuned_samples: bool = True,
        keep_warning_stat: bool = False,
        var_names: Sequence[str] | None = None,
        idata_kwargs: dict[str, Any] | None = None,
        compute_convergence_checks: bool = True,
        compile_kwargs: dict[str, Any] | None = None,
    ):
        """Sample the model with this sampler's configuration.

        Deliberately not named ``sample`` so it cannot be confused with the
        flat per-algorithm entry points (e.g. ``pm.nutpie.nuts.sample``),
        which combine algorithm and run arguments in a single call.

        Every argument is run configuration that any sampler understands;
        subclasses that cannot honor one (e.g. ``compile_kwargs`` for a
        jax-only sampler) must warn or raise rather than silently ignore it.
        """
        # Deliberately no **kwargs: the run argument set is a closed
        # contract. If `pm.sample` grows a new forwarded argument, every
        # sampler must explicitly decide its disposition.


class ExternalSampler(Sampler):
    """Shared implementation logic for samplers backed by other libraries.

    This is implementation inheritance, not interface: it adds nothing to the
    :class:`Sampler` contract, only conveniences such as the dependency check.
    """

    package: str
    version_range: str | None = None

    def __init__(self):
        require_installed(self.package, self.version_range)


class SamplerEntry:
    """Both user surfaces of one sampler in a single object.

    Calling the entry configures a sampler for the ``pm.sample`` funnel::

        pm.sample(sampler=pm.nutpie.nuts(target_accept=0.9))

    Calling ``.sample`` is the flat entry point — algorithm and run arguments
    in one call::

        pm.nutpie.nuts.sample(model=model, target_accept=0.9, draws=1000)

    The flat entry point is defined in terms of the first, so the two
    surfaces cannot disagree.
    """

    def __init__(self, name: str, factory: Callable[..., Sampler], doc: str | None = None):
        self._name = name
        self._factory = factory
        self.__doc__ = doc if doc is not None else factory.__doc__

    def __repr__(self) -> str:
        return f"<pymc.{self._name}>"

    def __call__(self, **algorithm_kwargs) -> Sampler:
        return self._factory(**algorithm_kwargs)

    def sample(
        self,
        *,
        model: Model | None = None,
        draws: int = 1000,
        tune: int | None = 1000,
        chains: int | None = None,
        cores: int | None = None,
        initvals: StartDict | Sequence[StartDict | None] | None = None,
        random_seed: RandomState = None,
        progressbar: bool = True,
        quiet: bool = False,
        discard_tuned_samples: bool = True,
        keep_warning_stat: bool = False,
        var_names: Sequence[str] | None = None,
        idata_kwargs: dict[str, Any] | None = None,
        compute_convergence_checks: bool = True,
        backend: str | None = None,
        compile_kwargs: dict[str, Any] | None = None,
        **algorithm_kwargs,
    ):
        """Configure and run the sampler in one flat call.

        Keyword arguments beyond the shared run configuration are passed to
        the sampler's constructor. ``backend`` is resolved into
        ``compile_kwargs["mode"]``, exactly as in ``pm.sample``.
        """
        from pymc.pytensorf import resolve_backend_compile_kwargs

        compile_kwargs = resolve_backend_compile_kwargs(backend, compile_kwargs)
        return self(**algorithm_kwargs).sample_from_init(
            model=model,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            initvals=initvals,
            random_seed=random_seed,
            progressbar=progressbar,
            quiet=quiet,
            discard_tuned_samples=discard_tuned_samples,
            keep_warning_stat=keep_warning_stat,
            var_names=var_names,
            idata_kwargs=idata_kwargs,
            compute_convergence_checks=compute_convergence_checks,
            compile_kwargs=compile_kwargs,
        )


def require_installed(package: str, version_range: str | None = None) -> None:
    """Raise with an install hint if ``package`` is missing or too old/new."""
    if importlib.util.find_spec(package) is None:
        raise ImportError(
            f"This sampler requires {package}. Install it with `pip install {package}`."
        )
    if version_range is not None:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        version = getattr(importlib.import_module(package), "__version__", None)
        if version is not None and Version(version) not in SpecifierSet(version_range):
            raise ImportError(
                f"This sampler requires {package}{version_range}, found {version}. "
                f"Install a compatible version with `pip install '{package}{version_range}'`."
            )


def require_continuous_model(model: Model, *, sampler_name: str) -> None:
    """Raise if the model has discrete free random variables."""
    if any(var.dtype in discrete_types for var in model.free_RVs):
        raise ValueError(
            f"The {sampler_name} sampler can only sample models "
            "where all free variables are continuous."
        )
