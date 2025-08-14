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
import warnings

from arviz import InferenceData, dict_to_dataset

from pymc.backends.arviz import coords_and_dims_for_inferencedata, find_constants, find_observations
from pymc.sampling.external.base import NUTSExternalSampler
from pymc.stats.convergence import log_warnings, run_convergence_checks
from pymc.util import _get_seeds_per_chain


class Nutpie(NUTSExternalSampler):
    def __init__(
        self,
        model=None,
        backend="numba",
        gradient_backend="pytensor",
        compile_kwargs=None,
        sample_kwargs=None,
    ):
        super().__init__(model)
        self.backend = backend
        self.gradient_backend = gradient_backend
        self.compile_kwargs = compile_kwargs or {}
        self.sample_kwargs = sample_kwargs or {}
        self.compiled_model = None

    def sample(
        self,
        *,
        tune,
        draws,
        chains,
        initvals,
        random_seed,
        progressbar,
        var_names,
        idata_kwargs,
        compute_convergence_checks,
        **kwargs,
    ):
        try:
            import nutpie
        except ImportError as err:
            raise ImportError(
                "nutpie not found. Install it with conda install -c conda-forge nutpie"
            ) from err

        from nutpie.sample import _BackgroundSampler

        if initvals:
            warnings.warn(
                "initvals are currently ignored by the nutpie sampler.",
                UserWarning,
            )
        if idata_kwargs:
            warnings.warn(
                "idata_kwargs are currently ignored by the nutpie sampler.",
                UserWarning,
            )

        compiled_model = nutpie.compile_pymc_model(
            self.model,
            var_names=var_names,
            backend=self.backend,
            gradient_backend=self.gradient_backend,
            **self.compile_kwargs,
        )

        result = nutpie.sample(
            compiled_model,
            tune=tune,
            draws=draws,
            chains=chains,
            seed=_get_seeds_per_chain(random_seed, 1)[0],
            progress_bar=progressbar,
            **self.sample_kwargs,
            **kwargs,
        )
        if isinstance(result, _BackgroundSampler):
            # Wrap _BackgroundSampler so that when sampling is finished we run post_process_sampler
            class NutpieBackgroundSamplerWrapper(_BackgroundSampler):
                def __init__(self, *args, pymc_model, compute_convergence_checks, **kwargs):
                    self.pymc_model = pymc_model
                    self.compute_convergence_checks = compute_convergence_checks
                    super().__init__(*args, **kwargs, return_raw_trace=False)

                def _extract(self, *args, **kwargs):
                    idata = super()._extract(*args, **kwargs)
                    return Nutpie._post_process_sample(
                        model=self.pymc_model,
                        idata=idata,
                        compute_convergence_checks=self.compute_convergence_checks,
                    )

            # non-blocked sampling
            return NutpieBackgroundSamplerWrapper(
                result,
                pymc_model=self.model,
                compute_convergence_checks=compute_convergence_checks,
            )
        else:
            return self._post_process_sample(self.model, result, compute_convergence_checks)

    @staticmethod
    def _post_process_sample(
        model, idata: InferenceData, compute_convergence_checks
    ) -> InferenceData:
        # Temporary work-around. Revert once https://github.com/pymc-devs/nutpie/issues/74 is fixed
        # gather observed and constant data as nutpie.sample() has no access to the PyMC model
        if compute_convergence_checks:
            log_warnings(run_convergence_checks(idata, model))

        coords, dims = coords_and_dims_for_inferencedata(model)
        constant_data = dict_to_dataset(
            find_constants(model),
            library=idata.attrs.get("library", None),
            coords=coords,
            dims=dims,
            default_dims=[],
        )
        observed_data = dict_to_dataset(
            find_observations(model),
            library=idata.attrs.get("library", None),
            coords=coords,
            dims=dims,
            default_dims=[],
        )
        idata.add_groups(
            {"constant_data": constant_data, "observed_data": observed_data},
            coords=coords,
            dims=dims,
        )
        return idata
