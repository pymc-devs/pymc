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

import logging
import warnings

from typing import Literal

from arviz import InferenceData

from pymc.backends.arviz import coords_and_dims_for_inferencedata, find_constants, find_observations
from pymc.model import Model
from pymc.step_methods.compound import Competence
from pymc.step_methods.external.base import ExternalSampler
from pymc.vartypes import continuous_types

logger = logging.getLogger("pymc")

try:
    import nutpie

    # Check if it's actually installed and not just an empty mock module
    NUTPIE_AVAILABLE = hasattr(nutpie, "compile_pymc_model")
except ImportError:
    NUTPIE_AVAILABLE = False


class NutPie(ExternalSampler):
    """NutPie No-U-Turn Sampler.

    This class provides an interface to the NutPie sampler, which is a high-performance
    implementation of the No-U-Turn Sampler (NUTS). Unlike PyMC's native NUTS implementation,
    NutPie samples chains sequentially in a single CPU, which can be more efficient for some
    models.

    Parameters
    ----------
    vars : list, optional
        Variables to be sampled
    model : Model, optional
        PyMC model
    backend : {"numba", "jax"}, default="numba"
        Which backend to use for computation
    target_accept : float, default=0.8
        Target acceptance rate for step size adaptation
    max_treedepth : int, default=10
        Maximum tree depth for NUTS (passed as 'maxdepth' to NutPie)
    **kwargs
        Additional parameters passed to nutpie.sample()

    Notes
    -----
    Requires the nutpie package to be installed:
    pip install nutpie
    """

    name = "nutpie"

    def __init__(
        self,
        vars=None,
        *,
        model=None,
        backend: Literal["numba", "jax"] = "numba",
        target_accept: float = 0.8,
        max_treedepth: int = 10,
        **kwargs,
    ):
        """Initialize NutPie sampler."""
        if not NUTPIE_AVAILABLE:
            raise ImportError("nutpie not found. Install it with: pip install nutpie")

        super().__init__(vars=vars, model=model)

        self.backend = backend
        self.target_accept = target_accept
        self.max_treedepth = max_treedepth
        self.nutpie_kwargs = kwargs

    def sample(
        self,
        draws: int,
        tune: int = 1000,
        chains: int = 4,
        random_seed=None,
        initvals=None,
        progressbar=True,
        cores=None,
        idata_kwargs=None,
        compute_convergence_checks=True,
        **kwargs,
    ) -> InferenceData:
        """Run NutPie sampler and return results as InferenceData.

        Parameters
        ----------
        draws : int
            Number of draws per chain
        tune : int
            Number of tuning draws per chain
        chains : int
            Number of chains to sample
        random_seed : int or sequence, optional
            Random seed(s) for reproducibility
        initvals : dict or list of dict, optional
            Initial values for variables (currently not used by NutPie)
        progressbar : bool
            Whether to display progress bar
        cores : int, optional
            Number of CPU cores to use (ignored by NutPie)
        idata_kwargs : dict, optional
            Additional arguments for arviz.InferenceData conversion
        compute_convergence_checks : bool
            Whether to compute convergence diagnostics
        **kwargs
            Additional sampler-specific parameters

        Returns
        -------
        InferenceData
            ArviZ InferenceData object with sampling results
        """
        model = kwargs.pop("model", self.model)
        if model is None:
            model = Model.get_context()

        # Handle variables
        vars = kwargs.pop("vars", self._vars)
        if vars is None:
            vars = model.value_vars

        # Create a NutPie model
        logger.info("Compiling NutPie model")
        nutpie_model = nutpie.compile_pymc_model(
            model,
            backend=self.backend,
        )

        # Set up sampling parameters - NutPie does this internally
        # Keep these for other nutpie parameters to pass
        nuts_kwargs = {
            **self.nutpie_kwargs,
            **kwargs,
        }

        if initvals is not None:
            warnings.warn(
                "`initvals` are currently not passed to nutpie sampler. "
                "Use `init_mean` kwarg following nutpie specification instead.",
                UserWarning,
            )

        # Set up random seed
        if random_seed is not None:
            nuts_kwargs["seed"] = random_seed

        # Run the sampler
        logger.info(
            f"Running NutPie sampler with {chains} chains, {tune} tuning steps, and {draws} draws"
        )

        # Add target acceptance and max tree depth
        nutpie_kwargs = {
            "target_accept": self.target_accept,
            "maxdepth": self.max_treedepth,
            **nuts_kwargs,
        }

        # Update parameter names to match NutPie's API
        if "progressbar" in nutpie_kwargs:
            nutpie_kwargs["progress_bar"] = nutpie_kwargs.pop("progressbar")

        # Pass progressbar from the sample function arguments
        if progressbar is not None:
            nutpie_kwargs["progress_bar"] = progressbar

        # Call NutPie's sample function
        nutpie_trace = nutpie.sample(
            nutpie_model,
            draws=draws,
            tune=tune,
            chains=chains,
            **nutpie_kwargs,
        )

        # Convert to InferenceData
        if idata_kwargs is None:
            idata_kwargs = {}

        # Extract relevant variables and data for InferenceData
        coords, dims = coords_and_dims_for_inferencedata(model)
        constants_data = find_constants(model)
        observed_data = find_observations(model)

        # Always include sampler stats
        if "include_sampler_stats" not in idata_kwargs:
            idata_kwargs["include_sampler_stats"] = True

        # NutPie already returns an InferenceData object
        idata = nutpie_trace

        # Set tuning steps attribute if possible
        try:
            idata.posterior.attrs["tuning_steps"] = tune
        except (AttributeError, KeyError):
            logger.warning("Could not set tuning_steps attribute on InferenceData")

        # Skip compute_convergence_checks for now
        # NutPie's InferenceData structure is different from PyMC's expectations

        return idata

    @staticmethod
    def competence(var, has_grad):
        """Determine competence level for sampling var.

        Parameters
        ----------
        var : Variable
            Variable to be sampled
        has_grad : bool
            Whether gradient information is available

        Returns
        -------
        Competence
            Enum indicating competence level for this variable
        """
        if var.dtype in continuous_types and has_grad:
            return Competence.IDEAL
        return Competence.INCOMPATIBLE
