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

from abc import ABC, abstractmethod

from arviz import InferenceData

from pymc.step_methods.compound import BlockedStep, Competence


class ExternalSampler(BlockedStep, ABC):
    """Base class for external samplers.

    External samplers manage their own MCMC loop rather than using PyMC's.
    These samplers (like NutPie, BlackJax, etc.) are designed to run
    their own efficient loop inside their implementation.

    Attributes
    ----------
    is_external : bool
        Flag indicating that this is an external sampler that needs
        special handling in PyMC's sampling loops.
    """

    is_external = True

    def __init__(
        self,
        vars=None,
        model=None,
        **kwargs,
    ):
        """Initialize external sampler.

        Parameters
        ----------
        vars : list, optional
            Variables to be sampled
        model : Model, optional
            PyMC model
        **kwargs
            Sampler-specific arguments
        """
        self.model = model
        self._vars = vars
        self._kwargs = kwargs

    @abstractmethod
    def sample(
        self,
        draws: int,
        tune: int = 1000,
        chains: int = 4,
        random_seed=None,
        initvals=None,
        progressbar=True,
        cores=None,
        **kwargs,
    ) -> InferenceData:
        """Run external sampler and return results as InferenceData.

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
            Initial values for variables
        progressbar : bool
            Whether to display progress bar
        cores : int, optional
            Number of CPU cores to use
        **kwargs
            Additional sampler-specific parameters

        Returns
        -------
        InferenceData
            ArviZ InferenceData object with sampling results
        """
        pass

    def step(self, point):
        """Do not use this method. External samplers use their own sampling loop.

        External samplers do not use PyMC's step() mechanism.
        """
        raise NotImplementedError(
            "External samplers use their own sampling loop rather than PyMC's step() method."
        )

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
        return Competence.COMPATIBLE
