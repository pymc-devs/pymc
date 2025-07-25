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
"""PyMC-ArviZ conversion code."""

import logging
import warnings

from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
    cast,
)

import numpy as np
import xarray

from arviz import InferenceData, concat, rcParams
from arviz.data.base import CoordSpec, DimSpec, dict_to_dataset, requires
from pytensor.graph import ancestors
from pytensor.tensor.sharedvar import SharedVariable
from rich.progress import Console
from rich.theme import Theme
from xarray import Dataset

import pymc

from pymc.model import Model, modelcontext
from pymc.progress_bar import CustomProgress, default_progress_theme
from pymc.pytensorf import PointFunc, extract_obs_data
from pymc.util import get_default_varnames

if TYPE_CHECKING:
    from pymc.backends.base import MultiTrace

___all__ = [""]

_log = logging.getLogger(__name__)


RAISE_ON_INCOMPATIBLE_COORD_LENGTHS = False


# random variable object ...
Var = Any


def dict_to_dataset_drop_incompatible_coords(vars_dict, *args, dims, coords, **kwargs):
    safe_coords = coords

    if not RAISE_ON_INCOMPATIBLE_COORD_LENGTHS:
        coords_lengths = {k: len(v) for k, v in coords.items()}
        for var_name, var in vars_dict.items():
            # Iterate in reversed because of chain/draw batch dimensions
            for dim, dim_length in zip(reversed(dims.get(var_name, ())), reversed(var.shape)):
                coord_length = coords_lengths.get(dim, None)
                if (coord_length is not None) and (coord_length != dim_length):
                    warnings.warn(
                        f"Incompatible coordinate length of {coord_length} for dimension '{dim}' of variable '{var_name}'.\n"
                        "This usually happens when a sliced or concatenated variable is wrapped as a `pymc.dims.Deterministic`."
                        "The originate coordinates for this dim will not be included in the returned dataset for any of the variables. "
                        "Instead they will default to `np.arange(var_length)` and the shorter variables will be right-padded with nan.\n"
                        "To make this warning into an error set `pymc.backends.arviz.RAISE_ON_INCOMPATIBLE_COORD_LENGTHS` to `True`",
                        UserWarning,
                    )
                    if safe_coords is coords:
                        safe_coords = coords.copy()
                    safe_coords.pop(dim)
                    coords_lengths.pop(dim)

    # FIXME: Would be better to drop coordinates altogether, but arviz defaults to `np.arange(var_length)`
    return dict_to_dataset(vars_dict, *args, dims=dims, coords=safe_coords, **kwargs)


def find_observations(model: "Model") -> dict[str, Var]:
    """If there are observations available, return them as a dictionary."""
    observations = {}
    for obs in model.observed_RVs:
        aux_obs = model.rvs_to_values.get(obs, None)
        if aux_obs is not None:
            try:
                obs_data = extract_obs_data(aux_obs)
                observations[obs.name] = obs_data
            except TypeError:
                warnings.warn(f"Could not extract data from symbolic observation {obs}")
        else:
            warnings.warn(f"No data for observation {obs}")

    return observations


def find_constants(model: "Model") -> dict[str, Var]:
    """If there are constants available, return them as a dictionary."""
    model_vars = model.basic_RVs + model.deterministics + model.potentials
    value_vars = set(model.rvs_to_values.values())

    constant_data = {}
    for var in model.data_vars:
        if var in value_vars:
            # An observed value variable could also be part of the generative graph
            if var not in ancestors(model_vars):
                continue

        if isinstance(var, SharedVariable):
            var_value = var.get_value()
        else:
            var_value = var.data
        constant_data[var.name] = var_value

    return constant_data


def coords_and_dims_for_inferencedata(model: Model) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse PyMC model coords and dims format to one accepted by InferenceData."""
    coords = {
        cname: np.array(cvals) if isinstance(cvals, tuple) else cvals
        for cname, cvals in model.coords.items()
        if cvals is not None
    }
    dims = {dname: list(dvals) for dname, dvals in model.named_vars_to_dims.items()}

    return coords, dims


class _DefaultTrace:
    """
    Utility for collecting samples into a dictionary.

    Name comes from its similarity to ``defaultdict``:
    entries are lazily created.

    Parameters
    ----------
    samples : int
        The number of samples that will be collected, per variable,
        into the trace.

    Attributes
    ----------
    trace_dict : Dict[str, np.ndarray]
        A dictionary constituting a trace.  Should be extracted
        after a procedure has filled the `_DefaultTrace` using the
        `insert()` method
    """

    def __init__(self, samples: int):
        self._len: int = samples
        self.trace_dict: dict[str, np.ndarray] = {}

    def insert(self, k: str, v, idx: int):
        """
        Insert `v` as the value of the `idx`th sample for the variable `k`.

        Parameters
        ----------
        k: str
            Name of the variable.
        v: anything that can go into a numpy array (including a numpy array)
            The value of the `idx`th sample from variable `k`
        ids: int
            The index of the sample we are inserting into the trace.
        """
        value_shape = np.shape(v)

        # initialize if necessary
        if k not in self.trace_dict:
            array_shape = (self._len, *value_shape)
            self.trace_dict[k] = np.empty(array_shape, dtype=np.array(v).dtype)

        # do the actual insertion
        if value_shape == ():
            self.trace_dict[k][idx] = v
        else:
            self.trace_dict[k][idx, :] = v


class InferenceDataConverter:
    """Encapsulate InferenceData specific logic."""

    model: Model | None = None
    posterior_predictive: Mapping[str, np.ndarray] | None = None
    predictions: Mapping[str, np.ndarray] | None = None
    prior: Mapping[str, np.ndarray] | None = None

    def __init__(
        self,
        *,
        trace=None,
        prior=None,
        posterior_predictive=None,
        log_likelihood=False,
        log_prior=False,
        predictions=None,
        coords: CoordSpec | None = None,
        dims: DimSpec | None = None,
        sample_dims: list | None = None,
        model=None,
        save_warmup: bool | None = None,
        include_transformed: bool = False,
    ):
        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup
        self.include_transformed = include_transformed
        self.trace = trace

        # this permits us to get the model from command-line argument or from with model:
        self.model = modelcontext(model)

        self.attrs = None
        if trace is not None:
            self.nchains = trace.nchains if hasattr(trace, "nchains") else 1
            if hasattr(trace.report, "n_draws") and trace.report.n_draws is not None:
                self.ndraws = trace.report.n_draws
                self.attrs = {
                    "sampling_time": trace.report.t_sampling,
                    "tuning_steps": trace.report.n_tune,
                }
            else:
                self.ndraws = len(trace)
                if self.save_warmup:
                    warnings.warn(
                        "Warmup samples will be stored in posterior group and will not be"
                        " excluded from stats and diagnostics."
                        " Do not slice the trace manually before conversion",
                        UserWarning,
                    )
            self.ntune = len(self.trace) - self.ndraws
            self.posterior_trace, self.warmup_trace = self.split_trace()
        else:
            self.nchains = self.ndraws = 0

        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.predictions = predictions

        if all(elem is None for elem in (trace, predictions, posterior_predictive, prior)):
            raise ValueError(
                "When constructing InferenceData you must pass at least"
                " one of trace, prior, posterior_predictive or predictions."
            )

        user_coords = {} if coords is None else coords
        user_dims = {} if dims is None else dims
        model_coords, model_dims = coords_and_dims_for_inferencedata(self.model)
        self.coords = {**model_coords, **user_coords}
        self.dims = {**model_dims, **user_dims}

        if sample_dims is None:
            sample_dims = ["chain", "draw"]
        self.sample_dims = sample_dims

        self.observations = find_observations(self.model)

    def split_trace(self) -> tuple[Union[None, "MultiTrace"], Union[None, "MultiTrace"]]:
        """Split MultiTrace object into posterior and warmup.

        Returns
        -------
        trace_posterior: MultiTrace or None
            The slice of the trace corresponding to the posterior. If the posterior
            trace is empty, None is returned
        trace_warmup: MultiTrace or None
            The slice of the trace corresponding to the warmup. If the warmup trace is
            empty or ``save_warmup=False``, None is returned
        """
        trace_posterior = None
        trace_warmup = None
        if self.save_warmup and self.ntune > 0:
            trace_warmup = self.trace[: self.ntune]
        if self.ndraws > 0:
            trace_posterior = self.trace[self.ntune :]
        return trace_posterior, trace_warmup

    @requires("trace")
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        var_names = get_default_varnames(
            self.trace.varnames, include_transformed=self.include_transformed
        )
        data = {}
        data_warmup = {}
        for var_name in var_names:
            if self.warmup_trace:
                data_warmup[var_name] = np.array(
                    self.warmup_trace.get_values(var_name, combine=False, squeeze=False)
                )
            if self.posterior_trace:
                data[var_name] = np.array(
                    self.posterior_trace.get_values(var_name, combine=False, squeeze=False)
                )
        return (
            dict_to_dataset(
                data,
                library=pymc,
                coords=self.coords,
                dims=self.dims,
                attrs=self.attrs,
            ),
            dict_to_dataset(
                data_warmup,
                library=pymc,
                coords=self.coords,
                dims=self.dims,
                attrs=self.attrs,
            ),
        )

    @requires("trace")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from PyMC trace."""
        data = {}
        rename_key = {
            "model_logp": "lp",
            "mean_tree_accept": "acceptance_rate",
            "depth": "tree_depth",
            "tree_size": "n_steps",
        }
        data = {}
        data_warmup = {}
        for stat in self.trace.stat_names:
            name = rename_key.get(stat, stat)
            if name == "tune":
                continue
            if self.warmup_trace:
                data_warmup[name] = np.array(
                    self.warmup_trace.get_sampler_stats(stat, combine=False, squeeze=False)
                )
            if self.posterior_trace:
                data[name] = np.array(
                    self.posterior_trace.get_sampler_stats(stat, combine=False, squeeze=False)
                )

        return (
            dict_to_dataset(
                data,
                library=pymc,
                dims=None,
                coords=self.coords,
                attrs=self.attrs,
            ),
            dict_to_dataset(
                data_warmup,
                library=pymc,
                dims=None,
                coords=self.coords,
                attrs=self.attrs,
            ),
        )

    @requires(["posterior_predictive"])
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data = self.posterior_predictive
        dims = {var_name: self.sample_dims + self.dims.get(var_name, []) for var_name in data}
        return dict_to_dataset(
            data, library=pymc, coords=self.coords, dims=dims, default_dims=self.sample_dims
        )

    @requires(["predictions"])
    def predictions_to_xarray(self):
        """Convert predictions (out of sample predictions) to xarray."""
        data = self.predictions
        dims = {var_name: self.sample_dims + self.dims.get(var_name, []) for var_name in data}
        return dict_to_dataset(
            data, library=pymc, coords=self.coords, dims=dims, default_dims=self.sample_dims
        )

    def priors_to_xarray(self):
        """Convert prior samples (and if possible prior predictive too) to xarray."""
        if self.prior is None:
            return {"prior": None, "prior_predictive": None}
        if self.observations is not None:
            prior_predictive_vars = list(set(self.observations).intersection(self.prior))
            prior_vars = [key for key in self.prior.keys() if key not in prior_predictive_vars]
        else:
            prior_vars = list(self.prior.keys())
            prior_predictive_vars = None

        priors_dict = {}
        for group, var_names in zip(
            ("prior", "prior_predictive"), (prior_vars, prior_predictive_vars)
        ):
            priors_dict[group] = (
                None
                if var_names is None
                else dict_to_dataset_drop_incompatible_coords(
                    {k: np.expand_dims(self.prior[k], 0) for k in var_names},
                    library=pymc,
                    coords=self.coords,
                    dims=self.dims,
                )
            )
        return priors_dict

    @requires("observations")
    @requires("model")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        if self.predictions:
            return None
        return dict_to_dataset(
            self.observations,
            library=pymc,
            coords=self.coords,
            dims=self.dims,
            default_dims=[],
        )

    @requires("model")
    def constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        constant_data = find_constants(self.model)
        if not constant_data:
            return None

        xarray_dataset = dict_to_dataset(
            constant_data,
            library=pymc,
            coords=self.coords,
            dims=self.dims,
            default_dims=[],
        )

        # provisional handling of scalars in constant
        # data to prevent promotion to rank 1
        # in the future this will be handled by arviz
        scalars = [var_name for var_name, value in constant_data.items() if np.ndim(value) == 0]
        for s in scalars:
            s_dim_0_name = f"{s}_dim_0"
            xarray_dataset = xarray_dataset.squeeze(s_dim_0_name, drop=True)

        return xarray_dataset

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (e.g., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        id_dict = {
            "posterior": self.posterior_to_xarray(),
            "sample_stats": self.sample_stats_to_xarray(),
            "posterior_predictive": self.posterior_predictive_to_xarray(),
            "predictions": self.predictions_to_xarray(),
            **self.priors_to_xarray(),
            "observed_data": self.observed_data_to_xarray(),
        }
        if self.predictions:
            id_dict["predictions_constant_data"] = self.constant_data_to_xarray()
        else:
            id_dict["constant_data"] = self.constant_data_to_xarray()
        idata = InferenceData(save_warmup=self.save_warmup, **id_dict)
        if self.log_likelihood:
            from pymc.stats.log_density import compute_log_likelihood

            idata = compute_log_likelihood(
                idata,
                var_names=None if self.log_likelihood is True else self.log_likelihood,
                extend_inferencedata=True,
                model=self.model,
                sample_dims=self.sample_dims,
                progressbar=False,
            )
        if self.log_prior:
            from pymc.stats.log_density import compute_log_prior

            idata = compute_log_prior(
                idata,
                var_names=None if self.log_prior is True else self.log_prior,
                extend_inferencedata=True,
                model=self.model,
                sample_dims=self.sample_dims,
                progressbar=False,
            )
        return idata


def to_inference_data(
    trace: Optional["MultiTrace"] = None,
    *,
    prior: Mapping[str, Any] | None = None,
    posterior_predictive: Mapping[str, Any] | None = None,
    log_likelihood: bool | Iterable[str] = False,
    log_prior: bool | Iterable[str] = False,
    coords: CoordSpec | None = None,
    dims: DimSpec | None = None,
    sample_dims: list | None = None,
    model: Optional["Model"] = None,
    save_warmup: bool | None = None,
    include_transformed: bool = False,
) -> InferenceData:
    """Convert pymc data into an InferenceData object.

    All three of them are optional arguments, but at least one of ``trace``,
    ``prior`` and ``posterior_predictive`` must be present.
    For a usage example read the
    :ref:`Creating InferenceData section on from_pymc <creating_InferenceData>`

    Parameters
    ----------
    trace : MultiTrace, optional
        Trace generated from MCMC sampling. Output of
        :func:`~pymc.sampling.sample`.
    prior : dict, optional
        Dictionary with the variable names as keys, and values numpy arrays
        containing prior and prior predictive samples.
    posterior_predictive : dict, optional
        Dictionary with the variable names as keys, and values numpy arrays
        containing posterior predictive samples.
    log_likelihood : bool or array_like of str, optional
        List of variables to calculate `log_likelihood`. Defaults to False.
        If set to True, computes `log_likelihood` for all observed variables.
    log_prior : bool or array_like of str, optional
        List of variables to calculate `log_prior`. Defaults to False.
        If set to True, computes `log_prior` for all unobserved variables.
    coords : dict of {str: array-like}, optional
        Map of coordinate names to coordinate values
    dims : dict of {str: list of str}, optional
        Map of variable names to the coordinate names to use to index its dimensions.
    model : Model, optional
        Model used to generate ``trace``. It is not necessary to pass ``model`` if in
        ``with`` context.
    save_warmup : bool, optional
        Save warmup iterations InferenceData object. If not defined, use default
        defined by the rcParams.
    include_transformed : bool, optional
        Save the transformed parameters in the InferenceData object. By default, these are
        not saved.

    Returns
    -------
    arviz.InferenceData
    """
    if isinstance(trace, InferenceData):
        return trace

    return InferenceDataConverter(
        trace=trace,
        prior=prior,
        posterior_predictive=posterior_predictive,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        coords=coords,
        dims=dims,
        sample_dims=sample_dims,
        model=model,
        save_warmup=save_warmup,
        include_transformed=include_transformed,
    ).to_inference_data()


### Later I could have this return ``None`` if the ``idata_orig`` argument is supplied.  But
### perhaps we should have an inplace argument?
def predictions_to_inference_data(
    predictions,
    posterior_trace: Optional["MultiTrace"] = None,
    model: Optional["Model"] = None,
    coords: CoordSpec | None = None,
    dims: DimSpec | None = None,
    sample_dims: list | None = None,
    idata_orig: InferenceData | None = None,
    inplace: bool = False,
) -> InferenceData:
    """Translate out-of-sample predictions into ``InferenceData``.

    Parameters
    ----------
    predictions: Dict[str, np.ndarray]
        The predictions are the return value of :func:`~pymc.sample_posterior_predictive`,
        a dictionary of strings (variable names) to numpy ndarrays (draws).
        Requires the arrays to follow the convention ``chain, draw, *shape``.
    posterior_trace: MultiTrace
        This should be a trace that has been thinned appropriately for
        ``pymc.sample_posterior_predictive``. Specifically, any variable whose shape is
        a deterministic function of the shape of any predictor (explanatory, independent, etc.)
        variables must be *removed* from this trace.
    model: Model
        The pymc model. It can be omitted if within a model context.
    coords: Dict[str, array-like[Any]]
        Coordinates for the variables.  Map from coordinate names to coordinate values.
    dims: Dict[str, array-like[str]]
        Map from variable name to ordered set of coordinate names.
    idata_orig: InferenceData, optional
        If supplied, then modify this inference data in place, adding ``predictions`` and
        (if available) ``predictions_constant_data`` groups. If this is not supplied, make a
        fresh InferenceData
    inplace: boolean, optional
        If idata_orig is supplied and inplace is True, merge the predictions into idata_orig,
        rather than returning a fresh InferenceData object.

    Returns
    -------
    InferenceData:
        May be modified ``idata_orig``.
    """
    if inplace and not idata_orig:
        raise ValueError(
            "Do not pass True for inplace unless passing an existing InferenceData as idata_orig"
        )
    converter = InferenceDataConverter(
        trace=posterior_trace,
        predictions=predictions,
        model=model,
        coords=coords,
        dims=dims,
        sample_dims=sample_dims,
        log_likelihood=False,
    )
    if hasattr(idata_orig, "posterior"):
        assert idata_orig is not None
        converter.nchains = idata_orig["posterior"].sizes["chain"]
        converter.ndraws = idata_orig["posterior"].sizes["draw"]
    else:
        aelem = next(iter(predictions.values()))
        converter.nchains, converter.ndraws = aelem.shape[:2]
    new_idata = converter.to_inference_data()
    if idata_orig is None:
        return new_idata
    elif inplace:
        concat([idata_orig, new_idata], dim=None, inplace=True)
        return idata_orig
    else:
        # if we are not returning in place, then merge the old groups into the new inference
        # data and return that.
        concat([new_idata, idata_orig], dim=None, copy=True, inplace=True)
        return new_idata


def dataset_to_point_list(
    ds: xarray.Dataset | dict[str, xarray.DataArray], sample_dims: Sequence[str]
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    # All keys of the dataset must be a str
    var_names = cast(list[str], list(ds.keys()))
    for vn in var_names:
        if not isinstance(vn, str):
            raise ValueError(f"Variable names must be str, but dataset key {vn} is a {type(vn)}.")

    num_sample_dims = len(sample_dims)
    stacked_dims = {dim_name: ds[var_names[0]][dim_name] for dim_name in sample_dims}
    transposed_dict = {vn: da.transpose(*sample_dims, ...) for vn, da in ds.items()}
    stacked_size = np.prod(transposed_dict[var_names[0]].shape[:num_sample_dims], dtype=int)
    stacked_dict = {
        vn: da.values.reshape((stacked_size, *da.shape[num_sample_dims:]))
        for vn, da in transposed_dict.items()
    }
    points = [
        {vn: stacked_dict[vn][i, ...] for vn in var_names}
        for i in range(np.prod([len(coords) for coords in stacked_dims.values()]))
    ]
    # use the list of points
    return cast(list[dict[str, np.ndarray]], points), stacked_dims


def apply_function_over_dataset(
    fn: PointFunc,
    dataset: Dataset,
    *,
    output_var_names: Sequence[str],
    coords,
    dims,
    sample_dims: Sequence[str] = ("chain", "draw"),
    progressbar: bool = True,
    progressbar_theme: Theme | None = default_progress_theme,
) -> Dataset:
    posterior_pts, stacked_dims = dataset_to_point_list(dataset, sample_dims)

    n_pts = len(posterior_pts)
    out_dict = _DefaultTrace(n_pts)
    indices = range(n_pts)

    with CustomProgress(
        console=Console(theme=progressbar_theme), disable=not progressbar
    ) as progress:
        task = progress.add_task("Computing ...", total=n_pts)
        for idx in indices:
            out = fn(posterior_pts[idx])
            fn.f.trust_input = True  # If we arrive here the dtypes are valid
            for var_name, val in zip(output_var_names, out, strict=True):
                out_dict.insert(var_name, val, idx)

            progress.advance(task)
        progress.update(task, refresh=True, completed=n_pts)

    out_trace = out_dict.trace_dict
    for key, val in out_trace.items():
        out_trace[key] = val.reshape(
            (
                *[len(coord) for coord in stacked_dims.values()],
                *val.shape[1:],
            )
        )

    return dict_to_dataset(
        out_trace,
        library=pymc,
        dims=dims,
        coords=coords,
        default_dims=list(sample_dims),
        skip_event_dims=True,
    )
