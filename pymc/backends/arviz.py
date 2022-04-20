"""PyMC-ArviZ conversion code."""
import logging
import warnings

from typing import (  # pylint: disable=unused-import
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr

from aesara.graph.basic import Constant
from aesara.tensor.sharedvar import SharedVariable
from aesara.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1
from arviz import InferenceData, concat, rcParams
from arviz.data.base import CoordSpec, DimSpec, dict_to_dataset, requires

import pymc

from pymc.aesaraf import extract_obs_data
from pymc.model import modelcontext
from pymc.util import get_default_varnames

if TYPE_CHECKING:
    from typing import Set  # pylint: disable=ungrouped-imports

    from pymc.backends.base import MultiTrace  # pylint: disable=invalid-name
    from pymc.model import Model

___all__ = [""]

_log = logging.getLogger("pymc")

# random variable object ...
Var = Any  # pylint: disable=invalid-name


def find_observations(model: Optional["Model"]) -> Optional[Dict[str, Var]]:
    """If there are observations available, return them as a dictionary."""
    if model is None:
        return None

    observations = {}
    for obs in model.observed_RVs:
        aux_obs = getattr(obs.tag, "observations", None)
        if aux_obs is not None:
            try:
                obs_data = extract_obs_data(aux_obs)
                observations[obs.name] = obs_data
            except TypeError:
                warnings.warn(f"Could not extract data from symbolic observation {obs}")
        else:
            warnings.warn(f"No data for observation {obs}")

    return observations


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
        self.trace_dict: Dict[str, np.ndarray] = {}

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
            array_shape = (self._len,) + value_shape
            self.trace_dict[k] = np.empty(array_shape, dtype=np.array(v).dtype)

        # do the actual insertion
        if value_shape == ():
            self.trace_dict[k][idx] = v
        else:
            self.trace_dict[k][idx, :] = v


class InferenceDataConverter:  # pylint: disable=too-many-instance-attributes
    """Encapsulate InferenceData specific logic."""

    model = None  # type: Optional[Model]
    nchains = None  # type: int
    ndraws = None  # type: int
    posterior_predictive = None  # Type: Optional[Mapping[str, np.ndarray]]
    predictions = None  # Type: Optional[Mapping[str, np.ndarray]]
    prior = None  # Type: Optional[Mapping[str, np.ndarray]]

    def __init__(
        self,
        *,
        trace=None,
        prior=None,
        posterior_predictive=None,
        log_likelihood=True,
        predictions=None,
        coords: Optional[CoordSpec] = None,
        dims: Optional[DimSpec] = None,
        model=None,
        save_warmup: Optional[bool] = None,
    ):

        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup
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
        self.predictions = predictions

        if all(elem is None for elem in (trace, predictions, posterior_predictive, prior)):
            raise ValueError(
                "When constructing InferenceData you must pass at least"
                " one of trace, prior, posterior_predictive or predictions."
            )

        # Make coord types more rigid
        untyped_coords: Dict[str, Optional[Sequence[Any]]] = {**self.model.coords}
        if coords:
            untyped_coords.update(coords)
        self.coords = {
            cname: np.array(cvals) if isinstance(cvals, tuple) else cvals
            for cname, cvals in untyped_coords.items()
            if cvals is not None
        }

        self.dims = {} if dims is None else dims
        if hasattr(self.model, "RV_dims"):
            model_dims = {
                var_name: [dim for dim in dims if dim is not None]
                for var_name, dims in self.model.RV_dims.items()
            }
            self.dims = {**model_dims, **self.dims}

        self.observations = find_observations(self.model)

    def split_trace(self) -> Tuple[Union[None, "MultiTrace"], Union[None, "MultiTrace"]]:
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

    def log_likelihood_vals_point(self, point, var, log_like_fun):
        """Compute log likelihood for each observed point."""
        # TODO: This is a cheap hack; we should filter-out the correct
        # variables some other way
        point = {i.name: point[i.name] for i in log_like_fun.f.maker.inputs if i.name in point}
        log_like_val = np.atleast_1d(log_like_fun(point))

        if isinstance(var.owner.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1)):
            try:
                obs_data = extract_obs_data(var.tag.observations)
            except TypeError:
                warnings.warn(f"Could not extract data from symbolic observation {var}")

            mask = obs_data.mask
            if np.ndim(mask) > np.ndim(log_like_val):
                mask = np.any(mask, axis=-1)
            log_like_val = np.where(mask, np.nan, log_like_val)
        return log_like_val

    def _extract_log_likelihood(self, trace):
        """Compute log likelihood of each observation."""
        if self.trace is None:
            return None
        if self.model is None:
            return None

        # TODO: We no longer need one function per observed variable
        if self.log_likelihood is True:
            cached = [
                (
                    var,
                    self.model.compile_fn(
                        self.model.logpt(var, sum=False)[0],
                        inputs=self.model.value_vars,
                        on_unused_input="ignore",
                    ),
                )
                for var in self.model.observed_RVs
            ]
        else:
            cached = [
                (
                    var,
                    self.model.compile_fn(
                        self.model.logpt(var, sum=False)[0],
                        inputs=self.model.value_vars,
                        on_unused_input="ignore",
                    ),
                )
                for var in self.model.observed_RVs
                if var.name in self.log_likelihood
            ]
        log_likelihood_dict = _DefaultTrace(len(trace.chains))
        for var, log_like_fun in cached:
            for k, chain in enumerate(trace.chains):
                log_like_chain = [
                    self.log_likelihood_vals_point(point, var, log_like_fun)
                    for point in trace.points([chain])
                ]
                log_likelihood_dict.insert(var.name, np.stack(log_like_chain), k)
        return log_likelihood_dict.trace_dict

    @requires("trace")
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        var_names = get_default_varnames(self.trace.varnames, include_transformed=False)
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
                    self.warmup_trace.get_sampler_stats(stat, combine=False)
                )
            if self.posterior_trace:
                data[name] = np.array(self.posterior_trace.get_sampler_stats(stat, combine=False))

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

    @requires("trace")
    @requires("model")
    def log_likelihood_to_xarray(self):
        """Extract log likelihood and log_p data from PyMC trace."""
        if self.predictions or not self.log_likelihood:
            return None
        data_warmup = {}
        data = {}
        warn_msg = (
            "Could not compute log_likelihood, it will be omitted. "
            "Check your model object or set log_likelihood=False"
        )
        if self.posterior_trace:
            try:
                data = self._extract_log_likelihood(self.posterior_trace)
            except TypeError:
                warnings.warn(warn_msg)
        if self.warmup_trace:
            try:
                data_warmup = self._extract_log_likelihood(self.warmup_trace)
            except TypeError:
                warnings.warn(warn_msg)
        return (
            dict_to_dataset(
                data,
                library=pymc,
                dims=self.dims,
                coords=self.coords,
                skip_event_dims=True,
            ),
            dict_to_dataset(
                data_warmup,
                library=pymc,
                dims=self.dims,
                coords=self.coords,
                skip_event_dims=True,
            ),
        )

    def translate_posterior_predictive_dict_to_xarray(self, dct, kind) -> xr.Dataset:
        """Take Dict of variables to numpy ndarrays (samples) and translate into dataset."""
        data = {}
        warning_vars = []
        for k, ary in dct.items():
            if (ary.shape[0] == self.nchains) and (ary.shape[1] == self.ndraws):
                data[k] = ary
            else:
                data[k] = np.expand_dims(ary, 0)
                warning_vars.append(k)
        if warning_vars:
            warnings.warn(
                f"The shape of variables {', '.join(warning_vars)} in {kind} group is not compatible "
                "with number of chains and draws. The automatic dimension naming might not have worked. "
                "This can also mean that some draws or even whole chains are not represented.",
                UserWarning,
            )
        return dict_to_dataset(data, library=pymc, coords=self.coords, dims=self.dims)

    @requires(["posterior_predictive"])
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        return self.translate_posterior_predictive_dict_to_xarray(
            self.posterior_predictive, "posterior_predictive"
        )

    @requires(["predictions"])
    def predictions_to_xarray(self):
        """Convert predictions (out of sample predictions) to xarray."""
        return self.translate_posterior_predictive_dict_to_xarray(self.predictions, "predictions")

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
                else dict_to_dataset(
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

    @requires(["trace", "predictions"])
    @requires("model")
    def constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        # For constant data, we are concerned only with deterministics and
        # data.  The constant data vars must be either pm.Data
        # (TensorConstant/SharedVariable) or pm.Deterministic
        constant_data_vars = {}  # type: Dict[str, Var]

        def is_data(name, var) -> bool:
            assert self.model is not None
            return (
                var not in self.model.deterministics
                and var not in self.model.observed_RVs
                and var not in self.model.free_RVs
                and var not in self.model.potentials
                and var not in self.model.value_vars
                and (self.observations is None or name not in self.observations)
                and isinstance(var, (Constant, SharedVariable))
            )

        # I don't know how to find pm.Data, except that they are named
        # variables that aren't observed or free RVs, nor are they
        # deterministics, and then we eliminate observations.
        for name, var in self.model.named_vars.items():
            if is_data(name, var):
                constant_data_vars[name] = var

        if not constant_data_vars:
            return None

        constant_data = {}
        for name, vals in constant_data_vars.items():
            if hasattr(vals, "get_value"):
                vals = vals.get_value()
            elif hasattr(vals, "data"):
                vals = vals.data
            constant_data[name] = vals

        return dict_to_dataset(
            constant_data,
            library=pymc,
            coords=self.coords,
            dims=self.dims,
            default_dims=[],
        )

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (e.g., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        id_dict = {
            "posterior": self.posterior_to_xarray(),
            "sample_stats": self.sample_stats_to_xarray(),
            "log_likelihood": self.log_likelihood_to_xarray(),
            "posterior_predictive": self.posterior_predictive_to_xarray(),
            "predictions": self.predictions_to_xarray(),
            **self.priors_to_xarray(),
            "observed_data": self.observed_data_to_xarray(),
        }
        if self.predictions:
            id_dict["predictions_constant_data"] = self.constant_data_to_xarray()
        else:
            id_dict["constant_data"] = self.constant_data_to_xarray()
        return InferenceData(save_warmup=self.save_warmup, **id_dict)


def to_inference_data(
    trace: Optional["MultiTrace"] = None,
    *,
    prior: Optional[Mapping[str, Any]] = None,
    posterior_predictive: Optional[Mapping[str, Any]] = None,
    log_likelihood: Union[bool, Iterable[str]] = True,
    coords: Optional[CoordSpec] = None,
    dims: Optional[DimSpec] = None,
    model: Optional["Model"] = None,
    save_warmup: Optional[bool] = None,
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
        List of variables to calculate `log_likelihood`. Defaults to True which calculates
        `log_likelihood` for all observed variables. If set to False, log_likelihood is skipped.
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
        coords=coords,
        dims=dims,
        model=model,
        save_warmup=save_warmup,
    ).to_inference_data()


### Later I could have this return ``None`` if the ``idata_orig`` argument is supplied.  But
### perhaps we should have an inplace argument?
def predictions_to_inference_data(
    predictions,
    posterior_trace: Optional["MultiTrace"] = None,
    model: Optional["Model"] = None,
    coords: Optional[CoordSpec] = None,
    dims: Optional[DimSpec] = None,
    idata_orig: Optional[InferenceData] = None,
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
        The pymc model. It can be ommited if within a model context.
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
        log_likelihood=False,
    )
    if hasattr(idata_orig, "posterior"):
        assert idata_orig is not None
        converter.nchains = idata_orig["posterior"].dims["chain"]
        converter.ndraws = idata_orig["posterior"].dims["draw"]
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
