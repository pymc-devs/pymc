import cloudpickle
from pathlib import Path
import os
import arviz as az
import numpy as np
import pymc as pm
import trace 

import logging
import sys
import time
import warnings

from collections import defaultdict
from copy import copy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from pymc.initial_point import (
    PointType,
    StartDict,
    filter_rvs_to_jitter,
    make_initial_point_fns_per_chain,
)

RandomSeed = Optional[Union[int, Sequence[int], np.ndarray]]
RandomState = Union[RandomSeed, np.random.RandomState, np.random.Generator]

class base:
    def __init__(self):
        self.model = None
        self.trace = None
        self.saved = False
        self.build = False
        self.Model = None

    def build_model(self):
        self.model = pm.Model()

    def save(self,file_prefix,filepath,save_format=None,overwrite=False):
        if overwrite==False and self.saved:
            print("Model already exists")
            return
        if save_format == 'h5':
            extension = '.hdf5'
        else:
            extension = '.pickle'
        filepath = Path(filepath+str(file_prefix)+extension)
        Model = cloudpickle.dumps(self.model)
        file = open(filepath, 'wb')
        file.write(Model)
        self.saved = True
        print("Model Saved")

    def load(self,filepath):
        filepath = Path(filepath)
        file = open(filepath,'rb')
        self.model = cloudpickle.loads(file.read())

    def fit(
        self,
        draws: int = 1000,
        step=None,
        init: str = "auto",
        n_init: int = 200_000,
        initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
        trace = None,
        chain_idx: int = 0,
        chains: Optional[int] = None,
        cores: Optional[int] = None,
        tune: int = 1000,
        progressbar: bool = True,
        model=None,
        random_seed: RandomState = None,
        discard_tuned_samples: bool = True,
        compute_convergence_checks: bool = True,
        callback=None,
        jitter_max_retries: int = 10,
        return_inferencedata: bool = True,
        idata_kwargs: dict = None,
        mp_ctx=None,
        **kwargs,
        ):
        with self.model:
            trace = pm.sample(
                draws,
                step,
                init,
                n_init,
                initvals,
                trace,
                chain_idx,
                chains,
                cores,
                tune,
                progressbar,
                model,
                random_seed,
                discard_tuned_samples,
                compute_convergence_checks,
                callback,
                jitter_max_retries,
                return_inferencedata=return_inferencedata,
                idata_kwargs=idata_kwargs,
                mp_ctx=mp_ctx,
                **kwargs,
                )

        self.trace = trace
        return trace

    def predict(self,
                X,
                samples: int = 500,
                var_names: Optional[Iterable[str]] = None,
                random_seed=None,
                return_inferencedata: bool = True,
                idata_kwargs: dict = None,
                compile_kwargs: dict = None
                ):

        # if self.trace is None:
        #     raise NotFittedError('Model is not yet fitted')

        # res_dct = {X[i]: X[i + 1] for i in range(0, len(X), 2)}

        with self.model:
             pm.set_data(X)
             idata = self.trace
             y_pred = pm.sample_posterior_predictive(idata)
             return  y_pred

class MyModel(base):
    def init():
        return

    def build_model(self):
        super().build_model()

    def save(self,file_prefix,filepath,save_format=None,overwrite=False):
        super().save(file_prefix,filepath,save_format,overwrite)

    def load(self,filepath):
        model = super().load(filepath)
        return model

    def fit(
        self,
        draws: int = 1000,
        step=None,
        init: str = "auto",
        n_init: int = 200_000,
        initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
        trace = None,
        chain_idx: int = 0,
        chains: Optional[int] = None,
        cores: Optional[int] = None,
        tune: int = 1000,
        progressbar: bool = True,
        model=None,
        random_seed: RandomState = None,
        discard_tuned_samples: bool = True,
        compute_convergence_checks: bool = True,
        callback=None,
        jitter_max_retries: int = 10,
        return_inferencedata: bool = True,
        idata_kwargs: dict = None,
        mp_ctx=None,
        **kwargs,
        ):

        trace = super().fit(
                draws,
                step,
                init,
                n_init,
                initvals,
                trace,
                chain_idx,
                chains,
                cores,
                tune,
                progressbar,
                model,
                random_seed,
                discard_tuned_samples,
                compute_convergence_checks,
                callback,
                jitter_max_retries,
                return_inferencedata,
                idata_kwargs,
                mp_ctx,
                **kwargs,
                )
        return trace

    def predict(self,
                X,
                samples: int = 500,
                var_names: Optional[Iterable[str]] = None,
                random_seed=None,
                return_inferencedata: bool = True,
                idata_kwargs: dict = None,
                compile_kwargs: dict = None):

        predicted = super().predict(
                X,
                samples,
                var_names,
                random_seed,
                return_inferencedata,
                idata_kwargs,
                compile_kwargs)
        return predicted
