import pymc as pm
import arviz
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict
import cloudpickle
import arviz as az
import hashlib

class ModelBuilder(pm.Model):

    _model_type = 'BaseClass'
    version = 'None'

    def __init__(self, data : pd.DataFrame, model_config : Dict, sampler_config : Dict):
        super().__init__()
        self.data = data # input and output data
        self.model_config = model_config # parameters for priors etc.
        self.sample_config = sampler_config # parameters for sampling

        self.idata = None

    def _build(self):
    	raise NotImplementedError


    def _data_setter(self, data : pd.DataFrame, x_only : bool = True):

        raise NotImplementedError


    @classmethod
    def create_sample_input(cls):
        '''
        Needs to be implemented by the user in the inherited class.
        Returns examples for data, model_config, samples_config.
        This is useful for understanding the required 
        data structures for the user model.
        '''
        raise NotImplementedError


    def build(self):
        with self:
            self._build()


    def save(self,file_prefix,filepath,save_model=True,save_idata=True):
        if save_idata:
            file = Path(filepath+str(file_prefix)+'.nc')
            self.idata.to_netcdf(file)
        if save_model:
            filepath = Path(str(filepath)+str(file_prefix)+'.pickle')
            Model = cloudpickle.dumps(self)
            file = open(filepath, 'wb')
            file.write(Model)
        self.saved = True


    @classmethod
    def load(cls,file_prefix,filepath,load_model=True,laod_idata=True):
        file = Path(str(filepath)+str(file_prefix)+'.pickle')
        file = open(file,'rb')
        self = cloudpickle.loads(file.read())
        filepath = Path(str(filepath)+str(file_prefix)+'.nc')
        data = az.from_netcdf(filepath)
        self.idata = data
        return self

    # fit and predict methods
    def fit(self, data : pd.DataFrame = None):
        if data is not None: 
            self.data = data

        if self.basic_RVs == []:
            print('No model found, building model...')
            self.build()

        with self:
            self.idata = pm.sample(**self.sample_config)
            self.idata.extend(pm.sample_prior_predictive())
            self.idata.extend(pm.sample_posterior_predictive(self.idata))

        self.idata.attrs['id']=self.id()
        self.idata.attrs['model_type']=self._model_type
        self.idata.attrs['version']=self.version
        self.idata.attrs['sample_conifg']=self.sample_config
        self.idata.attrs['model_config']=self.model_config
        # model,model_type,version,sample_conifg,model_config
        return self.idata


    def predict(self, data_prediction : pd.DataFrame = None, point_estimate : bool = True):
        '''
        Prediction for new inputs
        '''
        if data_prediction is not None: # set new input data
            self._data_setter(data_prediction)

        with self.model: # sample with new input data
            post_pred = pm.sample_posterior_predictive(self.idata.posterior)

        # reshape output
        post_pred = self._extract_samples(post_pred)

        if point_estimate: # average, if point-like estimate desired
            for key in post_pred:
                post_pred[key] = post_pred[key].mean(axis=0)

        if data_prediction is not None: # set back original data in model
            self._data_setter(self.data)
            # is this necessary?

        return post_pred


    @staticmethod
    def _extract_samples(post_pred : arviz.data.inference_data.InferenceData) -> Dict[str, np.array]:
        '''
        Returns dict of numpy arrays from InferenceData object
        '''
        post_pred_dict = dict()
        for key in post_pred.posterior_predictive:
            post_pred_dict[key] = post_pred.posterior_predictive[key].to_numpy()[0]

        return post_pred_dict

    def id(self):
    	hasher = hashlib.sha256()
    	hasher.update(str(self.model_config.values()).encode())
    	hasher.update(self.version.encode())
    	hasher.update(self._model_type.encode())
    	hasher.update(str(self.sample_config.values()).encode())
    	return hasher.hexdigest()[:16]
