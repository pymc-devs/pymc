import theano
import theano.tensor as tt
import numpy as np
from .distribution import NoDistribution, Distribution

__all__ = ['Simulator']

class Simulator(NoDistribution):

    def __init__(self, function, parameters, sum_stat=None, distance_metric=None, *args, **kwargs):
   
        self.function = function
        self.parameters = parameters
        self.epsilon = theano.shared(np.inf)
        self.sum_stat = sum_stat
        observed = self.data
        self.observed_stat = get_sum_stats(observed, sum_stat=sum_stat)
        self.distance_metric = distance_metric
        super(Simulator, self).__init__(shape=np.prod(observed.shape), dtype=observed.dtype, *args, **kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from Uniform distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        
        print('Not implemented yet')
        
    def logp(self, value):
        """
        Calculate log-probability of Uniform distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value for which log-probability is calculated.
        Returns
        -------
        TensorVariable
        """
        parameters = self.parameters
        epsilon = self.epsilon
        simulated = self.function(*parameters)
        simulated_stat = get_sum_stats(simulated.eval(), sum_stat=self.sum_stat)
        distance_function = get_distance(self.distance_metric)
        distance = distance_function(self.observed_stat, simulated_stat)
        logp = tt.switch(tt.le(distance, epsilon) , 0, -np.inf)

        return logp

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        name = r'\text{%s}' % name
        function = dist.function
        params = dist.parameters
        sum_stat = dist.sum_stat
        return r'${} \sim \text{{Simulator}}(\mathit{{function}}={},~\mathit{{parameters}}={},~\mathit{{summary statistics}}={})$'.format(
            name, function, params, sum_stat)

def get_sum_stats(data, sum_stat=None):
    """
    Parameters:
    -----------
    data : array
        Observed or simulated data
    sum_stat : list
        List of summary statistics to be computed. Accepted strings are mean, std, var. 
        Python functions can be passed in this argument.

    Returns:
    --------
    sum_stat_vector : array
        Array contaning the summary statistics.
    """
    
    if data.ndim == 1:
        data = data[:,np.newaxis]
    sum_stat_vector = np.zeros((len(sum_stat), data.shape[1]))

    for i, stat in enumerate(sum_stat):
        for j in range(sum_stat_vector.shape[1]):
            if stat == 'mean':
                sum_stat_vector[i, j] =  data[:,j].mean()
            elif stat == 'std':
                sum_stat_vector[i, j] =  data[:,j].std()
            elif stat == 'var':
                sum_stat_vector[i, j] =  data[:,j].var()
            else:
                sum_stat_vector[i, j] =  stat(data[:,j])

    return np.atleast_1d(np.squeeze(sum_stat_vector))

def absolute_difference(a, b):
    return tt.sum(np.abs(a - b))

def sum_of_squared_distance(a, b):
    return tt.sum((a - b)**2)

def mean_absolute_error(a, b):
    return tt.sum(np.abs(a - b))/len(a)

def mean_squared_error(a, b):
    return tt.sum((a - b)**2)/len(a)

def euclidean_distance(a, b):
    return np.sqrt(tt.sum((a - b)**2))

def get_distance(func_name):
    d = {'absolute_difference': absolute_difference,
         'sum_of_squared_distance' : sum_of_squared_distance,
         'mean_absolute_error' : mean_absolute_error,
         'mean_squared_error' : mean_squared_error,
         'euclidean' : euclidean_distance}
    for key, value in d.items():
        if func_name == key:
            return value
