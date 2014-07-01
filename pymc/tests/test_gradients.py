from numpy import *
from pymc import *
from numpy.testing import *
import nose
import sys
from pymc import utils
from pymc import six
import pymc

float_dtypes = [float, single, float_, longfloat]


def generate_model():

    # Simulated data
    ReData = arange(200, 3000, 25)
    measured = 10.2 * (
        ReData) ** .5 + random.normal(
            scale=55,
            size=size(
                ReData))

    sd = pymc.Uniform('sd', lower=5, upper=100)

    a = pymc.Uniform('a', lower=0, upper=100)
    b = pymc.Uniform('b', lower=.05, upper=2.0)

    nonlinear = a * (ReData) ** b
    precision = sd ** -2

    results = pymc.Normal(
        'results',
        mu=nonlinear,
        tau=precision,
        value=measured,
        observed=True)

    return locals()


def find_variable_set(stochastic):
    set = [stochastic]
    for parameter, variable in six.iteritems(stochastic.parents):
        if isinstance(variable, Variable):
            set.append(variable)
    return set


def check_jacobians(deterministic):
    for parameter, pvalue in six.iteritems(deterministic.parents):

        if isinstance(pvalue, Variable):

            grad = random.normal(.5, .1, size=shape(deterministic.value))
            a_partial_grad = get_analytic_partial_gradient(
                deterministic, parameter, pvalue, grad)

            n_partial_grad = get_numeric_partial_gradient(
                deterministic, pvalue, grad)

            assert_array_almost_equal(a_partial_grad, n_partial_grad, 4,
                                      "analytic partial gradient for " + str(deterministic) +
                                      " with respect to parameter " + str(parameter) +
                                      " is not correct.")


def get_analytic_partial_gradient(deterministic, parameter, variable, grad):
    try:
        jacobian = deterministic._jacobians[parameter].get()
    except KeyError:
        raise ValueError(str(
            deterministic) + " has no jacobian for " + str(parameter))
    mapping = deterministic._jacobian_formats.get(parameter, 'full')

    return deterministic._format_mapping[mapping](
        deterministic, variable, jacobian, grad)


def get_numeric_partial_gradient(deterministic, pvalue, grad):
    j = get_numeric_jacobian(deterministic, pvalue)
    pg = deterministic._format_mapping['full'](deterministic, pvalue, j, grad)
    return reshape(pg, shape(pvalue.value))


def get_numeric_jacobian(deterministic, pvalue):
    e = 1e-9
    initial_pvalue = pvalue.value
    shape = initial_pvalue.shape
    size = initial_pvalue.size

    initial_value = ravel(deterministic.value)

    numeric_jacobian = zeros((deterministic.value.size, size))
    for i in range(size):

        delta = zeros(size)
        delta[i] += e

        pvalue.value = reshape(initial_pvalue.ravel() + delta, shape)
        value = ravel(deterministic.value)

        numeric_jacobian[:, i] = (value - initial_value) / e

    pvalue.value = initial_pvalue
    return numeric_jacobian


def check_model_gradients(model):

    model = set(model)
    # find the markov blanket
    children = set([])
    for s in model:
        for s2 in s.extended_children:
            if isinstance(s2, Stochastic) and s2.observed:
                children.add(s2)

    # self.markov_blanket is a list, because we want self.stochastics to have the chance to
    # raise ZeroProbability exceptions before self.children.
    markov_blanket = list(model) + list(children)

    gradients = utils.logp_gradient_of_set(model)
    for variable in model:

        analytic_gradient = gradients[variable]

        numeric_gradient = get_numeric_gradient(markov_blanket, variable)

        assert_array_almost_equal(numeric_gradient, analytic_gradient, 3,
                                  "analytic gradient for model " + str(model) +
                                  " with respect to variable " + str(variable) +
                                  " is not correct.")


def check_gradients(stochastic):

    stochastics = find_variable_set(stochastic)
    gradients = utils.logp_gradient_of_set(stochastics, stochastics)

    for s, analytic_gradient in six.iteritems(gradients):

        numeric_gradient = get_numeric_gradient(stochastics, s)

        assert_array_almost_equal(numeric_gradient, analytic_gradient, 3,
                                  "analytic gradient for " + str(stochastic) +
                                  " with respect to parameter " + str(s) +
                                  " is not correct.")


def get_numeric_gradient(stochastic, pvalue):
    e = 1e-9
    initial_value = pvalue.value
    i_shape = shape(initial_value)
    i_size = size(initial_value)

    initial_logp = utils.logp_of_set(stochastic)
    numeric_gradient = zeros(i_shape)
    if not (pvalue.dtype in float_dtypes):
        return numeric_gradient

    for i in range(i_size):

        delta = zeros(i_shape)

        try:
            delta[unravel_index(i, i_shape or (1,))] += e
        except IndexError:
            delta += e

        pvalue.value = initial_value + delta
        logp = utils.logp_of_set(stochastic)

        try:
            numeric_gradient[
                unravel_index(
                    i,
                    i_shape or (
                        1,
                    ))] = (
                        logp - initial_logp) / e
        except IndexError:
            numeric_gradient = (logp - initial_logp) / e

    pvalue.value = initial_value
    return numeric_gradient


class test_gradients(TestCase):

    def test_jacobians(self):
        shape = (3, 10)
        a = Normal('a', mu=zeros(shape), tau=ones(shape))
        b = Normal('b', mu=zeros(shape), tau=ones(shape))
        c = Uniform('c', lower=ones(shape) * .1, upper=ones(shape) * 10)
        d = Uniform('d', lower=ones(shape) * -10, upper=ones(shape) * -.1)

        addition = a + b
        check_jacobians(addition)

        subtraction = a - b
        check_jacobians(subtraction)

        multiplication = a * b
        check_jacobians(subtraction)

        division1 = a / c
        check_jacobians(division1)

        division2 = a / d
        check_jacobians(division2)

        a2 = Uniform('a2', lower=.1 * ones(shape), upper=2.0 * ones(shape))
        powering = a2 ** b
        check_jacobians(powering)

        negation = -a
        check_jacobians(negation)

        absing = abs(a)
        check_jacobians(absing)

        indexing1 = a[0:1, 5:8]
        check_jacobians(indexing1)

        indexing3 = a[::-1, :]
        check_jacobians(indexing3)

        # this currently does not work because scalars use the Index deterministic
        # which is special and needs more thought
        indexing2 = a[0]
        check_jacobians(indexing2)

    def test_numpy_deterministics_jacobians(self):

        shape = (2, 3)
        a = Normal('a', mu=zeros(shape), tau=ones(shape) * 5)
        b = Normal('b', mu=zeros(shape), tau=ones(shape))
        c = Uniform('c', lower=ones(shape) * .1, upper=ones(shape) * 10)
        d = Uniform('d', lower=ones(shape) * -1.0, upper=ones(shape) * 1.0)
        e = Normal('e', mu=zeros(shape), tau=ones(shape))
        f = Uniform('c', lower=ones(shape) * 1.0, upper=ones(shape) * 10)

        summing = sum(a, axis=0)
        check_jacobians(summing)

        summing2 = sum(a)
        check_jacobians(summing2)

        absing = abs(a)
        check_jacobians(absing)

        exping = exp(a)
        check_jacobians(exping)

        logging = log(c)
        check_jacobians(logging)

        sqrting = sqrt(c)
        check_jacobians(sqrting)

        sining = sin(a)
        check_jacobians(sining)

        cosing = cos(a)
        check_jacobians(cosing)

        taning = tan(a)
        check_jacobians(taning)

        arcsining = arcsin(d)
        check_jacobians(arcsining)

        arcosing = arccos(d)
        check_jacobians(arcosing)

        arctaning = arctan(d)
        check_jacobians(arctaning)

        sinhing = sinh(a)
        check_jacobians(sinhing)

        coshing = cosh(a)
        check_jacobians(coshing)

        tanhing = tanh(a)
        check_jacobians(tanhing)

        arcsinhing = arcsinh(a)
        check_jacobians(arcsinhing)

        arccoshing = arccosh(f)
        check_jacobians(arccoshing)

        arctanhing = arctanh(d)
        check_jacobians(arctanhing)

        arctan2ing = arctan2(b, e)
        check_jacobians(arctan2ing)

        hypoting = hypot(b, e)
        check_jacobians(hypoting)

    def test_gradients(self):

        shape = (5,)
        a = Normal('a', mu=zeros(shape), tau=ones(shape))
        b = Normal('b', mu=zeros(shape), tau=ones(shape))
        b2 = Normal('b2', mu=2, tau=1.0)
        c = Uniform('c', lower=ones(shape) * .7, upper=ones(shape) * 2.5)
        d = Uniform('d', lower=ones(shape) * .7, upper=ones(shape) * 2.5)
        e = Uniform('e', lower=ones(shape) * .2, upper=ones(shape) * 10)
        f = Uniform('f', lower=ones(shape) * 2, upper=ones(shape) * 30)
        p = Uniform(
            'p',
            lower=zeros(
                shape) + .05,
            upper=ones(
                shape) - .05)
        n = 5

        a.value = 2 * ones(shape)
        b.value = 2.5 * ones(shape)
        b2.value = 2.5

        norm = Normal('norm', mu=a, tau=b)
        check_gradients(norm)

        norm2 = Normal('norm2', mu=0, tau=b2)
        check_gradients(norm2)

        gamma = Gamma('gamma', alpha=a, beta=b)
        check_gradients(gamma)

        bern = Bernoulli('bern', p=p)
        check_gradients(bern)

        beta = Beta('beta', alpha=c, beta=d)

        check_gradients(beta)

        cauchy = Cauchy('cauchy', alpha=a, beta=d)
        check_gradients(cauchy)

        chi2 = Chi2('chi2', nu=e)
        check_gradients(chi2)

        exponential = Exponential('expon', beta=d)
        check_gradients(exponential)

        t = T('t', nu=f)
        check_gradients(t)

        half_normal = HalfNormal('half_normal', tau=e)
        check_gradients(half_normal)

        inverse_gamma = InverseGamma('inverse_gamma', alpha=c, beta=d)
        check_gradients(inverse_gamma)

        laplace = Laplace('laplace', mu=a, tau=c)
        check_gradients(laplace)

        lognormal = Lognormal('lognormal', mu=a, tau=c)
        check_gradients(lognormal)

        weibull = Weibull('weibull', alpha=c, beta=d)
        check_gradients(weibull)

        binomial = Binomial('binomial', p=p, n=n)
        check_gradients(binomial)

        geometric = Geometric('geometric', p=p)
        check_gradients(geometric)

        poisson = Poisson('poisson', mu=c)
        check_gradients(poisson)

        u = Uniform('u', lower=a, upper=b)
        check_gradients(u)

        negative_binomial = NegativeBinomial(
            'negative_binomial',
            mu=c,
            alpha=d)
        check_gradients(negative_binomial)

        # exponweib still not working for unknown reasons
        # exponweib = Exponweib('exponweib', alpha = c, k =d , loc = a, scale = e )
        # check_gradients(exponweib)

    def test_model(self):

        model = generate_model()
        model["sd"].value = 55.0
        model["a"].value = 10.2
        model["b"].value = .5

        check_gradients(model["sd"])

        M = pymc.MCMC(model)
        check_model_gradients(M.stochastics)


if __name__ == '__main__':
    C = nose.config.Config(verbosity=1)
    nose.runmodule(config=C)
