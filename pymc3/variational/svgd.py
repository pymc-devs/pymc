import theano.tensor as tt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pymc3 as pm
from pymc3.model import modelcontext
from tqdm import tqdm

def svgd_kernel(theta, h = -1):
    sq_dist = pdist(theta)
    pairwise_dists = squareform(sq_dist)**2
    if h < 0: # if h < 0, using median trick
        h = np.median(pairwise_dists)
        h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

    # compute the rbf kernel
    Kxy = np.exp( -pairwise_dists / h**2 / 2)

    dxkxy = -np.matmul(Kxy, theta)
    sumkxy = np.sum(Kxy, axis=1)
    for i in range(theta.shape[1]):
        dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i], sumkxy)
    dxkxy = dxkxy / (h**2)
    return (Kxy, dxkxy)


def _svgd_run(x0, lnprob, n_iter=1000, stepsize=1e-3,
              bandwidth=-1, alpha=0.9, progressbar=True,
              kernel=svgd_kernel):

    theta = np.copy(x0)

    # adagrad with momentum
    fudge_factor = 1e-6
    historical_grad = 0

    if progressbar:
        progress = tqdm(np.arange(n_iter))
    else:
        progress = np.arange(n_iter)

    lnpgrad = np.empty_like(theta)
    for i in progress:
        for j, t in enumerate(theta):
            lnpgrad[j, :] = lnprob(t)

        #lnpgrad = lnprob(theta)

        # calculating the kernel matrix
        kxy, dxkxy = kernel(theta, h = -1)
        grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

        # adagrad
        if iter == 0:
            historical_grad = historical_grad + grad_theta ** 2
        else:
            historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
        adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
        theta = theta + stepsize * adj_grad

    return theta

def svgd(n=5000, stepsize=0.01, n_particles=100, jitter=.01, kernel=svgd_kernel,
         start=None, progressbar=True, random_seed=None, model=None):

    if random_seed is not None:
        seed(random_seed)

    model = modelcontext(model)

    if start is None:
        start = model.test_point

    start = model.dict_to_array(start)

    x0 = np.tile(start, (n_particles, 1))
    x0 += np.random.normal(0, jitter, x0.shape)

    theta = _svgd_run(x0, model.dlogp_array,
                      n_iter=n, stepsize=stepsize, kernel=kernel,
                      progressbar=progressbar)

    # Build trade
    strace = pm.backends.NDArray()
    strace.setup(theta.shape[0], 1)

    for p in theta:
        strace.record(model.bijection.rmap(p))
    strace.close()

    trace = pm.backends.base.MultiTrace([strace])

    return trace
