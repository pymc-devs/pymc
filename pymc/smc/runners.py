import multiprocessing as mp

from itertools import repeat

import cloudpickle

from fastprogress.fastprogress import progress_bar


def run_chains_parallel(chains, progressbar, to_run, params, random_seed, kernel_kwargs, cores):
    pbar = progress_bar((), total=100, display=progressbar)
    pbar.update(0)
    pbars = [pbar] + [None] * (chains - 1)

    pool = mp.Pool(cores)

    # "manually" (de)serialize params before/after multiprocessing
    params = tuple(cloudpickle.dumps(p) for p in params)
    kernel_kwargs = {key: cloudpickle.dumps(value) for key, value in kernel_kwargs.items()}
    results = _starmap_with_kwargs(
        pool,
        to_run,
        [(*params, random_seed[chain], chain, pbars[chain]) for chain in range(chains)],
        repeat(kernel_kwargs),
    )
    results = tuple(cloudpickle.loads(r) for r in results)
    pool.close()
    pool.join()
    return results


def run_chains_sequential(chains, progressbar, to_run, params, random_seed, kernel_kwargs):
    results = []
    pbar = progress_bar((), total=100 * chains, display=progressbar)
    pbar.update(0)
    for chain in range(chains):
        pbar.offset = 100 * chain
        pbar.base_comment = f"Chain: {chain + 1}/{chains}"
        results.append(to_run(*params, random_seed[chain], chain, pbar, **kernel_kwargs))
    return results


def _starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    # Helper function to allow kwargs with Pool.starmap
    # Copied from https://stackoverflow.com/a/53173433/13311693
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(_apply_args_and_kwargs, args_for_starmap)


def _apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)
