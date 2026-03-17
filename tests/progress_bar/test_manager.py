#   Copyright 2025 - present The PyMC Developers
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

from unittest.mock import patch

import pytest

import pymc as pm

from pymc.progress_bar import MCMCProgressBarManager, SMCProgressBarManager
from pymc.smc.kernels import IMH

NUTS_DUMMY_STATS = [{"divergences": 0, "step_size": 0.5, "tree_size": 7}]


@pytest.fixture
def imh_kernel():
    with pm.Model():
        pm.Normal("x", 0, 1)
        kernel = IMH()
    return kernel


def _get_task(manager, idx=0):
    return manager._backend._progress.tasks[idx]


def test_mcmc_split_bar_starts_at_zero_ends_at_total():
    draws, tune, chains = 10, 5, 2
    captured = {}

    orig_init = MCMCProgressBarManager.__init__

    def capturing_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        captured["manager"] = self

    with patch.object(MCMCProgressBarManager, "__init__", capturing_init):
        with pm.Model():
            pm.Normal("x")
            pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                progressbar=True,
                compute_convergence_checks=False,
            )

    manager = captured["manager"]
    total = draws + tune
    for chain in range(chains):
        task = _get_task(manager, chain)
        assert task.completed == task.total == total
        assert task.fields["draws"] == total


def test_mcmc_combined_bar_ends_at_total():
    draws, tune, chains = 10, 5, 2
    captured = {}

    orig_init = MCMCProgressBarManager.__init__

    def capturing_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        captured["manager"] = self

    with patch.object(MCMCProgressBarManager, "__init__", capturing_init):
        with pm.Model():
            pm.Normal("x")
            pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                progressbar="combined+stats",
                compute_convergence_checks=False,
            )

    manager = captured["manager"]
    total = (draws + tune) * chains
    task = _get_task(manager, 0)
    assert task.completed == task.total == total
    assert task.fields["draws"] == total


def test_mcmc_draws_stat_shows_completed_count():
    with pm.Model():
        pm.Normal("x")
        step = pm.NUTS()

    manager = MCMCProgressBarManager(step_method=step, chains=1, draws=10, tune=5, progressbar=True)

    with manager:
        manager.update(chain_idx=0, is_last=False, draw=0, tuning=True, stats=NUTS_DUMMY_STATS)
        assert _get_task(manager).fields["draws"] == 1

        for i in range(1, 15):
            manager.update(
                chain_idx=0, is_last=i == 14, draw=i, tuning=i < 5, stats=NUTS_DUMMY_STATS
            )
        assert _get_task(manager).fields["draws"] == 15


def test_smc_bar_starts_at_zero_ends_at_one(imh_kernel):
    manager = SMCProgressBarManager(kernel=imh_kernel, chains=1, progressbar=True)

    with manager:
        task = _get_task(manager)
        assert task.total == 1.0
        assert task.completed == 0
        assert task.percentage == 0.0

        betas = [0.2, 0.5, 0.8, 1.0]
        old_beta = 0.0
        for stage, beta in enumerate(betas):
            manager.update(
                chain_idx=0, stage=stage, beta=beta, old_beta=old_beta, is_last=beta >= 1.0
            )
            old_beta = beta

        task = _get_task(manager)
        assert task.completed == pytest.approx(task.total)
        assert task.fields["beta"] == pytest.approx(1.0)


def test_smc_multi_chain(imh_kernel):
    chains = 3
    manager = SMCProgressBarManager(kernel=imh_kernel, chains=chains, progressbar=True)

    with manager:
        for chain in range(chains):
            assert _get_task(manager, chain).completed == 0

        for chain in range(chains):
            manager.update(chain_idx=chain, stage=0, beta=0.4, old_beta=0.0)
            manager.update(chain_idx=chain, stage=1, beta=1.0, old_beta=0.4, is_last=True)

        for chain in range(chains):
            assert _get_task(manager, chain).completed == pytest.approx(1.0)


def test_progressbar_nested_compound():
    """Regression test for https://github.com/pymc-devs/pymc/issues/7721"""
    with pm.Model():
        a = pm.Poisson("a", mu=10)
        b = pm.Binomial("b", n=a, p=0.8)
        c = pm.Poisson("c", mu=11)
        d = pm.Dirichlet("d", a=[c, b])

        step = pm.CompoundStep(
            [
                pm.CompoundStep([pm.Metropolis(a), pm.Metropolis(b), pm.Metropolis(c)]),
                pm.NUTS([d]),
            ]
        )

        kwargs = {
            "draws": 10,
            "tune": 10,
            "chains": 2,
            "compute_convergence_checks": False,
            "step": step,
        }

        for cores in (1, 2):
            pm.sample(**kwargs, cores=cores, progressbar=True)
            pm.sample(**kwargs, cores=cores, progressbar="combined")
            pm.sample(**kwargs, cores=cores, progressbar="split")
            pm.sample(**kwargs, cores=cores, progressbar=False)
