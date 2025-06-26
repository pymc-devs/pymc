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
"""
WALNUTS (Within-orbit Adaptive Step-length No-U-Turn Sampler) implementation.

Based on WALNUTSpy by Tore Selland Kleppe.
Reference: Bou-Rabee, N., Carpenter, B., Kleppe, T. S., & Liu, S. (2025).
The Within-Orbit Adaptive Leapfrog No-U-Turn Sampler.
arXiv preprint arXiv:2506.18746.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from pymc.stats.convergence import SamplerWarning
from pymc.step_methods.compound import Competence
from pymc.step_methods.hmc.adaptive_integrators import fixedLeapFrog, integratorAuxPar
from pymc.step_methods.hmc.base_hmc import BaseHMC, DivergenceInfo, HMCStepData
from pymc.step_methods.hmc.p2_quantile import P2quantile
from pymc.step_methods.hmc.walnuts_constants import __logZero as _logZero
from pymc.step_methods.hmc.walnuts_constants import __wtSumThresh as _wtSumThresh
from pymc.vartypes import continuous_types

__all__ = ["WALNUTS"]


# Sequence of stop-checks during NUTS iteration (precomputed)
checks = None
k = 0


def subTreePlan(nleaf):
    """Create sequence of U-turn checks for tree building."""
    global checks
    global k
    checks = np.zeros((nleaf - 1, 2), dtype=int)
    k = 0

    def Uturn(a, b):
        global k
        global checks
        checks[k, 0] = a
        checks[k, 1] = b
        k += 1

    def subUturn(a, b):
        if a != b:
            m = math.floor((a + b) / 2)
            subUturn(a, m)
            subUturn(m + 1, b)
            Uturn(a, b)

    subUturn(1, nleaf)
    return checks


class stateStore:
    """Utilities for storing states (avoiding storing the complete orbit)."""

    def __init__(self, d, n):
        self.__stateStack = np.zeros((d, n))
        self.__stateStackId = np.zeros(n, dtype=int)
        self.__stateStackUsed = np.full(n, False, dtype=bool)

    def statePush(self, id_, state):
        full = True
        for i in range(len(self.__stateStackId)):
            if not self.__stateStackUsed[i]:
                self.__stateStack[:, i] = state
                self.__stateStackId[i] = id_
                self.__stateStackUsed[i] = True
                full = False
                break
        if full:
            sys.exit("stack full")

    def stateRead(self, id_):
        inds = np.where(self.__stateStackId == id_)[0]
        if len(inds > 0):
            for i in range(len(inds)):
                if self.__stateStackUsed[inds[i]]:
                    return self.__stateStack[:, inds[i]]
        sys.exit("element not found for read in stack")

    def stateDeleteRange(self, from_, to_):
        self.__stateStackUsed[
            np.logical_and(self.__stateStackId >= from_, self.__stateStackId <= to_)
        ] = False

    def stateReset(self):
        self.__stateStackUsed = np.full(len(self.__stateStackUsed), False, dtype=bool)

    def dump(self):
        """Dump state stack for debugging."""
        return {
            "values": self.__stateStack,
            "ids": self.__stateStackId,
            "used": self.__stateStackUsed,
        }


def stopCondition(qm, vm, qp, vp):
    """NUT stop condition."""
    tmp = qp - qm
    return sum(vp * tmp) < 0.0 or sum(vm * tmp) < 0.0


class WALNUTS(BaseHMC):
    """Within-orbit Adaptive Step-length No-U-Turn Sampler.

    WALNUTS (Bou-Rabee et al., 2025) extends NUTS by adapting the integration step size within
    each trajectory. This can improve numerical stability in models with varying curvature.

    Parameters
    ----------
    vars : list, optional
        Variables to sample. If None, all continuous variables in the model.
    H0 : float, default=0.2
        Initial big step size / fixed big step size if adaptH=False.
    stepSizeRandScale : float, default=0.2
        Step size randomization scale.
    delta0 : float, default=0.05
        Initial integrator tolerance.
    M : int, default=10
        Number of NUTS iterations (maximum tree depth).
    integrator : function, default=fixedLeapFrog
        Integrator function to use.
    igrAux : integratorAuxPar, optional
        Auxiliary parameters for integrators.
    adaptH : bool, default=True
        Whether to adapt the big step size H.
    adaptHtarget : float, default=0.8
        Desired fraction of steps where crudest step size is accepted.
    adaptDelta : bool, default=True
        Whether to adapt the integrator tolerance delta.
    adaptDeltaTarget : float, default=0.6
        Target for delta adaptation.
    adaptDeltaQuantile : float, default=0.9
        Quantile for delta adaptation.
    recordOrbitStats : bool, default=False
        Whether to record orbit statistics.
    **kwargs
        Additional arguments passed to BaseHMC.

    References
    ----------
    .. [1] Bou-Rabee, N., Carpenter, B., Kleppe, T. S., & Liu, S. (2025).
       The Within-Orbit Adaptive Leapfrog No-U-Turn Sampler.
       arXiv preprint arXiv:2506.18746.
       https://arxiv.org/abs/2506.18746v1
    """

    name = "walnuts"

    default_blocked = True

    stats_dtypes_shapes = {
        "L_": (np.int64, []),
        "NdoublSampled_": (np.int64, []),
        "orbitLen_": (np.float64, []),
        "orbitLenSam_": (np.float64, []),
        "maxFint": (np.int64, []),
        "maxBint": (np.int64, []),
        "nevalF": (np.int64, []),
        "nevalB": (np.int64, []),
        "min_Ifs": (np.int64, []),
        "max_Ifs": (np.int64, []),
        "min_lwts": (np.float64, []),
        "max_lwts": (np.float64, []),
        "bothEndsPassive": (bool, []),
        "oneEndPassive": (bool, []),
        "mean_IfsNeqIbs": (np.float64, []),
        "H": (np.float64, []),
        "mean_IfsEq0": (np.float64, []),
        "orbitEnergyError": (np.float64, []),
        "delta": (np.float64, []),
        "stopCode": (np.int64, []),
        "NdoublComputed_": (np.int64, []),
        "min_cs": (np.int64, []),
        "max_cs": (np.int64, []),
        "indexStat_": (np.float64, []),
        # PyMC compatibility fields
        "depth": (np.int64, []),
        "step_size": (np.float64, []),
        "tune": (bool, []),
        "mean_tree_accept": (np.float64, []),
        "step_size_bar": (np.float64, []),
        "tree_size": (np.float64, []),
        "diverging": (bool, []),
        "energy_error": (np.float64, []),
        "energy": (np.float64, []),
        "max_energy_error": (np.float64, []),
        "model_logp": (np.float64, []),
        "process_time_diff": (np.float64, []),
        "perf_counter_diff": (np.float64, []),
        "perf_counter_start": (np.float64, []),
        "largest_eigval": (np.float64, []),
        "smallest_eigval": (np.float64, []),
        "index_in_trajectory": (np.int64, []),
        "reached_max_treedepth": (bool, []),
        "warning": (SamplerWarning, None),
        "n_steps_total": (np.int64, []),
        "avg_steps_per_proposal": (np.float64, []),
    }

    def __init__(
        self,
        vars=None,
        H0=0.2,
        stepSizeRandScale=0.2,
        delta0=0.05,
        M=10,
        integrator=None,
        igrAux=None,
        adaptH=True,
        adaptHtarget=0.8,
        adaptDelta=True,
        adaptDeltaTarget=0.6,
        adaptDeltaQuantile=0.9,
        recordOrbitStats=False,
        max_error=None,
        max_treedepth=None,
        **kwargs,
    ):
        """Initialize WALNUTS sampler."""
        # WALNUTSpy parameters
        self.H = H0
        self.stepSizeRandScale = stepSizeRandScale
        self.delta = delta0
        # Allow max_treedepth to override M for PyMC compatibility
        self.M = max_treedepth if max_treedepth is not None else M
        self.walnuts_integrator = integrator if integrator is not None else fixedLeapFrog
        self.igrAux = igrAux if igrAux is not None else integratorAuxPar()
        self.adaptH = adaptH
        self.adaptHtarget = adaptHtarget
        self.adaptDelta = adaptDelta
        self.adaptDeltaTarget = adaptDeltaTarget
        self.adaptDeltaQuantile = adaptDeltaQuantile
        self.recordOrbitStats = recordOrbitStats

        # For PyMC compatibility
        self.max_treedepth = self.M
        self.early_max_treedepth = min(8, self.M)
        self.max_error = max_error if max_error is not None else delta0

        # Adaptation setup
        if self.adaptH:
            if self.adaptHtarget < 0.0 or self.adaptHtarget > 1.0:
                sys.exit("bad adaptHtarget")
            self.igrConstQ = P2quantile(1.0 - self.adaptHtarget)

        if self.adaptDelta:
            if self.adaptDeltaTarget < 0.0:
                sys.exit("bad adaptDeltaTarget")

        # Make tables for all of the sub-uturn checks
        self.plans = []
        for i in range(0, self.M):
            self.plans.append(subTreePlan(2**i))

        # Initialize parent class with remaining kwargs
        super().__init__(vars, **kwargs)

        # Track iteration count for warmup
        self.iterN = 0
        self.warmupIter = 1000  # Default warmup iterations

        # Energy error tracking for adaptation
        if self.adaptDelta:
            self.energyErrorInfFacs = np.zeros(self.warmupIter)

    def _hamiltonian_step(self, start, p0, step_size):
        """Perform a single WALNUTS iteration."""
        # Use PyMC's step size if provided, otherwise use WALNUTS H
        H = step_size if step_size is not None else self.H

        # Extract position and gradient from PyMC State
        qc = start.q.data
        grad0 = start.q_grad
        d = len(qc)

        # Create lpFun wrapper for PyMC compatibility
        def lpFun(q):
            # Create a new State with updated position
            new_q = start.q._replace(data=q)
            new_state = self.integrator.compute_state(new_q, p0)
            return new_state.model_logp, new_state.q_grad

        # Track iteration
        self.iterN += 1
        warmup = self.tune and self.iterN <= self.warmupIter

        # Per iteration diagnostics info
        nevalF = 0
        nevalB = 0
        Lold_ = 0
        L_ = 0
        orbitLen_ = 0.0
        orbitLenSam_ = 0.0
        NdoublSampled_ = 0
        NdoublComputed_ = 0
        indexStat_ = 0.0
        indexStatOld_ = 0.0
        timeLenF_ = 0.0
        timeLenB_ = 0.0

        # Integration directions: 1=backward, 0=forward
        B = np.floor(self.rng.uniform(low=0.0, high=2.0, size=self.M)).astype(int)

        # How many backward steps could there possibly be
        nleft = sum(B * (2 ** np.arange(0, self.M)))

        # Allocate memory for intermediate states
        states = stateStore(3 * d, 2 * (self.M + 1) + 1)

        # Memory for quantities stored for all states in orbit
        Hs = np.zeros(2**self.M)
        Ifs = np.zeros(2**self.M, dtype=int)
        Ibs = np.zeros(2**self.M, dtype=int)
        cs = np.zeros(2**self.M, dtype=int)
        lwts = np.zeros(2**self.M)

        # I0 is the index of the zeroth state
        I0 = nleft
        Ifs[I0] = 0
        Ibs[I0] = 0
        cs[I0] = 0
        lwts[I0] = 0.0

        # Endpoints of accepted and proposed trajectory
        a = 0
        b = 0
        maxFint = 0
        maxBint = 0

        # Full momentum refresh
        v = self.rng.normal(size=d)

        # Endpoints of current orbit (p=forward, m=backward)
        qp = qc
        qm = qc
        vp = v
        vm = v

        # Current proposal
        qProp = qc
        qPropLast = qc

        # Evaluate at current state
        f0, grad0 = lpFun(qc)

        # Gradients at either endpoint
        gp = grad0
        gm = grad0

        # Hamiltonian at initial point
        Hs[I0] = -f0 + 0.5 * sum(v**2)

        # Index selection-related quantities
        multinomialLscale = Hs[I0]
        WoldSum = 1.0

        lwtSumb = 0.0
        lwtSumf = 0.0

        # Reject orbit if numerical problems occur
        forcedReject = False

        # Stop if both multinomial bias at both ends are zero
        bothEndsPassive = False
        stopCode = 0

        # NUT iteration loop
        for i in range(self.M):
            # Integration direction
            xi = (-1) ** B[i]
            # Proposed new endpoints
            at = a + xi * (2**i)
            bt = b + xi * (2**i)

            # More bookkeeping
            expandFurther = True
            qPropLast = qProp
            Lold_ = L_
            indexStatOld_ = indexStat_

            if i == 0:  # Single first integration step required
                HLoc = self.rng.uniform(
                    low=H * (1 - self.stepSizeRandScale),
                    high=H * (1 + self.stepSizeRandScale),
                    size=1,
                )[0]
                orbitLen_ += HLoc

                if xi == 1:  # Forward integration
                    intOut = self.walnuts_integrator(
                        qp, vp, gp, Hs[I0], HLoc, xi, lpFun, self.delta, self.igrAux
                    )
                    qp = intOut.q
                    vp = intOut.v
                    gp = intOut.grad
                    nevalF += intOut.nEvalF
                    nevalB += intOut.nEvalB
                    Hs[I0 + 1] = -intOut.lp + 0.5 * sum(vp * vp)
                    Ifs[I0 + 1] = intOut.If
                    Ibs[I0 + 1] = intOut.Ib
                    cs[I0 + 1] = intOut.c
                    lwts[I0 + 1] = intOut.lwt
                    if warmup and self.adaptH:
                        self.igrConstQ.push(np.log(intOut.igrConst))
                    maxFint = 1
                    timeLenF_ = HLoc
                    if not np.isfinite(Hs[I0 + 1]):
                        forcedReject = True
                        stopCode = 999
                        break

                    lwtSumf = lwts[I0 + 1]
                    Wnew = np.exp(-Hs[I0 + 1] + multinomialLscale + lwtSumf)

                    qProp = qp
                    L_ = 1
                    indexStat_ = timeLenF_

                else:  # Backward integration
                    intOut = self.walnuts_integrator(
                        qm, vm, gm, Hs[I0], HLoc, xi, lpFun, self.delta, self.igrAux
                    )
                    qm = intOut.q
                    vm = intOut.v
                    gm = intOut.grad
                    nevalF += intOut.nEvalF
                    nevalB += intOut.nEvalB
                    Hs[I0 - 1] = -intOut.lp + 0.5 * sum(vm * vm)
                    Ifs[I0 - 1] = intOut.If
                    Ibs[I0 - 1] = intOut.Ib
                    cs[I0 - 1] = intOut.c
                    lwts[I0 - 1] = intOut.lwt
                    if warmup and self.adaptH:
                        self.igrConstQ.push(np.log(intOut.igrConst))
                    maxBint = -1
                    timeLenB_ = HLoc
                    if not np.isfinite(Hs[I0 - 1]):
                        forcedReject = True
                        stopCode = 999
                        break
                    lwtSumb = lwts[I0 - 1]
                    Wnew = np.exp(-Hs[I0 - 1] + multinomialLscale + lwtSumb)

                    qProp = qm
                    L_ = -1
                    indexStat_ = -timeLenB_

                WoldSum = 1.0
                WnewSum = Wnew

            else:  # More than a single integration step, these require sub-u-turn checks
                # Work out which sub-u-turn-checks we are doing
                plan = 0
                if xi == 1:
                    plan = b + self.plans[i]
                else:
                    plan = a - self.plans[i]

                WnewSum = 0.0

                for j in range(len(plan)):  # Loop over U-turn-checks
                    if abs(plan[j, 0] - plan[j, 1]) == 1:  # New integration steps needed
                        HLoc1 = self.rng.uniform(
                            low=H * (1 - self.stepSizeRandScale),
                            high=H * (1 + self.stepSizeRandScale),
                            size=2,
                        )

                        if xi == -1:  # Backward integration
                            i1 = plan[j, 0]
                            intOut = self.walnuts_integrator(
                                qm,
                                vm,
                                gm,
                                Hs[I0 + i1 + 1],
                                HLoc1[0],
                                xi,
                                lpFun,
                                self.delta,
                                self.igrAux,
                            )
                            qm = intOut.q
                            vm = intOut.v
                            gm = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0 + i1] = -intOut.lp + 0.5 * sum(vm * vm)
                            Ifs[I0 + i1] = intOut.If
                            Ibs[I0 + i1] = intOut.Ib
                            cs[I0 + i1] = intOut.c
                            lwts[I0 + i1] = intOut.lwt
                            if warmup and self.adaptH:
                                self.igrConstQ.push(np.log(intOut.igrConst))
                            maxBint = i1
                            timeLenB_ += HLoc1[0]
                            if not np.isfinite(Hs[I0 + i1]):
                                forcedReject = True
                                stopCode = 999
                                break

                            lwtSumb += lwts[I0 + i1]

                            Wnew = np.exp(-Hs[I0 + i1] + multinomialLscale + lwtSumb)
                            WnewSum += Wnew

                            # Online categorical sampling
                            if WnewSum > _wtSumThresh and self.rng.uniform() < Wnew / WnewSum:
                                qProp = qm
                                L_ = i1
                                indexStat_ = -timeLenB_

                            states.statePush(i1, np.concatenate([qm, vm, gm]))
                            orbitLen_ += HLoc1[0]

                            qtmp = qm
                            vtmp = vm

                            # Second integration step
                            i2 = plan[j, 1]
                            intOut = self.walnuts_integrator(
                                qm,
                                vm,
                                gm,
                                Hs[I0 + i2 + 1],
                                HLoc1[1],
                                xi,
                                lpFun,
                                self.delta,
                                self.igrAux,
                            )
                            qm = intOut.q
                            vm = intOut.v
                            gm = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0 + i2] = -intOut.lp + 0.5 * sum(vm * vm)
                            Ifs[I0 + i2] = intOut.If
                            Ibs[I0 + i2] = intOut.Ib
                            cs[I0 + i2] = intOut.c
                            lwts[I0 + i2] = intOut.lwt
                            if warmup and self.adaptH:
                                self.igrConstQ.push(np.log(intOut.igrConst))
                            maxBint = i2
                            timeLenB_ += HLoc1[1]
                            if not np.isfinite(Hs[I0 + i2]):
                                forcedReject = True
                                break

                            # Online categorical sampling
                            Wnew = np.exp(-Hs[I0 + i2] + multinomialLscale + lwtSumb)
                            WnewSum += Wnew
                            if WnewSum > _wtSumThresh and self.rng.uniform() < Wnew / WnewSum:
                                qProp = qm
                                L_ = i2
                                indexStat_ = -timeLenB_

                            # Store state for future u-turn-checking
                            states.statePush(i2, np.concatenate([qm, vm, gm]))
                            orbitLen_ += HLoc1[1]

                            # Uturn check
                            if stopCondition(qm, vm, qtmp, vtmp):
                                expandFurther = False
                                break

                        else:  # Forward integration
                            i1 = plan[j, 0]
                            intOut = self.walnuts_integrator(
                                qp,
                                vp,
                                gp,
                                Hs[I0 + i1 - 1],
                                HLoc1[0],
                                xi,
                                lpFun,
                                self.delta,
                                self.igrAux,
                            )
                            qp = intOut.q
                            vp = intOut.v
                            gp = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0 + i1] = -intOut.lp + 0.5 * sum(vp * vp)
                            Ifs[I0 + i1] = intOut.If
                            Ibs[I0 + i1] = intOut.Ib
                            cs[I0 + i1] = intOut.c
                            lwts[I0 + i1] = intOut.lwt
                            if warmup and self.adaptH:
                                self.igrConstQ.push(np.log(intOut.igrConst))
                            maxFint = i1
                            timeLenF_ += HLoc1[0]
                            if not np.isfinite(Hs[I0 + i1]):
                                forcedReject = True
                                stopCode = 999
                                break

                            lwtSumf += lwts[I0 + i1]

                            # Online categorical sampling
                            Wnew = np.exp(-Hs[I0 + i1] + multinomialLscale + lwtSumf)
                            WnewSum += Wnew
                            if WnewSum > _wtSumThresh and self.rng.uniform() < Wnew / WnewSum:
                                qProp = qp
                                L_ = i1
                                indexStat_ = timeLenF_

                            # Store state for future u-turn-checking
                            states.statePush(i1, np.concatenate([qp, vp, gp]))
                            orbitLen_ += HLoc1[0]

                            qtmp = qp
                            vtmp = vp

                            # Second integration step
                            i2 = plan[j, 1]
                            intOut = self.walnuts_integrator(
                                qp,
                                vp,
                                gp,
                                Hs[I0 + i2 - 1],
                                HLoc1[1],
                                xi,
                                lpFun,
                                self.delta,
                                self.igrAux,
                            )
                            qp = intOut.q
                            vp = intOut.v
                            gp = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0 + i2] = -intOut.lp + 0.5 * sum(vp * vp)
                            Ifs[I0 + i2] = intOut.If
                            Ibs[I0 + i2] = intOut.Ib
                            cs[I0 + i2] = intOut.c
                            lwts[I0 + i2] = intOut.lwt
                            if warmup and self.adaptH:
                                self.igrConstQ.push(np.log(intOut.igrConst))
                            maxFint = i2
                            timeLenF_ += HLoc1[1]
                            if not np.isfinite(Hs[I0 + i2]):
                                forcedReject = True
                                break

                            # Multinomial/progressive sampling
                            lwtSumf += lwts[I0 + i2]

                            Wnew = np.exp(-Hs[I0 + i2] + multinomialLscale + lwtSumf)
                            WnewSum += Wnew
                            if WnewSum > _wtSumThresh and self.rng.uniform() < Wnew / WnewSum:
                                qProp = qp
                                L_ = i2
                                indexStat_ = timeLenF_

                            # Store state for future u-turn-checking
                            states.statePush(i2, np.concatenate([qp, vp, gp]))
                            orbitLen_ += HLoc1[1]

                            if stopCondition(qtmp, vtmp, qp, vp):
                                expandFurther = False
                                break
                        # Done forward integration
                    else:  # No new integration steps needed, only U-turn checks
                        # Delete states not needed further
                        im = min(plan[j, :])
                        ip = max(plan[j, :])
                        states.stateDeleteRange(im + 1, ip - 1)
                        statep = states.stateRead(ip)
                        statem = states.stateRead(im)

                        if stopCondition(
                            statem[0:d], statem[d : (2 * d)], statep[0:d], statep[d : (2 * d)]
                        ):
                            expandFurther = False
                            break
                # Done loop over j

            if forcedReject:
                break

            indexStat_ = indexStat_ / (timeLenF_ + timeLenB_)

            if not expandFurther:
                # Proposed subOrbit had a sub-U-turn
                qProp = qPropLast
                L_ = Lold_
                indexStat_ = indexStatOld_
                NdoublSampled_ = i
                NdoublComputed_ = i + 1
                stopCode = 5
                break
            else:
                # The proposed sub-orbit was found to be free of u-turns
                # Now check if proposed state should be from old or new sub-orbit
                if not (self.rng.uniform() < WnewSum / WoldSum):
                    L_ = Lold_
                    indexStat_ = indexStatOld_
                    qProp = qPropLast

                # Proposed suborbit free of U-turns
                # Final U-turn check
                joinedCrit = stopCondition(qm, vm, qp, vp)
                # Stop simulation if multinomial weights at either end are effectively zero
                bothEndsPassive = lwtSumb < _logZero + 1.0 and lwtSumf < _logZero + 1.0
                if joinedCrit or bothEndsPassive:
                    if joinedCrit:
                        stopCode = 4
                    else:
                        stopCode = -4

                    NdoublSampled_ = i + 1
                    NdoublComputed_ = i + 1
                    orbitLenSam_ = orbitLen_
                    break

            # From now on, it is clear that a new doubling will be attempted
            WoldSum += WnewSum

            orbitLenSam_ = orbitLen_
            NdoublSampled_ = i + 1
            NdoublComputed_ = i + 1
            a = min(a, at)
            b = max(b, bt)
            states.stateReset()

        # Done NUTS loop

        # Store samples and diagnostics info
        if maxBint < 0 and maxFint > 0:
            usedSteps = np.r_[maxBint:0, 1 : (maxFint + 1)]
        elif maxBint < 0:
            usedSteps = np.r_[maxBint:0]
        else:
            usedSteps = np.r_[1 : (maxFint + 1)]

        enUsedSteps = np.r_[0, usedSteps]
        orbitEnergyError = np.max(Hs[I0 + enUsedSteps]) - np.min(Hs[I0 + enUsedSteps])

        # Tuning parameter adaptation
        if warmup:
            # Tuning of local error threshold delta
            if self.adaptDelta:
                self.energyErrorInfFacs[self.iterN - 1] = orbitEnergyError / self.delta
            if self.adaptDelta and self.iterN > 10:
                self.delta = self.adaptDeltaTarget / np.quantile(
                    self.energyErrorInfFacs[0 : self.iterN], self.adaptDeltaQuantile
                )

            # Tuning of big step size H
            if self.adaptH and self.igrConstQ.npush > 10:
                self.H = ((self.delta) ** (1.0 / 3.0)) * np.exp(self.igrConstQ.quantile())

        # Create final state for PyMC
        qc = qProp
        new_q = start.q._replace(data=qc)
        final_state = self.integrator.compute_state(new_q, p0)

        # Prepare statistics
        divergence_info = None
        if forcedReject:
            divergence_info = DivergenceInfo(
                "Numerical problems in WALNUTS",
                None,
                final_state,
                None,
            )

        stats = {
            # WALNUTSpy statistics
            "L_": L_,
            "NdoublSampled_": NdoublSampled_,
            "orbitLen_": orbitLen_,
            "orbitLenSam_": orbitLenSam_,
            "maxFint": maxFint,
            "maxBint": maxBint,
            "nevalF": nevalF,
            "nevalB": nevalB,
            "min_Ifs": np.min(Ifs[I0 + usedSteps]) if len(usedSteps) > 0 else 0,
            "max_Ifs": np.max(Ifs[I0 + usedSteps]) if len(usedSteps) > 0 else 0,
            "min_lwts": np.min(lwts[I0 + usedSteps]) if len(usedSteps) > 0 else 0.0,
            "max_lwts": np.max(lwts[I0 + usedSteps]) if len(usedSteps) > 0 else 0.0,
            "bothEndsPassive": bothEndsPassive,
            "oneEndPassive": lwtSumb < _logZero + 1.0 or lwtSumf < _logZero + 1.0,
            "mean_IfsNeqIbs": (
                np.mean(Ifs[I0 + usedSteps] != Ibs[I0 + usedSteps]) if len(usedSteps) > 0 else 0.0
            ),
            "H": self.H,
            "mean_IfsEq0": np.mean(Ifs[I0 + usedSteps] == 0) if len(usedSteps) > 0 else 0.0,
            "orbitEnergyError": orbitEnergyError,
            "delta": self.delta,
            "stopCode": stopCode,
            "NdoublComputed_": NdoublComputed_,
            "min_cs": np.min(cs[I0 + usedSteps]) if len(usedSteps) > 0 else 0,
            "max_cs": np.max(cs[I0 + usedSteps]) if len(usedSteps) > 0 else 0,
            "indexStat_": indexStat_,
            # PyMC compatibility statistics
            "depth": NdoublSampled_,
            "step_size": self.H,
            "tune": self.tune,
            "mean_tree_accept": np.exp(-orbitEnergyError),  # Approximation
            "step_size_bar": self.H,
            "tree_size": nevalF + nevalB,
            "diverging": forcedReject,
            "energy_error": orbitEnergyError,
            "energy": final_state.energy,
            "max_energy_error": orbitEnergyError,
            "model_logp": final_state.model_logp,
            "index_in_trajectory": final_state.index_in_trajectory,
            "reached_max_treedepth": NdoublSampled_ >= self.M,
            "n_steps_total": nevalF + nevalB,
            "avg_steps_per_proposal": (nevalF + nevalB) / max(1, NdoublSampled_),
            "largest_eigval": np.nan,
            "smallest_eigval": np.nan,
        }

        return HMCStepData(final_state, 1, divergence_info, stats)

    @staticmethod
    def competence(var, has_grad):
        """Check if WALNUTS can sample this variable."""
        if var.dtype in continuous_types and has_grad:
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE
