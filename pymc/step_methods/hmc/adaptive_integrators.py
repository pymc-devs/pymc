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
Adaptive integrators for WALNUTS sampler.

Based on adaptiveIntegrators.py from WALNUTSpy by Tore Selland Kleppe.
"""

import sys

import numpy as np

from pymc.step_methods.hmc.walnuts_constants import __logZero


class integratorReturn:
    """Common return object for integrators."""

    def __init__(self, q1, v1, lp1, grad1, nEvalF, nEvalB, If, Ib, c, lwt, igrConst):
        self.q = q1
        self.v = v1
        self.lp = lp1
        self.grad = grad1
        self.nEvalF = nEvalF
        self.nEvalB = nEvalB
        self.If = If
        self.Ib = Ib
        self.c = c
        self.lwt = lwt
        self.igrConst = igrConst

    def __str__(self):
        return (
            "dim: "
            + str(len(self.q))
            + " If: "
            + str(self.If)
            + " Ib: "
            + str(self.Ib)
            + " c: "
            + str(self.c)
        )


class integratorAuxPar:
    """Class for passing assorted tuning parameters to the integrators."""

    def __init__(
        self,
        minC=0,
        maxC=10,
        R2Pprob0=2.0 / 3.0,
        maxFPiter=30,
        FPtol=1.0e-8,
        FPNewton=False,
        rescaledGradThresh=5.0,
    ):
        self.minC = minC
        self.maxC = maxC
        self.R2Pprob0 = R2Pprob0
        self.maxFPiter = maxFPiter
        self.FPtol = FPtol
        self.FPNewton = FPNewton
        self.rescaledGradThresh = rescaledGradThresh


def fixedLeapFrog(q, v, g, Ham0, h, xi, lpFun, delta, auxPar):
    """Implement basic leapfrog integration step."""
    vh = xi * v + 0.5 * h * g
    qq = q + h * vh
    fnew, gnew = lpFun(qq)
    vv = vh + 0.5 * h * gnew

    H1 = -fnew + 0.5 * sum(vv * vv)

    return integratorReturn(
        qq,
        xi * vv,
        fnew,
        gnew,
        1,
        0,
        0,
        0,
        0,
        0.0,
        h * (max(1.0e-10, abs(Ham0 - H1)) ** (-1.0 / 3.0)),
    )


def adaptLeapFrogD(q, v, g, Ham0, h, xi, lpFun, delta, auxPar):
    """Adaptive Leap Frog with Deterministic choice of precision parameter."""
    nEvalF = 0
    If = auxPar.maxC
    for c in range(auxPar.minC, auxPar.maxC + 1):
        nstep = 2**c
        hh = h / nstep
        qq = q
        vv = xi * v
        gg = g
        Hams = np.zeros(nstep + 1)
        Hams[0] = Ham0

        for i in range(1, nstep + 1):
            vh = vv + 0.5 * hh * gg
            qq = qq + hh * vh
            fnew, gg = lpFun(qq)
            nEvalF += 1
            vv = vh + 0.5 * hh * gg
            Hams[i] = -fnew + 0.5 * sum(vv * vv)

        maxErr = abs(Hams[0] - Hams[-1])

        if all(np.isfinite(Hams)) and maxErr < delta:
            If = c
            break

    qOut = qq
    vOut = vv
    fOut = fnew
    gOut = gg

    igrConst = hh * (np.max(np.abs(np.diff(Hams))) ** (-1.0 / 3.0))

    Ib = If
    nEvalB = 0

    if If > auxPar.minC:
        H0b = Hams[-1]
        for c in range(auxPar.minC, If):
            nstep = 2**c
            hh = h / nstep
            qq = qOut
            vv = -vOut
            gg = gOut
            Hams = np.zeros(nstep + 1)
            Hams[0] = H0b

            for i in range(1, nstep + 1):
                vh = vv + 0.5 * hh * gg
                qq = qq + hh * vh
                fnew, gg = lpFun(qq)
                nEvalB += 1
                vv = vh + 0.5 * hh * gg
                Hams[i] = -fnew + 0.5 * sum(vv * vv)

            maxErr = abs(Hams[0] - Hams[-1])
            if all(np.isfinite(Hams)) and maxErr < delta:
                Ib = c
                break

    return integratorReturn(
        qOut, xi * vOut, fOut, gOut, nEvalF, nEvalB, If, Ib, If, (If != Ib) * __logZero, igrConst
    )


def adaptLeapFrogFlowD(q, v, g, Ham0, h, xi, lpFun, delta, auxPar):
    """Adaptive Leap Frog with Deterministic choice of precision parameter and flow-based error criterion."""
    nEvalF = 0
    If = auxPar.maxC
    for c in range(0, auxPar.maxC + 1):
        nstep = 2**c
        hh = h / nstep
        qq = q
        vv = xi * v
        gg = g
        Hams = np.zeros(nstep + 1)
        Errs = np.zeros(nstep)
        Hams[0] = Ham0

        for i in range(1, nstep + 1):
            vh = vv + 0.5 * hh * gg
            qold = qq
            gold = gg
            vold = vv
            qq = qq + hh * vh
            fnew, gg = lpFun(qq)
            nEvalF += 1
            vv = vh + 0.5 * hh * gg

            qMid = 0.5 * (qq + qold) + (hh / 8.0) * (vold - vv)
            fMid, gMid = lpFun(qMid)
            nEvalF += 1

            qf = qold + hh * vold + hh * hh * ((1.0 / 6.0) * gold + (1.0 / 3.0) * gMid)
            err = np.max(np.abs(qf - qq))

            vf = vold + (hh / 6.0) * (gold + gg + 4.0 * gMid)
            err = max(err, np.max(np.abs(vf - vv)))

            qb = qq - hh * vv + hh * hh * ((1.0 / 6.0) * gg + (1.0 / 3.0) * gMid)
            err = max(err, np.max(np.abs(qb - qold)))

            vb = -(-(vv + (hh / 6.0) * (gold + gg + 4.0 * gMid)))

            err = max(err, np.max(np.abs(vb - vold)))
            Errs[i - 1] = err
            Hams[i] = -fnew + 0.5 * sum(vv * vv)

        maxErr = np.max(Errs)

        if all(np.isfinite(Hams)) and maxErr < delta:
            If = c
            break

    qOut = qq
    vOut = vv
    fOut = fnew
    gOut = gg

    igrConst = hh * (np.max(np.abs(np.diff(Hams))) ** (-1.0 / 3.0))

    Ib = If
    nEvalB = 0

    if If > 0:
        H0b = Hams[-1]
        for c in range(0, If):
            nstep = 2**c
            hh = h / nstep
            qq = qOut
            vv = -vOut
            gg = gOut
            Hams = np.zeros(nstep + 1)
            Hams[0] = H0b
            Errs = np.zeros(nstep)

            for i in range(1, nstep + 1):
                vh = vv + 0.5 * hh * gg
                qold = qq
                gold = gg
                vold = vv
                qq = qq + hh * vh
                fnew, gg = lpFun(qq)
                nEvalB += 1
                vv = vh + 0.5 * hh * gg

                qMid = 0.5 * (qq + qold) + (hh / 8.0) * (vold - vv)
                fMid, gMid = lpFun(qMid)
                nEvalB += 1
                qf = qold + hh * vold + hh * hh * ((1.0 / 6.0) * gold + (1.0 / 3.0) * gMid)
                err = np.max(np.abs(qf - qq))
                vf = vold + (hh / 6.0) * (gold + gg + 4.0 * gMid)
                err = max(err, np.max(np.abs(vf - vv)))
                qb = qq - hh * vv + hh * hh * ((1.0 / 6.0) * gg + (1.0 / 3.0) * gMid)
                err = max(err, np.max(np.abs(qb - qold)))
                vb = -(-(vv + (hh / 6.0) * (gold + gg + 4.0 * gMid)))

                err = max(err, np.max(np.abs(vb - vold)))
                Errs[i - 1] = err
                Hams[i] = -fnew + 0.5 * sum(vv * vv)

            maxErr = np.max(Errs)
            if all(np.isfinite(Hams)) and maxErr < delta:
                Ib = c
                break

    return integratorReturn(
        qOut, xi * vOut, fOut, gOut, nEvalF, nEvalB, If, Ib, If, (If != Ib) * __logZero, igrConst
    )


def adaptLeapFrogR2P(q, v, g, Ham0, h, xi, lpFun, delta, auxPar):
    """Adaptive Leap Frog with Randomized-to-Probabilistic choice."""
    nEvalF = 0
    If = auxPar.maxC
    for c in range(auxPar.minC, auxPar.maxC + 1):
        nstep = 2**c
        hh = h / nstep
        qq = q
        vv = xi * v
        gg = g
        Hams = np.zeros(nstep + 1)
        Hams[0] = Ham0

        for i in range(1, nstep + 1):
            vh = vv + 0.5 * hh * gg
            qq = qq + hh * vh
            fnew, gg = lpFun(qq)
            nEvalF += 1
            vv = vh + 0.5 * hh * gg
            Hams[i] = -fnew + 0.5 * sum(vv * vv)

        maxErr = abs(Hams[0] - Hams[-1])

        if all(np.isfinite(Hams)) and maxErr < delta:
            If = c
            break

    if np.random.uniform() < auxPar.R2Pprob0:
        # simulation occur at minimal accepted precision
        qOut = qq
        vOut = vv
        fOut = fnew
        gOut = gg
        cSim = If
        igrConst = hh * (np.max(np.abs(np.diff(Hams))) ** (-1.0 / 3.0))
    else:
        # simulation occur at minimal + 1
        c = If + 1
        nstep = 2**c
        hh = h / nstep
        qq = q
        vv = xi * v
        gg = g
        Hams = np.zeros(nstep + 1)
        Hams[0] = Ham0

        for i in range(1, nstep + 1):
            vh = vv + 0.5 * hh * gg
            qq = qq + hh * vh
            fnew, gg = lpFun(qq)
            nEvalF += 1
            vv = vh + 0.5 * hh * gg
            Hams[i] = -fnew + 0.5 * sum(vv * vv)

        qOut = qq
        vOut = vv
        fOut = fnew
        gOut = gg
        cSim = If + 1
        igrConst = hh * (np.max(np.abs(np.diff(Hams))) ** (-1.0 / 3.0))

    # done forward simulation pass, now do backward simulations
    nEvalB = 0

    if cSim == If:
        maxTry = If - 1
        Ib = If
        lwtf = np.log(auxPar.R2Pprob0)
    else:
        maxTry = auxPar.maxC
        Ib = auxPar.maxC
        lwtf = np.log(1.0 - auxPar.R2Pprob0)

    if maxTry >= auxPar.minC:
        H0b = Hams[-1]
        for c in range(auxPar.minC, maxTry + 1):
            nstep = 2**c
            hh = h / nstep
            qq = qOut
            vv = -vOut
            gg = gOut
            Hams = np.zeros(nstep + 1)
            Hams[0] = H0b

            for i in range(1, nstep + 1):
                vh = vv + 0.5 * hh * gg
                qq = qq + hh * vh
                fnew, gg = lpFun(qq)
                nEvalB += 1
                vv = vh + 0.5 * hh * gg
                Hams[i] = -fnew + 0.5 * sum(vv * vv)

            maxErr = abs(Hams[0] - Hams[-1])
            if all(np.isfinite(Hams)) and maxErr < delta:
                Ib = c
                break

    # done backward simulation pass, now work out backward probability
    lwtb = __logZero
    if cSim == Ib:
        lwtb = np.log(auxPar.R2Pprob0)
    elif cSim == Ib + 1:
        lwtb = np.log(1.0 - auxPar.R2Pprob0)

    return integratorReturn(
        qOut, xi * vOut, fOut, gOut, nEvalF, nEvalB, If, Ib, cSim, lwtb - lwtf, igrConst
    )


def adaptImplicitMidpointD(q, v, g, Ham0, h, xi, lpFun, delta, auxPar):
    """Adaptive Implicit Midpoint integrator."""
    nEvalF = 0
    If = auxPar.maxC
    for c in range(0, auxPar.maxC + 1):
        nstep = 2**c
        hh = h / nstep
        qq = q
        vv = xi * v
        gg = g
        Hams = np.zeros(nstep + 1)
        Hams[0] = Ham0
        numCompleted = 0
        for i in range(1, nstep + 1):
            # initial guess based on leap frog
            qt = qq + hh * (vv + 0.5 * hh * gg)

            # controls for fixed point iterations
            converged = False
            oldMaxErr = 1.0e100

            for iter in range(1, auxPar.maxFPiter + 1):
                mpq = 0.5 * (qt + qq)
                if auxPar.FPNewton:
                    fmp, gmp, Hmp = lpFun(mpq, True)
                    HH = 0.25 * hh * hh * Hmp - np.identity(len(gmp))
                    qtNew = qt - np.linalg.solve(HH, qq + hh * vv + (0.5 * hh * hh) * gmp - qt)
                else:
                    fmp, gmp = lpFun(mpq)
                    qtNew = qq + hh * vv + (0.5 * hh * hh) * gmp

                nEvalF += 1
                maxErr = np.max(np.abs(qtNew - qt))
                qt = qtNew
                if maxErr < auxPar.FPtol:
                    converged = True
                    break

                if maxErr > 1.1 * oldMaxErr:
                    break
                oldMaxErr = maxErr

            if not converged:
                break

            # step used and evaluation at mesh times
            mpq = 0.5 * (qt + qq)

            fmp, gmp = lpFun(mpq)
            nEvalF += 1
            qq = qq + hh * vv + (0.5 * hh * hh) * gmp
            vv = vv + hh * gmp

            fnew, gg = lpFun(qq)
            nEvalF += 1
            Hams[i] = -fnew + 0.5 * sum(vv * vv)
            numCompleted += 1

        maxHErr = abs(Hams[0] - Hams[-1])
        if all(np.isfinite(Hams)) and maxHErr < delta and numCompleted == nstep:
            If = c
            break

    if not converged:
        import warnings

        warnings.warn("Numerical problems in adaptImplicitMidpoint, consider increasing maxC")
        sys.exit()

    qOut = qq
    vOut = vv
    fOut = fnew
    gOut = gg

    igrConst = hh * (np.max(np.abs(np.diff(Hams))) ** (-1.0 / 3.0))

    Ib = auxPar.maxC
    nEvalB = 0

    Hb0 = -fOut + 0.5 * sum(vOut * vOut)
    for c in range(0, auxPar.maxC + 1):
        nstep = 2**c
        hh = h / nstep
        qq = qOut
        vv = -vOut
        gg = gOut
        Hams = np.zeros(nstep + 1)
        Hams[0] = Hb0
        numCompleted = 0
        for i in range(1, nstep + 1):
            # initial guess based on leap frog
            qt = qq + hh * (vv + 0.5 * hh * gg)

            # controls for fixed point iterations
            converged = False
            oldMaxErr = 1.0e100

            for iter in range(1, auxPar.maxFPiter + 1):
                mpq = 0.5 * (qt + qq)
                if auxPar.FPNewton:
                    fmp, gmp, Hmp = lpFun(mpq, True)
                    HH = 0.25 * hh * hh * Hmp - np.identity(len(gmp))
                    qtNew = qt - np.linalg.solve(HH, qq + hh * vv + (0.5 * hh * hh) * gmp - qt)
                else:
                    fmp, gmp = lpFun(mpq)
                    qtNew = qq + hh * vv + (0.5 * hh * hh) * gmp

                nEvalB += 1
                maxErr = np.max(np.abs(qtNew - qt))
                qt = qtNew

                if maxErr < auxPar.FPtol:
                    converged = True
                    break

                if maxErr > 1.1 * oldMaxErr:
                    break
                oldMaxErr = maxErr

            if not converged:
                break

            # step used
            mpq = 0.5 * (qt + qq)
            fmp, gmp = lpFun(mpq)
            nEvalB += 1
            qq = qq + hh * vv + (0.5 * hh * hh) * gmp
            vv = vv + hh * gmp

            fnew, gg = lpFun(qq)
            nEvalB += 1
            Hams[i] = -fnew + 0.5 * sum(vv * vv)
            numCompleted += 1

        maxHErr = abs(Hams[0] - Hams[-1])
        if all(np.isfinite(Hams)) and maxHErr < delta and numCompleted == nstep:
            Ib = c
            break

    return integratorReturn(
        qOut, xi * vOut, fOut, gOut, nEvalF, nEvalB, If, Ib, If, (If != Ib) * __logZero, igrConst
    )


def rescaledLeapFrogTest(q, v, g, Ham0, h, xi, lpFun, delta, auxPar):
    """Test rescaled leapfrog integrator."""
    d = len(q)
    Sd = np.exp(np.random.normal(scale=0.5, size=d))

    vv = xi * v
    qb = q / Sd
    gb = Sd * g
    vh = vv + 0.5 * h * gb
    qbn = qb + h * vh
    fout, g = lpFun(qbn * Sd)
    vout = vh + 0.5 * h * Sd * g
    return integratorReturn(qbn * Sd, xi * vout, fout, g, 1, 0, 0, 0, 0, 0, 1.0)


def adaptRescaledLeapFrogD(q, v, g, Ham0, h, xi, lpFun, delta, auxPar):
    """Adaptive Rescaled Leap Frog with Deterministic choice."""
    gradThresh = auxPar.rescaledGradThresh
    d = len(q)
    Sd = np.ones(d)
    Sred = np.zeros(d, dtype=np.int64)
    vv = xi * v
    If = auxPar.maxC
    nEvalF = 0
    nEvalB = 0
    for c in range(0, auxPar.maxC + 1):
        qb = q / Sd
        gb = Sd * g
        vh = vv + 0.5 * h * gb
        qbn = qb + h * vh
        q1 = qbn * Sd
        ff, gnew = lpFun(q1)
        nEvalF += 1
        gb1 = Sd * gnew
        v1 = vh + 0.5 * h * gb1
        gbmean = 0.5 * (np.abs(gb) + np.abs(gb1))
        Ham1 = -ff + 0.5 * sum(v1 * v1)

        grTooBig = gbmean > gradThresh

        if not np.isfinite(Ham1):
            Sred += 1
        elif np.any(grTooBig):
            Sred[grTooBig] += 1
        elif np.abs(Ham0 - Ham1) > delta:
            Sred += 1
        else:
            If = c
            break

        Sd = 2.0 ** (-Sred)

    qOut = q1
    vOut = v1
    fOut = ff
    gOut = gnew
    SredForw = Sred

    Hb0 = Ham1

    Sd = np.ones(d)
    Sred = np.zeros(d, dtype=np.int64)
    vv = -vOut
    Ib = If

    if If > 0:
        for c in range(0, auxPar.maxC + 1):
            qb = qOut / Sd
            gb = Sd * gOut
            vh = vv + 0.5 * h * gb
            qbn = qb + h * vh
            q1 = qbn * Sd
            ff, gnew = lpFun(q1)
            nEvalB += 1
            gb1 = Sd * gnew
            v1 = vh + 0.5 * h * gb1
            gbmean = 0.5 * (np.abs(gb) + np.abs(gb1))
            Ham1 = -ff + 0.5 * sum(v1 * v1)

            grTooBig = gbmean > gradThresh

            if not np.isfinite(Ham1):
                Sred += 1
            elif np.any(grTooBig):
                Sred[grTooBig] += 1
            elif np.abs(Hb0 - Ham1) > delta:
                Sred += 1
            else:
                Ib = c
                break

            if np.all(SredForw == Sred):
                Ib = c + 1
                break
            Sd = 2.0 ** (-Sred)

    lpw = __logZero * (not np.all(Sred == SredForw))

    return integratorReturn(qOut, xi * vOut, fOut, gOut, nEvalF, nEvalB, If, Ib, If, lpw, 1.0)
