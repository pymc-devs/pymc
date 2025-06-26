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
Online estimation/approximation of quantiles using the P-squared algorithm.

Based on P2quantile.py from WALNUTSpy by Tore Selland Kleppe.
Implementation of Jain and Chlamtac, Communications of the ACM, 28(10) 1985.
"""

import numpy as np


class P2quantile:
    def __init__(self, prob=0.5):
        self.npush = 0
        self.p = prob
        self.x = np.zeros(5)
        self.q = np.zeros(5)
        self.n = np.r_[1:6]
        self.npp = np.array([1.0, 1.0 + 2.0 * prob, 1.0 + 4.0 * prob, 3.0 + 2.0 * prob, 5.0])

    def quantile(self):
        return self.q[2]

    def findInterval(self, xi):
        if xi < self.q[0]:
            return 0
        elif xi > self.q[4]:
            return 5
        for i in range(0, 4):
            if xi < self.q[i + 1]:
                return i + 1

    def pushCore(self, xi):
        self.npush += 1
        if self.npush <= 5:
            self.x[self.npush - 1] = xi

        if self.npush == 5:
            self.x = np.sort(self.x)
            self.q = self.x
        elif self.npush > 5:
            k = self.findInterval(xi)

            if k == 0:
                self.q[0] = xi
                k = 1
            elif k == 5:
                self.q[4] = xi
                k = 4

            self.n[k:5] += 1

            nn = self.npush
            pp = self.p
            self.npp = np.array(
                [
                    1.0,
                    0.5 * (nn - 1) * pp + 1.0,
                    (nn - 1) * pp + 1.0,
                    (nn - 1) * (1 + pp) / 2.0 + 1,
                    nn,
                ]
            )

            for i in range(2, 5):
                ni = self.n[i - 1]
                nip = self.n[i]
                nim = self.n[i - 2]
                di = self.npp[i - 1] - ni

                if (di >= 1.0 and nip - ni > 1) or (di <= -1.0 and nim - ni < -1):
                    di = np.sign(di).astype(np.int64)

                    qi = self.q[i - 1]
                    qip = qi + (di / (nip - nim)) * (
                        (ni - nim + di) * (self.q[i] - qi) / (nip - ni)
                        + (nip - ni - di) * (qi - self.q[i - 2]) / (ni - nim)
                    )
                    if self.q[i - 2] < qip and qip < self.q[i]:
                        self.q[i - 1] = qip
                    else:
                        self.q[i - 1] = qi + di * (self.q[i + di - 1] - qi) / (
                            self.n[i + di - 1] - self.n[i - 1]
                        )
                    self.n[i - 1] += di

    def pushVec(self, x):
        for val in x:
            self.pushCore(val)

    def push(self, x):
        self.pushCore(x)
