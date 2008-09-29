"""DisasterModel time series"""

from pymc.examples import DisasterModel
import pylab as P
import numpy as np

y = DisasterModel.disasters_array

years = 1852 + np.arange(len(y))

P.rcdefaults()

P.figure(figsize=(6,3))
P.subplots_adjust(bottom=.2)
P.subplot(111)
P.plot(years, y, 'bo', ms=3)
P.plot(years, y, 'k-', lw=1, alpha=.5)
P.xlim(years.min(), years.max())
P.xlabel('Year')
P.ylabel('Number of disasters')
P.savefig('disastersts_web.png', dpi=80)
P.close()


P.figure(figsize=(6,3))
P.rcParams['font.size'] = 10.
P.subplots_adjust(bottom=.2)
P.subplot(111)
P.plot(years, y, 'ko', ms=3)
P.plot(years, y, 'k-', lw=1, alpha=.5)
P.xlim(years.min(), years.max())
P.xlabel('Year')
P.ylabel('Number of disasters')
P.savefig('disastersts.png', dpi=300)
P.close()
