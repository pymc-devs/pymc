from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


from aeppl.logprob import logprob  # isort: split

from aeppl.joint_logprob import factorized_joint_logprob, joint_logprob
from aeppl.printing import latex_pprint, pprint

# isort: off
# Add rewrites to the DBs
import aeppl.censoring
import aeppl.cumsum
import aeppl.mixture
import aeppl.scan
import aeppl.tensor
import aeppl.transforms

# isort: on
