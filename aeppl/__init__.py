from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


from aeppl.logprob import logprob  # isort: split

from aeppl.joint_logprob import joint_logprob
from aeppl.printing import latex_pprint, pprint

# isort: off
# Add optimizations to the DBs
import aeppl.mixture

# isort: on
