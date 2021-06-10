from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


from .logprob import logprob  # isort: split

from .joint_logprob import joint_logprob
from .printing import latex_pprint, pprint
