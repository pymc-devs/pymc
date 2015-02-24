try:
    from . import families
    from .glm import *
except ImportError:
    print("Warning: patsy not found, not importing glm submodule.")
