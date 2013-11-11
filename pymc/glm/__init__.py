try:
    import statsmodels.api as sm
    from . import links
    from . import families
    from .glm import *
except ImportError:
    print("Warning: statsmodels and/or patsy not found, not importing glm submodule.")
