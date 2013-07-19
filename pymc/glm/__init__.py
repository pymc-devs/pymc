try:
    import statsmodels.api as sm
    import links
    import families
    from .glm import *
except ImportError:
    print "Warning: statsmodels not found, only importing parts of glm."
    from .glm import linear_component
