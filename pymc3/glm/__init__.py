try:
    from . import families
    from .glm import glm, linear_component, plot_posterior_predictive
except ImportError:
    print("Warning: patsy not found, not importing glm submodule.")
