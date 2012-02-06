import numpy as np
from copy import copy
from pymc import six

__all__ = ['plot_GP_envelopes']

def plot_GP_envelopes(f, x, HPD = [.25, .5, .95], transx = None, transy = None):
    """
    plot_GP_envelopes(f, x[, HPD, transx, transy])


    Plots centered posterior probability envelopes for f, which is a GP instance,
    which is a function of one variable.


    :Arguments:

        -   `f`: A GaussianProcess object.

        -   `x`: The mesh on which to plot the envelopes.

        -   `HPD`: A list of values between 0 and 1 giving the probability mass
            contained in each envelope.

        -   `transx`: Any transformation of the x-axis.

        -   `transy`: Any transformation of the y-axis.
    """
    try:
        from pymc.Matplot import centered_envelope

        f_trace = f.trace()
        x = x.ravel()
        N = len(f_trace)
        func_stacks = np.zeros((N,len(x)),dtype=float)

        def identity(y):
            return y

        if transy is None:
            transy = identity
        if transx is None:
            transx = identity

        # Get evaluations
        for i in range(N):
            f = copy(f_trace[i])
            func_stacks[i,:] = transy(f(transx(x)))

        # Plot envelopes
        HPD = np.sort(HPD)
        sorted_func_stack = np.sort(func_stacks,0)
        for m in HPD[::-1]:
            env = centered_envelope(sorted_func_stack, m)
            # from IPython.Debugger import Pdb
            # Pdb(color_scheme='LightBG').set_trace()
            env.display(x, alpha=1.-m*.5,new=False)
        centered_envelope(sorted_func_stack, 0.).display(x, alpha=1., new=False)
    except ImportError:
        six.print_('Plotter could not be imported; plotting disabled')
