import re
import functools
from numpy import asscalar

LATEX_ESCAPE_RE = re.compile(r'(%|_|\$|#|&)', re.MULTILINE)


def escape_latex(strng):
    """Consistently escape LaTeX special characters for _repr_latex_ in IPython

    Implementation taken from the IPython magic `format_latex`

    Examples
    --------
        escape_latex('disease_rate')  # 'disease\_rate'

    Parameters
    ----------
    strng : str
        string to escape LaTeX characters

    Returns
    -------
    str
        A string with LaTeX escaped
    """
    if strng is None:
        return 'None'
    return LATEX_ESCAPE_RE.sub(r'\\\1', strng)


def get_transformed_name(name, transform):
    """
    Consistent way of transforming names

    Parameters
    ----------
    name : str
        Name to transform
    transform : transforms.Transform
        Should be a subclass of `transforms.Transform`

    Returns
    -------
    str
        A string to use for the transformed variable
    """
    return "{}_{}__".format(name, transform.name)


def is_transformed_name(name):
    """
    Quickly check if a name was transformed with `get_transormed_name`

    Parameters
    ----------
    name : str
        Name to check

    Returns
    -------
    bool
        Boolean, whether the string could have been produced by `get_transormed_name`
    """
    return name.endswith('__') and name.count('_') >= 3


def get_untransformed_name(name):
    """
    Undo transformation in `get_transformed_name`. Throws ValueError if name wasn't transformed

    Parameters
    ----------
    name : str
        Name to untransform

    Returns
    -------
    str
        String with untransformed version of the name.
    """
    if not is_transformed_name(name):
        raise ValueError('{} does not appear to be a transformed name'.format(name))
    return '_'.join(name.split('_')[:-3])


def get_default_varnames(var_iterator, include_transformed):
    """Helper to extract default varnames from a trace.

    Parameters
    ----------
    varname_iterator : iterator
        Elements will be cast to string to check whether it is transformed, and optionally filtered
    include_transformed : boolean
        Should transformed variable names be included in return value

    Returns
    -------
    list
        List of variables, possibly filtered
    """
    if include_transformed:
        return list(var_iterator)
    else:
        return [var for var in var_iterator if not is_transformed_name(str(var))]


def get_variable_name(variable):
    """Returns the variable data type if it is a constant, otherwise
    returns the argument name.
    """
    name = variable.name
    if name is None:
        if hasattr(variable, 'get_parents'):
            try:
                names = [get_variable_name(item)
                         for item in variable.get_parents()[0].inputs]
                # do not escape_latex these, since it is not idempotent
                return 'f(%s)' % ',~'.join([n for n in names if isinstance(n, str)])
            except IndexError:
                pass
        value = variable.eval()
        if not value.shape:
            return asscalar(value)
        return 'array'
    return r'\text{%s}' % name


def update_start_vals(a, b, model):
    """Update a with b, without overwriting existing keys. Values specified for
    transformed variables on the original scale are also transformed and inserted.
    """
    if model is not None:
        for free_RV in model.free_RVs:
            tname = free_RV.name
            for name in a:
                if is_transformed_name(tname) and get_untransformed_name(tname) == name:
                    transform_func = [
                        d.transformation for d in model.deterministics if d.name == name]
                    if transform_func:
                        b[tname] = transform_func[0].forward_val(
                            a[name], point=b)

    a.update({k: v for k, v in b.items() if k not in a})


def get_transformed(z):
    if hasattr(z, 'transformed'):
        z = z.transformed
    return z


def biwrap(wrapper):
    @functools.wraps(wrapper)
    def enhanced(*args, **kwargs):
        is_bound_method = hasattr(args[0], wrapper.__name__) if args else False
        if is_bound_method:
            count = 1
        else:
            count = 0
        if len(args) > count:
            newfn = wrapper(*args, **kwargs)
            return newfn
        else:
            newwrapper = functools.partial(wrapper, *args, **kwargs)
            return newwrapper
    return enhanced
