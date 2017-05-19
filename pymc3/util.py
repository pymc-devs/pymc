from numpy import asscalar

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
        raise ValueError(u'{} does not appear to be a transformed name'.format(name))
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
                names = [get_variable_name(item) for item in variable.get_parents()[0].inputs]
                return 'f(%s)' % ','.join([n for n in names if isinstance(n, str)]) 
            except IndexError:
                pass
        value = variable.eval()
        if not value.shape:
            return asscalar(value)
        return 'array'
    return name
