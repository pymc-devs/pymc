import functools
import inspect
import warnings
import regex as re
import textwrap


def deprecator(reason=None, version=None,action='deprecate', deprecated_args=None, docs=True):
    """
    PyMC deprecation helper. This decorator emits deprecated warnings.
    """    
    if deprecated_args is not None:
        deprecated_args = set(deprecated_args.split())
    cause = reason
    if cause is not None and version is not None:
        reason = ", since version " + version + " " + "("+cause+")"

    if cause is not None and version is None:
        reason = "("+ str(cause) + ")"

    if cause is None and version is not None :
        reason = ", since version " + version
    #this function is an edited version of the source code taken from the library Deprecated
    #https://github.com/tantale/deprecated (MIT License)
    def sphinxformatter(version, reason):
        fmtsphinx = ".. deprecated:: {version}" if version else ".. deprecated::"
        sphinxtext = [fmtsphinx.format(directive = "deprecated", version=version)]
        width = 2 ** 16
        reason = reason.lstrip()
        for paragraph in reason.splitlines():
            if paragraph:
                sphinxtext.extend(
                    textwrap.fill(
                        paragraph,
                        width=width,
                        initial_indent="   ",
                        subsequent_indent="   ",
                    ).splitlines()
                )
            else:
                sphinxtext.append("")
        return sphinxtext


    def decorator(func):
        if deprecated_args is None:
            if inspect.isclass(func):
                fmt = "Class {name} is deprecated{reason}."
            else:
                fmt = "Function or method {name} is deprecated{reason}."
        else:
            fmt = "Parameter(s) {name} deprecated{reason}"
            
        if docs==True:
            docstring = textwrap.dedent(func.__doc__ or "")
            if docstring:
                docstring = re.sub(r"\n+$", "", docstring, flags=re.DOTALL) + "\n\n"
            else:
                docstring = "\n"

            sphinxtext = sphinxformatter(version, cause)
            if deprecated_args is None:
                docstring += "".join("{}\n".format(line) for line in sphinxtext)
            else:
                for deprecated_arg in deprecated_args:
                    doc=docstring.split(f'\n{deprecated_arg}:')[1]
                    nextitem = re.search("\\n{1}\\w+:{1}", doc)
                    nextsection = re.search("\\n{1}\\w+\\n{1}-+\\n{1}",doc)
                    last = len(doc)
                    n = min(nextitem.start(), nextsection.start(), last)
                    y = len(docstring.split(f'\n{deprecated_arg}:')[0]) + len(f'\n{deprecated_arg}:')
                    docstring = docstring[:y+n] + str(sphinxtext) + docstring[y+n:]

            func.__doc__ = docstring       

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            if deprecated_args is None:
                name = func.__name__
            else:
                argstodeprecate = deprecated_args.intersection(kwargs)
                if argstodeprecate is not None:
                    name = ', '.join(repr(arg) for arg in argstodeprecate)

            if name!="":
                if action!="ignore":
                    warnings.simplefilter('always', DeprecationWarning)
                    warnings.warn(
                        fmt.format(name=name, reason=reason),
                        category=DeprecationWarning,
                        stacklevel=2
                )
            return func
        
        return new_func
    
    return decorator
