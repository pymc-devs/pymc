import re
import functools
from numpy import asscalar
import os
import inspect
import theano
from six import with_metaclass


THEANO_PACKAGE_PATH = os.path.dirname(theano.__file__)

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
        return u'None'
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
        Boolean, whether the string could have been produced by
        `get_transormed_name`
    """
    return name.endswith('__') and name.count('_') >= 3


def get_untransformed_name(name):
    """
    Undo transformation in `get_transformed_name`. Throws ValueError if
    name wasn't transformed

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
        raise ValueError(
            u'{} does not appear to be a transformed name'.format(name))
    return '_'.join(name.split('_')[:-3])


def get_default_varnames(var_iterator, include_transformed):
    """Helper to extract default varnames from a trace.

    Parameters
    ----------
    varname_iterator : iterator
        Elements will be cast to string to check whether it is transformed,
        and optionally filtered
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
        return [var for var in var_iterator
                if not is_transformed_name(str(var))]


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
                return 'f(%s)' % ',~'.join([n for n in names
                                            if isinstance(n, str)])
            except IndexError:
                pass
        value = variable.eval()
        if not value.shape:
            return asscalar(value)
        return 'array'
    return r'\text{%s}' % name


def update_start_vals(a, b, model):
    """
    Update a with b, without overwriting existing keys. Values specified for
    transformed variables on the original scale are also transformed and
    inserted.

    """
    if model is not None:
        for free_RV in model.free_RVs:
            tname = free_RV.name
            for name in a:
                if is_transformed_name(tname) and \
                   get_untransformed_name(tname) == name:
                    transform_func = [
                        d.transformation for d in model.deterministics
                        if d.name == name]
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


# Methods that will wrap the attribute getting and setting methods to handle
# name
def called_from_inside_theano(look_behind=1):
    """
    Look at the call stack trace of the previous `look_behind` number of frames
    to see if the call came from theano

    """
    # Twisted way to control the precise look_behind steps to make, because
    # sometimes inspect.stack finds code objects at some outer frame and
    # crashes
    frame = inspect.currentframe().f_back
    for rewind in range(look_behind):
        frame = frame.f_back
    try:
        caller_path = inspect.getframeinfo(frame)[0]
    except:
        # Failed to get the caller_path, maybe because it originated at a code
        # object so the default is to say the call was external to theano
        caller_path = None
    return os.path.abspath(caller_path).startswith(THEANO_PACKAGE_PATH)


def name_wrapped_getattribute(self, at, override=False):
    if at == 'name' and not override and not called_from_inside_theano():
        return self.pymc_name
    else:
        return object.__getattribute__(self, at)


def name_wrapped_setattr(self, at, value):
    if at == 'name' and not called_from_inside_theano():
        self.pymc_name = value
    else:
        if at not in self._reserved_dict_keys:
            # We changed an attribute that is in the masked theano
            # variable, so we mark it must be reconstructed in
            # get_theano_instance
            self._masked_theano_var = None
        object.__setattr__(self, at, value)


class MetaNameWrapped(theano.gof.utils.MetaObject):
    registry = {}

    def __new__(cls, clsname, bases, dct, theano_class=None):
        """
        MetaNameWrapped.__new__(cls, clsname, bases, dct, theano_class=None)

        This function is used to create the dynamic NameWrapped classes. It has
        two distinct behaviors.
        A) If theano_class is None, this method just calls super.__new__.
           This first call to __new__ creates an empty class on which __call__
           can operate to generate the correct dynamic class, which depends on
           the input passed at the variable's construction.

        B) If theano_class is not None, it must be a class. If not, a TypeError
           will be raised. If theano_class is a class then __new__ does the
           following:
           1) Changes clsname to clsname + theano_class.__name__
           2) Ensures that theano_class is added as the leftmost base
           3) Changes __getattribute__ and __setattribute__ methods in dct to
              name_wrapped_getattribute and name_wrapped_setattr
           4) Changes __eq__ and __ne__. __hash__ will give the same result as
              a call to the hash of the theano variable used for construction.
              __eq__ and __ne__ will work conditionally to whether the call
              comes from within theano or not. If the call is from inside
              theano they will work with the theano variable used for
              construction's __eq__ and __ne__. If not, they will simply
              follow up the simply use the theano implementation.
           5) Registers the new class (if it did not already exist)
           6) Returns the newly created dynamic class

        """
        if theano_class is None:
            key = tuple(clsname)+bases
            try:
                return cls.registry[key]
            except KeyError:
                out_cls = super(MetaNameWrapped, cls).__new__(cls, clsname,
                                                              bases, dct)
                cls.registry[key] = out_cls
                return out_cls
        else:
            if not inspect.isclass(theano_class):
                raise TypeError("MetaNameWrapped's theano_class input must be "
                                "a class. However, theano_class's type is "
                                "{}".format(type(theano_class)))
            # Step 1. Change clsname
            clsname += theano_class.__name__

            # Step 2. Prepend theano_class to bases
            if bases:
                bases = list(bases)
                # Ensure that theano_class is placed as the leftmost base
                if theano_class in bases:
                    bases.pop(bases.index(theano_class))
                bases = tuple([theano_class, ] + bases)
            else:
                bases = (theano_class, )

            key = tuple(clsname)+bases
            try:
                return cls.registry[key]
            except KeyError:
                # Step 3. Change __getattribute__ and __setattr__ to name
                # wrapped versions
                dct['__getattribute__'] = name_wrapped_getattribute
                dct['__setattr__'] = name_wrapped_setattr
                out_cls = super(MetaNameWrapped, cls).__new__(cls, clsname,
                                                              bases, dct)
                # Step 4. Register the new class
                cls.registry[key] = out_cls
                return out_cls

    def __call__(cls, theano_var, *args, **kwargs):
        """
        MetaNameWrapped.__call__(cls, theano_var, *args, **kwargs)

        This class method intercepts NameWrapped instance creation, and based
        on the theano_var's class, it dynamically creates the correct class
        with name wrapping behavior.

        Input:
            theano_var: (Mandatory) an instance of a theano class variable
            *args, **kwargs: Are passed to NameWrapped.__init__
        Output:
            An instance of
            'NameWrapped{}'.format(theano_var.__class__.__name__) class.

        This function does the following
        1) When __call__ is executed, __new__ has already run a first time.
           However, this first "dry" was in no way aware of the theano_var
           input passed to __call__. So, when __call__ is first executed,
           and not before, the class of the output instance will be known. This
           means that MetaNameWrapped.__new__ must be called here again with
           an added parametrization, in order to get the correct wrapped class.
        2) An instance of the class created in step 1 must be created with
           cls.__new__
        3) The theano variable's __dict__ must be copied into the new
           instance's __dict__ (maybe a shallow copy is enough).
        4) Call cls.__init__ on the newly created instance
        5) Check if the original theano_var has owner. If so, the outputs that
           match the theano_var must be changed to the wrapped instance.
        6) Returned the created instance. The rules of metaclass __call__ say
           that if __call__ returns an istance of cls, it will also call
           cls.__init__ on it. As we are changing the input cls during this
           call, the output will never have the same cls that was inputed, so
           __init__ will not be called again.

        """
        # Prepare the call for the __new__ statement that depends on the
        # theano_var
        dct = vars(cls).copy()
        slots = dct.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                dct.pop(slots_var)
        dct.pop('__dict__', None)
        dct.pop('__weakref__', None)

        # Step 1. Call the MetaNameWrapped.__new__ method with the
        # theano_class, to get the actual modified class
        try:
            _masked_class = theano_var._masked_class
            # ~ _masked_dict_keys = theano_var._masked_dict_keys
        except AttributeError:
            _masked_class = type(theano_var)
            # ~ _masked_dict_keys = set(theano_var.__dict__.keys())

        cls = MetaNameWrapped.__new__(MetaNameWrapped, cls.__name__,
                                      cls.__bases__, dct,
                                      theano_class=_masked_class)
        # Step 2. Create the instance with cls.__new__
        instance = cls.__new__(cls)
        # Step 3. Copy the theano_var __dict__ into the newly created instance
        # ~ instance._reserved_dict_keys = _reserved_dict_keys
        instance._masked_theano_var = None
        instance._masked_class = _masked_class
        instance.__dict__.update(theano_var.__dict__)

        # Step 4. Call __init__ as this will not be done automatically for us
        instance.__init__(*args, **kwargs)

        # Step 5. Change owner outputs to the wrapped variable
        try:
            owner = instance.owner
        except AttributeError:
            owner = None
        if owner:
            if hasattr(owner, 'outputs'):
                for i, out in enumerate(owner.outputs):
                    if out == theano_var:
                        instance.owner.outputs[i] = instance
        return instance

    @classmethod
    def is_name_wrapped_instance(cls, obj):
        return isinstance(obj, tuple(cls.registry.values()))


class NameWrapped(object, with_metaclass(MetaNameWrapped)):
    _reserved_dict_keys = set(['pymc_name', '_masked_theano_var',
                              '_masked_class'])

    def __init__(self, pymc_name=None, override_name=False):
        """
        Class that is intended to wrap instances of theano variables of any
        type. This class is intended to change the way in which the `name`
        attribute is handled from outside of theano's scope. From outside of
        `theano`, the `name` attribute should be an alias for the `pymc_name`
        attribute, whilst from inside of `theano`, the `name` should stay the
        same.

        """
        # We only set pymc_name here, the theano variable's original name is
        # left untouched during the init, but later calls to
        # setattr(self, 'name',...) will change both the name attribute and the
        # pymc name
        if override_name:
            # setattr(self, 'name', val) will also set pymc_name. The contrary
            # does not occur
            self.name = pymc_name
        else:
            self.pymc_name = pymc_name

    def __reduce__(self):
        theano_var = self.get_theano_instance()
        return (NameWrapped, (theano_var, self.pymc_name))

    def get_theano_instance(self):
        """
        self.get_theano_instance()

        Return a new instance of the theano class that was used in the
        construction of self.

        """
        if self._masked_theano_var is None:
            tv = self._masked_class.__new__(self._masked_class)
            dct = {k: self.__getattribute__(k, override=True) for k in
                   self.__dict__ if k not in self._reserved_dict_keys}
            tv.__dict__.update(dct)
            self._masked_theano_var = tv
        return self._masked_theano_var

    def __eq__(self, other):
        if MetaNameWrapped.is_name_wrapped_instance(other):
            other_theano_var = other.get_theano_instance()
            other_wrapped = True
        else:
            other_theano_var = other
            other_wrapped = False
        theano_var = self.get_theano_instance()
        eq_theano_vars = theano_var == other_theano_var
        if called_from_inside_theano():
            return eq_theano_vars
        else:
            return other_wrapped and eq_theano_vars and \
                   self.pymc_name == other.pymc_name

    def __ne__(self, other):
        if MetaNameWrapped.is_name_wrapped_instance(other):
            other_theano_var = other.get_theano_instance()
            other_wrapped = True
        else:
            other_theano_var = other
            other_wrapped = False
        theano_var = self.get_theano_instance()
        eq_theano_vars = theano_var == other_theano_var
        if called_from_inside_theano():
            return not eq_theano_vars
        else:
            return not (other_wrapped and eq_theano_vars and
                        self.pymc_name == other.pymc_name)

    def __hash__(self):
        return hash(self.get_theano_instance())

    def clone(self):
        """
        self.clone()

        Return a clone of the name wrapped instance, by calling clone on the
        output from self.get_theano_instance()

        """
        tc = self.get_theano_instance().clone()
        cp = NameWrapped(tc, self.pymc_name)
        return cp

    def clone_with_new_inputs(self, *args, **kwargs):
        tc = self.get_theano_instance().clone_with_new_inputs(*args, **kwargs)
        cp = NameWrapped(tc, self.pymc_name)
        return cp
