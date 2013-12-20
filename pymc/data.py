import pkgutil
from io import StringIO

__all__ = ['get_data_file']

def get_data_file(pkg, path):
        return StringIO(unicode(pkgutil.get_data(pkg, path)))

