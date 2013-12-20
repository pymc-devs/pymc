import pkgutil
import io

__all__ = ['get_data_file']

def get_data_file(pkg, path):
        return io.BytesIO(pkgutil.get_data(pkg, path))

