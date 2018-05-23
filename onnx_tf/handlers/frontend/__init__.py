import os
import pkgutil

__all__ = [
    modname for _, modname, _ in pkgutil.walk_packages(
        path=[os.path.split(__file__)[0]])
]
