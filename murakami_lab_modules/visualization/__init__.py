"""Matplotlib-based plotting helpers.

The plotting implementation depends on the optional ``plot`` extra. Objects are
imported lazily so importing this subpackage does not require matplotlib until a
plotting object is actually requested.
"""

__all__ = ['Plotter', 'plot_histogram']


def __getattr__(name: str):
    if name in __all__:
        from .plotter import Plotter, plot_histogram

        objects = {
            'Plotter': Plotter,
            'plot_histogram': plot_histogram,
        }
        globals().update(objects)
        return objects[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
