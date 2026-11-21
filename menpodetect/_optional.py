"""What menpodetect says when the library behind a detector (OpenCV, dlib) cannot be imported."""

import textwrap

from menpo.base import MenpoMissingDependencyError


class MissingExtraError(MenpoMissingDependencyError):
    r"""
    The library that a group of detectors needs cannot be imported. It says which
    extra of menpodetect to install when the library is simply not installed
    (``pip install "menpodetect[opencv]"``), and otherwise that the library could
    not be imported, with the error (a system library that it needs may be missing).

    Parameters
    ----------
    extra : `str`
        The extra of menpodetect that installs the library (``"opencv"``).
    library : `str`
        What to call the library in the message (``"OpenCV"``).
    module : `str`
        The name that it is imported as (``"cv2"``).
    error : `ImportError`, optional
        The error that was raised when it was imported.
    """

    def __init__(self, extra, library, module, error=None):
        super(MissingExtraError, self).__init__(extra)
        not_installed = error is None or (
            isinstance(error, ModuleNotFoundError)
            and (error.name or "").split(".")[0] == module
        )
        self.extra = extra
        self.missing_name = extra if not_installed else None
        if not_installed:
            message = textwrap.dedent(
                """
                The {lib} detectors need {lib}, which is an optional dependency of
                menpodetect. Install it with

                    pip install "menpodetect[{extra}]"
                """
            ).format(lib=library, extra=extra)
        else:
            message = textwrap.dedent(
                """
                {lib} could not be imported, so the {lib} detectors are not
                available. This is not a package that is simply missing: it may be a
                system library that it needs, or a version that is too old, so read
                the error below.
                """
            ).format(lib=library)
        if error is not None:
            message += "\nThe import failed with:\n\n    {}: {}\n".format(
                type(error).__name__, error
            )
        self.message = message


def names_of_a_missing_library(error, names, module_name):
    r"""
    A module ``__getattr__`` for a package whose names could not be imported
    because a library is missing. ``from package import name`` and ``package.name``
    then raise `error`, which says why and what to install, instead of an unhelpful
    "cannot import name". Any other attribute is an `AttributeError`, as usual.
    """

    def __getattr__(name):
        if name in names:
            raise error
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(module_name, name)
        )

    return __getattr__
