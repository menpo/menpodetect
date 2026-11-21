from menpo.base import MenpoMissingDependencyError

from menpodetect._optional import names_of_a_missing_library

_NAMES = ("load_dlib_frontal_face_detector", "DlibDetector", "train_dlib_detector")

try:
    from .detect import load_dlib_frontal_face_detector, DlibDetector
    from .train import train_dlib_detector
except MenpoMissingDependencyError as _error:
    # dlib is not installed (or cannot be imported): the names are not here, and
    # asking for one says why
    __getattr__ = names_of_a_missing_library(_error, _NAMES, __name__)

# Remove from scope
del MenpoMissingDependencyError, names_of_a_missing_library
