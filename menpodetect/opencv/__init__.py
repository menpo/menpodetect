from menpo.base import MenpoMissingDependencyError

from menpodetect._optional import names_of_a_missing_library

_NAMES = (
    "load_opencv_frontal_face_detector",
    "load_opencv_profile_face_detector",
    "load_opencv_eye_detector",
    "OpenCVDetector",
)

try:
    from .detect import (
        load_opencv_frontal_face_detector,
        load_opencv_profile_face_detector,
        load_opencv_eye_detector,
        OpenCVDetector,
    )
except MenpoMissingDependencyError as _error:
    # OpenCV is not installed (or cannot be imported): the names are not here, and
    # asking for one says why
    __getattr__ = names_of_a_missing_library(_error, _NAMES, __name__)

# Remove from scope
del MenpoMissingDependencyError, names_of_a_missing_library
