from menpodetect.dlib import (load_dlib_frontal_face_detector, DlibDetector,
                              train_dlib_detector)
from menpodetect.opencv import (load_opencv_eye_detector,
                                load_opencv_frontal_face_detector,
                                load_opencv_profile_face_detector,
                                OpenCVDetector)
from menpodetect.pico import load_pico_frontal_face_detector, PicoDetector
from ._version import get_versions


__version__ = get_versions()['version']
del get_versions

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
