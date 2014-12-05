from menpodetect.dlib import load_dlib_frontal_face_detector, DlibDetector
from menpodetect.opencv import (load_opencv_eye_detector,
                                load_opencv_frontal_face_detector,
                                load_opencv_profile_face_detector,
                                OpenCVDetector)
from ._version import get_versions


__version__ = get_versions()['version']
del get_versions

