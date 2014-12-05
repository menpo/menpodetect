import os

from menpodetect.dlib import load_dlib_frontal_face_detector, DlibDetector
from menpodetect.opencv import (load_opencv_eye_detector,
                                load_opencv_frontal_face_detector,
                                load_opencv_profile_face_detector,
                                OpenCVDetector)
from ._version import get_versions


__version__ = get_versions()['version']
del get_versions


def src_dir_path():
    r"""The path to the top of the MenpoDetect Python package.

    Useful for locating where the models folder is stored.

    Returns
    -------
    path : str
        The full path to the top of the MenpoDetect package
    """
    return os.path.split(os.path.abspath(__file__))[0]


def models_dir_path():
    r"""The path to the models directory of the MenpoDetect Python package.

    Returns
    -------
    path : str
        The full path to the models directory of the MenpoDetect package
    """
    return os.path.join(src_dir_path(), 'models')
