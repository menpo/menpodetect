from __future__ import division
from functools import partial
from pathlib import Path

from menpo.base import MenpoMissingDependencyError

try:
    import cv2
except ImportError:
    raise MenpoMissingDependencyError('opencv')

from menpodetect.detect import detect
from menpodetect.compatibility import STRING_TYPES
from .conversion import (pointgraph_from_rect, opencv_frontal_face_path,
                         opencv_profile_face_path, opencv_eye_path)


def _get_default_flags():
    version = cv2.__version__.split('.')[0]
    if version == '2':
        return cv2.cv.CV_HAAR_SCALE_IMAGE
    elif version == '3' or version == '4':
        return cv2.CASCADE_SCALE_IMAGE
    else:
        raise ValueError('Unsupported OpenCV version: {}'.format(version))


class _opencv_detect(object):
    r"""
    A utility callable that allows the caching of an opencv detector.

    This callable is important for presenting the correct parameters to the
    user. It also marshalls the return type of the detector back to
    menpo.shape.PointDirectedGraph.

    Parameters
    ----------
    model : `Path` or `str` or `opencv.CascadeClassifier`
        Either a path to an `opencv.CascadeClassifier` or the detector itself.

    Raises
    ------
    ValueError
        If a path was provided and it does not exist.
    """
    def __init__(self, model):
        if isinstance(model, STRING_TYPES) or isinstance(model, Path):
            m_path = Path(model)
            if not Path(m_path).exists():
                raise ValueError('Model {} does not exist.'.format(m_path))
            model = cv2.CascadeClassifier(str(m_path))
        self._opencv_model = model

    def __call__(self, uint8_image, scale_factor=1.1, min_neighbours=5,
                 min_size=(30, 30), flags=None):
        r"""
        Perform a detection using the cached opencv detector.

        Parameters
        ----------
        uint8_image : `ndarray`
            A Greyscale (1 Channel) numpy array of uint8
        scale_factor : `float`, optional
            The amount to increase the sliding windows by over the second
            pass.
        min_neighbours : `int`, optional
            The minimum number of neighbours (close detections) before
            Non-Maximum suppression to be considered a detection. Use 0
            to return all detections.
        min_size : `tuple` of 2 ints
            The minimum object size in pixels that the detector will consider.
        flags : `int`
            The flags to be passed through to the detector.

        Returns
        ------
        bounding_boxes : `list` of `menpo.shape.PointDirectedGraph`
            The detected objects.
        """
        if flags is None:
            flags = _get_default_flags()
        rects = self._opencv_model.detectMultiScale(
            uint8_image, scaleFactor=scale_factor, minNeighbors=min_neighbours,
            minSize=min_size, flags=flags)
        return [pointgraph_from_rect(r) for r in rects]


class OpenCVDetector(object):
    r"""
    A generic opencv detector.

    Wraps an opencv object detector inside the menpodetect framework and
    provides a clean interface to expose the opencv arguments.
    """
    def __init__(self, model):
        self._detector = _opencv_detect(model)

    def __call__(self, image, image_diagonal=None, group_prefix='opencv',
                 scale_factor=1.1, min_neighbours=5,
                 min_size=(30, 30), flags=None):
        r"""
        Perform a detection using the cached opencv detector.

        The detections will also be attached to the image as landmarks.

        Parameters
        ----------
        image : `menpo.image.Image`
            A Menpo image to detect. The bounding boxes of the detected objects
            will be attached to this image.
        image_diagonal : `int`, optional
            The total size of the diagonal of the image that should be used for
            detection. This is useful for scaling images up and down for
            detection.
        group_prefix : `str`, optional
            The prefix string to be appended to each each landmark group that is
            stored on the image. Each detection will be stored as group_prefix_#
            where # is a count starting from 0.
        scale_factor : `float`, optional
            The amount to increase the sliding windows by over the second
            pass.
        min_neighbours : `int`, optional
            The minimum number of neighbours (close detections) before
            Non-Maximum suppression to be considered a detection. Use 0
            to return all detections.
        min_size : `tuple` of 2 ints
            The minimum object size in pixels that the detector will consider.
        flags : `int`, optional
            The flags to be passed through to the detector.

        Returns
        ------
        bounding_boxes : `list` of `menpo.shape.PointDirectedGraph`
            The detected objects.
        """
        if flags is None:
            flags = _get_default_flags()
        detect_partial = partial(self._detector, scale_factor=scale_factor,
                                 min_neighbours=min_neighbours,
                                 min_size=min_size, flags=flags)
        return detect(detect_partial, image, greyscale=True,
                      image_diagonal=image_diagonal, group_prefix=group_prefix)


def load_opencv_frontal_face_detector():
    r"""
    Load the opencv frontal face detector: haarcascade_frontalface_alt.xml

    Returns
    -------
    detector : OpenCVDetector
        The frontal face detector.
    """
    return OpenCVDetector(opencv_frontal_face_path)


def load_opencv_profile_face_detector():
    r"""
    Load the opencv profile face detector: haarcascade_profileface.xml

    Returns
    -------
    detector : OpenCVDetector
        The profile face detector.
    """
    return OpenCVDetector(opencv_profile_face_path)


def load_opencv_eye_detector():
    r"""
    Load the opencv eye detector: haarcascade_eye.xml

    Returns
    -------
    detector : OpenCVDetector
        The eye detector.
    """
    return OpenCVDetector(opencv_eye_path)

