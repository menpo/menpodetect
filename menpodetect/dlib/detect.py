from __future__ import division
from functools import partial

import dlib
from pathlib import Path

from menpodetect.detect import detect
from menpodetect.compatibility import STRING_TYPES
from .conversion import rect_to_pointgraph


class _dlib_detect(object):
    r"""
    A utility callable that allows the caching of a dlib detector.

    This callable is important for presenting the correct parameters to the
    user. It also marshalls the return type of the detector back to
    `menpo.shape.PointDirectedGraph`.

    Parameters
    ----------
    model : `Path` or `str` or `dlib.simple_object_detector`
        Either a path to a `dlib.simple_object_detector` or a
        `dlib.fhog_object_detector` or the detector itself.

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
            # There are two different kinds of object detector, the
            # simple_object_detector and the fhog_object_detector, but we
            # can't tell which is which from the file name. Therefore, try one
            # and then the other. Unfortunately, it throws a runtime error,
            # which we have to catch.
            try:
                model = dlib.simple_object_detector(str(m_path))
            except RuntimeError:
                model = dlib.fhog_object_detector(str(m_path))
        self._dlib_model = model

    def __call__(self, uint8_image, n_upscales=0):
        r"""
        Perform a detection using the cached dlib detector.

        Parameters
        ----------
        uint8_image : `ndarray`
            An RGB (3 Channels) or Greyscale (1 Channel) numpy array of uint8
        n_upscales : `int`, optional
            Number of times to upscale the image when performing the detection,
            may increase the chances of detecting smaller objects.

        Returns
        ------
        bounding_boxes : menpo.shape.PointDirectedGraph
            The detected objects.
        """
        # Dlib doesn't handle the dead last axis
        if uint8_image.shape[-1] == 1:
            uint8_image = uint8_image[..., 0]
        rects = self._dlib_model(uint8_image, n_upscales)
        return [rect_to_pointgraph(r) for r in rects]


class DlibDetector(object):
    r"""
    A generic dlib detector.

    Wraps a dlib object detector inside the menpodetect framework and provides
    a clean interface to expose the dlib arguments.
    """
    def __init__(self, model):
        self._detector = _dlib_detect(model)

    def __call__(self, image, greyscale=False, image_diagonal=None,
                 group_prefix='dlib', n_upscales=0):
        r"""
        Perform a detection using the cached dlib detector.

        The detections will also be attached to the image as landmarks.

        Parameters
        ----------
        image : `menpo.image.Image`
            A Menpo image to detect. The bounding boxes of the detected objects
            will be attached to this image.
        greyscale : `bool`, optional
            Convert the image to greyscale or not.
        image_diagonal : `int`, optional
            The total size of the diagonal of the image that should be used for
            detection. This is useful for scaling images up and down for
            detection.
        group_prefix : `str`, optional
            The prefix string to be appended to each each landmark group that is
            stored on the image. Each detection will be stored as group_prefix_#
            where # is a count starting from 0.
        n_upscales : `int`, optional
            Number of times to upscale the image when performing the detection,
            may increase the chances of detecting smaller objects.

        Returns
        ------
        bounding_boxes : `menpo.shape.PointDirectedGraph`
            The detected objects.
        """
        detect_partial = partial(self._detector, n_upscales=n_upscales)
        return detect(detect_partial, image, greyscale=greyscale,
                      image_diagonal=image_diagonal, group_prefix=group_prefix)


def load_dlib_frontal_face_detector():
    r"""
    Load the dlib frontal face detector.

    Returns
    -------
    detector : `DlibDetector`
        The frontal face detector.
    """
    return DlibDetector(dlib.get_frontal_face_detector())
