from __future__ import division
from functools import partial
from pathlib import Path
import numpy as np

from menpo.base import MenpoMissingDependencyError

try:
    from cypico import detect_objects, detect_frontal_faces
except ImportError:
    raise MenpoMissingDependencyError('cypico')

from menpodetect.detect import detect
from menpodetect.compatibility import STRING_TYPES
from .conversion import pointgraph_from_circle


class _pico_detect(object):
    r"""
    A utility callable that allows the caching of a pico detector.

    This callable is important for presenting the correct parameters to the
    user. It also marshalls the return type of the detector back to
    menpo.shape.PointDirectedGraph.

    Parameters
    ----------
    model : `pico detector`
        At the moment loading new pico detectors is not possible. Unless you
        have managed to load a pico detector as a 1D uint8 array.

    Raises
    ------
    ValueError
        If a path was provided.
    """
    def __init__(self, model):
        if isinstance(model, STRING_TYPES) or isinstance(model, Path):
            raise ValueError('Loading Pico trained models from disk '
                             'is not currently supported.')
        self._pico_model = model

    def __call__(self, uint8_image, max_detections=100, orientations=0.0,
                 degrees=True, scale_factor=1.2, stride_factor=0.1,
                 min_size=100, confidence_cutoff=3.0, axis_aligned_bb=True):
        r"""
        Perform a detection using the cached pico detector.

        Parameters
        ----------
        uint8_image : `ndarray`
            A Greyscale (1 Channel) numpy array of uint8
        max_detections : `int`, optional
            The maximum number of detections to return.
        orientations : `list` of `float`s or `float`, optional
            The orientations of the cascades to use. ``0.0`` will perform an
            axis aligned detection. Values greater than ``0.0`` will perform
            detections of the cascade rotated counterclockwise around a unit
            circle.
            If a list is passed, each item should be an orientation in
            radians around the unit circle, with ``0.0`` being axis aligned.
        scale_factor : `float`, optional
            The ratio to increase the cascade window at every iteration. Must
            be greater than 1.0
        stride_factor : `float`, optional
            The ratio to decrease the window step by at every iteration. Must be
            less than 1.0, optional
        min_size : `float`, optional
            The minimum size in pixels (diameter of the detection circle) that a
            face can be. This is the starting cascade window size.
        confidence_cutoff : `float`, optional
            The confidence value to trim the detections with. Any detections
            with confidence less than the cutoff will be discarded.
        axis_aligned_bb : `bool`, optional
            If ``True``, the returned detections will be axis aligned,
            regardless of which orientation they were detected at.
            If ``False``, the returned bounding box will be rotated by the
            orientation detected.

        Returns
        ------
        bounding_boxes : `list` of `menpo.shape.PointDirectedGraph`
            The detected objects.
        """
        fittings = detect_objects(
            uint8_image, self._pico_model, max_detections=max_detections,
            orientations=orientations, scale_factor=scale_factor,
            stride_factor=stride_factor, min_size=min_size,
            confidence_cutoff=confidence_cutoff,
            axis_aligned_bb=axis_aligned_bb)
        return [pointgraph_from_circle(f) for f in fittings]


class _pico_face_detect(_pico_detect):
    def __call__(self, uint8_image, max_detections=100, orientations=0.0,
                 scale_factor=1.2, stride_factor=0.1,
                 min_size=100, confidence_cutoff=3.0, axis_aligned_bb=True):
        r"""
        Perform a detection using the frontal face pico detector.

        The detections will also be attached to the image as landmarks.

        Parameters
        ----------
        image : `menpo.image.Image`
            A Menpo image to detect. The bounding boxes of the detected faces
            will be attached to this image.
        image_diagonal : `int`, optional
            The total size of the diagonal of the image that should be used for
            detection. This is useful for scaling images up and down for
            detection.
        group_prefix : `str`, optional
            The prefix string to be appended to each each landmark group that is
            stored on the image. Each detection will be stored as group_prefix_#
            where # is a count starting from 0.
        max_detections : `int`, optional
            The maximum number of detections to return.
        orientations : list of `float`s or `float`, optional
            The orientations of the cascades to use. ``0.0`` will perform an
            axis aligned detection. Values greater than ``0.0`` will perform
            detections of the cascade rotated counterclockwise around a unit
            circle.
            If a list is passed, each item should be an orientation in
            radians around the unit circle, with ``0.0`` being axis aligned.
        scale_factor : `float`, optional
            The ratio to increase the cascade window at every iteration. Must
            be greater than 1.0
        stride_factor : `float`, optional
            The ratio to decrease the window step by at every iteration. Must be
            less than 1.0, optional
        min_size : `float`, optional
            The minimum size in pixels (diameter of the detection circle) that a
            face can be. This is the starting cascade window size.
        confidence_cutoff : `float`, optional
            The confidence value to trim the detections with. Any detections
            with confidence less than the cutoff will be discarded.
        axis_aligned_bb : `bool`, optional
            If ``True``, the returned detections will be axis aligned,
            regardless of which orientation they were detected at.
            If ``False``, the returned bounding box will be rotated by the
            orientation detected.

        Returns
        ------
        bounding_boxes : `list` of `menpo.shape.PointDirectedGraph`
            The detected faces.
        """
        fittings = detect_frontal_faces(
            uint8_image, max_detections=max_detections,
            orientations=orientations, scale_factor=scale_factor,
            stride_factor=stride_factor, min_size=min_size,
            confidence_cutoff=confidence_cutoff)
        return [pointgraph_from_circle(f, axis_aligned_bb=axis_aligned_bb)
                for f in fittings]


# This is a slightly strange model because at the moment cypico is not
# really able to load arbitrary pico models. This is due to the fact that
# pico models are just saved down as raw data and loaded into a char* array
# then case into the members of the C-struct they actually represent. To make
# this cleaner, the C-struct would need to be created in cypico so we could
# properly load and save pico models
class PicoDetector(object):
    r"""
    A generic pico detector.

    Wraps a pico object detector inside the menpodetect framework and
    provides a clean interface to expose the pico arguments.

    At the moment this isn't particularly useful as loading Pico models is
    complex.
    """
    def __init__(self, model, detector=_pico_detect):
        self._detector = detector(model)

    def __call__(self, image, image_diagonal=None, group_prefix='pico',
                 max_detections=100, orientations=0.0, degrees=True,
                 scale_factor=1.2, stride_factor=0.1, min_size=100,
                 confidence_cutoff=3.0, axis_aligned_bb=True):
        r"""
        Perform a detection using the cached pico detector.

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
        max_detections : `int`, optional
            The maximum number of detections to return.
        orientations : list of `float`s or `float`, optional
            The orientations of the cascades to use. ``0.0`` will perform an
            axis aligned detection. Values greater than ``0.0`` will perform
            detections of the cascade rotated counterclockwise around a unit
            circle.
            If a list is passed, each item should be an orientation in
            either radians or degrees around the unit circle, with ``0.0``
            being axis aligned.
        degrees : `bool`, optional
            If ``True``, the ``orientations`` parameter is treated as
            rotations counterclockwise in degrees rather than radians.
        scale_factor : `float`, optional
            The ratio to increase the cascade window at every iteration. Must
            be greater than 1.0
        stride_factor : `float`, optional
            The ratio to decrease the window step by at every iteration. Must be
            less than 1.0, optional
        min_size : `float`, optional
            The minimum size in pixels (diameter of the detection circle) that a
            face can be. This is the starting cascade window size.
        confidence_cutoff : `float`, optional
            The confidence value to trim the detections with. Any detections
            with confidence less than the cutoff will be discarded.
        axis_aligned_bb : `bool`, optional
            If ``True``, the returned detections will be axis aligned,
            regardless of which orientation they were detected at.
            If ``False``, the returned bounding box will be rotated by the
            orientation detected.

        Returns
        ------
        bounding_boxes : `list` of `menpo.shape.PointDirectedGraph`
            The detected objects.
        """
        if degrees:
            # Cypico expects Radians
            orientations = np.deg2rad(orientations)
        detect_partial = partial(self._detector, max_detections=max_detections,
                                 orientations=orientations,
                                 scale_factor=scale_factor,
                                 stride_factor=stride_factor, min_size=min_size,
                                 confidence_cutoff=confidence_cutoff,
                                 axis_aligned_bb=axis_aligned_bb)
        return detect(detect_partial, image, greyscale=True,
                      image_diagonal=image_diagonal, group_prefix=group_prefix)


def load_pico_frontal_face_detector():
    r"""
    Load the pico frontal face detector.

    Returns
    -------
    detector : `PicoDetector`
        The frontal face detector.
    """
    return PicoDetector(None, detector=_pico_face_detect)
