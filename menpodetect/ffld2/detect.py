from __future__ import division
from functools import partial
from pathlib import Path

from menpo.base import MenpoMissingDependencyError

try:
    from cyffld2 import (load_model, detect_objects,
                         get_frontal_face_mixture_model)
except ImportError:
    raise MenpoMissingDependencyError('cyffld2')

from menpodetect.detect import detect
from menpodetect.compatibility import STRING_TYPES
from .conversion import pointgraph_from_rect, ensure_channel_axis


class _ffld2_detect(object):
    r"""
    A utility callable that allows the caching of an ffld2 detector.

    This callable is important for presenting the correct parameters to the
    user. It also marshalls the return type of the detector back to
    menpo.shape.PointDirectedGraph.

    Parameters
    ----------
    model : `Path` or `str` or `cyffld2.FFLDMixture`
        Either a path to an `cyffld2.FFLDMixture` or the detector itself.

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
            model = load_model(str(m_path))
        self._ffld2_model = model

    def __call__(self, uint8_image, padding=6, interval=5, threshold=0.5,
                 overlap=0.3):
        r"""
        Perform a detection using the cached ffld2 detector.

        Parameters
        ----------
        uint8_image : `ndarray`
            A Greyscale or RGB image.
        padding : `int`, optional
            Amount of zero padding in HOG cells
        interval : `int`, optional
            Number of levels per octave in the HOG pyramid
        threshold : `double`
            Minimum detection threshold. Detections with a score less than this
            value are not returned. Values can be negative.
        overlap : `double`, optional
            Minimum overlap in in latent positive search and
            non-maxima suppression.
            As discussed in the Face Detection Without Bells and Whistles paper,
            a sensible value for overlap is 0.3

        Returns
        ------
        bounding_boxes : `list` of `menpo.shape.PointDirectedGraph`
            The detected objects.
        """
        # Add the channel to a greyscale image.
        uint8_image = ensure_channel_axis(uint8_image)
        rects = detect_objects(self._ffld2_model, uint8_image,
                               padding=padding, interval=interval,
                               threshold=threshold, overlap=overlap)
        return [pointgraph_from_rect(r) for r in rects]


class FFLD2Detector(object):
    r"""
    A generic ffld2 detector.

    Wraps an ffld2 object detector inside the menpodetect framework and
    provides a clean interface to expose the ffld2 arguments.
    """
    def __init__(self, model):
        self._detector = _ffld2_detect(model)

    def __call__(self, image, greyscale=True, image_diagonal=None,
                 group_prefix='ffld2', padding=6, interval=5, threshold=0.5,
                 overlap=0.3):
        r"""
        Perform a detection using the cached ffdl2 detector.

        The detections will also be attached to the image as landmarks.

        Parameters
        ----------
        image : `menpo.image.Image`
            A Menpo image to detect. The bounding boxes of the detected objects
            will be attached to this image.
        greyscale : `bool`, optional
            Whether to convert the image to greyscale or not.
        image_diagonal : `int`, optional
            The total size of the diagonal of the image that should be used for
            detection. This is useful for scaling images up and down for
            detection.
        group_prefix : `str`, optional
            The prefix string to be appended to each each landmark group that is
            stored on the image. Each detection will be stored as group_prefix_#
            where # is a count starting from 0.
        padding : `int`, optional
            Amount of zero padding in HOG cells
        interval : `int`, optional
            Number of levels per octave in the HOG pyramid
        threshold : `double`, optional
            Minimum detection threshold. Detections with a score less than this
            value are not returned. Values can be negative.
        overlap : `double`, optional
            Minimum overlap in in latent positive search and
            non-maxima suppression.
            As discussed in the Face Detection Without Bells and Whistles paper,
            a sensible value for overlap is 0.3

        Returns
        ------
        bounding_boxes : `list` of `menpo.shape.PointDirectedGraph`
            The detected objects.
        """
        detect_partial = partial(self._detector, padding=padding,
                                 interval=interval, threshold=threshold,
                                 overlap=overlap)
        return detect(detect_partial, image, greyscale=greyscale,
                      image_diagonal=image_diagonal, group_prefix=group_prefix)


def load_ffld2_frontal_face_detector():
    r"""
    Load the ffld2 frontal face detector. This detector is the DPM baseline
    provided from [1]_.

    Returns
    -------
    detector : FFLD2Detector
        The frontal face detector.

    References
    ----------
    .. [1] M. Mathias and R. Benenson and M. Pedersoli and L. Van Gool
       Face detection without bells and whistles
       ECCV 2014
    """
    return FFLD2Detector(get_frontal_face_mixture_model())
