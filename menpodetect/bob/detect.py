from __future__ import division
from functools import partial
from pathlib import Path

from menpo.base import MenpoMissingDependencyError

try:
    import bob.ip.facedetect
    from bob.ip.facedetect.detector.cascade import Cascade
except ImportError:
    raise MenpoMissingDependencyError('bob.ip.facedetect')

from menpodetect.detect import detect
from menpodetect.compatibility import STRING_TYPES
from .conversion import bb_to_pointgraph


class _bob_detect(object):
    r"""
    A utility callable that allows the caching of a bob detector.

    This callable is important for presenting the correct parameters to the
    user. It also marshalls the return type of the detector back to
    `menpo.shape.PointDirectedGraph`.

    Parameters
    ----------
    model : `Path` or `str` or `bob.ip.facedetect.detector.cascade.Cascade`
        Either a path to a `bob.ip.facedetect.detector.cascade.Cascade` or a
        `bob.ip.facedetect.detector.cascade.Cascade` itself.

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
            model = Cascade(cascade_file=str(m_path))
        self._bob_model = model

    def __call__(self, uint8_image, threshold=20, minimum_overlap=0.2):
        r"""
        Perform a detection using the cached bob detector.

        Parameters
        ----------
        uint8_image : `ndarray`
            An RGB (3 Channels) or Greyscale (1 Channel) numpy array of uint8
            with **channels at the front**.
        threshold : `float`, optional
            The threshold of the quality of detected faces.
            Detections with a quality lower than this value will not be
            considered. Higher thresholds will not detect all faces, while lower
            thresholds will generate false detections.
        ``minimum_overlap`` : `float` ``[0, 1]``
            Computes the best detection using the given minimum overlap.

        Returns
        ------
        bounding_boxes : `list` of `menpo.shape.PointDirectedGraph`
            The detected objects.
        """
        result = bob.ip.facedetect.detect_all_faces(
            uint8_image, cascade=self._bob_model, threshold=threshold,
            minimum_overlap=minimum_overlap)
        if result is None:
            bbs = []
        else:
            bbs, confidences = result
        return [bb_to_pointgraph(b) for b in bbs]


class BobDetector(object):
    r"""
    A Bob cascade detector.

    Wraps a bob cascade detector inside the menpodetect framework and provides
    a clean interface to expose the bob arguments.
    """
    def __init__(self, model):
        self._detector = _bob_detect(model)

    def __call__(self, image, greyscale=False, image_diagonal=None,
                 group_prefix='bob', threshold=20, minimum_overlap=0.2):
        r"""
        Perform a detection using the cached bob detector.

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
        threshold : `float`, optional
            The threshold of the quality of detected faces.
            Detections with a quality lower than this value will not be
            considered. Higher thresholds will not detect all faces, while lower
            thresholds will generate false detections.
        ``minimum_overlap`` : `float` ``[0, 1]``
            Computes the best detection using the given minimum overlap.

        Returns
        ------
        bounding_boxes : `list` of `menpo.shape.PointDirectedGraph`
            The detected objects.
        """
        detect_partial = partial(self._detector, threshold=threshold,
                                 minimum_overlap=minimum_overlap)
        return detect(detect_partial, image, greyscale=greyscale,
                      image_diagonal=image_diagonal, group_prefix=group_prefix,
                      channels_at_back=False)


def load_bob_frontal_face_detector():
    r"""
    Load the bob frontal face detector.

    Returns
    -------
    detector : `BobDetector`
        The frontal face detector.
    """
    return BobDetector(bob.ip.facedetect.default_cascade())
