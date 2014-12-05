from __future__ import division
from functools import partial

import cv2
from pathlib import Path

from menpodetect.detect import detect
from menpodetect.compatibility import STRING_TYPES
from .conversion import (pointgraph_from_rect, opencv_frontal_face_path,
                         opencv_profile_face_path, opencv_eye_path)


class _opencv_detect(object):
    def __init__(self, model):
        if isinstance(model, STRING_TYPES) or isinstance(model, Path):
            m_path = Path(model)
            if not Path(m_path).exists():
                raise ValueError('Model {} does not exist.'.format(m_path))
            model = cv2.CascadeClassifier(str(m_path))
        self._opencv_model = model

    def __call__(self, uint8_image, scale_factor=1.1, min_neighbours=5,
                 min_size=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE):
        rects = self._opencv_model.detectMultiScale(
            uint8_image, scaleFactor=scale_factor, minNeighbors=min_neighbours,
            minSize=min_size, flags=flags)
        return [pointgraph_from_rect(r) for r in rects]


class OpenCVDetector(object):
    def __init__(self, model):
        self._detector = _opencv_detect(model)

    def __call__(self, image, image_diagonal=None, group_prefix='object',
                 scale_factor=1.1, min_neighbours=5,
                 min_size=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE):
        detect_partial = partial(self._detector, scale_factor=scale_factor,
                                 min_neighbours=min_neighbours,
                                 min_size=min_size, flags=flags)
        return detect(detect_partial, image, greyscale=True,
                      image_diagonal=image_diagonal, group_prefix=group_prefix)


def load_opencv_frontal_face_detector():
    return OpenCVDetector(opencv_frontal_face_path)


def load_opencv_profile_face_detector():
    return OpenCVDetector(opencv_profile_face_path)


def load_opencv_eye_detector():
    return OpenCVDetector(opencv_eye_path)
