from __future__ import division
from functools import partial

import dlib
from pathlib import Path
from menpodetect.detectors import detect
from menpodetect.compatibility import STRING_TYPES
from .conversion import rect_to_pointgraph


class _dlib_detect(object):
    def __init__(self, model):
        if isinstance(model, STRING_TYPES) or isinstance(model, Path):
            m_path = Path(model)
            if not Path(m_path).exists():
                raise ValueError('Model {} does not exist.'.format(m_path))
            model = dlib.simple_object_detector(str(m_path))
        self._dlib_model = model

    def __call__(self, uint8_image, n_upscales=0):
        rects = self._dlib_model(uint8_image, n_upscales)
        return [rect_to_pointgraph(r) for r in rects]


class DlibDetector(object):
    def __init__(self, model):
        self._detector = _dlib_detect(model)

    def __call__(self, image, greyscale=True, image_diagonal=None,
                 group_prefix='object', n_upscales=0):
        detect_partial = partial(self._detector, n_upscales=n_upscales)
        return detect(detect_partial, image, greyscale=greyscale,
                      image_diagonal=image_diagonal, group_prefix=group_prefix)


def load_dlib_frontal_face_detector():
    return DlibDetector(dlib.get_frontal_face_detector())
