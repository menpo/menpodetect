from __future__ import division
import dlib
from menpo.shape import PointDirectedGraph
import numpy as np
from pathlib import Path
from menpodetect.detectors import detect
from functools import partial
from menpodetect.compatibility import STRING_TYPES


def pointgraph_from_rect(rect):
    return PointDirectedGraph(np.array(((rect.top(), rect.left()),
                                        (rect.bottom(), rect.left()),
                                        (rect.bottom(), rect.right()),
                                        (rect.top(), rect.right()))),
                              np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))


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
        return [pointgraph_from_rect(r) for r in rects]


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
