from __future__ import division
from functools import partial

from cypico import detect_objects, detect_frontal_faces
from pathlib import Path

from menpodetect.detect import detect
from menpodetect.compatibility import STRING_TYPES
from menpodetect.pico.conversion import pointgraph_from_circle


class _pico_detect(object):
    def __init__(self, model):
        if isinstance(model, STRING_TYPES) or isinstance(model, Path):
            m_path = Path(model)
            if not Path(m_path).exists():
                raise ValueError('Model {} does not exist.'.format(m_path))
            else:
                raise ValueError('Loading Pico trained models from disk '
                                 'is not currently supported.')
        self._pico_model = model

    def __call__(self, uint8_image, max_detections=100, orientations=0.0,
                 scale_factor=1.2, stride_factor=0.1,
                 min_size=100, confidence_cutoff=3.0):
        fittings = detect_objects(
            uint8_image, self._pico_model, max_detections=max_detections,
            orientations=orientations, scale_factor=scale_factor,
            stride_factor=stride_factor, min_size=min_size,
            confidence_cutoff=confidence_cutoff)
        return [pointgraph_from_circle(f) for f in fittings]


class _pico_face_detect(_pico_detect):
    def __call__(self, uint8_image, max_detections=100, orientations=0.0,
                 scale_factor=1.2, stride_factor=0.1,
                 min_size=100, confidence_cutoff=3.0):
        fittings = detect_frontal_faces(
            uint8_image, max_detections=max_detections,
            orientations=orientations, scale_factor=scale_factor,
            stride_factor=stride_factor, min_size=min_size,
            confidence_cutoff=confidence_cutoff)
        return [pointgraph_from_circle(f) for f in fittings]


# This is a slightly strange model because at the moment cypico is not
# really able to load arbitrary pico models. This is due to the fact that
# pico models are just saved down as raw data and loaded into a char* array
# then case into the members of the C-struct they actually represent. To make
# this cleaner, the C-struct would need to be created in cypico so we could
# properly load and save pico models
class PicoDetector(object):
    def __init__(self, model, detector=_pico_detect):
        self._detector = detector(model)

    def __call__(self, image, image_diagonal=None, group_prefix='object',
                 max_detections=100, orientations=0.0, scale_factor=1.2,
                 stride_factor=0.1, min_size=100, confidence_cutoff=3.0):
        detect_partial = partial(self._detector, max_detections=max_detections,
                                 orientations=orientations,
                                 scale_factor=scale_factor,
                                 stride_factor=stride_factor, min_size=min_size,
                                 confidence_cutoff=confidence_cutoff)
        return detect(detect_partial, image, greyscale=True,
                      image_diagonal=image_diagonal, group_prefix=group_prefix)


def load_pico_frontal_face_detector():
    return PicoDetector(None, detector=_pico_face_detect)
