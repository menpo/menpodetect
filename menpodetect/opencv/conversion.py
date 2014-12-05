from menpo.shape import PointDirectedGraph
import numpy as np
from pathlib import Path

from menpodetect.paths import models_dir_path


opencv_models_path = Path(models_dir_path(), 'opencv')
opencv_frontal_face_path = Path(opencv_models_path,
                                'haarcascade_frontalface_alt.xml')
opencv_profile_face_path = Path(opencv_models_path,
                                'haarcascade_profileface.xml')
opencv_eye_path = Path(opencv_models_path,
                       'haarcascade_eye.xml')


def pointgraph_from_rect(rect):
    x, y, w, h = rect
    return PointDirectedGraph(np.array(((y, x),
                                        (y + h, x),
                                        (y + h, x + w),
                                        (y, x + w))),
                              np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
