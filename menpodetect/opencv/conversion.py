from menpo.shape import PointDirectedGraph
import numpy as np
from pathlib import Path

from menpodetect.paths import models_dir_path


# Paths to the OpenCV shipped with menpodetect
opencv_models_path = Path(models_dir_path(), 'opencv')
opencv_frontal_face_path = Path(opencv_models_path,
                                'haarcascade_frontalface_alt.xml')
opencv_profile_face_path = Path(opencv_models_path,
                                'haarcascade_profileface.xml')
opencv_eye_path = Path(opencv_models_path,
                       'haarcascade_eye.xml')


def pointgraph_from_rect(rect):
    r"""
    Convert an opencv detection to a menpo.shape.PointDirectedGraph.
    This enforces a particular point ordering.

    Parameters
    ----------
    rect : tuple
        The bounding box to convert. Result of calling an opencv detection.

    Returns
    -------
    bounding_box : menpo.shape.PointDirectedGraph
        A menpo PointDirectedGraph giving the bounding box.
    """
    x, y, w, h = rect
    return PointDirectedGraph(np.array(((y, x),
                                        (y + h, x),
                                        (y + h, x + w),
                                        (y, x + w))),
                              np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
