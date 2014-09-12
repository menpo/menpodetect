from __future__ import division
import dlib
from menpo.shape import PointCloud
from menpo.transform import UniformScale
import numpy as np

dlib_frontal_face_detector = None


def pointcloud_from_rect(rect):
    return PointCloud(np.array(((rect.top(), rect.left()),
                                (rect.bottom(), rect.right()))))


def dlib_detect_frontal_faces(image, add_as_landmarks=True, width=300):
    global dlib_frontal_face_detector
    if dlib_frontal_face_detector is None:
        dlib_frontal_face_detector = dlib.get_frontal_face_detector()
    dlib_image = image.as_greyscale(mode='average')
    did_rescale = False
    if width is not None and dlib_image.width > width:
        did_rescale = True
        scale_factor = width / dlib_image.width
        dlib_image = dlib_image.rescale(scale_factor)
    dlib_image = (dlib_image.pixels[..., 0] * 255.0).astype(np.uint8)
    faces = dlib_frontal_face_detector(dlib_image)
    pcs = [pointcloud_from_rect(f) for f in faces]
    if did_rescale:
        pcs = [UniformScale(1/scale_factor, n_dims=2).apply(pc) for pc in pcs]
    if add_as_landmarks:
        for i, pc in enumerate(pcs):
            image.landmarks['frontal_face_{:02d}'.format(i)] = pc
    return pcs
