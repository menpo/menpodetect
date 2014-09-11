import dlib
from menpo.shape import PointCloud
import numpy as np

dlib_frontal_face_detector = None


def pointcloud_from_rect(rect):
    return PointCloud(np.array(((rect.top(), rect.left()),
                                (rect.bottom(), rect.right()))))


def dlib_detect_frontal_faces(image, add_as_landmarks=True):
    global dlib_frontal_face_detector
    if dlib_frontal_face_detector is None:
        dlib_frontal_face_detector = dlib.get_frontal_face_detector()
    image_for_dlib = (image.as_greyscale().pixels[..., 0] *
                      255.0).astype(np.uint8)
    faces = dlib_frontal_face_detector(image_for_dlib)
    pcs = [pointcloud_from_rect(f) for f in faces]
    if add_as_landmarks:
        for i, pc in enumerate(pcs):
            image.landmarks['frontal_face_{:02d}'.format(i)] = pc
    return pcs
