from menpodetect.opencv import (load_opencv_frontal_face_detector,
                                load_opencv_eye_detector)
import menpo.io as mio

takeo = mio.import_builtin_asset.takeo_ppm()


def test_frontal_face_detector():
    takeo_copy = takeo.copy()
    opencv_detector = load_opencv_frontal_face_detector()
    pcs = opencv_detector(takeo_copy)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['opencv_0'].n_points == 4


def test_frontal_face_detector_min_neighbors():
    takeo_copy = takeo.copy()
    opencv_detector = load_opencv_frontal_face_detector()
    pcs = opencv_detector(takeo_copy, min_neighbours=100)
    assert len(pcs) == 0
    assert takeo_copy.n_channels == 3


def test_eye_detector():
    takeo_copy = takeo.copy()
    opencv_detector = load_opencv_eye_detector()
    pcs = opencv_detector(takeo_copy, min_size=(5, 5), min_neighbours=0)
    assert len(pcs) > 0
    assert takeo_copy.n_channels == 3
    # This is because appyveyor and travis (automated testing) return
    # a different number of detections
    first_l = list(takeo_copy.landmarks.items_matching('opencv_*'))[0][1]
    assert first_l.n_points == 4
