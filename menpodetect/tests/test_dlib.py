from menpodetect.dlib import load_dlib_frontal_face_detector
import menpo.io as mio

takeo = mio.import_builtin_asset.takeo_ppm()


def test_frontal_face_detector():
    takeo_copy = takeo.copy()
    dlib_detector = load_dlib_frontal_face_detector()
    pcs = dlib_detector(takeo_copy)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['dlib_0'].n_points == 4


def test_frontal_face_detector_rgb():
    takeo_copy = takeo.copy()
    assert takeo_copy.n_channels == 3
    dlib_detector = load_dlib_frontal_face_detector()
    pcs = dlib_detector(takeo_copy, greyscale=False)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['dlib_0'].n_points == 4


def test_frontal_face_detector_upscales():
    takeo_copy = takeo.copy()
    dlib_detector = load_dlib_frontal_face_detector()
    pcs = dlib_detector(takeo_copy, n_upscales=1)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['dlib_0'].n_points == 4
