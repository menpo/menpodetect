from menpodetect.ffld2 import load_ffld2_frontal_face_detector
import menpo.io as mio

takeo = mio.import_builtin_asset.takeo_ppm()


def test_frontal_face_detector():
    takeo_copy = takeo.copy()
    ffld2_detector = load_ffld2_frontal_face_detector()
    pcs = ffld2_detector(takeo_copy, threshold=2)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['ffld2_0'].n_points == 4


def test_frontal_face_detector_rgb():
    takeo_copy = takeo.copy()
    assert takeo_copy.n_channels == 3
    ffld2_detector = load_ffld2_frontal_face_detector()
    pcs = ffld2_detector(takeo_copy, greyscale=False, threshold=2)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['ffld2_0'].n_points == 4
