from menpodetect.pico import load_pico_frontal_face_detector
import menpo.io as mio

takeo = mio.import_builtin_asset.takeo_ppm()


def test_frontal_face_detector():
    takeo_copy = takeo.copy()
    pico_detector = load_pico_frontal_face_detector()
    pcs = pico_detector(takeo_copy)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['pico_0'].n_points == 4
