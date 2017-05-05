import unittest
try:
    from menpodetect.bob import load_bob_frontal_face_detector
    MISSING_BOB = False
except ImportError:
    MISSING_BOB = True

import menpo.io as mio

takeo = mio.import_builtin_asset.takeo_ppm()


@unittest.skipIf(MISSING_BOB, "requires bob.ip.facedetect")
def test_frontal_face_detector():
    takeo_copy = takeo.copy()
    bob_detector = load_bob_frontal_face_detector()
    pcs = bob_detector(takeo_copy)
    assert len(pcs) == 2
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['bob_0'].n_points == 4


@unittest.skipIf(MISSING_BOB, "requires bob.ip.facedetect")
def test_frontal_face_detector_no_result():
    takeo_copy = takeo.copy()
    bob_detector = load_bob_frontal_face_detector()
    pcs = bob_detector(takeo_copy, threshold=float('inf'))
    assert len(pcs) == 0


@unittest.skipIf(MISSING_BOB, "requires bob.ip.facedetect")
def test_frontal_face_detector_rgb():
    takeo_copy = takeo.copy()
    bob_detector = load_bob_frontal_face_detector()
    pcs = bob_detector(takeo_copy, greyscale=False)
    assert len(pcs) == 2
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['bob_0'].n_points == 4


@unittest.skipIf(MISSING_BOB, "requires bob.ip.facedetect")
def test_frontal_face_detector_threshold():
    takeo_copy = takeo.copy()
    bob_detector = load_bob_frontal_face_detector()
    pcs = bob_detector(takeo_copy, threshold=30)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['bob_0'].n_points == 4


@unittest.skipIf(MISSING_BOB, "requires bob.ip.facedetect")
def test_frontal_face_detector_minimum_overlap():
    takeo_copy = takeo.copy()
    bob_detector = load_bob_frontal_face_detector()
    pcs = bob_detector(takeo_copy, minimum_overlap=0.5)
    assert len(pcs) == 4
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['bob_0'].n_points == 4
