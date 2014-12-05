from menpo.shape import PointDirectedGraph
from menpodetect.detect import detect
import menpo.io as mio
import numpy as np
from numpy.testing import assert_allclose


takeo = mio.import_builtin_asset.takeo_ppm()
fake_box = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
fake_detector = lambda x: ([PointDirectedGraph(
    fake_box.copy(),
    np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))])


def test_rescaling_image():
    takeo_copy = takeo.copy()
    ratio = 200.0 / takeo_copy.diagonal
    pcs = detect(fake_detector, takeo_copy, image_diagonal=200)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['object_0'][None].n_points == 4
    assert_allclose(takeo_copy.landmarks['object_0'][None].points,
                    fake_box * (1.0 / ratio), atol=10e-2)
