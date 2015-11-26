from menpo.shape import PointDirectedGraph
from menpodetect.detect import (detect, menpo_image_to_uint8)
import menpo.io as mio
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises


takeo = mio.import_builtin_asset.takeo_ppm()
takeo_uint8 = mio.import_image(mio.data_path_to('takeo.ppm'), normalise=False)
fake_box = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
fake_detector = lambda x: ([PointDirectedGraph.init_from_edges(
    fake_box.copy(),
    np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))])


def test_rescaling_image():
    takeo_copy = takeo.copy()
    ratio = 200.0 / takeo_copy.diagonal()
    pcs = detect(fake_detector, takeo_copy, image_diagonal=200)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['object_0'][None].n_points == 4
    assert_allclose(takeo_copy.landmarks['object_0'][None].points,
                    fake_box * (1.0 / ratio), atol=10e-2)


def test_passing_uint8_image():
    takeo_copy = takeo_uint8.copy()
    pcs = detect(fake_detector, takeo_copy, greyscale=False)
    assert len(pcs) == 1
    assert takeo_copy.n_channels == 3
    assert takeo_copy.landmarks['object_0'][None].n_points == 4


@raises(ValueError)
def test_passing_uint8_image_greyscale_raises():
    takeo_copy = takeo_uint8.copy()
    pcs = detect(fake_detector, takeo_copy, greyscale=True)


def test_image_to_uint8():
    takeo_copy = takeo.copy()
    np_im = menpo_image_to_uint8(takeo_copy)
    shi = takeo_copy.pixels.shape
    shnp = np_im.shape
    assert np_im.dtype == np.uint8
    assert (shi[0] == shnp[2] and shi[1] == shnp[0] and shi[2] == shnp[1])