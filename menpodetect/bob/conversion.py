import bob.ip.facedetect
from menpo.shape import bounding_box


def bb_to_pointgraph(bb):
    r"""
    Convert a `bob.ip.facedetect.BoundingBox` to a
    `menpo.shape.PointDirectedGraph`.
    This enforces a particular point ordering.

    Parameters
    ----------
    bb : `bob.ip.facedetect.BoundingBox`
        The bounding box to convert.

    Returns
    -------
    bounding_box : `menpo.shape.PointDirectedGraph`
        A menpo PointDirectedGraph giving the bounding box.
    """
    return bounding_box(bb.topleft_f, bb.bottomright_f)


def pointgraph_to_bb(pg):
    r"""
    Convert a `menpo.shape.PointCloud` to a `bob.ip.facedetect.BoundingBox`.

    Parameters
    ----------
    pg : `menpo.shape.PointDirectedGraph`
        The Menpo PointDirectedGraph to convert into a rect. No check is done
        to see if the PointDirectedGraph actually is a rectangle.

    Returns
    -------
    bounding_box : `bob.ip.facedetect.BoundingBox`
        A bob BoundingBox.
    """
    return bob.ip.facedetect.BoundingBox(pg.bounds()[0], pg.range())
