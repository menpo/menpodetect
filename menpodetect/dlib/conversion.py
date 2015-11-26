import dlib
from menpo.shape import bounding_box


def rect_to_pointgraph(rect):
    r"""
    Convert a dlib.rect to a menpo.shape.PointDirectedGraph.
    This enforces a particular point ordering.

    Parameters
    ----------
    rect : `dlib.rect`
        The bounding box to convert.

    Returns
    -------
    bounding_box : `menpo.shape.PointDirectedGraph`
        A menpo PointDirectedGraph giving the bounding box.
    """
    return bounding_box((rect.top(), rect.left()),
                        (rect.bottom(), rect.right()))


def pointgraph_to_rect(pg):
    r"""
    Convert a `menpo.shape.PointCloud` to a `dlib.rect`.

    Parameters
    ----------
    pg : `menpo.shape.PointDirectedGraph`
        The Menpo PointDirectedGraph to convert into a rect. No check is done
        to see if the PointDirectedGraph actually is a rectangle.

    Returns
    -------
    bounding_rect : `dlib.rect`
        A dlib Rectangle.
    """
    min_p, max_p = pg.bounds()
    return dlib.rectangle(left=int(min_p[1]), top=int(min_p[0]),
                          right=int(max_p[1]), bottom=int(max_p[0]))
