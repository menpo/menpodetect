import numpy as np
from menpo.shape import PointDirectedGraph


def bounding_box(min_point, max_point):
    r"""
    Return the bounding box from the given minimum and maximum points.
    The the first point (0) will be nearest the origin. Therefore, the point
    adjacency is:

    ::

        0<--3
        |   ^
        |   |
        v   |
        1-->2

    Returns
    -------
    bounding_box : :map:`PointDirectedGraph`
        The axis aligned bounding box from the given points.
    """
    return PointDirectedGraph.init_from_edges(
        np.array([min_point, [max_point[0], min_point[1]],
                  max_point, [min_point[0], max_point[1]]]),
        np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), copy=False)
