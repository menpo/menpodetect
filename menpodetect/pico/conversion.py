from menpo.shape import PointDirectedGraph
import numpy as np


def pointgraph_from_circle(fitting):
    y, x = fitting.center
    radius = fitting.diameter / 2.0
    return PointDirectedGraph(np.array(((y, x),
                                        (y + radius, x),
                                        (y + radius, x + radius),
                                        (y, x + radius))),
                              np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
