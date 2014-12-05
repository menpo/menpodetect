from menpo.shape import PointDirectedGraph
import numpy as np


def pointgraph_from_circle(fitting):
    diameter = fitting.diameter
    radius = diameter / 2.0
    y, x = fitting.center
    y -= radius
    x -= radius
    return PointDirectedGraph(np.array(((y, x),
                                        (y + diameter, x),
                                        (y + diameter, x + diameter),
                                        (y, x + diameter))),
                              np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
