from menpo.shape import PointDirectedGraph
import numpy as np


def pointgraph_from_circle(fitting):
    r"""
    Convert a Pico detection to a menpo.shape.PointDirectedGraph.
    This enforces a particular point ordering. The Pico detections are
    circles with a given diameter. Here we convert them to the tighest
    possible bounding box around the circle. No orientaton is currently
    provided.

    Parameters
    ----------
    fitting : cypico.PicoDetection
        The Pico detection to convert. Result of calling a pico detection.
        A namedtuple with a diameter and a centre.

    Returns
    -------
    bounding_box : menpo.shape.PointDirectedGraph
        A menpo PointDirectedGraph giving the bounding box.
    """
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
