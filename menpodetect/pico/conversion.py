from menpodetect.conversion import bounding_box


def pointgraph_from_circle(fitting):
    r"""
    Convert a Pico detection to a menpo.shape.PointDirectedGraph.
    This enforces a particular point ordering. The Pico detections are
    circles with a given diameter. Here we convert them to the tighest
    possible bounding box around the circle. No orientaton is currently
    provided.

    Parameters
    ----------
    fitting : `cypico.PicoDetection`
        The Pico detection to convert. Result of calling a pico detection.
        A namedtuple with a diameter and a centre.

    Returns
    -------
    bounding_box : `menpo.shape.PointDirectedGraph`
        A menpo PointDirectedGraph giving the bounding box.
    """
    diameter = fitting.diameter
    radius = diameter / 2.0
    y, x = fitting.center
    y -= radius
    x -= radius
    return bounding_box((y, x), (y + diameter, x + diameter))
