from menpo.shape import bounding_box
from menpo.transform import rotate_ccw_about_centre


def pointgraph_from_circle(fitting, axis_aligned_bb=True):
    r"""
    Convert a Pico detection to a menpo.shape.PointDirectedGraph.
    This enforces a particular point ordering. The Pico detections are
    circles with a given diameter. Here we convert them to the tightest
    possible bounding box around the circle. No orientation is currently
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
    bb = bounding_box((y, x), (y + diameter, x + diameter))
    if not axis_aligned_bb:
        t = rotate_ccw_about_centre(bb, fitting.orientation, degrees=False)
        bb = t.apply(bb)
    return bb
