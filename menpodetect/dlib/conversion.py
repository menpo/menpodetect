from menpo.shape import PointDirectedGraph
import numpy as np
import dlib


def rect_to_pointgraph(rect):
    return PointDirectedGraph(np.array(((rect.top(), rect.left()),
                                        (rect.bottom(), rect.left()),
                                        (rect.bottom(), rect.right()),
                                        (rect.top(), rect.right()))),
                              np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))


def pointgraph_to_rect(pg):
    min_p, max_p = pg.bounds()
    return dlib.rectangle(left=int(min_p[1]), top=int(min_p[0]),
                          right=int(max_p[1]), bottom=int(max_p[0]))
