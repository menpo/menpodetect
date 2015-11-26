.. _api-opencv-index:

:mod:`menpodetect.opencv`
=========================
This module contains a wrapper of the detector provided by the OpenCV [1]_
project. At the moment, we assume the use of OpenCV v2.x and therefore
this detector will not be available for Python 3.x. We provide a number
of pre-trained models that have been provided by the OpenCV community, all
of which are implementations of the Viola-Jones method [2]_.

Detection
---------

.. toctree::
  :maxdepth: 1

  OpenCVDetector
  load_opencv_frontal_face_detector
  load_opencv_profile_face_detector
  load_opencv_eye_detector

References
----------
.. [1] http://opencv.org/
.. [2] Viola, Paul, and Michael Jones. "Rapid object detection using a boosted
       cascade of simple features." Computer Vision and Pattern Recognition,
       2001. CVPR 2001.
