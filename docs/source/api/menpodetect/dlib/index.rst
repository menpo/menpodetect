.. _api-dlib-index:

:mod:`menpodetect.dlib`
=======================
This module contains a wrapper of the detector provided by the Dlib [1]_ [2]_
project. In particular, it provides access to a frontal face detector that
implements the work from [3]_. The Dlib detector is also trainable.

Detection
---------

.. toctree::
  :maxdepth: 1

  DlibDetector
  load_dlib_frontal_face_detector

Training
--------

.. toctree::
  :maxdepth: 1

  train_dlib_detector


References
----------
.. [1] http://dlib.net/
.. [2] King, Davis E. "Dlib-ml: A machine learning toolkit." The Journal of
       Machine Learning Research 10 (2009): 1755-1758.
.. [3] King, Davis E. "Max-Margin Object Detection." arXiv preprint
       arXiv:1502.00046 (2015).
