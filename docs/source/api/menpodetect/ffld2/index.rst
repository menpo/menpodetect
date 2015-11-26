.. _api-ffld2-index:

:mod:`menpodetect.ffld2`
========================
This module contains a wrapper of the detector provided by the FFLD2 [1]_ [2]_
project. This module also provides the very powerful DPM model provided by
[3]_.

The FFLD2 detector is also trainable.

Detection
---------

.. toctree::
  :maxdepth: 1

  FFLD2Detector
  load_ffld2_frontal_face_detector

Training
--------

.. toctree::
  :maxdepth: 1

  train_ffld2_detector


References
----------
.. [1] https://www.idiap.ch/scientific-research/resources/exact-acceleration-of-linear-object-detectors
.. [2] Dubout, Charles, and François Fleuret. "Exact acceleration of linear
       object detectors." Computer Vision–ECCV 2012. Springer Berlin Heidelberg,
       2012. 301-311.
.. [3] Mathias, Markus, et al. "Face detection without bells and whistles."
       Computer Vision–ECCV 2014. Springer International Publishing, 2014.
       720-735.
