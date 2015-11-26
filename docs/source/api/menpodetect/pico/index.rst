.. _api-pico-index:

:mod:`menpodetect.pico`
=======================
This module contains a wrapper of the detector provided by the Pico [1]_
project. In particular, it provides access to a frontal face detector that
implements the work from [2]_. At the moment no other Pico models can be loaded
due to a technical limitation in how the models are provided.

Pico is of particularly useful for images where the face has undergone in-plane
rotation as Pico is capable of performing in-plane detections.

Detection
---------

.. toctree::
  :maxdepth: 1

  PicoDetector
  load_pico_frontal_face_detector

References
----------
.. [1] https://github.com/nenadmarkus/pico
.. [2] N. Markus, M. Frljak, I. S. Pandzic, J. Ahlberg and R. Forchheimer,
       "Object Detection with Pixel Intensity Comparisons Organized in Decision
       Trees", http://arxiv.org/abs/1305.4537
