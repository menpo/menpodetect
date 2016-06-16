Welcome
=======
**Welcome to the MenpoDetect documentation!**

MenpoDetect is a Python package designed to make object detection, in particular
face detection, simple. MenpoDetect relies on the core package of Menpo, and
thus the output of MenpoDetect is always assumed to be Menpo core types. If you
aren't sure what Menpo is, please take a look over at
`Menpo.org <http://www.menpo.org/>`_.

A short example is often more illustrative than a verbose explanation. Let's
assume that you want to load a set of images and that we want to detect
all the faces in the images. We could do this using the Viola-Jones detector
provided by OpenCV as follows:

.. code-block:: python

    import menpo.io as mio
    from menpodetect import load_opencv_frontal_face_detector

    opencv_detector = load_opencv_frontal_face_detector()

    images = []
    for image in mio.import_images('./images_folder'):
        opencv_detector(image)
        images.append(image)

Where we use Menpo to load the images from disk and then detect as many
faces as possible using OpenCV. The detections are automatically attached
to each image in the form of a set of landmarks. These are then easily viewed
within a Jupyter notebook using the MenpoWidgets package:

.. code-block:: python

    %matplotlib inline
    from menpowidgets import visualize_images

    visualize_images(images)

Supported Detectors
-------------------
MenpoDetect was not designed for performing novel object detection research.
Therefore, it relies on a number of existing packages and merely normalizes
the inputs and outputs so that they are consistent with core Menpo types.
These projects are as follows:

  - `cypico <https://github.com/menpo/cypico>`_ - Provides the detection
    capabilities of the Pico detector. This has similar performance to a
    Viola-Jones detector but allows for in-plane rotation detection (detecting
    faces that are rotated in the Roll angle).
  - `cyffld2 <https://github.com/menpo/cyffld2>`_ - Provides the detection
    capabilities of the FFLD2 project. This is an FFT based DPM detection
    package. It also ships with the highly performant detector provided by
    `Mathias et. al <http://markusmathias.bitbucket.org/2014_eccv_face_detection/>`_.
  - `dlib <http://dlib.net>`_ - Provides the detection capabilities of the
    Dlib project. This is a HOG-SVM based detector that will return a very
    low number of false positives.
  - `OpenCV <http://opencv.org>`_ - Provides the detection capabilities of the
    OpenCV project. This is only available for Python 2.x due to limitations
    of the OpenCV project. OpenCV implements a Viola-Jones detector
    and provides models for both frontal and profile faces as well as eyes.
  - `bob.ip.facedetect <https://pythonhosted.org/bob.ip.facedetect/>`_ -
    Provides the detection capabilities of the Bob detector.
    This detector is based on the PhD thesis of Cosmin Atanasoaei of
    EPFL and is comprised of an ensemble of weak LBP classifiers.
    **Not currently shipped using conda and therefore must be installed
    independently.**

We would be very happy to see this collection expand, so pull requests
are very welcome!

.. toctree::
  :maxdepth: 2
  :hidden:

  api/index
