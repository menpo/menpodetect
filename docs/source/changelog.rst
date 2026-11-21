.. _changelog:

Changelog
#########

For releases before 0.7.0 see the `GitHub releases <https://github.com/menpo/menpodetect/releases>`_.

0.7.2 (2026/11/21)
------------------

**Fixed.** The error for a detector whose library is not installed told you to
``pip install opencv`` (not a PyPI name) or ``conda install dlib``, and was only
raised by ``import menpodetect.opencv.detect``. ``from menpodetect.opencv import
OpenCVDetector`` (and the other names, and the same for ``menpodetect.dlib``) just
said "cannot import name". Asking for any of those names now raises an
``ImportError`` that says to ``pip install "menpodetect[opencv]"`` (or
``[dlib]``) and shows the original error. If the library is installed but cannot
be loaded (OpenCV without ``libGL``, for example), it says so instead of calling it
missing. Importing ``menpodetect`` itself still works without the libraries, and
a name that is not one of the detectors is still an ``AttributeError``.

Github Pull Requests
....................
- `#26`_ Say which extra to install when OpenCV or dlib cannot be imported (@patricksnape)

.. _#26: https://github.com/menpo/menpodetect/pull/26

0.7.1 (2026/11/19)
------------------

Documentation and release checks only, no change to the package. The legacy
``readthedocs.yml`` (a Python 3.7 conda environment that Read the Docs preferred
over the new ``.readthedocs.yaml``) is removed and guarded against, and a
release is now blocked unless Read the Docs has built the commit as ``latest``.

Github Pull Requests
....................
- `#24`_ Remove the legacy readthedocs.yml; block releases when Read the Docs cannot build (@patricksnape)

.. _#24: https://github.com/menpo/menpodetect/pull/24

0.7.0 (2026/11/18)
------------------

The first release built and published from GitHub Actions with PyPI trusted
publishing, so that menpodetect can be installed alongside menpo 0.12.

**Requirements.** Python 3.10 or newer, and ``menpo>=0.12`` (it was
``menpo<0.12``). ``pyproject.toml`` with ``setuptools-scm`` (the version comes
from the git tag) replaces ``setup.py`` and the vendored miniver, whose
``distutils`` import fails on Python 3.12 and newer. The detectors are optional
extras: ``menpodetect[opencv]`` (the Haar cascade face and eye detectors) and
``menpodetect[dlib]`` (the frontal face detector and training).

**OpenCV 5.** OpenCV 5 moved ``cv2.CascadeClassifier`` out of the main modules
into opencv_contrib, so the main ``opencv-python-headless`` 5.x wheel has no
Haar cascades. The ``opencv`` extra is therefore
``opencv-contrib-python-headless>=4.5`` (a superset of the main wheel on 4.x),
and the check that rejected every OpenCV version except 2 to 4 is removed.

**Tests and CI.** The dlib and OpenCV tests are skipped when the library is
missing, ``test_black.py`` is replaced by ruff, and CI runs on Linux, macOS and
Windows with Python 3.10 and 3.14, with a dlib job that fails if anything is
skipped. The documentation builds on Read the Docs again.

Github Pull Requests
....................
- `#23`_ Move to pyproject.toml, GitHub Actions CI and trusted-publishing releases (@patricksnape)

.. _#23: https://github.com/menpo/menpodetect/pull/23
