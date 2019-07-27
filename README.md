
<p align="center">
  <img src="menpodetect-logo.png" alt="menpodetect" width="30%"></center>
  <br><br>
  <a href="https://pypi.python.org/pypi/menpodetect"><img src="http://img.shields.io/pypi/v/menpodetect.svg?style=flat" alt="PyPI Release"/></a>
  <a href="https://github.com/menpo/menpodetect/blob/master/LICENSE.txt"><img src="http://img.shields.io/badge/License-BSD-green.svg" alt="BSD License"/></a>
  <br>
  <img src="https://img.shields.io/badge/Python-3.6-green.svg" alt="Python 3.6 Support"/>
  <img src="https://img.shields.io/badge/Python-3.7-green.svg" alt="Python 3.7 Support"/>
</p>


menpodetect - Simple object detection
=====================================
Simple object detection within the [Menpo Project](http://www.menpo.org/). We do not attempt
to implement novel techniques, but instead wrap existing projects so that they
integrate nicely with Menpo. At the moment the current libraries are wrapped:

  - **[dlib](http://dlib.net/) (Boost Software License - Version 1.0)**  
    Frontal face detection, arbitrary dlib models and training code is all
    wrapped.
  - **[opencv](http://opencv.org/) (BSD)**
    Frontal face detection, profile face detection, eye detection and arbitrary
    OpenCV cascade files (via loading from disk) are all provided.

Important
---------
This project aims to wrap existing object detection libraries for easy
integration with Menpo. The core project is under a BSD license, but since
other projects are wrapped, they may not be compatible with this BSD license.
Therefore, we urge caution be taken when interacting with this library for
non-academic purposes.

Installation
------------
Here in the Menpo team, we are firm believers in making installation as simple
as possible. Unfortunately, we are a complex project that relies on satisfying
a number of complex 3rd party library dependencies. The default Python packing
environment does not make this an easy task. Therefore, we evangelise the use
of the conda ecosystem, provided by
[Anaconda](https://store.continuum.io/cshop/anaconda/). In order to make things
as simple as possible, we suggest that you use conda too! To try and persuade
you, go to the [Menpo website](http://www.menpo.io/installation/) to find
installation instructions for all major platforms.

If you want to try pip installing this package, note that you will need
to satisfy the dependencies as specified in the meta.yaml BEFORE install.

#### Build Status

|  CI Host |                       OS                  |                      Build Status                     |
|:--------:|:-----------------------------------------:|:-----------------------------------------------------:|
| Travis   | Ubuntu 16.04 (x64) and OSX 10.12 (x64)    | [![Travis Build Status][travis_shield]][travis]       |


[travis]: https://travis-ci.org/menpo/menpodetect
[travis_shield]: http://img.shields.io/travis/menpo/menpodetect.svg?style=flat


Documentation
-------------
See our documentation on [ReadTheDocs](http://menpodetect.readthedocs.org)
