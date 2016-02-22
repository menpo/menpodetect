[![Coverage Status][coveralls_shield]][coveralls]
[![PyPI Release][pypi_shield]][pypi]
[![BSD License][bsd_shield]][bsd]


![Python 2.7 Support][python27]
![Python 3.4 Support][python34]
![Python 3.5 Support][python35]

[coveralls]: https://coveralls.io/r/menpo/menpodetect
[coveralls_shield]: http://img.shields.io/coveralls/menpo/menpodetect.svg?style=flat
[pypi]: https://pypi.python.org/pypi/menpodetect
[pypi_shield]: http://img.shields.io/pypi/v/menpodetect.svg?style=flat
[bsd]: https://github.com/menpo/menpodetect/blob/master/LICENSE.txt
[bsd_shield]: http://img.shields.io/badge/License-BSD-green.svg
[python27]: https://img.shields.io/badge/Python-2.7-green.svg
[python34]: https://img.shields.io/badge/Python-3.4-green.svg
[python35]: https://img.shields.io/badge/Python-3.5-green.svg

menpodetect - Simple object detection
=====================================
Simple object detection within the Menpo project environment. We do not attempt 
to implement novel techniques, but instead wrap existing projects so that they 
integrate nicely with Menpo. At the moment the current libraries are wrapped:

  - **[dlib](http://dlib.net/) (Boost Software License - Version 1.0)**  
    Frontal face detection, arbitrary dlib models and training code is all
    wrapped.
  - **[opencv](http://opencv.org/) (BSD)**
    Frontal face detection, profile face detection, eye detection and arbitrary
    OpenCV cascade files (via loading from disk) are all provided.
  - **[pico](https://github.com/nenadmarkus/pico) (Academic Only)**
    Frontal face detection and arbitrary pico models are provided. Loading
    arbitrary Pico models is likely to be very difficult, however.
  - **[ffld2](http://charles.dubout.ch/en/index.html) (GNU AGPL)**
    Frontal face detection using the DPM Baseline model provided by
    [Mathias et. al.](http://markusmathias.bitbucket.org/2014_eccv_face_detection/).
    Training code is also wrapped, but requires explicit negative samples.

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

|  CI Host |                 OS                |                      Build Status                     |
|:--------:|:---------------------------------:|:-----------------------------------------------------:|
| Travis   | Ubuntu 12.04 (x64)                | [![Travis Build Status][travis_shield]][travis]       |
| Jenkins  | OSX 10.10 (x64)                   | [![Jenkins Build Status][jenkins_shield]][jenkins]    |
| Appveyor | Windows Server 2012 R2 (x86, x64) | [![Appveyor Build Status][appveyor_shield]][appveyor] |


[travis]: https://travis-ci.org/menpo/menpodetect
[travis_shield]: http://img.shields.io/travis/menpo/menpodetect.svg?style=flat
[appveyor]: https://ci.appveyor.com/project/jabooth/menpodetect
[appveyor_shield]: https://ci.appveyor.com/api/projects/status/github/menpo/menpodetect?svg=true
[jenkins]: http://jenkins.menpo.org/view/menpo/job/menpodetect
[jenkins_shield]: http://jenkins.menpo.org/buildStatus/icon?job=menpodetect
