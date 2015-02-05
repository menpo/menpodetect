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
  - **[pico](https://github.com/nenadmarkus/pico)(Academic Only)**
    Frontal face detection and arbitrary pico models are provided. Loading
    arbitrary Pico models is likely to be very difficult, however.

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
to satisfy the following dependencies BEFORE install:

  - numpy 1.9*
  - dlib  18.13
  - opencv 2.4.9*
  - cypico 0.2.1
  - menpo 0.4*
