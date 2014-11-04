menpodetect - Simple object detection
=====================================
Simple object detection within the Menpo project environment. We do not attempt 
to implement novel techniques, but instead wrap existing projects so that they 
integrate nicely with Menpo. At the moment the current libraries are wrapped:

  - **[dlib](http://dlib.net/) (Boost Software License - Version 1.0)**  
    Only the frontal face detection code is currently wrapped.

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
