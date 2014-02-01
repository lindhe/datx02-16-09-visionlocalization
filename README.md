GulliView
=============

C++ port of the APRIL tags library, using OpenCV (and optionally, CGAL).

Code has been modified for Vision Based Localization of Autonomous
Vehicles in the Gulliver Project at Chalmers University, Sweden

Modification Authors:
Andrew Soderberg-Rivkin <sandrew@student.chalmers.se>
Sanjana Hangal <sanjana@student.chalmers.se>

Requirements
============

Please be sure that you have the latest update to your Linux system
and that all build-essentials are installed.

GulliView requires the following to build:

  * OpenCV >= 2.3 (2.4.8 is now out and stable)
  * GLUT or freeglut (freeglut3-dev)
  * Cairo (libcairo2-dev)
  * 32bit libraries (if using 64-bit machine --> la32-libs)

You must have cmake installed to build the software as well.

Building
========

To compile the code, 

    cd /path/to/visionlocalization
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make

Demo/utility programs
=====================

The APRIL tags library is intended to be used as a library, from C++,
but there are also three demo/utility programs included in this
distribution:


   *   `gltest` - Demonstrate 3D tag locations using OpenGL to
       visualize, with an attached camera.

   *   `quadtest` - Demonstrate/test tag position refinement using
       a template tracking approach.

   *   `maketags` - Create PDF files for printing tags.
