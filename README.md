GulliView
=============

C++ port of the APRIL tags library, using OpenCV (and optionally, CGAL).

Original author: Edwin Olson <ebolson@umich.edu>
C++ port and modifications: Matt Zucker <mzucker1@swarthmore.edu>

Code has been modified for Vision Based Localization of Autonomous
Vehicles in the Gulliver Project at Chalmers University, Sweden

Modification Authors:
Andrew Soderberg-Rivkin <sandrew@student.chalmers.se>
Sanjana Hangal <sanjana@student.chalmers.se>

Helper files and other functions have also been merged with the original
project to help with certain functionalities:

Copyright (c) 2010  Chen Feng (cforrest (at) umich.edu)

	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation; either version 2 of the License, or
	(at your option) any later version.

More information can be found at: https://code.google.com/p/cv2cg/

Requirements
============

Please be sure that you have the latest update to your Linux system
and that all build-essentials are installed.

GulliView requires the following to build:

  * OpenCV >= 2.3 (2.4.8 is now out and stable)
  * GLUT or freeglut (freeglut3-dev)
  * Cairo (libcairo2-dev)
	* For creating tags (note: margins set to -10 for A4 paper)
  * 32bit libraries (if using 64-bit machine --> la32-libs)
  * Boost version 1.49 (sudo apt-get install libboost-all-dev)
	* Used for sending Tag IDs, coordinates and timestamp to
	  server over UDP

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

Running GulliView
=================

Simply clone visionlocalization from github directory 
(git clone https://username@bitbucket.org/thpe/visionlocalization.git) 
to a local directory of your choice. Note: Change "username" to username 
given to you to access repository. Follow building section to build all
executables. 

* cd /path/to/visionlocalization/build
* ./GulliViewServer (sudo if access is denied for binding process)
* in another terminal: ./GulliView -f Tag16h5 (or tag family of your choice)
* Start detecting and sending info to UDP server :)
