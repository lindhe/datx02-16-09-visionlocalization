# swatbotics/apriltags-cpp/CMakeLists.txt

cmake_minimum_required(VERSION 2.6)

project(APRILTAGS)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

if(APPLE)
  include_directories(/opt/local/include) # MacPorts
  link_directories(/opt/local/lib)
  find_library(OPENGL_LIBRARY OpenGL)
else()
  find_library(OPENGL_LIBRARY GL)
  find_library(GLU_LIBRARY GLU)
  set(OPENGL_LIBRARY ${OPENGL_LIBRARY} ${GLU_LIBRARY})
endif()

find_package(Boost 1.49 COMPONENTS program_options system REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

find_library(GLUT_LIBRARY glut)

include(FindPkgConfig)

pkg_search_module(OPENCV REQUIRED opencv>=2.3 opencv-2.3.1)
include_directories(${OPENCV_INCLUDE_DIRS})

pkg_check_modules(CAIRO cairo)

if (${CAIRO_FOUND})
add_definitions(-DMZ_HAVE_CAIRO)
endif ()

#find_package( CGAL QUIET COMPONENTS  )

#if (CGAL_FOUND)
 # include( ${CGAL_USE_FILE} )
  #add_definitions(-DHAVE_CGAL)
  #find_package( Boost REQUIRED )
#else()
 # message("CGAL not found, can not use new quad detector")
#endif()

set(CMAKE_C_FLAGS "-Wall -Wpedantic -g -I /usr/include/boost -l boost_system -pthread")
set(CMAKE_CXX_FLAGS "-Wall -Wpedantic -g -I /usr/include/boost -l boost_system -pthread")

if (APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wsign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wsign-compare")
endif (APPLE)

set(CMAKE_C_FLAGS_DEBUG "-O")
set(CMAKE_CXX_FLAGS_DEBUG "-O")

set(CMAKE_C_FLAGS_RELEASE "-Ofast -march=native -funroll-loops -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -funroll-loops -DNDEBUG")

add_subdirectory(src)

include_directories(/usr/include/)
target_link_libraries(/usr/lib/libueye_api64.so)
